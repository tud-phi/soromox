# ruff: noqa: I001, UP018
"""Cooperative CUDA kernels for fused linear-threadlike PCS forces.

Each 32-lane block owns one environment-segment pair and prepares its strain
in shared memory. Medium layouts distribute path-point pairs across lanes;
large layouts distribute complete paths to reduce FP64 atomic traffic. The
launchers operate on caller-owned arrays without allocation, synchronization,
JAX calls, or device transfers, and remain CUDA-graph-capturable.
"""

from __future__ import annotations

import warp as wp

from soromox.execution.warp.common.se3 import Vec6d

wp.set_module_options({"enable_backward": False})

PCS_THREADLIKE_FORCE_BLOCK_DIM = 32
PCS_THREADLIKE_SERIAL_MAX_PATHS = 4
PCS_THREADLIKE_PATH_POINT_MAX_PATHS = 32


@wp.func
def _planar_contribution(
    strain: wp.tile(dtype=wp.float64, shape=(3,)),
    segment: int,
    point: int,
    path: int,
    physical_points: wp.array2d[wp.float64],
    physical_weights: wp.array2d[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    epsilons: wp.array[wp.float64],
) -> wp.vec3d:
    """Evaluate one weighted PlanarPCS path-point contribution."""
    s = physical_points[segment, point]
    offset = routing_intercepts[path, 1] + s * routing_slopes[path, 1]
    axial = strain[1] + offset * strain[0]
    shear = strain[2] + routing_slopes[path, 1]
    norm = wp.sqrt(axial * axial + shear * shear)
    axial_ratio = wp.float64(0.0)
    shear_ratio = wp.float64(0.0)
    if norm > epsilons[0]:
        axial_ratio = axial / norm
        shear_ratio = shear / norm
    weight = physical_weights[segment, point]
    return wp.vec3d(
        weight * offset * axial_ratio,
        weight * axial_ratio,
        weight * shear_ratio,
    )


@wp.func
def _spatial_contribution(
    strain: wp.tile(dtype=wp.float64, shape=(6,)),
    segment: int,
    point: int,
    path: int,
    physical_points: wp.array2d[wp.float64],
    physical_weights: wp.array2d[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    epsilons: wp.array[wp.float64],
) -> Vec6d:
    """Evaluate one weighted spatial PCS path-point contribution."""
    s = physical_points[segment, point]
    offset = wp.vec3d(
        routing_intercepts[path, 0] + s * routing_slopes[path, 0],
        routing_intercepts[path, 1] + s * routing_slopes[path, 1],
        routing_intercepts[path, 2] + s * routing_slopes[path, 2],
    )
    omega = wp.vec3d(strain[0], strain[1], strain[2])
    linear = wp.vec3d(strain[3], strain[4], strain[5])
    slope = wp.vec3d(
        routing_slopes[path, 0],
        routing_slopes[path, 1],
        routing_slopes[path, 2],
    )
    tangent_raw = slope + wp.cross(omega, offset) + linear
    norm = wp.length(tangent_raw)
    tangent = wp.vec3d()
    if norm > epsilons[0]:
        tangent = tangent_raw / norm
    moment = wp.cross(offset, tangent)
    weight = physical_weights[segment, point]
    return Vec6d(
        weight * moment[0],
        weight * moment[1],
        weight * moment[2],
        weight * tangent[0],
        weight * tangent[1],
        weight * tangent[2],
    )


@wp.kernel(enable_backward=False)
def planar_threadlike_force_path_kernel(
    q: wp.array2d[wp.float64],
    controls: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    physical_points: wp.array2d[wp.float64],
    physical_weights: wp.array2d[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    path_start_segments: wp.array[wp.int32],
    path_end_segments: wp.array[wp.int32],
    coordinate_scales: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    output: wp.array2d[wp.float64],
):
    """Reduce complete PlanarPCS paths within one segment-owned block."""
    work_item, lane = wp.tid()
    num_segments = active_indices.shape[0]
    segment = work_item % num_segments
    environment = work_item // num_segments
    strain = wp.tile_zeros(shape=(3,), dtype=wp.float64, storage="shared")

    row = lane
    value = wp.float64(0.0)
    if row < 3:
        value = reference_strain[segment, row]
        active = active_indices[segment, row]
        if active >= 0:
            value += active_scales[segment, row] * q[environment, active]
            output[environment, active] = wp.float64(0.0)
    wp.tile_scatter_masked(strain, row, value, row < 3)

    path = lane
    while path < routing_intercepts.shape[0]:
        if segment >= path_start_segments[path] and segment <= path_end_segments[path]:
            integrated = wp.vec3d()
            point = int(0)
            while point < physical_points.shape[1]:
                integrated += _planar_contribution(
                    strain,
                    segment,
                    point,
                    path,
                    physical_points,
                    physical_weights,
                    routing_intercepts,
                    routing_slopes,
                    epsilons,
                )
                point += 1
            effort = coordinate_scales[path] * controls[environment, path]
            row = int(0)
            while row < 3:
                active = active_indices[segment, row]
                if active >= 0:
                    wp.atomic_add(
                        output,
                        environment,
                        active,
                        active_scales[segment, row] * effort * integrated[row],
                    )
                row += 1
        path += wp.block_dim()


@wp.kernel(enable_backward=False)
def spatial_threadlike_force_path_kernel(
    q: wp.array2d[wp.float64],
    controls: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    physical_points: wp.array2d[wp.float64],
    physical_weights: wp.array2d[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    path_start_segments: wp.array[wp.int32],
    path_end_segments: wp.array[wp.int32],
    coordinate_scales: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    output: wp.array2d[wp.float64],
):
    """Reduce complete spatial PCS paths within one segment-owned block."""
    work_item, lane = wp.tid()
    num_segments = active_indices.shape[0]
    segment = work_item % num_segments
    environment = work_item // num_segments
    strain = wp.tile_zeros(shape=(6,), dtype=wp.float64, storage="shared")

    row = lane
    value = wp.float64(0.0)
    if row < 6:
        value = reference_strain[segment, row]
        active = active_indices[segment, row]
        if active >= 0:
            value += active_scales[segment, row] * q[environment, active]
            output[environment, active] = wp.float64(0.0)
    wp.tile_scatter_masked(strain, row, value, row < 6)

    path = lane
    while path < routing_intercepts.shape[0]:
        if segment >= path_start_segments[path] and segment <= path_end_segments[path]:
            integrated = Vec6d()
            point = int(0)
            while point < physical_points.shape[1]:
                integrated += _spatial_contribution(
                    strain,
                    segment,
                    point,
                    path,
                    physical_points,
                    physical_weights,
                    routing_intercepts,
                    routing_slopes,
                    epsilons,
                )
                point += 1
            effort = coordinate_scales[path] * controls[environment, path]
            row = int(0)
            while row < 6:
                active = active_indices[segment, row]
                if active >= 0:
                    wp.atomic_add(
                        output,
                        environment,
                        active,
                        active_scales[segment, row] * effort * integrated[row],
                    )
                row += 1
        path += wp.block_dim()


@wp.kernel(enable_backward=False)
def planar_threadlike_force_path_point_kernel(
    q: wp.array2d[wp.float64],
    controls: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    physical_points: wp.array2d[wp.float64],
    physical_weights: wp.array2d[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    path_start_segments: wp.array[wp.int32],
    path_end_segments: wp.array[wp.int32],
    coordinate_scales: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    output: wp.array2d[wp.float64],
):
    """Reduce PlanarPCS path-point pairs within one segment-owned block."""
    work_item, lane = wp.tid()
    num_segments = active_indices.shape[0]
    segment = work_item % num_segments
    environment = work_item // num_segments
    strain = wp.tile_zeros(shape=(3,), dtype=wp.float64, storage="shared")

    row = lane
    value = wp.float64(0.0)
    if row < 3:
        value = reference_strain[segment, row]
        active = active_indices[segment, row]
        if active >= 0:
            value += active_scales[segment, row] * q[environment, active]
            output[environment, active] = wp.float64(0.0)
    wp.tile_scatter_masked(strain, row, value, row < 3)

    num_points = physical_points.shape[1]
    path_point = lane
    while path_point < routing_intercepts.shape[0] * num_points:
        path = path_point // num_points
        point = path_point - path * num_points
        if segment >= path_start_segments[path] and segment <= path_end_segments[path]:
            contribution = _planar_contribution(
                strain,
                segment,
                point,
                path,
                physical_points,
                physical_weights,
                routing_intercepts,
                routing_slopes,
                epsilons,
            )
            effort = coordinate_scales[path] * controls[environment, path]
            row = int(0)
            while row < 3:
                active = active_indices[segment, row]
                if active >= 0:
                    wp.atomic_add(
                        output,
                        environment,
                        active,
                        active_scales[segment, row] * effort * contribution[row],
                    )
                row += 1
        path_point += wp.block_dim()


@wp.kernel(enable_backward=False)
def spatial_threadlike_force_path_point_kernel(
    q: wp.array2d[wp.float64],
    controls: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    physical_points: wp.array2d[wp.float64],
    physical_weights: wp.array2d[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    path_start_segments: wp.array[wp.int32],
    path_end_segments: wp.array[wp.int32],
    coordinate_scales: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    output: wp.array2d[wp.float64],
):
    """Reduce spatial PCS path-point pairs within one segment-owned block."""
    work_item, lane = wp.tid()
    num_segments = active_indices.shape[0]
    segment = work_item % num_segments
    environment = work_item // num_segments
    strain = wp.tile_zeros(shape=(6,), dtype=wp.float64, storage="shared")

    row = lane
    value = wp.float64(0.0)
    if row < 6:
        value = reference_strain[segment, row]
        active = active_indices[segment, row]
        if active >= 0:
            value += active_scales[segment, row] * q[environment, active]
            output[environment, active] = wp.float64(0.0)
    wp.tile_scatter_masked(strain, row, value, row < 6)

    num_points = physical_points.shape[1]
    path_point = lane
    while path_point < routing_intercepts.shape[0] * num_points:
        path = path_point // num_points
        point = path_point - path * num_points
        if segment >= path_start_segments[path] and segment <= path_end_segments[path]:
            contribution = _spatial_contribution(
                strain,
                segment,
                point,
                path,
                physical_points,
                physical_weights,
                routing_intercepts,
                routing_slopes,
                epsilons,
            )
            effort = coordinate_scales[path] * controls[environment, path]
            row = int(0)
            while row < 6:
                active = active_indices[segment, row]
                if active >= 0:
                    wp.atomic_add(
                        output,
                        environment,
                        active,
                        active_scales[segment, row] * effort * contribution[row],
                    )
                row += 1
        path_point += wp.block_dim()


def _launch(
    kernel,
    q,
    controls,
    active_indices,
    active_scales,
    reference_strain,
    physical_points,
    physical_weights,
    routing_intercepts,
    routing_slopes,
    path_start_segments,
    path_end_segments,
    coordinate_scales,
    epsilons,
    output,
):
    """Enqueue one fixed 32-lane cooperative force launch."""
    wp.launch_tiled(
        kernel,
        dim=q.shape[0] * active_indices.shape[0],
        inputs=[
            q,
            controls,
            active_indices,
            active_scales,
            reference_strain,
            physical_points,
            physical_weights,
            routing_intercepts,
            routing_slopes,
            path_start_segments,
            path_end_segments,
            coordinate_scales,
            epsilons,
        ],
        outputs=[output],
        block_dim=PCS_THREADLIKE_FORCE_BLOCK_DIM,
    )


def launch_planar_threadlike_force_path(
    q: wp.array2d[wp.float64],
    controls: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    physical_points: wp.array2d[wp.float64],
    physical_weights: wp.array2d[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    path_start_segments: wp.array[wp.int32],
    path_end_segments: wp.array[wp.int32],
    coordinate_scales: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    output: wp.array2d[wp.float64],
):
    """Fill a caller-owned PlanarPCS force using path-parallel reduction."""
    _launch(
        planar_threadlike_force_path_kernel,
        q,
        controls,
        active_indices,
        active_scales,
        reference_strain,
        physical_points,
        physical_weights,
        routing_intercepts,
        routing_slopes,
        path_start_segments,
        path_end_segments,
        coordinate_scales,
        epsilons,
        output,
    )


def launch_planar_threadlike_force_path_point(
    q: wp.array2d[wp.float64],
    controls: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    physical_points: wp.array2d[wp.float64],
    physical_weights: wp.array2d[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    path_start_segments: wp.array[wp.int32],
    path_end_segments: wp.array[wp.int32],
    coordinate_scales: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    output: wp.array2d[wp.float64],
):
    """Fill a caller-owned PlanarPCS force using path-point reduction."""
    _launch(
        planar_threadlike_force_path_point_kernel,
        q,
        controls,
        active_indices,
        active_scales,
        reference_strain,
        physical_points,
        physical_weights,
        routing_intercepts,
        routing_slopes,
        path_start_segments,
        path_end_segments,
        coordinate_scales,
        epsilons,
        output,
    )


def launch_spatial_threadlike_force_path(
    q: wp.array2d[wp.float64],
    controls: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    physical_points: wp.array2d[wp.float64],
    physical_weights: wp.array2d[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    path_start_segments: wp.array[wp.int32],
    path_end_segments: wp.array[wp.int32],
    coordinate_scales: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    output: wp.array2d[wp.float64],
):
    """Fill a caller-owned spatial PCS force using path-parallel reduction."""
    _launch(
        spatial_threadlike_force_path_kernel,
        q,
        controls,
        active_indices,
        active_scales,
        reference_strain,
        physical_points,
        physical_weights,
        routing_intercepts,
        routing_slopes,
        path_start_segments,
        path_end_segments,
        coordinate_scales,
        epsilons,
        output,
    )


def launch_spatial_threadlike_force_path_point(
    q: wp.array2d[wp.float64],
    controls: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    physical_points: wp.array2d[wp.float64],
    physical_weights: wp.array2d[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    path_start_segments: wp.array[wp.int32],
    path_end_segments: wp.array[wp.int32],
    coordinate_scales: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    output: wp.array2d[wp.float64],
):
    """Fill a caller-owned spatial PCS force using path-point reduction."""
    _launch(
        spatial_threadlike_force_path_point_kernel,
        q,
        controls,
        active_indices,
        active_scales,
        reference_strain,
        physical_points,
        physical_weights,
        routing_intercepts,
        routing_slopes,
        path_start_segments,
        path_end_segments,
        coordinate_scales,
        epsilons,
        output,
    )


planar_threadlike_force_path = wp.jax_kernel(
    planar_threadlike_force_path_kernel,
    num_outputs=1,
    block_dim=PCS_THREADLIKE_FORCE_BLOCK_DIM,
)
planar_threadlike_force_path_point = wp.jax_kernel(
    planar_threadlike_force_path_point_kernel,
    num_outputs=1,
    block_dim=PCS_THREADLIKE_FORCE_BLOCK_DIM,
)
spatial_threadlike_force_path = wp.jax_kernel(
    spatial_threadlike_force_path_kernel,
    num_outputs=1,
    block_dim=PCS_THREADLIKE_FORCE_BLOCK_DIM,
)
spatial_threadlike_force_path_point = wp.jax_kernel(
    spatial_threadlike_force_path_point_kernel,
    num_outputs=1,
    block_dim=PCS_THREADLIKE_FORCE_BLOCK_DIM,
)


__all__ = [
    "PCS_THREADLIKE_FORCE_BLOCK_DIM",
    "PCS_THREADLIKE_PATH_POINT_MAX_PATHS",
    "PCS_THREADLIKE_SERIAL_MAX_PATHS",
    "launch_planar_threadlike_force_path",
    "launch_planar_threadlike_force_path_point",
    "launch_spatial_threadlike_force_path",
    "launch_spatial_threadlike_force_path_point",
    "planar_threadlike_force_path_kernel",
    "planar_threadlike_force_path_point_kernel",
    "spatial_threadlike_force_path_kernel",
    "spatial_threadlike_force_path_point_kernel",
]
