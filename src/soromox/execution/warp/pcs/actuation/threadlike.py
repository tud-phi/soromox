# ruff: noqa: I001, UP018
"""Allocation-free Warp kernels for linear threadlike PCS actuation.

The public launchers enqueue one write-complete integration kernel. Physical
quadrature coordinates and weights are prepared in the operand bundle, and
inactive path/segment entries are written as exact zeros. The launchers
allocate no arrays, call no JAX functions, perform no synchronization, and
transfer no data, so callers may capture them in a CUDA graph.
"""

from __future__ import annotations

import warp as wp

from soromox.execution.warp.common.se3 import Vec6d
from soromox.execution.warp.pcs.actuation.cooperative import (
    PCS_THREADLIKE_PATH_POINT_MAX_PATHS,
    PCS_THREADLIKE_SERIAL_MAX_PATHS,
    launch_planar_threadlike_force_path,
    launch_planar_threadlike_force_path_point,
    launch_spatial_threadlike_force_path,
    launch_spatial_threadlike_force_path_point,
)

wp.set_module_options({"enable_backward": False})


@wp.kernel(enable_backward=False)
def planar_threadlike_matrix_kernel(
    q: wp.array2d[wp.float64],
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
    output: wp.array3d[wp.float64],
):
    """Integrate one PlanarPCS environment-segment-path contribution."""
    item = wp.tid()
    num_paths = routing_intercepts.shape[0]
    num_segments = active_indices.shape[0]
    path = item % num_paths
    item = item // num_paths
    segment = item % num_segments
    environment = item // num_segments
    active_path = (
        segment >= path_start_segments[path] and segment <= path_end_segments[path]
    )

    strain = wp.vec3d()
    row = int(0)
    while row < 3:
        strain[row] = reference_strain[segment, row]
        active = active_indices[segment, row]
        if active >= 0:
            strain[row] += active_scales[segment, row] * q[environment, active]
        row += 1

    integrated = wp.vec3d()
    point = int(0)
    while point < physical_points.shape[1]:
        if active_path:
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
            integrated[0] += weight * offset * axial_ratio
            integrated[1] += weight * axial_ratio
            integrated[2] += weight * shear_ratio
        point += 1

    scale = coordinate_scales[path]
    row = int(0)
    while row < 3:
        active = active_indices[segment, row]
        if active >= 0:
            output[environment, active, path] = (
                active_scales[segment, row] * scale * integrated[row]
            )
        row += 1


@wp.kernel(enable_backward=False)
def planar_threadlike_force_kernel(
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
    """Fuse PlanarPCS path integration and direct-effort reduction."""
    item = wp.tid()
    num_segments = active_indices.shape[0]
    segment = item % num_segments
    environment = item // num_segments

    strain = wp.vec3d()
    row = int(0)
    while row < 3:
        strain[row] = reference_strain[segment, row]
        active = active_indices[segment, row]
        if active >= 0:
            strain[row] += active_scales[segment, row] * q[environment, active]
        row += 1

    accumulated = wp.vec3d()
    path = int(0)
    while path < routing_intercepts.shape[0]:
        if segment >= path_start_segments[path] and segment <= path_end_segments[path]:
            effort = coordinate_scales[path] * controls[environment, path]
            point = int(0)
            while point < physical_points.shape[1]:
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
                weight = physical_weights[segment, point] * effort
                accumulated[0] += weight * offset * axial_ratio
                accumulated[1] += weight * axial_ratio
                accumulated[2] += weight * shear_ratio
                point += 1
        path += 1

    row = int(0)
    while row < 3:
        active = active_indices[segment, row]
        if active >= 0:
            output[environment, active] = active_scales[segment, row] * accumulated[row]
        row += 1


@wp.kernel(enable_backward=False)
def spatial_threadlike_matrix_kernel(
    q: wp.array2d[wp.float64],
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
    output: wp.array3d[wp.float64],
):
    """Integrate one spatial PCS environment-segment-path contribution."""
    item = wp.tid()
    num_paths = routing_intercepts.shape[0]
    num_segments = active_indices.shape[0]
    path = item % num_paths
    item = item // num_paths
    segment = item % num_segments
    environment = item // num_segments
    active_path = (
        segment >= path_start_segments[path] and segment <= path_end_segments[path]
    )

    strain = Vec6d()
    row = int(0)
    while row < 6:
        strain[row] = reference_strain[segment, row]
        active = active_indices[segment, row]
        if active >= 0:
            strain[row] += active_scales[segment, row] * q[environment, active]
        row += 1

    integrated = Vec6d()
    point = int(0)
    while point < physical_points.shape[1]:
        if active_path:
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
            integrated[0] += weight * moment[0]
            integrated[1] += weight * moment[1]
            integrated[2] += weight * moment[2]
            integrated[3] += weight * tangent[0]
            integrated[4] += weight * tangent[1]
            integrated[5] += weight * tangent[2]
        point += 1

    scale = coordinate_scales[path]
    row = int(0)
    while row < 6:
        active = active_indices[segment, row]
        if active >= 0:
            output[environment, active, path] = (
                active_scales[segment, row] * scale * integrated[row]
            )
        row += 1


@wp.kernel(enable_backward=False)
def spatial_threadlike_force_kernel(
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
    """Fuse spatial PCS path integration and direct-effort reduction."""
    item = wp.tid()
    num_segments = active_indices.shape[0]
    segment = item % num_segments
    environment = item // num_segments

    strain = Vec6d()
    row = int(0)
    while row < 6:
        strain[row] = reference_strain[segment, row]
        active = active_indices[segment, row]
        if active >= 0:
            strain[row] += active_scales[segment, row] * q[environment, active]
        row += 1

    accumulated = Vec6d()
    path = int(0)
    while path < routing_intercepts.shape[0]:
        if segment >= path_start_segments[path] and segment <= path_end_segments[path]:
            effort = coordinate_scales[path] * controls[environment, path]
            point = int(0)
            while point < physical_points.shape[1]:
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
                weight = physical_weights[segment, point] * effort
                accumulated[0] += weight * moment[0]
                accumulated[1] += weight * moment[1]
                accumulated[2] += weight * moment[2]
                accumulated[3] += weight * tangent[0]
                accumulated[4] += weight * tangent[1]
                accumulated[5] += weight * tangent[2]
                point += 1
        path += 1

    row = int(0)
    while row < 6:
        active = active_indices[segment, row]
        if active >= 0:
            output[environment, active] = active_scales[segment, row] * accumulated[row]
        row += 1


def _launch_matrix(
    kernel,
    q,
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
    """Launch one write-complete dimension-specific matrix kernel."""
    wp.launch(
        kernel,
        dim=q.shape[0] * active_indices.shape[0] * routing_intercepts.shape[0],
        inputs=[
            q,
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
        block_dim=128,
    )


def _launch_force(
    serial_kernel,
    path_launcher,
    path_point_launcher,
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
    """Launch the static-layout-selected dimension-specific force kernel."""
    inputs = [
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
    ]
    num_paths = routing_intercepts.shape[0]
    if not q.device.is_cuda or num_paths <= PCS_THREADLIKE_SERIAL_MAX_PATHS:
        wp.launch(
            serial_kernel,
            dim=q.shape[0] * active_indices.shape[0],
            inputs=inputs,
            outputs=[output],
            block_dim=128,
        )
    elif num_paths <= PCS_THREADLIKE_PATH_POINT_MAX_PATHS:
        path_point_launcher(*inputs, output)
    else:
        path_launcher(*inputs, output)


def launch_planar_threadlike_actuation_matrix(
    q: wp.array2d[wp.float64],
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
    output: wp.array3d[wp.float64],
):
    """Fill a caller-owned ``(E, D, A)`` PlanarPCS matrix."""
    _launch_matrix(
        planar_threadlike_matrix_kernel,
        q,
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


def launch_spatial_threadlike_actuation_matrix(
    q: wp.array2d[wp.float64],
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
    output: wp.array3d[wp.float64],
):
    """Fill a caller-owned ``(E, D, A)`` spatial PCS matrix."""
    _launch_matrix(
        spatial_threadlike_matrix_kernel,
        q,
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


def launch_planar_threadlike_actuation_force(
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
    """Fill a caller-owned ``(E, D)`` PlanarPCS fused force."""
    _launch_force(
        planar_threadlike_force_kernel,
        launch_planar_threadlike_force_path,
        launch_planar_threadlike_force_path_point,
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


def launch_spatial_threadlike_actuation_force(
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
    """Fill a caller-owned ``(E, D)`` spatial PCS fused force."""
    _launch_force(
        spatial_threadlike_force_kernel,
        launch_spatial_threadlike_force_path,
        launch_spatial_threadlike_force_path_point,
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


planar_threadlike_actuation_matrix = wp.jax_callable(
    launch_planar_threadlike_actuation_matrix, num_outputs=1
)
spatial_threadlike_actuation_matrix = wp.jax_callable(
    launch_spatial_threadlike_actuation_matrix, num_outputs=1
)
planar_threadlike_actuation_force = wp.jax_callable(
    launch_planar_threadlike_actuation_force,
    num_outputs=1,
    module_preload_mode=wp.JaxModulePreloadMode.NONE,
)
spatial_threadlike_actuation_force = wp.jax_callable(
    launch_spatial_threadlike_actuation_force,
    num_outputs=1,
    module_preload_mode=wp.JaxModulePreloadMode.NONE,
)
planar_threadlike_matrix = wp.jax_kernel(
    planar_threadlike_matrix_kernel,
    num_outputs=1,
    block_dim=128,
)
spatial_threadlike_matrix = wp.jax_kernel(
    spatial_threadlike_matrix_kernel,
    num_outputs=1,
    block_dim=128,
)
planar_threadlike_force = wp.jax_kernel(
    planar_threadlike_force_kernel,
    num_outputs=1,
    block_dim=128,
)
spatial_threadlike_force = wp.jax_kernel(
    spatial_threadlike_force_kernel,
    num_outputs=1,
    block_dim=128,
)


__all__ = [
    "PCS_THREADLIKE_PATH_POINT_MAX_PATHS",
    "PCS_THREADLIKE_SERIAL_MAX_PATHS",
    "launch_planar_threadlike_actuation_force",
    "launch_planar_threadlike_actuation_matrix",
    "launch_spatial_threadlike_actuation_force",
    "launch_spatial_threadlike_actuation_matrix",
    "planar_threadlike_force_kernel",
    "planar_threadlike_matrix_kernel",
    "spatial_threadlike_force_kernel",
    "spatial_threadlike_matrix_kernel",
]
