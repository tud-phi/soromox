# ruff: noqa: I001, UP018
"""Allocation-free Warp kernels for linear threadlike GVS actuation.

Each launcher first clears its caller-owned result, then integrates link-local
threadlike moments and maps link coordinates into the global active space.
Joint-only coordinates therefore remain zero. The sequence allocates nothing,
does not call JAX, synchronize, or transfer data, and is CUDA-graph-capturable.
"""

from __future__ import annotations

import warp as wp

from soromox.systems.execution.warp.common.se3 import Vec6d
from soromox.systems.execution.warp.actuation.kernels import (
    clear_threadlike_force_kernel,
    clear_threadlike_matrix_kernel,
)

wp.set_module_options({"enable_backward": False})


@wp.func
def _gvs_local_basis(
    q: wp.array2d[wp.float64],
    environment: int,
    segment: int,
    point: int,
    path: int,
    link_local_to_global: wp.array2d[wp.int32],
    link_basis: wp.array4d[wp.float64],
    reference_strain: wp.array3d[wp.float64],
    integration_points: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    options: wp.array[wp.int32],
    epsilons: wp.array[wp.float64],
) -> Vec6d:
    """Evaluate the routed local moment basis at one GVS quadrature point."""
    length = segment_lengths[segment]
    strain = Vec6d()
    row = int(0)
    while row < 6:
        value = reference_strain[segment, point, row]
        local = int(0)
        while local < link_local_to_global.shape[1]:
            global_dof = link_local_to_global[segment, local]
            if global_dof >= 0:
                basis_value = link_basis[segment, point, row, local]
                if options[0] != 0 and row < 3:
                    basis_value = basis_value / length
                value += basis_value * q[environment, global_dof]
            local += 1
        strain[row] = value
        row += 1

    s = segment_starts[segment] + integration_points[segment, point] * length
    offset = wp.vec3d(
        routing_intercepts[path, 0] + s * routing_slopes[path, 0],
        routing_intercepts[path, 1] + s * routing_slopes[path, 1],
        routing_intercepts[path, 2] + s * routing_slopes[path, 2],
    )
    slope = wp.vec3d(
        routing_slopes[path, 0],
        routing_slopes[path, 1],
        routing_slopes[path, 2],
    )
    omega = wp.vec3d(strain[0], strain[1], strain[2])
    linear = wp.vec3d(strain[3], strain[4], strain[5])
    tangent_raw = slope + wp.cross(omega, offset) + linear
    norm = wp.length(tangent_raw)
    tangent = wp.vec3d()
    if norm > epsilons[0]:
        tangent = tangent_raw / norm
    moment = wp.cross(offset, tangent)
    return Vec6d(
        moment[0],
        moment[1],
        moment[2],
        tangent[0],
        tangent[1],
        tangent[2],
    )


@wp.kernel(enable_backward=False)
def gvs_threadlike_matrix_kernel(
    q: wp.array2d[wp.float64],
    link_local_to_global: wp.array2d[wp.int32],
    link_basis: wp.array4d[wp.float64],
    reference_strain: wp.array3d[wp.float64],
    integration_points: wp.array2d[wp.float64],
    integration_weights: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    path_start_segments: wp.array[wp.int32],
    path_end_segments: wp.array[wp.int32],
    coordinate_scales: wp.array[wp.float64],
    options: wp.array[wp.int32],
    epsilons: wp.array[wp.float64],
    output: wp.array3d[wp.float64],
):
    """Integrate one environment-segment-path-local-coordinate entry."""
    item = wp.tid()
    max_dof = link_local_to_global.shape[1]
    num_paths = routing_intercepts.shape[0]
    num_segments = link_local_to_global.shape[0]
    local = item % max_dof
    item = item // max_dof
    path = item % num_paths
    item = item // num_paths
    segment = item % num_segments
    environment = item // num_segments
    global_dof = link_local_to_global[segment, local]
    if global_dof < 0:
        return
    if segment < path_start_segments[path] or segment > path_end_segments[path]:
        return

    value = wp.float64(0.0)
    point = int(0)
    while point < integration_points.shape[1]:
        local_moment = _gvs_local_basis(
            q,
            environment,
            segment,
            point,
            path,
            link_local_to_global,
            link_basis,
            reference_strain,
            integration_points,
            segment_lengths,
            segment_starts,
            routing_intercepts,
            routing_slopes,
            options,
            epsilons,
        )
        row = int(0)
        projected = wp.float64(0.0)
        while row < 6:
            basis_value = link_basis[segment, point, row, local]
            if options[0] != 0 and row < 3:
                basis_value = basis_value / segment_lengths[segment]
            projected += basis_value * local_moment[row]
            row += 1
        value += integration_weights[segment, point] * projected
        point += 1
    output[environment, global_dof, path] = coordinate_scales[path] * value


@wp.kernel(enable_backward=False)
def gvs_threadlike_force_kernel(
    q: wp.array2d[wp.float64],
    controls: wp.array2d[wp.float64],
    link_local_to_global: wp.array2d[wp.int32],
    link_basis: wp.array4d[wp.float64],
    reference_strain: wp.array3d[wp.float64],
    integration_points: wp.array2d[wp.float64],
    integration_weights: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    path_start_segments: wp.array[wp.int32],
    path_end_segments: wp.array[wp.int32],
    coordinate_scales: wp.array[wp.float64],
    options: wp.array[wp.int32],
    epsilons: wp.array[wp.float64],
    output: wp.array2d[wp.float64],
):
    """Fuse GVS local path integration with direct-effort reduction."""
    item = wp.tid()
    max_dof = link_local_to_global.shape[1]
    num_segments = link_local_to_global.shape[0]
    local = item % max_dof
    item = item // max_dof
    segment = item % num_segments
    environment = item // num_segments
    global_dof = link_local_to_global[segment, local]
    if global_dof < 0:
        return

    value = wp.float64(0.0)
    path = int(0)
    while path < routing_intercepts.shape[0]:
        if segment >= path_start_segments[path] and segment <= path_end_segments[path]:
            effort = coordinate_scales[path] * controls[environment, path]
            point = int(0)
            while point < integration_points.shape[1]:
                local_moment = _gvs_local_basis(
                    q,
                    environment,
                    segment,
                    point,
                    path,
                    link_local_to_global,
                    link_basis,
                    reference_strain,
                    integration_points,
                    segment_lengths,
                    segment_starts,
                    routing_intercepts,
                    routing_slopes,
                    options,
                    epsilons,
                )
                row = int(0)
                projected = wp.float64(0.0)
                while row < 6:
                    basis_value = link_basis[segment, point, row, local]
                    if options[0] != 0 and row < 3:
                        basis_value = basis_value / segment_lengths[segment]
                    projected += basis_value * local_moment[row]
                    row += 1
                value += integration_weights[segment, point] * projected * effort
                point += 1
        path += 1
    output[environment, global_dof] = value


def launch_gvs_threadlike_actuation_matrix(
    q: wp.array2d[wp.float64],
    link_local_to_global: wp.array2d[wp.int32],
    link_basis: wp.array4d[wp.float64],
    reference_strain: wp.array3d[wp.float64],
    integration_points: wp.array2d[wp.float64],
    integration_weights: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    path_start_segments: wp.array[wp.int32],
    path_end_segments: wp.array[wp.int32],
    coordinate_scales: wp.array[wp.float64],
    options: wp.array[wp.int32],
    epsilons: wp.array[wp.float64],
    output: wp.array3d[wp.float64],
):
    """Clear then fill caller-owned GVS matrix output ``(E, D, A)``."""
    wp.launch(
        clear_threadlike_matrix_kernel,
        dim=output.shape[0] * output.shape[1] * output.shape[2],
        outputs=[output],
    )
    wp.launch(
        gvs_threadlike_matrix_kernel,
        dim=(
            q.shape[0]
            * link_local_to_global.shape[0]
            * routing_intercepts.shape[0]
            * link_local_to_global.shape[1]
        ),
        inputs=[
            q,
            link_local_to_global,
            link_basis,
            reference_strain,
            integration_points,
            integration_weights,
            segment_lengths,
            segment_starts,
            routing_intercepts,
            routing_slopes,
            path_start_segments,
            path_end_segments,
            coordinate_scales,
            options,
            epsilons,
        ],
        outputs=[output],
        block_dim=128,
    )


def launch_gvs_threadlike_actuation_force(
    q: wp.array2d[wp.float64],
    controls: wp.array2d[wp.float64],
    link_local_to_global: wp.array2d[wp.int32],
    link_basis: wp.array4d[wp.float64],
    reference_strain: wp.array3d[wp.float64],
    integration_points: wp.array2d[wp.float64],
    integration_weights: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    path_start_segments: wp.array[wp.int32],
    path_end_segments: wp.array[wp.int32],
    coordinate_scales: wp.array[wp.float64],
    options: wp.array[wp.int32],
    epsilons: wp.array[wp.float64],
    output: wp.array2d[wp.float64],
):
    """Clear then fill caller-owned fused GVS force output ``(E, D)``."""
    wp.launch(
        clear_threadlike_force_kernel,
        dim=output.shape[0] * output.shape[1],
        outputs=[output],
    )
    wp.launch(
        gvs_threadlike_force_kernel,
        dim=(
            q.shape[0]
            * link_local_to_global.shape[0]
            * link_local_to_global.shape[1]
        ),
        inputs=[
            q,
            controls,
            link_local_to_global,
            link_basis,
            reference_strain,
            integration_points,
            integration_weights,
            segment_lengths,
            segment_starts,
            routing_intercepts,
            routing_slopes,
            path_start_segments,
            path_end_segments,
            coordinate_scales,
            options,
            epsilons,
        ],
        outputs=[output],
        block_dim=128,
    )


gvs_threadlike_actuation_matrix = wp.jax_callable(
    launch_gvs_threadlike_actuation_matrix, num_outputs=1
)
gvs_threadlike_actuation_force = wp.jax_callable(
    launch_gvs_threadlike_actuation_force, num_outputs=1
)


__all__ = [
    "gvs_threadlike_force_kernel",
    "gvs_threadlike_matrix_kernel",
    "launch_gvs_threadlike_actuation_force",
    "launch_gvs_threadlike_actuation_matrix",
]
