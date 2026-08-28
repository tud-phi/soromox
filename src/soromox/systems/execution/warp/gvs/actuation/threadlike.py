# ruff: noqa: I001, UP018
"""Allocation-free Warp kernels for linear threadlike GVS actuation.

Both public launchers first prepare quadrature-point strains in a caller-owned
``(E, S, Q, 6)`` workspace. Matrix projection then writes every ``(E, D, A)``
entry exactly once. Fused CUDA force execution uses one 64-lane block per
environment and segment to reduce paths in parallel; the CPU fallback writes
one active coordinate per thread. Joint-only coordinates and inactive paths
are written as exact zeros without a separate clear launch.

The fixed two-launch sequences allocate nothing, call no JAX functions,
perform no synchronization or transfer, and are CUDA-graph-capturable.
"""

from __future__ import annotations

import warp as wp

from soromox.systems.execution.warp.common.se3 import Vec6d

wp.set_module_options({"enable_backward": False})

GVS_THREADLIKE_FORCE_BLOCK_DIM = 64


@wp.kernel(enable_backward=False)
def prepare_gvs_threadlike_strain_kernel(
    q: wp.array2d[wp.float64],
    link_local_to_global: wp.array2d[wp.int32],
    link_basis: wp.array4d[wp.float64],
    reference_strain: wp.array3d[wp.float64],
    strain_workspace: wp.array4d[wp.float64],
):
    """Prepare each ``(environment, segment, quadrature)`` strain once."""
    item = wp.tid()
    num_points = reference_strain.shape[1]
    num_segments = reference_strain.shape[0]
    point = item % num_points
    item = item // num_points
    segment = item % num_segments
    environment = item // num_segments

    row = int(0)
    while row < 6:
        value = reference_strain[segment, point, row]
        local = int(0)
        while local < link_local_to_global.shape[1]:
            global_dof = link_local_to_global[segment, local]
            if global_dof >= 0:
                value += (
                    link_basis[segment, point, row, local] * q[environment, global_dof]
                )
            local += 1
        strain_workspace[environment, segment, point, row] = value
        row += 1


@wp.func
def _gvs_local_moment(
    strain_workspace: wp.array4d[wp.float64],
    environment: int,
    segment: int,
    point: int,
    path: int,
    physical_points: wp.array2d[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    epsilons: wp.array[wp.float64],
) -> Vec6d:
    """Evaluate one routed local moment from a prepared GVS strain."""
    s = physical_points[segment, point]
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
    omega = wp.vec3d(
        strain_workspace[environment, segment, point, 0],
        strain_workspace[environment, segment, point, 1],
        strain_workspace[environment, segment, point, 2],
    )
    linear = wp.vec3d(
        strain_workspace[environment, segment, point, 3],
        strain_workspace[environment, segment, point, 4],
        strain_workspace[environment, segment, point, 5],
    )
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


@wp.func
def _find_link_coordinate(
    global_dof: int,
    link_global_to_local: wp.array2d[wp.int32],
) -> wp.vec2i:
    """Return ``(segment, local)`` or ``(-1, -1)`` for a joint coordinate."""
    segment = int(0)
    while segment < link_global_to_local.shape[0]:
        local = link_global_to_local[segment, global_dof]
        if local >= 0:
            return wp.vec2i(segment, local)
        segment += 1
    return wp.vec2i(-1, -1)


@wp.kernel(enable_backward=False)
def gvs_threadlike_matrix_kernel(
    strain_workspace: wp.array4d[wp.float64],
    link_global_to_local: wp.array2d[wp.int32],
    link_basis: wp.array4d[wp.float64],
    physical_points: wp.array2d[wp.float64],
    integration_weights: wp.array2d[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    path_start_segments: wp.array[wp.int32],
    path_end_segments: wp.array[wp.int32],
    coordinate_scales: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    output: wp.array3d[wp.float64],
):
    """Integrate and write one complete ``(environment, DOF, path)`` entry."""
    item = wp.tid()
    num_paths = routing_intercepts.shape[0]
    num_dofs = link_global_to_local.shape[1]
    path = item % num_paths
    item = item // num_paths
    global_dof = item % num_dofs
    environment = item // num_dofs
    coordinate = _find_link_coordinate(global_dof, link_global_to_local)
    segment = coordinate[0]
    local = coordinate[1]

    value = wp.float64(0.0)
    if (
        local >= 0
        and segment >= path_start_segments[path]
        and segment <= path_end_segments[path]
    ):
        point = int(0)
        while point < physical_points.shape[1]:
            local_moment = _gvs_local_moment(
                strain_workspace,
                environment,
                segment,
                point,
                path,
                physical_points,
                routing_intercepts,
                routing_slopes,
                epsilons,
            )
            projected = wp.float64(0.0)
            row = int(0)
            while row < 6:
                projected += link_basis[segment, point, row, local] * local_moment[row]
                row += 1
            value += integration_weights[segment, point] * projected
            point += 1
        value *= coordinate_scales[path]
    output[environment, global_dof, path] = value


@wp.kernel(enable_backward=False)
def gvs_threadlike_force_kernel(
    strain_workspace: wp.array4d[wp.float64],
    controls: wp.array2d[wp.float64],
    link_global_to_local: wp.array2d[wp.int32],
    link_basis: wp.array4d[wp.float64],
    physical_points: wp.array2d[wp.float64],
    integration_weights: wp.array2d[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    path_start_segments: wp.array[wp.int32],
    path_end_segments: wp.array[wp.int32],
    coordinate_scales: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    output: wp.array2d[wp.float64],
):
    """CPU fallback: fuse path reduction for one active coordinate."""
    item = wp.tid()
    num_dofs = link_global_to_local.shape[1]
    global_dof = item % num_dofs
    environment = item // num_dofs
    coordinate = _find_link_coordinate(global_dof, link_global_to_local)
    segment = coordinate[0]
    local = coordinate[1]

    value = wp.float64(0.0)
    if local >= 0:
        path = int(0)
        while path < routing_intercepts.shape[0]:
            if (
                segment >= path_start_segments[path]
                and segment <= path_end_segments[path]
            ):
                effort = coordinate_scales[path] * controls[environment, path]
                point = int(0)
                while point < physical_points.shape[1]:
                    local_moment = _gvs_local_moment(
                        strain_workspace,
                        environment,
                        segment,
                        point,
                        path,
                        physical_points,
                        routing_intercepts,
                        routing_slopes,
                        epsilons,
                    )
                    projected = wp.float64(0.0)
                    row = int(0)
                    while row < 6:
                        projected += (
                            link_basis[segment, point, row, local] * local_moment[row]
                        )
                        row += 1
                    value += integration_weights[segment, point] * projected * effort
                    point += 1
            path += 1
    output[environment, global_dof] = value


@wp.kernel(enable_backward=False)
def gvs_threadlike_force_block_kernel(
    strain_workspace: wp.array4d[wp.float64],
    controls: wp.array2d[wp.float64],
    link_local_to_global: wp.array2d[wp.int32],
    link_global_to_local: wp.array2d[wp.int32],
    link_basis: wp.array4d[wp.float64],
    physical_points: wp.array2d[wp.float64],
    integration_weights: wp.array2d[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    path_start_segments: wp.array[wp.int32],
    path_end_segments: wp.array[wp.int32],
    coordinate_scales: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    output: wp.array2d[wp.float64],
):
    """CUDA path-parallel reduction owned by one environment/segment block."""
    work_item, lane = wp.tid()
    num_segments = link_local_to_global.shape[0]
    segment = work_item % num_segments
    environment = work_item // num_segments
    barrier = wp.tile_zeros(shape=(1,), dtype=wp.int32, storage="shared")

    local = lane
    while local < link_local_to_global.shape[1]:
        global_dof = link_local_to_global[segment, local]
        if global_dof >= 0:
            output[environment, global_dof] = wp.float64(0.0)
        local += wp.block_dim()

    if segment == 0:
        global_dof = lane
        while global_dof < output.shape[1]:
            coordinate = _find_link_coordinate(global_dof, link_global_to_local)
            if coordinate[1] < 0:
                output[environment, global_dof] = wp.float64(0.0)
            global_dof += wp.block_dim()
    wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)

    num_points = physical_points.shape[1]
    path_point = lane
    while path_point < routing_intercepts.shape[0] * num_points:
        path = path_point // num_points
        point = path_point - path * num_points
        if segment >= path_start_segments[path] and segment <= path_end_segments[path]:
            local_moment = _gvs_local_moment(
                strain_workspace,
                environment,
                segment,
                point,
                path,
                physical_points,
                routing_intercepts,
                routing_slopes,
                epsilons,
            )
            effort_weight = (
                coordinate_scales[path]
                * controls[environment, path]
                * integration_weights[segment, point]
            )
            local = int(0)
            while local < link_local_to_global.shape[1]:
                global_dof = link_local_to_global[segment, local]
                if global_dof >= 0:
                    projected = wp.float64(0.0)
                    row = int(0)
                    while row < 6:
                        projected += (
                            link_basis[segment, point, row, local] * local_moment[row]
                        )
                        row += 1
                    wp.atomic_add(
                        output,
                        environment,
                        global_dof,
                        effort_weight * projected,
                    )
                local += 1
        path_point += wp.block_dim()


def _launch_strain_preparation(
    q,
    link_local_to_global,
    link_basis,
    reference_strain,
    strain_workspace,
):
    """Enqueue the shared first-stage GVS strain preparation."""
    wp.launch(
        prepare_gvs_threadlike_strain_kernel,
        dim=q.shape[0] * reference_strain.shape[0] * reference_strain.shape[1],
        inputs=[q, link_local_to_global, link_basis, reference_strain],
        outputs=[strain_workspace],
        block_dim=128,
    )


def launch_gvs_threadlike_actuation_matrix(
    q: wp.array2d[wp.float64],
    link_local_to_global: wp.array2d[wp.int32],
    link_global_to_local: wp.array2d[wp.int32],
    link_basis: wp.array4d[wp.float64],
    reference_strain: wp.array3d[wp.float64],
    physical_points: wp.array2d[wp.float64],
    integration_weights: wp.array2d[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    path_start_segments: wp.array[wp.int32],
    path_end_segments: wp.array[wp.int32],
    coordinate_scales: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    strain_workspace: wp.array4d[wp.float64],
    output: wp.array3d[wp.float64],
):
    """Prepare strains, then fill caller-owned GVS matrix output ``(E,D,A)``."""
    _launch_strain_preparation(
        q,
        link_local_to_global,
        link_basis,
        reference_strain,
        strain_workspace,
    )
    wp.launch(
        gvs_threadlike_matrix_kernel,
        dim=q.shape[0] * output.shape[1] * output.shape[2],
        inputs=[
            strain_workspace,
            link_global_to_local,
            link_basis,
            physical_points,
            integration_weights,
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


def launch_gvs_threadlike_actuation_force(
    q: wp.array2d[wp.float64],
    controls: wp.array2d[wp.float64],
    link_local_to_global: wp.array2d[wp.int32],
    link_global_to_local: wp.array2d[wp.int32],
    link_basis: wp.array4d[wp.float64],
    reference_strain: wp.array3d[wp.float64],
    physical_points: wp.array2d[wp.float64],
    integration_weights: wp.array2d[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    path_start_segments: wp.array[wp.int32],
    path_end_segments: wp.array[wp.int32],
    coordinate_scales: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    strain_workspace: wp.array4d[wp.float64],
    output: wp.array2d[wp.float64],
):
    """Prepare strains, then fill caller-owned fused GVS force ``(E,D)``."""
    _launch_strain_preparation(
        q,
        link_local_to_global,
        link_basis,
        reference_strain,
        strain_workspace,
    )
    common = [
        strain_workspace,
        controls,
        link_global_to_local,
        link_basis,
        physical_points,
        integration_weights,
        routing_intercepts,
        routing_slopes,
        path_start_segments,
        path_end_segments,
        coordinate_scales,
        epsilons,
    ]
    if q.device.is_cuda:
        wp.launch_tiled(
            gvs_threadlike_force_block_kernel,
            dim=q.shape[0] * link_local_to_global.shape[0],
            inputs=[
                strain_workspace,
                controls,
                link_local_to_global,
                link_global_to_local,
                link_basis,
                physical_points,
                integration_weights,
                routing_intercepts,
                routing_slopes,
                path_start_segments,
                path_end_segments,
                coordinate_scales,
                epsilons,
            ],
            outputs=[output],
            block_dim=GVS_THREADLIKE_FORCE_BLOCK_DIM,
        )
    else:
        wp.launch(
            gvs_threadlike_force_kernel,
            dim=q.shape[0] * output.shape[1],
            inputs=common,
            outputs=[output],
            block_dim=128,
        )


gvs_threadlike_actuation_matrix = wp.jax_callable(
    launch_gvs_threadlike_actuation_matrix, num_outputs=2
)
gvs_threadlike_actuation_force = wp.jax_callable(
    launch_gvs_threadlike_actuation_force, num_outputs=2
)


__all__ = [
    "GVS_THREADLIKE_FORCE_BLOCK_DIM",
    "gvs_threadlike_force_block_kernel",
    "gvs_threadlike_force_kernel",
    "gvs_threadlike_matrix_kernel",
    "launch_gvs_threadlike_actuation_force",
    "launch_gvs_threadlike_actuation_matrix",
    "prepare_gvs_threadlike_strain_kernel",
]
