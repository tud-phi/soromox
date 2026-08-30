# ruff: noqa: I001, UP018
"""Allocation-free Warp kernels for linear threadlike GVS actuation.

Both public launchers first prepare quadrature-point strains in a caller-owned
``(E, S, Q, 6)`` workspace. Matrix projection then writes every ``(E, D, A)``
entry exactly once. Fused CUDA force execution uses one 128-lane block per
environment and segment to reduce paths in parallel; the CPU fallback writes
one active coordinate per thread. Joint-only coordinates and inactive paths
are written as exact zeros without a separate clear launch.

The fixed two-launch sequences allocate nothing, call no JAX functions,
perform no synchronization or transfer, and are CUDA-graph-capturable.
"""

from __future__ import annotations

import warp as wp

from soromox.execution.warp.common.se3 import Vec6d

wp.set_module_options({"enable_backward": False})

GVS_THREADLIKE_FORCE_BLOCK_DIM = 128


@wp.kernel(enable_backward=False)
def prepare_gvs_threadlike_strain_kernel(
    q: wp.array2d[wp.float64],
    link_local_to_global: wp.array2d[wp.int32],
    link_basis: wp.array4d[wp.float64],
    reference_strain: wp.array3d[wp.float64],
    strain_workspace: wp.array4d[wp.float64],
):
    """Prepare each environment-segment-quadrature strain once.

    Args:
        q: Batched active configurations with shape ``(E, D)``.
        link_local_to_global: Padded link-local to active-coordinate map.
        link_basis: Length-scaled link basis with shape ``(S, Q, 6, L)``.
        reference_strain: Reference strain with shape ``(S, Q, 6)``.
        strain_workspace: Caller-owned output workspace ``(E, S, Q, 6)``.

    Returns:
        None. Prepared strains are written to ``strain_workspace``.
    """
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
    """Evaluate one routed local moment from a prepared GVS strain.

    Args:
        strain_workspace: Prepared quadrature strains ``(E, S, Q, 6)``.
        environment: Environment index.
        segment: Segment index.
        point: Interior quadrature-point index.
        path: Ordered routing-path index.
        physical_points: Global backbone coordinates ``(S, Q)``.
        routing_intercepts: Linear-routing intercept vectors ``(A, 3)``.
        routing_slopes: Linear-routing slope vectors ``(A, 3)``.
        epsilons: One-entry array containing the normalization threshold.

    Returns:
        Six-dimensional local moment and tangent basis.
    """
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
    """Map an active coordinate to its owning link and local coordinate.

    Args:
        global_dof: Active generalized-coordinate index.
        link_global_to_local: Active-coordinate to link-local map.

    Returns:
        Pair ``(segment, local)`` or ``(-1, -1)`` for a joint coordinate.
    """
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
    """Integrate one complete environment-DOF-path matrix entry.

    Args:
        strain_workspace: Prepared quadrature strains ``(E, S, Q, 6)``.
        link_global_to_local: Active-coordinate to link-local map.
        link_basis: Length-scaled link basis ``(S, Q, 6, L)``.
        physical_points: Global backbone coordinates ``(S, Q)``.
        integration_weights: Physical quadrature weights ``(S, Q)``.
        routing_intercepts: Linear-routing intercept vectors ``(A, 3)``.
        routing_slopes: Linear-routing slope vectors ``(A, 3)``.
        path_start_segments: Inclusive first segment for every path.
        path_end_segments: Inclusive last segment for every path.
        coordinate_scales: Signed or physical scale for every path.
        epsilons: One-entry array containing the normalization threshold.
        output: Caller-owned actuation matrix ``(E, D, A)``.

    Returns:
        None. Every matrix entry, including zeros, is written to ``output``.
    """
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
    """Reduce all paths for one active coordinate on Warp CPU.

    Args:
        strain_workspace: Prepared quadrature strains ``(E, S, Q, 6)``.
        controls: Batched direct-effort controls ``(E, A)``.
        link_global_to_local: Active-coordinate to link-local map.
        link_basis: Length-scaled link basis ``(S, Q, 6, L)``.
        physical_points: Global backbone coordinates ``(S, Q)``.
        integration_weights: Physical quadrature weights ``(S, Q)``.
        routing_intercepts: Linear-routing intercept vectors ``(A, 3)``.
        routing_slopes: Linear-routing slope vectors ``(A, 3)``.
        path_start_segments: Inclusive first segment for every path.
        path_end_segments: Inclusive last segment for every path.
        coordinate_scales: Signed or physical scale for every path.
        epsilons: One-entry array containing the normalization threshold.
        output: Caller-owned generalized force ``(E, D)``.

    Returns:
        None. Every generalized-force entry is written to ``output``.
    """
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
    """Reduce paths with one CUDA block per environment and segment.

    Args:
        strain_workspace: Prepared quadrature strains ``(E, S, Q, 6)``.
        controls: Batched direct-effort controls ``(E, A)``.
        link_local_to_global: Padded link-local to active-coordinate map.
        link_global_to_local: Active-coordinate to link-local map.
        link_basis: Length-scaled link basis ``(S, Q, 6, L)``.
        physical_points: Global backbone coordinates ``(S, Q)``.
        integration_weights: Physical quadrature weights ``(S, Q)``.
        routing_intercepts: Linear-routing intercept vectors ``(A, 3)``.
        routing_slopes: Linear-routing slope vectors ``(A, 3)``.
        path_start_segments: Inclusive first segment for every path.
        path_end_segments: Inclusive last segment for every path.
        coordinate_scales: Signed or physical scale for every path.
        epsilons: One-entry array containing the normalization threshold.
        output: Caller-owned generalized force ``(E, D)``.

    Returns:
        None. Link-coordinate contributions are accumulated into ``output``;
        joint-only coordinates are written as exact zeros.
    """
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
    """Enqueue the shared first-stage GVS strain preparation.

    Args:
        q: Batched active configurations with shape ``(E, D)``.
        link_local_to_global: Padded link-local to active-coordinate map.
        link_basis: Length-scaled link basis ``(S, Q, 6, L)``.
        reference_strain: Reference strain ``(S, Q, 6)``.
        strain_workspace: Caller-owned output workspace ``(E, S, Q, 6)``.

    Returns:
        None. The launch is enqueued without synchronization.
    """
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
    """Prepare strains and fill a caller-owned GVS actuation matrix.

    Args:
        q: Batched active configurations with shape ``(E, D)``.
        link_local_to_global: Padded link-local to active-coordinate map.
        link_global_to_local: Active-coordinate to link-local map.
        link_basis: Length-scaled link basis ``(S, Q, 6, L)``.
        reference_strain: Reference strain ``(S, Q, 6)``.
        physical_points: Global backbone coordinates ``(S, Q)``.
        integration_weights: Physical quadrature weights ``(S, Q)``.
        routing_intercepts: Linear-routing intercept vectors ``(A, 3)``.
        routing_slopes: Linear-routing slope vectors ``(A, 3)``.
        path_start_segments: Inclusive first segment for every path.
        path_end_segments: Inclusive last segment for every path.
        coordinate_scales: Signed or physical scale for every path.
        epsilons: One-entry array containing the normalization threshold.
        strain_workspace: Caller-owned workspace ``(E, S, Q, 6)``.
        output: Caller-owned actuation matrix ``(E, D, A)``.

    Returns:
        None. Both launches are enqueued without synchronization.

    Notes:
        The launcher allocates no arrays and performs no JAX call or data
        transfer, so callers may capture the fixed sequence in a CUDA graph.
    """
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
    """Prepare strains and fill a caller-owned fused GVS force.

    Args:
        q: Batched active configurations with shape ``(E, D)``.
        controls: Batched direct-effort controls ``(E, A)``.
        link_local_to_global: Padded link-local to active-coordinate map.
        link_global_to_local: Active-coordinate to link-local map.
        link_basis: Length-scaled link basis ``(S, Q, 6, L)``.
        reference_strain: Reference strain ``(S, Q, 6)``.
        physical_points: Global backbone coordinates ``(S, Q)``.
        integration_weights: Physical quadrature weights ``(S, Q)``.
        routing_intercepts: Linear-routing intercept vectors ``(A, 3)``.
        routing_slopes: Linear-routing slope vectors ``(A, 3)``.
        path_start_segments: Inclusive first segment for every path.
        path_end_segments: Inclusive last segment for every path.
        coordinate_scales: Signed or physical scale for every path.
        epsilons: One-entry array containing the normalization threshold.
        strain_workspace: Caller-owned workspace ``(E, S, Q, 6)``.
        output: Caller-owned generalized force ``(E, D)``.

    Returns:
        None. Both launches are enqueued without synchronization.

    Notes:
        CUDA execution reduces path-point contributions cooperatively; Warp CPU
        execution uses the scalar fallback. Neither path allocates arrays,
        invokes JAX, synchronizes, or transfers data.
    """
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
