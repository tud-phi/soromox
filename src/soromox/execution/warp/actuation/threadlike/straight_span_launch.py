"""JAX FFI launch boundaries for native free-space routing.

Every runtime numerical operation is a Warp launch. Workspaces are provided by
JAX FFI; native integrations use StraightSpanActuation to retain allocations.
"""

import warp as wp

from soromox.execution.warp.common.relative_kinematics import segment_boundaries_kernel

from .straight_span import (
    guide_geometry_kernel,
    path_force_kernel,
    path_lengths_kernel,
    path_matrix_kernel,
    prepare_path_data,
    straight_spans_kernel,
)


def launch_force(
    q: wp.array2d[wp.float64],
    basis1: wp.array4d[wp.float64],
    basis2: wp.array4d[wp.float64],
    reference1: wp.array3d[wp.float64],
    reference2: wp.array3d[wp.float64],
    distances: wp.array2d[wp.float64],
    intervals: wp.array2d[wp.int32],
    span_intervals: wp.array[wp.int32],
    paths: wp.array[wp.int32],
    predecessors: wp.array[wp.int32],
    coincident: wp.array[wp.bool],
    start_offsets: wp.array2d[wp.float64],
    end_offsets: wp.array2d[wp.float64],
    tensions: wp.array2d[wp.float64],
    strains: wp.array4d[wp.float64],
    links: wp.array4d[wp.float64],
    link_jacobians: wp.array4d[wp.float64],
    joints: wp.array4d[wp.float64],
    joint_jacobians: wp.array4d[wp.float64],
    transforms: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
    force: wp.array2d[wp.float64],
):
    """Evaluate native free-space force with caller-owned output arrays.

    E denotes environments, L links, C cells including the joint, D internal
    coordinates, Q configurations, V velocities, S spans, I intervals, and P paths.
    All buffers use FP64 except int32 topology and boolean flags.

    Args:
        q: Configurations ``(E, Q)`` including any base coordinates.
        basis1: First strain bases ``(L, C, 6, D)`` in internal coordinate order.
        basis2: Second strain bases at the other Magnus sample, same shape.
        reference1: First reference strains ``(L, C, 6)``.
        reference2: Second reference strains ``(L, C, 6)``.
        distances: Integration distances ``(L, C)``; cell zero represents the joint.
        intervals: Inclusive first/last link indices ``(I, 2)``.
        span_intervals: Interval index of each span ``(S,)``.
        paths: Physical path index of each span ``(S,)``.
        predecessors: Previous span on the same path, or -1, ``(S,)``.
        coincident: Invariant coincidence flags for preceding connections ``(S,)``.
        start_offsets: Start guides in link-entry frames ``(S, 3)``.
        end_offsets: End guides in link-exit frames ``(S, 3)``.
        tensions: Signed axial loads ``(E, P)``; positive denotes tension.
        strains: Shared strain output workspace ``(E, L*C, 2, 6)``.
        links: Relative link transform output workspace ``(E, L, 4, 4)``.
        link_jacobians: Link body Jacobian output workspace ``(E, L, 6, D)``.
        joints: Joint transform output workspace ``(E, L, 4, 4)``.
        joint_jacobians: Joint body Jacobian output workspace ``(E, L, 6, D)``.
        transforms: Composed interval workspace ``(E, I, 4, 4)``; I=0 reuses links.
        jacobians: Interval body Jacobian workspace ``(E, I, 6, D)``.
        force: Generalized-force output ``(E, V)``; cleared before accumulation.

    Returns:
        None. Enqueues kinematics and force accumulation on q's device.
    """
    data = prepare_path_data(
        q,
        (basis1, basis2, reference1, reference2, distances),
        (intervals, span_intervals, paths, predecessors, coincident),
        (start_offsets, end_offsets),
        (
            strains,
            links,
            link_jacobians,
            joints,
            joint_jacobians,
            transforms,
            jacobians,
        ),
    )
    force.zero_()
    if force.shape[1] and paths.shape[0]:
        wp.launch(
            path_force_kernel,
            dim=(force.shape[0], paths.shape[0], force.shape[1]),
            inputs=[data, tensions],
            outputs=[force],
            device=q.device,
        )


force = wp.jax_callable(launch_force, num_outputs=8)


def launch_matrix(
    q: wp.array2d[wp.float64],
    basis1: wp.array4d[wp.float64],
    basis2: wp.array4d[wp.float64],
    reference1: wp.array3d[wp.float64],
    reference2: wp.array3d[wp.float64],
    distances: wp.array2d[wp.float64],
    intervals: wp.array2d[wp.int32],
    span_intervals: wp.array[wp.int32],
    paths: wp.array[wp.int32],
    predecessors: wp.array[wp.int32],
    coincident: wp.array[wp.bool],
    start_offsets: wp.array2d[wp.float64],
    end_offsets: wp.array2d[wp.float64],
    strains: wp.array4d[wp.float64],
    links: wp.array4d[wp.float64],
    link_jacobians: wp.array4d[wp.float64],
    joints: wp.array4d[wp.float64],
    joint_jacobians: wp.array4d[wp.float64],
    transforms: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
    matrix: wp.array3d[wp.float64],
):
    """Evaluate native free-space matrix with caller-owned output arrays.

    E denotes environments, L links, C cells including the joint, D internal
    coordinates, Q configurations, V velocities, S spans, I intervals, and P paths.
    All buffers use FP64 except int32 topology and boolean flags.

    Args:
        q: Configurations ``(E, Q)`` including any base coordinates.
        basis1: First strain bases ``(L, C, 6, D)`` in internal coordinate order.
        basis2: Second strain bases at the other Magnus sample, same shape.
        reference1: First reference strains ``(L, C, 6)``.
        reference2: Second reference strains ``(L, C, 6)``.
        distances: Integration distances ``(L, C)``; cell zero represents the joint.
        intervals: Inclusive first/last link indices ``(I, 2)``.
        span_intervals: Interval index of each span ``(S,)``.
        paths: Physical path index of each span ``(S,)``.
        predecessors: Previous span on the same path, or -1, ``(S,)``.
        coincident: Invariant coincidence flags for preceding connections ``(S,)``.
        start_offsets: Start guides in link-entry frames ``(S, 3)``.
        end_offsets: End guides in link-exit frames ``(S, 3)``.
        strains: Shared strain output workspace ``(E, L*C, 2, 6)``.
        links: Relative link transform output workspace ``(E, L, 4, 4)``.
        link_jacobians: Link body Jacobian output workspace ``(E, L, 6, D)``.
        joints: Joint transform output workspace ``(E, L, 4, 4)``.
        joint_jacobians: Joint body Jacobian output workspace ``(E, L, 6, D)``.
        transforms: Composed interval workspace ``(E, I, 4, 4)``; I=0 reuses links.
        jacobians: Interval body Jacobian workspace ``(E, I, 6, D)``.
        matrix: Actuation matrix output ``(E, V, P)``.

    Returns:
        None. Enqueues kinematics and writes every matrix entry on q's device.
    """
    data = prepare_path_data(
        q,
        (basis1, basis2, reference1, reference2, distances),
        (intervals, span_intervals, paths, predecessors, coincident),
        (start_offsets, end_offsets),
        (
            strains,
            links,
            link_jacobians,
            joints,
            joint_jacobians,
            transforms,
            jacobians,
        ),
    )
    if matrix.shape[1] and matrix.shape[2]:
        wp.launch(
            path_matrix_kernel,
            dim=matrix.shape,
            inputs=[data, True],
            outputs=[matrix],
            device=q.device,
        )


matrix = wp.jax_callable(launch_matrix, num_outputs=8)


def launch_routing(
    q: wp.array2d[wp.float64],
    basis1: wp.array4d[wp.float64],
    basis2: wp.array4d[wp.float64],
    reference1: wp.array3d[wp.float64],
    reference2: wp.array3d[wp.float64],
    distances: wp.array2d[wp.float64],
    intervals: wp.array2d[wp.int32],
    span_intervals: wp.array[wp.int32],
    paths: wp.array[wp.int32],
    predecessors: wp.array[wp.int32],
    coincident: wp.array[wp.bool],
    start_offsets: wp.array2d[wp.float64],
    end_offsets: wp.array2d[wp.float64],
    fixed_base: wp.array2d[wp.float64],
    strains: wp.array4d[wp.float64],
    links: wp.array4d[wp.float64],
    link_jacobians: wp.array4d[wp.float64],
    joints: wp.array4d[wp.float64],
    joint_jacobians: wp.array4d[wp.float64],
    transforms: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
    entries: wp.array4d[wp.float64],
    exits: wp.array4d[wp.float64],
    geometry: wp.array3d[wp.float64],
    span_jacobians: wp.array3d[wp.float64],
    valid: wp.array2d[wp.bool],
    guide_geometry: wp.array3d[wp.float64],
    guide_valid: wp.array2d[wp.bool],
    lengths: wp.array2d[wp.float64],
    matrix: wp.array3d[wp.float64],
):
    """Evaluate native free-space routing with caller-owned output arrays.

    E denotes environments, L links, C cells including the joint, D internal
    coordinates, Q configurations, V velocities, S spans, I intervals, and P paths.
    All buffers use FP64 except int32 topology and boolean flags.

    Args:
        q: Configurations ``(E, Q)`` including any base coordinates.
        basis1: First strain bases ``(L, C, 6, D)`` in internal coordinate order.
        basis2: Second strain bases at the other Magnus sample, same shape.
        reference1: First reference strains ``(L, C, 6)``.
        reference2: Second reference strains ``(L, C, 6)``.
        distances: Integration distances ``(L, C)``; cell zero represents the joint.
        intervals: Inclusive first/last link indices ``(I, 2)``.
        span_intervals: Interval index of each span ``(S,)``.
        paths: Physical path index of each span ``(S,)``.
        predecessors: Previous span on the same path, or -1, ``(S,)``.
        coincident: Invariant coincidence flags for preceding connections ``(S,)``.
        start_offsets: Start guides in link-entry frames ``(S, 3)``.
        end_offsets: End guides in link-exit frames ``(S, 3)``.
        fixed_base: Fixed base transform ``(4, 4)``; unused for floating bases.
        strains: Shared strain output workspace ``(E, L*C, 2, 6)``.
        links: Relative link transform output workspace ``(E, L, 4, 4)``.
        link_jacobians: Link body Jacobian output workspace ``(E, L, 6, D)``.
        joints: Joint transform output workspace ``(E, L, 4, 4)``.
        joint_jacobians: Joint body Jacobian output workspace ``(E, L, 6, D)``.
        transforms: Composed interval workspace ``(E, I, 4, 4)``; I=0 reuses links.
        jacobians: Interval body Jacobian workspace ``(E, I, 6, D)``.
        entries: World link-entry transform output ``(E, L, 4, 4)``.
        exits: World link-exit transform output ``(E, L, 4, 4)``.
        geometry: Span start/end world positions and length output ``(E, S, 7)``.
        span_jacobians: Internal span length derivative output ``(E, S, D)``.
        valid: Finite, noncollapsed span output flags ``(E, S)``.
        guide_geometry: Joint connection displacement and length output ``(E, S, 4)``.
        guide_valid: Joint connection differentiability output flags ``(E, S)``.
        lengths: Physical path length output ``(E, P)`` in metres.
        matrix: Negative configuration length Jacobian output ``(E, Q, P)``.

    Returns:
        None. Enqueues kinematics, geometry, and path assembly on q's device.
    """
    data = prepare_path_data(
        q,
        (basis1, basis2, reference1, reference2, distances),
        (intervals, span_intervals, paths, predecessors, coincident),
        (start_offsets, end_offsets),
        (
            strains,
            links,
            link_jacobians,
            joints,
            joint_jacobians,
            transforms,
            jacobians,
        ),
    )
    wp.launch(
        segment_boundaries_kernel,
        dim=q.shape[0],
        inputs=[q, joints, link_jacobians.shape[3], links, fixed_base],
        outputs=[entries, exits],
        device=q.device,
    )
    if paths.shape[0]:
        wp.launch(
            straight_spans_kernel,
            dim=(q.shape[0], paths.shape[0], jacobians.shape[3] + 1),
            inputs=[
                data.transforms,
                data.jacobians,
                entries,
                exits,
                intervals,
                span_intervals,
                start_offsets,
                end_offsets,
            ],
            outputs=[geometry, span_jacobians, valid],
            device=q.device,
        )
        wp.launch(
            guide_geometry_kernel,
            dim=guide_valid.shape,
            inputs=[data],
            outputs=[guide_geometry, guide_valid],
            device=q.device,
        )
    if lengths.shape[1]:
        wp.launch(
            path_lengths_kernel,
            dim=lengths.shape,
            inputs=[data],
            outputs=[lengths],
            device=q.device,
        )
    if matrix.shape[1] and matrix.shape[2]:
        wp.launch(
            path_matrix_kernel,
            dim=matrix.shape,
            inputs=[data, False],
            outputs=[matrix],
            device=q.device,
        )


routing = wp.jax_callable(launch_routing, num_outputs=16)
