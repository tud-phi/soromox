"""JAX FFI launch boundaries for native free-space routing.

Every runtime numerical operation is a Warp launch. Workspaces are provided by
JAX FFI; native integrations use StraightSpanPipeline to retain allocations.
"""

import warp as wp

from .straight_span import straight_spans_kernel
from .straight_span_actuation import (
    guide_geometry_kernel,
    path_force_kernel,
    path_lengths_kernel,
    path_matrix_kernel,
    prepare_path_data,
    routing_boundaries_kernel,
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
    """Evaluate native free-space force with caller-owned output arrays."""
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
    """Evaluate native free-space matrix with caller-owned output arrays."""
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
    """Evaluate native free-space routing with caller-owned output arrays."""
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
        routing_boundaries_kernel,
        dim=q.shape[0],
        inputs=[q, data, links, fixed_base],
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
