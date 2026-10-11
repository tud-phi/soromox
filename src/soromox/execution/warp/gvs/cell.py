# ruff: noqa: I001, UP018
"""Shape-generic Warp kernels for GVS link-cell dynamics.

The kernels keep spatial dimension six, but obtain the number of generalized
coordinates and local strain coordinates from array shapes. Runtime ``while``
loops prevent Warp from expanding work proportional to the number of segments
into the generated CUDA source.
"""

from __future__ import annotations
import warp as wp
from soromox.execution.warp.common.se3 import (
    Vec6d,
    _ad_action,
    _adjoint_inverse_action,
    _adjoint_inverse_entry,
    _forward_coefficients,
    _left_action,
    _left_coefficient_x_derivatives,
    _left_coefficients,
    _left_total_derivative_action,
    _translation,
)

wp.set_module_options({"enable_backward": False})
SPATIAL_DIM = 6


@wp.func
def _load_sparse_column(
    values: wp.array2d[wp.float64],
    rows: wp.array2d[wp.int32],
    cell_item: int,
    segment: int,
    column: int,
) -> Vec6d:
    """Load one sparse spatial basis column.

    Args:
        values: Nonzero basis values for every cell and local column.
        rows: Spatial row occupied by each segment-local column.
        cell_item: Flattened segment-cell row in ``values``.
        segment: Segment row in ``rows``.
        column: Padded local basis-column index.

    Returns:
        Six-dimensional column with at most one nonzero entry.
    """
    result = Vec6d()
    row = rows[segment, column]
    if row >= 0:
        result[row] = values[cell_item, column]
    return result


@wp.func
def _dynamic_strain(
    basis_values: wp.array2d[wp.float64],
    basis_rows: wp.array2d[wp.int32],
    reference: wp.array2d[wp.float64],
    coordinates: wp.array2d[wp.float64],
    cell_item: int,
    segment: int,
    coordinate_item: int,
) -> Vec6d:
    """Assemble spatially varying strain at one integration cell endpoint.

    Args:
        basis_values: Nonzero strain-basis values for every cell.
        basis_rows: Spatial row occupied by each segment-local coordinate.
        reference: Reference strains for every flattened cell.
        coordinates: Batched padded segment-local coordinates.
        cell_item: Flattened segment-cell index.
        segment: Segment containing the cell.
        coordinate_item: Flattened environment-segment coordinate row.

    Returns:
        Six-dimensional strain at the selected cell endpoint.
    """
    value = Vec6d(
        reference[cell_item, 0],
        reference[cell_item, 1],
        reference[cell_item, 2],
        reference[cell_item, 3],
        reference[cell_item, 4],
        reference[cell_item, 5],
    )
    column = int(0)
    while column < coordinates.shape[1]:
        coordinate = coordinates[coordinate_item, column]
        row = basis_rows[segment, column]
        if row >= 0:
            value[row] += basis_values[cell_item, column] * coordinate
        column += 1
    return value


@wp.func
def _dynamic_strain_rate(
    basis_values: wp.array2d[wp.float64],
    basis_rows: wp.array2d[wp.int32],
    velocities: wp.array2d[wp.float64],
    cell_item: int,
    segment: int,
    coordinate_item: int,
) -> Vec6d:
    """Assemble spatially varying strain rate at one cell endpoint.

    Args:
        basis_values: Nonzero strain-basis values for every cell.
        basis_rows: Spatial row occupied by each segment-local coordinate.
        velocities: Batched padded segment-local velocities.
        cell_item: Flattened segment-cell index.
        segment: Segment containing the cell.
        coordinate_item: Flattened environment-segment velocity row.

    Returns:
        Six-dimensional strain rate at the selected cell endpoint.
    """
    value = Vec6d()
    column = int(0)
    while column < velocities.shape[1]:
        velocity = velocities[coordinate_item, column]
        row = basis_rows[segment, column]
        if row >= 0:
            value[row] += basis_values[cell_item, column] * velocity
        column += 1
    return value


@wp.func
def cell_terms_at(
    work_item: int,
    lane: int,
    lanes: int,
    q_link: wp.array2d[wp.float64],
    qd_link: wp.array2d[wp.float64],
    basis_z1: wp.array2d[wp.float64],
    basis_z2: wp.array2d[wp.float64],
    basis_rows: wp.array2d[wp.int32],
    reference_z1: wp.array2d[wp.float64],
    reference_z2: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    cell_widths: wp.array[wp.float64],
    num_cells: wp.array[wp.int32],
    order_zero: wp.array[wp.int32],
    adjoint: wp.array2d[wp.float64],
    tangent_local: wp.array2d[wp.float64],
    link_velocity: wp.array2d[wp.float64],
    step_velocity: wp.array2d[wp.float64],
    tangent_velocity_dot: wp.array2d[wp.float64],
):
    """Compute general cell-local Lie data for one batched cell work item.

    Args:
        work_item: Flattened environment-segment-cell index in the supplied arrays.
        lane: Thread index within the cooperative work item.
        lanes: Number of threads sharing this cell; one on CPU or two on CUDA.
        q_link: Batched padded segment-local coordinates.
        qd_link: Batched padded segment-local velocities.
        basis_z1: Sparse basis values at the first Magnus node.
        basis_z2: Sparse basis values at the second Magnus node.
        basis_rows: Spatial row occupied by each sparse basis column.
        reference_z1: Reference strains at first Magnus nodes.
        reference_z2: Reference strains at second Magnus nodes.
        segment_lengths: Physical segment lengths.
        cell_widths: Normalized integration-cell widths.
        num_cells: One-entry array containing cells per segment.
        order_zero: One-entry flag selecting the constant-strain shortcut.
        adjoint: Caller-owned flattened cell-adjoint output.
        tangent_local: Caller-owned flattened local-tangent output.
        link_velocity: Caller-owned flattened link-velocity output.
        step_velocity: Caller-owned flattened step-velocity output.
        tangent_velocity_dot: Caller-owned tangent-derivative action output.

    Returns:
        None. All results are written to the caller-owned output arrays.
    """
    cells_per_segment = num_cells[0]
    cells_per_environment = segment_lengths.shape[0] * cells_per_segment
    environment = work_item // cells_per_environment
    environment_cell = work_item - environment * cells_per_environment
    segment = environment_cell // cells_per_segment
    cell = environment_cell - segment * cells_per_segment
    cell_item = segment * cells_per_segment + cell
    coordinate_item = environment * segment_lengths.shape[0] + segment
    output_base_row = work_item * SPATIAL_DIM
    xi1 = _dynamic_strain(
        basis_z1, basis_rows, reference_z1, q_link, cell_item, segment, coordinate_item
    )
    xid1 = _dynamic_strain_rate(
        basis_z1, basis_rows, qd_link, cell_item, segment, coordinate_item
    )
    length = segment_lengths[segment]
    width = cell_widths[cell_item]
    alpha = length * width * wp.float64(0.5)
    commutator_coefficient = (
        wp.sqrt(wp.float64(3.0)) * length * length * width * width / wp.float64(12.0)
    )
    xi2 = xi1
    xid2 = xid1
    magnus = length * width * xi1
    magnus_dot = length * width * xid1
    if order_zero[0] == 0:
        xi2 = _dynamic_strain(
            basis_z2,
            basis_rows,
            reference_z2,
            q_link,
            cell_item,
            segment,
            coordinate_item,
        )
        xid2 = _dynamic_strain_rate(
            basis_z2, basis_rows, qd_link, cell_item, segment, coordinate_item
        )
        magnus = alpha * (xi1 + xi2) + commutator_coefficient * _ad_action(xi1, xi2)
        magnus_dot = alpha * (xid1 + xid2) + commutator_coefficient * (
            _ad_action(xid1, xi2) + _ad_action(xi1, xid2)
        )
    angle_sq = magnus[0] * magnus[0] + magnus[1] * magnus[1] + magnus[2] * magnus[2]
    coefficients = _left_coefficients(angle_sq)
    omega = wp.vec3d(magnus[0], magnus[1], magnus[2])
    linear = wp.vec3d(magnus[3], magnus[4], magnus[5])
    exponential_coefficients = _forward_coefficients(angle_sq)
    translation = _translation(omega, linear, exponential_coefficients)
    element = lane
    while element < SPATIAL_DIM * SPATIAL_DIM:
        row = element // SPATIAL_DIM
        column = element % SPATIAL_DIM
        adjoint[output_base_row + row, column] = _adjoint_inverse_entry(
            omega, translation, angle_sq, exponential_coefficients, row, column
        )
        element += lanes
    local_column = lane
    while local_column < q_link.shape[1]:
        basis1 = _load_sparse_column(
            basis_z1, basis_rows, cell_item, segment, local_column
        )
        magnus_basis = length * width * basis1
        if order_zero[0] == 0:
            basis2 = _load_sparse_column(
                basis_z2, basis_rows, cell_item, segment, local_column
            )
            magnus_basis = alpha * (basis1 + basis2) + commutator_coefficient * (
                _ad_action(xi1, basis2) - _ad_action(xi2, basis1)
            )
        local_tangent = _left_action(magnus, magnus_basis, coefficients)
        row = int(0)
        while row < SPATIAL_DIM:
            tangent_local[output_base_row + row, local_column] = local_tangent[row]
            row += 1
        local_column += lanes
    if lane == 0:
        link = _left_action(magnus, magnus_dot, coefficients)
        step = _adjoint_inverse_action(
            omega, translation, angle_sq, exponential_coefficients, link
        )
        magnus_basis_dot_qd = Vec6d()
        if order_zero[0] == 0:
            magnus_basis_dot_qd = (
                wp.float64(2.0) * commutator_coefficient * _ad_action(xid1, xid2)
            )
        coefficient_derivatives = _left_coefficient_x_derivatives(angle_sq)
        angle_sq_dot = wp.float64(2.0) * (
            magnus[0] * magnus_dot[0]
            + magnus[1] * magnus_dot[1]
            + magnus[2] * magnus_dot[2]
        )
        tangent_dot_velocity_value = _left_total_derivative_action(
            magnus,
            magnus_dot,
            magnus_dot,
            magnus_basis_dot_qd,
            coefficients,
            coefficient_derivatives * angle_sq_dot,
        )
        row = int(0)
        while row < SPATIAL_DIM:
            link_velocity[output_base_row + row, 0] = link[row]
            step_velocity[output_base_row + row, 0] = step[row]
            tangent_velocity_dot[output_base_row + row, 0] = tangent_dot_velocity_value[
                row
            ]
            row += 1


@wp.kernel(enable_backward=False)
def cell_terms_kernel(
    q_link: wp.array2d[wp.float64],
    qd_link: wp.array2d[wp.float64],
    basis_z1: wp.array2d[wp.float64],
    basis_z2: wp.array2d[wp.float64],
    basis_rows: wp.array2d[wp.int32],
    reference_z1: wp.array2d[wp.float64],
    reference_z2: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    cell_widths: wp.array[wp.float64],
    num_cells: wp.array[wp.int32],
    order_zero: wp.array[wp.int32],
    adjoint: wp.array2d[wp.float64],
    tangent_local: wp.array2d[wp.float64],
    link_velocity: wp.array2d[wp.float64],
    step_velocity: wp.array2d[wp.float64],
    tangent_velocity_dot: wp.array2d[wp.float64],
):
    """Compute general cell-local Lie data for one batched cell work item.

    Args:
        q_link: Batched padded segment-local coordinates.
        qd_link: Batched padded segment-local velocities.
        basis_z1: Sparse basis values at the first Magnus node.
        basis_z2: Sparse basis values at the second Magnus node.
        basis_rows: Spatial row occupied by each sparse basis column.
        reference_z1: Reference strains at first Magnus nodes.
        reference_z2: Reference strains at second Magnus nodes.
        segment_lengths: Physical segment lengths.
        cell_widths: Normalized integration-cell widths.
        num_cells: One-entry array containing cells per segment.
        order_zero: One-entry flag selecting the constant-strain shortcut.
        adjoint: Caller-owned flattened cell-adjoint output.
        tangent_local: Caller-owned flattened local-tangent output.
        link_velocity: Caller-owned flattened link-velocity output.
        step_velocity: Caller-owned flattened step-velocity output.
        tangent_velocity_dot: Caller-owned tangent-derivative action output.

    Returns:
        None. All results are written to the caller-owned output arrays.
    """
    work_item = wp.tid()
    cell_terms_at(
        work_item,
        0,
        1,
        q_link,
        qd_link,
        basis_z1,
        basis_z2,
        basis_rows,
        reference_z1,
        reference_z2,
        segment_lengths,
        cell_widths,
        num_cells,
        order_zero,
        adjoint,
        tangent_local,
        link_velocity,
        step_velocity,
        tangent_velocity_dot,
    )


@wp.kernel(enable_backward=False)
def cooperative_cell_terms_kernel(
    q_link: wp.array2d[wp.float64],
    qd_link: wp.array2d[wp.float64],
    basis_z1: wp.array2d[wp.float64],
    basis_z2: wp.array2d[wp.float64],
    basis_rows: wp.array2d[wp.int32],
    reference_z1: wp.array2d[wp.float64],
    reference_z2: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    cell_widths: wp.array[wp.float64],
    num_cells: wp.array[wp.int32],
    order_zero: wp.array[wp.int32],
    adjoint: wp.array2d[wp.float64],
    tangent_local: wp.array2d[wp.float64],
    link_velocity: wp.array2d[wp.float64],
    step_velocity: wp.array2d[wp.float64],
    tangent_velocity_dot: wp.array2d[wp.float64],
):
    """Evaluate cell Lie operators with two CUDA threads per cell.

    Launch a one-dimensional grid of twice the number of cells. Threads
    share independent adjoint and tangent columns; no block barrier is required.

    Args:
        q_link: Batched padded segment-local coordinates.
        qd_link: Batched padded segment-local velocities.
        basis_z1: Sparse basis values at the first Magnus node.
        basis_z2: Sparse basis values at the second Magnus node.
        basis_rows: Spatial row occupied by each sparse basis column.
        reference_z1: Reference strains at first Magnus nodes.
        reference_z2: Reference strains at second Magnus nodes.
        segment_lengths: Physical segment lengths.
        cell_widths: Normalized integration-cell widths.
        num_cells: One-entry array containing cells per segment.
        order_zero: One-entry flag selecting the constant-strain shortcut.
        adjoint: Caller-owned flattened cell-adjoint output.
        tangent_local: Caller-owned flattened local-tangent output.
        link_velocity: Caller-owned flattened link-velocity output.
        step_velocity: Caller-owned flattened step-velocity output.
        tangent_velocity_dot: Caller-owned tangent-derivative action output.

    Returns:
        None. All results are written to the caller-owned output arrays.
    """
    item = wp.tid()
    cell_terms_at(
        item // 2,
        item % 2,
        2,
        q_link,
        qd_link,
        basis_z1,
        basis_z2,
        basis_rows,
        reference_z1,
        reference_z2,
        segment_lengths,
        cell_widths,
        num_cells,
        order_zero,
        adjoint,
        tangent_local,
        link_velocity,
        step_velocity,
        tangent_velocity_dot,
    )


@wp.struct
class GVSCellTermsData:
    """GVS link state, strain bases and cell Lie-operator buffers for one world.

    Array fields are non-owning views; the caller must retain their allocations
    for every launch and captured graph. Batched fields use one environment
    (``E = 1``); structural fields use this world's own dimensions.

    Attributes:
        q_link: Batched padded segment-local coordinates.
        qd_link: Batched padded segment-local velocities.
        basis_z1: Sparse basis values at the first Magnus node.
        basis_z2: Sparse basis values at the second Magnus node.
        basis_rows: Spatial row occupied by each sparse basis column.
        reference_z1: Reference strains at first Magnus nodes.
        reference_z2: Reference strains at second Magnus nodes.
        segment_lengths: Physical segment lengths.
        cell_widths: Normalized integration-cell widths.
        num_cells: One-entry array containing cells per segment.
        order_zero: One-entry flag selecting the constant-strain shortcut.
        adjoint: Caller-owned flattened cell-adjoint output.
        tangent_local: Caller-owned flattened local-tangent output.
        link_velocity: Caller-owned flattened link-velocity output.
        step_velocity: Caller-owned flattened step-velocity output.
        tangent_velocity_dot: Caller-owned tangent-derivative action output.
    """

    q_link: wp.array2d[wp.float64]
    qd_link: wp.array2d[wp.float64]
    basis_z1: wp.array2d[wp.float64]
    basis_z2: wp.array2d[wp.float64]
    basis_rows: wp.array2d[wp.int32]
    reference_z1: wp.array2d[wp.float64]
    reference_z2: wp.array2d[wp.float64]
    segment_lengths: wp.array[wp.float64]
    cell_widths: wp.array[wp.float64]
    num_cells: wp.array[wp.int32]
    order_zero: wp.array[wp.int32]
    adjoint: wp.array2d[wp.float64]
    tangent_local: wp.array2d[wp.float64]
    link_velocity: wp.array2d[wp.float64]
    step_velocity: wp.array2d[wp.float64]
    tangent_velocity_dot: wp.array2d[wp.float64]


@wp.kernel(enable_backward=False)
def packed_cell_terms_kernel(
    world_data: wp.array(dtype=GVSCellTermsData), work_items: wp.array(dtype=wp.vec3i)
):
    """Compute general cell-local Lie data for one batched cell work item.

    Launch with ``dim=len(work_items)``; each thread handles one work item.

    Args:
        world_data: Device array of per-world data views, including caller-owned
            outputs and workspaces. Keep every referenced allocation alive.
        work_items: INT32 triples ``(world_index, local_index, 0)``. The local
            index is local cell. The third component is reserved.

    Returns:
        None. All results are written to the caller-owned output arrays.
    """
    item = wp.tid()
    entry = work_items[item]
    data = world_data[entry[0]]
    cell_terms_at(
        entry[1],
        0,
        1,
        data.q_link,
        data.qd_link,
        data.basis_z1,
        data.basis_z2,
        data.basis_rows,
        data.reference_z1,
        data.reference_z2,
        data.segment_lengths,
        data.cell_widths,
        data.num_cells,
        data.order_zero,
        data.adjoint,
        data.tangent_local,
        data.link_velocity,
        data.step_velocity,
        data.tangent_velocity_dot,
    )


@wp.kernel(enable_backward=False)
def packed_cooperative_cell_terms_kernel(
    world_data: wp.array(dtype=GVSCellTermsData), work_items: wp.array(dtype=wp.vec3i)
):
    """Compute general cell-local Lie data for one batched cell work item.

    Launch with ``dim=(len(work_items), 2)`` and ``block_dim=128``.
    Two threads share each cell without a block barrier.

    Args:
        world_data: Device array of per-world data views, including caller-owned
            outputs and workspaces. Keep every referenced allocation alive.
        work_items: INT32 triples ``(world_index, local_index, 0)``. The local
            index is local cell. The third component is reserved.

    Returns:
        None. All results are written to the caller-owned output arrays.
    """
    item, lane = wp.tid()
    entry = work_items[item]
    data = world_data[entry[0]]
    cell_terms_at(
        entry[1],
        lane,
        2,
        data.q_link,
        data.qd_link,
        data.basis_z1,
        data.basis_z2,
        data.basis_rows,
        data.reference_z1,
        data.reference_z2,
        data.segment_lengths,
        data.cell_widths,
        data.num_cells,
        data.order_zero,
        data.adjoint,
        data.tangent_local,
        data.link_velocity,
        data.step_velocity,
        data.tangent_velocity_dot,
    )


def launch_cell_terms(
    q_link: wp.array2d[wp.float64],
    qd_link: wp.array2d[wp.float64],
    basis_z1: wp.array2d[wp.float64],
    basis_z2: wp.array2d[wp.float64],
    basis_rows: wp.array2d[wp.int32],
    reference_z1: wp.array2d[wp.float64],
    reference_z2: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    cell_widths: wp.array[wp.float64],
    num_cells: wp.array[wp.int32],
    order_zero: wp.array[wp.int32],
    adjoint: wp.array2d[wp.float64],
    tangent_local: wp.array2d[wp.float64],
    link_velocity: wp.array2d[wp.float64],
    step_velocity: wp.array2d[wp.float64],
    tangent_velocity_dot: wp.array2d[wp.float64],
):
    """Launch parallel GVS link-cell Lie-operator evaluation.

    Args:
        q_link: Padded FP64 link coordinates with shape
            ``(batch_size * num_segments, max_dof)``.
        qd_link: Padded link velocities with the same shape as ``q_link``.
        basis_z1: Sparse basis values at the first Magnus node.
        basis_z2: Sparse basis values at the second Magnus node.
        basis_rows: Spatial row index for each sparse local basis column.
        reference_z1: Reference strains at first Magnus nodes.
        reference_z2: Reference strains at second Magnus nodes.
        segment_lengths: FP64 length of every segment.
        cell_widths: Flattened width of every integration cell.
        num_cells: One-entry INT32 array containing cells per segment.
        order_zero: Reserved one-entry INT32 array used by the shape-generic
            JAX/Warp call contract.
        adjoint: Preallocated flattened cell-adjoint output.
        tangent_local: Preallocated flattened local-tangent output.
        link_velocity: Preallocated flattened link-velocity output.
        step_velocity: Preallocated flattened cell-step velocity output.
        tangent_velocity_dot: Preallocated flattened tangent derivative action.

    Returns:
        None. Outputs are written in place.
    """
    work_items = q_link.shape[0] * reference_z1.shape[0] // segment_lengths.shape[0]
    cooperative = q_link.device.is_cuda
    wp.launch(
        cooperative_cell_terms_kernel if cooperative else cell_terms_kernel,
        dim=work_items * 2 if cooperative else work_items,
        inputs=[
            q_link,
            qd_link,
            basis_z1,
            basis_z2,
            basis_rows,
            reference_z1,
            reference_z2,
            segment_lengths,
            cell_widths,
            num_cells,
            order_zero,
        ],
        outputs=[
            adjoint,
            tangent_local,
            link_velocity,
            step_velocity,
            tangent_velocity_dot,
        ],
        block_dim=128,
    )


scalable_cell_terms = wp.jax_callable(launch_cell_terms, num_outputs=5)
__all__ = ["cell_terms_kernel", "launch_cell_terms"]


__all__ += [
    "cooperative_cell_terms_kernel",
    "GVSCellTermsData",
    "packed_cell_terms_kernel",
    "packed_cooperative_cell_terms_kernel",
]
