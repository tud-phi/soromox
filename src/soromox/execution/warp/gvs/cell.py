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
        basis_z1,
        basis_rows,
        reference_z1,
        q_link,
        cell_item,
        segment,
        coordinate_item,
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
    forward = _forward_coefficients(angle_sq)
    translation = _translation(omega, linear, forward)

    row = int(0)
    while row < SPATIAL_DIM:
        column = int(0)
        while column < SPATIAL_DIM:
            adjoint[output_base_row + row, column] = _adjoint_inverse_entry(
                omega,
                translation,
                angle_sq,
                forward,
                row,
                column,
            )
            column += 1
        row += 1

    local_column = int(0)
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
        local_column += 1

    link = _left_action(magnus, magnus_dot, coefficients)
    step = _adjoint_inverse_action(omega, translation, angle_sq, forward, link)
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
        tangent_velocity_dot[output_base_row + row, 0] = tangent_dot_velocity_value[row]
        row += 1


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

    # ``num_cells`` has one entry whose value is not available to Python here.
    # Derive the work count from the reference-strain rows instead.
    work_items = (q_link.shape[0] * reference_z1.shape[0]) // segment_lengths.shape[0]
    wp.launch(
        cell_terms_kernel,
        dim=work_items,
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
