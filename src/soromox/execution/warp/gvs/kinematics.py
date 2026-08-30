# ruff: noqa: I001, SIM109, UP018
"""Fused, runtime-shaped GVS pose and inertial-Jacobian Warp kernels."""

from __future__ import annotations

import warp as wp

from soromox.execution.warp.common.se3 import (
    Vec6d,
    _ad_action,
    _adjoint_inverse_action,
    _forward_coefficients,
    _left_action,
    _left_coefficients,
    _rotation_entry,
    _translation,
)

wp.set_module_options({"enable_backward": False})

SPATIAL_DIM = 6
BASIS_FAMILY_MONOMIAL = wp.constant(0)
BASIS_FAMILY_LEGENDRE = wp.constant(1)
BASIS_FAMILY_CHEBYSHEV = wp.constant(2)
BASIS_FAMILY_FOURIER = wp.constant(3)
BASIS_FAMILY_GAUSSIAN = wp.constant(4)
BASIS_FAMILY_IMQ = wp.constant(5)


@wp.func
def _relative_pose_from_inverse_adjoint(
    adjoint: wp.array2d[wp.float64], base_row: int
) -> wp.mat44d:
    """Reconstruct one relative SE(3) pose from its inverse adjoint.

    Args:
        adjoint: Flattened inverse-adjoint matrix storage.
        base_row: First row of the selected six-by-six operator.

    Returns:
        Relative homogeneous transform represented by the operator.
    """

    result = wp.mat44d()
    row = int(0)
    while row < 3:
        column = int(0)
        while column < 3:
            result[row, column] = adjoint[base_row + column, row]
            column += 1
        row += 1
    p_hat = wp.mat33d()
    row = int(0)
    while row < 3:
        column = int(0)
        while column < 3:
            value = wp.float64(0.0)
            k = int(0)
            while k < 3:
                value -= result[row, k] * adjoint[base_row + 3 + k, column]
                k += 1
            p_hat[row, column] = value
            column += 1
        row += 1
    result[0, 3] = p_hat[2, 1]
    result[1, 3] = p_hat[0, 2]
    result[2, 3] = p_hat[1, 0]
    result[3, 3] = wp.float64(1.0)
    return result


@wp.func
def _relative_pose_from_magnus(magnus: Vec6d) -> wp.mat44d:
    """Evaluate an SE(3) exponential from one Magnus vector.

    Args:
        magnus: Integrated six-dimensional strain.

    Returns:
        Relative homogeneous transform ``exp(magnus)``.
    """

    omega = wp.vec3d(magnus[0], magnus[1], magnus[2])
    linear = wp.vec3d(magnus[3], magnus[4], magnus[5])
    angle_sq = wp.dot(omega, omega)
    forward = _forward_coefficients(angle_sq)
    translation = _translation(omega, linear, forward)
    result = wp.mat44d()
    row = int(0)
    while row < 3:
        column = int(0)
        while column < 3:
            result[row, column] = _rotation_entry(omega, angle_sq, forward, row, column)
            column += 1
        result[row, 3] = translation[row]
        row += 1
    result[3, 3] = wp.float64(1.0)
    return result


@wp.func
def _load_node_pose(node_pose: wp.array2d[wp.float64], node_item: int) -> wp.mat44d:
    """Load one homogeneous transform from flattened node storage.

    Args:
        node_pose: Flattened node-pose workspace.
        node_item: Selected node index.

    Returns:
        Four-by-four homogeneous transform at the node.
    """

    result = wp.mat44d()
    row = int(0)
    while row < 4:
        column = int(0)
        while column < 4:
            result[row, column] = node_pose[node_item * 4 + row, column]
            column += 1
        row += 1
    return result


@wp.func
def _store_node_pose(
    node_pose: wp.array2d[wp.float64], node_item: int, value: wp.mat44d
):
    """Store one homogeneous transform in flattened node storage.

    Args:
        node_pose: Flattened node-pose workspace.
        node_item: Destination node index.
        value: Four-by-four homogeneous transform to store.

    Returns:
        None. ``node_pose`` is updated in place.
    """

    row = int(0)
    while row < 4:
        column = int(0)
        while column < 4:
            node_pose[node_item * 4 + row, column] = value[row, column]
            column += 1
        row += 1


@wp.func
def _basis_width(
    basis_family: int,
    active_params: wp.array2d[wp.float64],
    order_params: wp.array2d[wp.float64],
    segment: int,
    row: int,
) -> int:
    """Return the padded coordinate width of one strain row.

    Args:
        basis_family: Integer basis-family identifier.
        active_params: Per-segment row activation parameters.
        order_params: Per-segment row basis orders.
        segment: Selected segment index.
        row: Selected spatial strain row.

    Returns:
        Number of local basis columns occupied by the row.
    """

    active = int(active_params[segment, row])
    order = int(order_params[segment, row])
    if basis_family == BASIS_FAMILY_FOURIER:
        return active * (2 * order + 1)
    return active * (order + 1)


@wp.func
def _basis_slot_value(
    basis_family: int, x: wp.float64, order: int, slot: int
) -> wp.float64:
    """Evaluate one scalar basis slot at a normalized coordinate.

    Args:
        basis_family: Integer basis-family identifier.
        x: Normalized segment coordinate.
        order: Configured basis order.
        slot: Basis slot within one spatial row.

    Returns:
        Scalar basis value for polynomial, orthogonal, Fourier, Gaussian, or
        inverse-multiquadric families.
    """

    value = wp.float64(0.0)
    if basis_family == BASIS_FAMILY_MONOMIAL:
        value = wp.pow(x, wp.float64(slot))
    elif (
        basis_family == BASIS_FAMILY_LEGENDRE or basis_family == BASIS_FAMILY_CHEBYSHEV
    ):
        z = wp.float64(2.0) * x - wp.float64(1.0)
        previous = wp.float64(1.0)
        current = z
        if slot == 0:
            value = previous
        elif slot == 1:
            value = current
        else:
            n = int(1)
            while n < slot:
                next_value = wp.float64(0.0)
                if basis_family == BASIS_FAMILY_LEGENDRE:
                    nf = wp.float64(n)
                    next_value = (
                        (wp.float64(2.0) * nf + wp.float64(1.0)) * z * current
                        - nf * previous
                    ) / (nf + wp.float64(1.0))
                else:
                    next_value = wp.float64(2.0) * z * current - previous
                previous = current
                current = next_value
                n += 1
            value = current
    elif basis_family == BASIS_FAMILY_FOURIER:
        if slot == 0:
            value = wp.float64(1.0)
        else:
            mode = (slot + 1) // 2
            angle = wp.float64(2.0) * wp.float64(wp.pi) * wp.float64(mode) * x
            if slot % 2 == 1:
                value = wp.cos(angle)
            else:
                value = wp.sin(angle)
    elif basis_family == BASIS_FAMILY_GAUSSIAN or basis_family == BASIS_FAMILY_IMQ:
        order_safe = wp.max(order, 1)
        center = wp.float64(slot) / wp.float64(order_safe)
        delta = x - center
        if order == 0:
            if slot == 0:
                value = wp.float64(1.0)
        elif basis_family == BASIS_FAMILY_GAUSSIAN:
            c = (
                wp.float64(2.0)
                * wp.sqrt(wp.log(wp.float64(2.0)))
                * wp.float64(order_safe)
            )
            value = wp.exp(-delta * delta * c * c)
        else:
            c = wp.float64(2.0) * wp.sqrt(wp.float64(3.0)) * wp.float64(order_safe)
            value = wp.float64(1.0) / wp.sqrt(wp.float64(1.0) + delta * delta * c * c)
    return value


@wp.func
def _basis_column_value(
    basis_type_index: wp.array[wp.int32],
    active_params: wp.array2d[wp.float64],
    order_params: wp.array2d[wp.float64],
    basis_rows: wp.array2d[wp.int32],
    segment_lengths: wp.array[wp.float64],
    scale_rotations: int,
    segment: int,
    column: int,
    x: wp.float64,
) -> wp.float64:
    """Evaluate one padded GVS link-basis column.

    Args:
        basis_type_index: Basis-family identifier for every segment.
        active_params: Per-segment row activation parameters.
        order_params: Per-segment row basis orders.
        basis_rows: Spatial row occupied by each local column.
        segment_lengths: Physical link lengths.
        scale_rotations: Whether angular rows are divided by link length.
        segment: Selected segment index.
        column: Selected padded local basis column.
        x: Normalized segment coordinate.

    Returns:
        Scalar value of the requested sparse basis column.
    """

    row = basis_rows[segment, column]
    if row < 0:
        return wp.float64(0.0)
    basis_family = basis_type_index[segment]
    offset = int(0)
    previous_row = int(0)
    while previous_row < row:
        offset += _basis_width(
            basis_family, active_params, order_params, segment, previous_row
        )
        previous_row += 1
    slot = column - offset
    value = _basis_slot_value(basis_family, x, int(order_params[segment, row]), slot)
    if scale_rotations != 0 and row < 3:
        value /= segment_lengths[segment]
    return value


@wp.kernel(enable_backward=False)
def gvs_node_poses_kernel(
    joint_adjoint: wp.array2d[wp.float64],
    cell_adjoint: wp.array2d[wp.float64],
    base_transform: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    num_cells_array: wp.array[wp.int32],
    node_pose: wp.array2d[wp.float64],
):
    """Build GVS cell-boundary poses without Jacobian propagation.

    Args:
        joint_adjoint: Flattened inverse adjoints for every joint.
        cell_adjoint: Flattened inverse adjoints for every full link cell.
        base_transform: Absolute robot base transform.
        segment_lengths: Physical link lengths used for the segment count.
        num_cells_array: One-entry array containing the padded cell count.
        node_pose: Caller-owned flattened node-pose workspace.

    Returns:
        None. Cell-boundary poses are written in place.
    """

    environment = wp.tid()
    num_segments = segment_lengths.shape[0]
    num_cells = num_cells_array[0]
    current_pose = wp.mat44d()
    row = int(0)
    while row < 4:
        column = int(0)
        while column < 4:
            current_pose[row, column] = base_transform[row, column]
            column += 1
        row += 1
    segment = int(0)
    while segment < num_segments:
        joint_base = (environment * num_segments + segment) * SPATIAL_DIM
        current_pose = current_pose * _relative_pose_from_inverse_adjoint(
            joint_adjoint, joint_base
        )
        node_item = (environment * num_segments + segment) * (num_cells + 1)
        _store_node_pose(node_pose, node_item, current_pose)
        cell = int(0)
        while cell < num_cells:
            cell_base = (
                (environment * num_segments + segment) * num_cells + cell
            ) * SPATIAL_DIM
            current_pose = current_pose * _relative_pose_from_inverse_adjoint(
                cell_adjoint, cell_base
            )
            _store_node_pose(node_pose, node_item + cell + 1, current_pose)
            cell += 1
        segment += 1


@wp.kernel(enable_backward=False)
def gvs_node_states_kernel(
    joint_adjoint: wp.array2d[wp.float64],
    joint_tangent: wp.array2d[wp.float64],
    cell_adjoint: wp.array2d[wp.float64],
    cell_tangent_local: wp.array2d[wp.float64],
    link_global_to_local: wp.array2d[wp.int32],
    base_transform: wp.array2d[wp.float64],
    num_cells_array: wp.array[wp.int32],
    node_pose: wp.array2d[wp.float64],
    node_jacobian: wp.array2d[wp.float64],
):
    """Build reusable GVS cell-boundary pose and body-Jacobian states.

    Args:
        joint_adjoint: Flattened inverse adjoints for every joint.
        joint_tangent: Joint tangents in active coordinates.
        cell_adjoint: Flattened inverse adjoints for every full link cell.
        cell_tangent_local: Full-cell tangents in padded local coordinates.
        link_global_to_local: Active-to-local link-coordinate map.
        base_transform: Absolute robot base transform.
        num_cells_array: One-entry array containing the padded cell count.
        node_pose: Caller-owned flattened node-pose workspace.
        node_jacobian: Caller-owned flattened body-Jacobian workspace.

    Returns:
        None. Boundary states are written in place.
    """

    environment = wp.tid()
    num_dofs = link_global_to_local.shape[1]
    num_segments = link_global_to_local.shape[0]
    num_cells = num_cells_array[0]
    current_pose = wp.mat44d()
    row = int(0)
    while row < 4:
        column = int(0)
        while column < 4:
            current_pose[row, column] = base_transform[row, column]
            column += 1
        row += 1

    segment = int(0)
    while segment < num_segments:
        joint_base = (environment * num_segments + segment) * SPATIAL_DIM
        relative = _relative_pose_from_inverse_adjoint(joint_adjoint, joint_base)
        current_pose = current_pose * relative
        node_item = (environment * num_segments + segment) * (num_cells + 1)
        _store_node_pose(node_pose, node_item, current_pose)
        row = int(0)
        while row < SPATIAL_DIM:
            column = int(0)
            while column < num_dofs:
                value = wp.float64(0.0)
                k = int(0)
                while k < SPATIAL_DIM:
                    previous = wp.float64(0.0)
                    if segment > 0:
                        previous = node_jacobian[
                            (
                                (environment * num_segments + segment - 1)
                                * (num_cells + 1)
                                + num_cells
                            )
                            * SPATIAL_DIM
                            + k,
                            column,
                        ]
                    value += joint_adjoint[joint_base + row, k] * (
                        previous + joint_tangent[joint_base + k, column]
                    )
                    k += 1
                node_jacobian[node_item * SPATIAL_DIM + row, column] = value
                column += 1
            row += 1

        cell = int(0)
        while cell < num_cells:
            cell_base = (
                (environment * num_segments + segment) * num_cells + cell
            ) * SPATIAL_DIM
            relative = _relative_pose_from_inverse_adjoint(cell_adjoint, cell_base)
            current_pose = current_pose * relative
            previous_item = node_item + cell
            next_item = previous_item + 1
            _store_node_pose(node_pose, next_item, current_pose)
            row = int(0)
            while row < SPATIAL_DIM:
                column = int(0)
                while column < num_dofs:
                    local = link_global_to_local[segment, column]
                    value = wp.float64(0.0)
                    k = int(0)
                    while k < SPATIAL_DIM:
                        tangent = wp.float64(0.0)
                        if local >= 0:
                            tangent = cell_tangent_local[cell_base + k, local]
                        value += cell_adjoint[cell_base + row, k] * (
                            node_jacobian[previous_item * SPATIAL_DIM + k, column]
                            + tangent
                        )
                        k += 1
                    node_jacobian[next_item * SPATIAL_DIM + row, column] = value
                    column += 1
                row += 1
            cell += 1
        segment += 1


@wp.kernel(enable_backward=False)
def gvs_cooperative_node_states_kernel(
    joint_adjoint: wp.array2d[wp.float64],
    joint_tangent: wp.array2d[wp.float64],
    cell_adjoint: wp.array2d[wp.float64],
    cell_tangent_local: wp.array2d[wp.float64],
    link_global_to_local: wp.array2d[wp.int32],
    base_transform: wp.array2d[wp.float64],
    num_cells_array: wp.array[wp.int32],
    node_pose: wp.array2d[wp.float64],
    node_jacobian: wp.array2d[wp.float64],
):
    """Build GVS boundary states cooperatively on CUDA.

    One persistent block owns an environment. Joint and cell traversal remains
    serial, while the six-by-DOF Jacobian propagation is distributed across
    lanes. This is the CUDA schedule for the same recurrence implemented by
    :func:`gvs_node_states_kernel`.

    Args:
        joint_adjoint: Flattened inverse adjoints for every joint.
        joint_tangent: Joint tangents in active coordinates.
        cell_adjoint: Flattened inverse adjoints for every full link cell.
        cell_tangent_local: Full-cell tangents in padded local coordinates.
        link_global_to_local: Active-to-local link-coordinate map.
        base_transform: Absolute robot base transform.
        num_cells_array: One-entry array containing the padded cell count.
        node_pose: Caller-owned flattened node-pose workspace.
        node_jacobian: Caller-owned flattened body-Jacobian workspace.

    Returns:
        None. Boundary states are written in place.
    """

    environment, lane = wp.tid()
    num_dofs = link_global_to_local.shape[1]
    num_segments = link_global_to_local.shape[0]
    num_cells = num_cells_array[0]
    barrier = wp.tile_zeros(shape=(1,), dtype=wp.int32, storage="shared")
    current_pose = wp.mat44d()
    if lane == 0:
        row = int(0)
        while row < 4:
            column = int(0)
            while column < 4:
                current_pose[row, column] = base_transform[row, column]
                column += 1
            row += 1
    wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)

    segment = int(0)
    while segment < num_segments:
        joint_base = (environment * num_segments + segment) * SPATIAL_DIM
        node_item = (environment * num_segments + segment) * (num_cells + 1)
        if lane == 0:
            current_pose = current_pose * _relative_pose_from_inverse_adjoint(
                joint_adjoint, joint_base
            )
            _store_node_pose(node_pose, node_item, current_pose)

        entry = lane
        while entry < SPATIAL_DIM * num_dofs:
            row = entry // num_dofs
            column = entry - row * num_dofs
            value = wp.float64(0.0)
            k = int(0)
            while k < SPATIAL_DIM:
                previous = wp.float64(0.0)
                if segment > 0:
                    previous = node_jacobian[
                        (
                            (environment * num_segments + segment - 1) * (num_cells + 1)
                            + num_cells
                        )
                        * SPATIAL_DIM
                        + k,
                        column,
                    ]
                value += joint_adjoint[joint_base + row, k] * (
                    previous + joint_tangent[joint_base + k, column]
                )
                k += 1
            node_jacobian[node_item * SPATIAL_DIM + row, column] = value
            entry += wp.block_dim()
        wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)

        cell = int(0)
        while cell < num_cells:
            cell_base = (
                (environment * num_segments + segment) * num_cells + cell
            ) * SPATIAL_DIM
            previous_item = node_item + cell
            next_item = previous_item + 1
            if lane == 0:
                current_pose = current_pose * _relative_pose_from_inverse_adjoint(
                    cell_adjoint, cell_base
                )
                _store_node_pose(node_pose, next_item, current_pose)

            entry = lane
            while entry < SPATIAL_DIM * num_dofs:
                row = entry // num_dofs
                column = entry - row * num_dofs
                local = link_global_to_local[segment, column]
                value = wp.float64(0.0)
                k = int(0)
                while k < SPATIAL_DIM:
                    tangent = wp.float64(0.0)
                    if local >= 0:
                        tangent = cell_tangent_local[cell_base + k, local]
                    value += cell_adjoint[cell_base + row, k] * (
                        node_jacobian[previous_item * SPATIAL_DIM + k, column] + tangent
                    )
                    k += 1
                node_jacobian[next_item * SPATIAL_DIM + row, column] = value
                entry += wp.block_dim()
            wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
            cell += 1
        segment += 1


@wp.kernel(enable_backward=False)
def gvs_pose_samples_kernel(
    q: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    link_local_to_global: wp.array2d[wp.int32],
    link_basis_rows: wp.array2d[wp.int32],
    link_reference_z1: wp.array2d[wp.float64],
    link_reference_z2: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    integration_points: wp.array2d[wp.float64],
    basis_type_index: wp.array[wp.int32],
    basis_active_params: wp.array2d[wp.float64],
    basis_order_params: wp.array2d[wp.float64],
    magnus_points: wp.array[wp.float64],
    scale_rotations_array: wp.array[wp.int32],
    epsilons: wp.array[wp.float64],
    num_cells_array: wp.array[wp.int32],
    node_pose: wp.array2d[wp.float64],
    poses: wp.array4d[wp.float64],
):
    """Evaluate GVS poses without Jacobian recurrence or output work.

    Args:
        q: Batched active configurations ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        link_local_to_global: Padded local-to-active link-coordinate map.
        link_basis_rows: Spatial row occupied by every local basis column.
        link_reference_z1: Reference strain at first Magnus nodes.
        link_reference_z2: Reference strain at second Magnus nodes.
        segment_lengths: Physical link lengths.
        segment_starts: Cumulative link-start positions.
        integration_points: Normalized cell-boundary coordinates.
        basis_type_index: Basis-family index for every segment.
        basis_active_params: Per-row basis activation parameters.
        basis_order_params: Per-row basis-order parameters.
        magnus_points: Two normalized Gauss points within a partial cell.
        scale_rotations_array: One-entry rotational-basis scaling flag.
        epsilons: Exponential and tangent thresholds.
        num_cells_array: One-entry array containing the padded cell count.
        node_pose: Precomputed flattened node poses.
        poses: Caller-owned SE(3) pose output.

    Returns:
        None. Pose outputs are written in place.
    """

    environment, sample = wp.tid()
    num_segments = segment_lengths.shape[0]
    num_cells = num_cells_array[0]
    abscissa = s[environment, sample]
    segment = int(0)
    i = int(0)
    while i < num_segments + 1:
        if abscissa > segment_starts[i]:
            segment = i
        i += 1
    segment = wp.max(0, wp.min(segment, num_segments - 1))
    length = segment_lengths[segment]
    x = wp.float64(0.0)
    if wp.abs(length) >= epsilons[0]:
        x = wp.clamp(
            (abscissa - segment_starts[segment]) / length,
            wp.float64(0.0),
            wp.float64(1.0),
        )
    cell = int(0)
    i = int(0)
    while i < num_cells + 1:
        if x >= integration_points[segment, i]:
            cell = i
        i += 1
    cell = wp.max(0, wp.min(cell, num_cells - 1))
    x_base = integration_points[segment, cell]
    width = x - x_base
    ds = width * length
    node_item = (environment * num_segments + segment) * (num_cells + 1) + cell
    base_pose = _load_node_pose(node_pose, node_item)
    xi1 = Vec6d()
    xi2 = Vec6d()
    local = int(0)
    reference_item = segment * num_cells + cell
    while local < link_local_to_global.shape[1]:
        global_index = link_local_to_global[segment, local]
        coordinate = wp.float64(0.0)
        if global_index >= 0:
            coordinate = q[environment, global_index]
        basis_row = link_basis_rows[segment, local]
        if basis_row >= 0:
            x1 = x_base + magnus_points[0] * width
            x2 = x_base + magnus_points[1] * width
            xi1[basis_row] += (
                _basis_column_value(
                    basis_type_index,
                    basis_active_params,
                    basis_order_params,
                    link_basis_rows,
                    segment_lengths,
                    scale_rotations_array[0],
                    segment,
                    local,
                    x1,
                )
                * coordinate
            )
            xi2[basis_row] += (
                _basis_column_value(
                    basis_type_index,
                    basis_active_params,
                    basis_order_params,
                    link_basis_rows,
                    segment_lengths,
                    scale_rotations_array[0],
                    segment,
                    local,
                    x2,
                )
                * coordinate
            )
        local += 1
    row = int(0)
    while row < SPATIAL_DIM:
        xi1[row] += link_reference_z1[reference_item, row]
        xi2[row] += link_reference_z2[reference_item, row]
        row += 1
    coefficient = wp.sqrt(wp.float64(3.0)) * ds * ds / wp.float64(12.0)
    magnus = ds * wp.float64(0.5) * (xi1 + xi2) + coefficient * _ad_action(xi1, xi2)
    pose = base_pose * _relative_pose_from_magnus(magnus)
    row = int(0)
    while row < 4:
        column = int(0)
        while column < 4:
            poses[environment, sample, row, column] = pose[row, column]
            column += 1
        row += 1


@wp.func
def write_gvs_sample_jacobian(
    environment: int,
    sample: int,
    segment: int,
    node_item: int,
    x_base: wp.float64,
    width: wp.float64,
    ds: wp.float64,
    magnus: Vec6d,
    xi1: Vec6d,
    xi2: Vec6d,
    pose: wp.mat44d,
    link_global_to_local: wp.array2d[wp.int32],
    link_basis_rows: wp.array2d[wp.int32],
    segment_lengths: wp.array[wp.float64],
    basis_type_index: wp.array[wp.int32],
    basis_active_params: wp.array2d[wp.float64],
    basis_order_params: wp.array2d[wp.float64],
    magnus_points: wp.array[wp.float64],
    scale_rotations_array: wp.array[wp.int32],
    node_jacobian: wp.array2d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Write one GVS inertial Jacobian from a partial-cell state.

    Args:
        environment: Environment index.
        sample: Backbone-sample index.
        segment: Segment owning the sample.
        node_item: Flattened boundary-state index.
        x_base: Normalized coordinate at the active cell base.
        width: Normalized partial-cell width.
        ds: Physical partial-cell width.
        magnus: Partial-cell Magnus vector.
        xi1: Strain at the first partial-cell Magnus node.
        xi2: Strain at the second partial-cell Magnus node.
        pose: Absolute SE(3) pose at the sample.
        link_global_to_local: Active-to-local link-coordinate map.
        link_basis_rows: Spatial row occupied by each local basis column.
        segment_lengths: Physical link lengths.
        basis_type_index: Basis-family index for every segment.
        basis_active_params: Per-row basis activation parameters.
        basis_order_params: Per-row basis-order parameters.
        magnus_points: Two normalized Gauss points within a partial cell.
        scale_rotations_array: One-entry rotational-basis scaling flag.
        node_jacobian: Precomputed flattened boundary body Jacobians.
        jacobians: Caller-owned inertial-Jacobian output.

    Returns:
        None. One Jacobian output slice is written in place.
    """

    omega = wp.vec3d(magnus[0], magnus[1], magnus[2])
    linear = wp.vec3d(magnus[3], magnus[4], magnus[5])
    angle_sq = wp.dot(omega, omega)
    coefficients = _left_coefficients(angle_sq)
    forward = _forward_coefficients(angle_sq)
    translation = _translation(omega, linear, forward)
    commutator_coefficient = wp.sqrt(wp.float64(3.0)) * ds * ds / wp.float64(12.0)
    x1 = x_base + magnus_points[0] * width
    x2 = x_base + magnus_points[1] * width
    column = int(0)
    while column < jacobians.shape[3]:
        tangent = Vec6d()
        local = link_global_to_local[segment, column]
        if local >= 0:
            basis_row = link_basis_rows[segment, local]
            if basis_row >= 0:
                value_z1 = _basis_column_value(
                    basis_type_index,
                    basis_active_params,
                    basis_order_params,
                    link_basis_rows,
                    segment_lengths,
                    scale_rotations_array[0],
                    segment,
                    local,
                    x1,
                )
                value_z2 = _basis_column_value(
                    basis_type_index,
                    basis_active_params,
                    basis_order_params,
                    link_basis_rows,
                    segment_lengths,
                    scale_rotations_array[0],
                    segment,
                    local,
                    x2,
                )
                basis_z1 = Vec6d()
                basis_z2 = Vec6d()
                basis_z1[basis_row] = value_z1
                basis_z2[basis_row] = value_z2
                magnus_basis = ds * wp.float64(0.5) * (
                    basis_z1 + basis_z2
                ) + commutator_coefficient * (
                    _ad_action(xi1, basis_z2) - _ad_action(xi2, basis_z1)
                )
                tangent = _left_action(magnus, magnus_basis, coefficients)
        source = Vec6d()
        k = int(0)
        while k < SPATIAL_DIM:
            source[k] = node_jacobian[node_item * SPATIAL_DIM + k, column] + tangent[k]
            k += 1
        body = _adjoint_inverse_action(omega, translation, angle_sq, forward, source)
        row = int(0)
        while row < SPATIAL_DIM:
            local_row = row if row < 3 else row - 3
            offset = 0 if row < 3 else 3
            output = wp.float64(0.0)
            k = int(0)
            while k < 3:
                output += pose[local_row, k] * body[offset + k]
                k += 1
            jacobians[environment, sample, row, column] = output
            row += 1
        column += 1


@wp.kernel(enable_backward=False)
def gvs_jacobian_samples_kernel(
    q: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    link_local_to_global: wp.array2d[wp.int32],
    link_global_to_local: wp.array2d[wp.int32],
    link_basis_rows: wp.array2d[wp.int32],
    link_reference_z1: wp.array2d[wp.float64],
    link_reference_z2: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    integration_points: wp.array2d[wp.float64],
    basis_type_index: wp.array[wp.int32],
    basis_active_params: wp.array2d[wp.float64],
    basis_order_params: wp.array2d[wp.float64],
    magnus_points: wp.array[wp.float64],
    scale_rotations_array: wp.array[wp.int32],
    epsilons: wp.array[wp.float64],
    num_cells_array: wp.array[wp.int32],
    node_pose: wp.array2d[wp.float64],
    node_jacobian: wp.array2d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Evaluate GVS inertial Jacobians without pose output writes.

    Args:
        q: Batched active configurations ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        link_local_to_global: Padded local-to-active link-coordinate map.
        link_global_to_local: Active-to-local link-coordinate map.
        link_basis_rows: Spatial row occupied by every local basis column.
        link_reference_z1: Reference strain at first Magnus nodes.
        link_reference_z2: Reference strain at second Magnus nodes.
        segment_lengths: Physical link lengths.
        segment_starts: Cumulative link-start positions.
        integration_points: Normalized cell-boundary coordinates.
        basis_type_index: Basis-family index for every segment.
        basis_active_params: Per-row basis activation parameters.
        basis_order_params: Per-row basis-order parameters.
        magnus_points: Two normalized Gauss points within a partial cell.
        scale_rotations_array: One-entry rotational-basis scaling flag.
        epsilons: Exponential and tangent thresholds.
        num_cells_array: One-entry array containing the padded cell count.
        node_pose: Precomputed flattened node poses.
        node_jacobian: Precomputed flattened boundary body Jacobians.
        jacobians: Caller-owned inertial-Jacobian output.

    Returns:
        None. Jacobian outputs are written in place.
    """

    environment, sample = wp.tid()
    num_segments = segment_lengths.shape[0]
    num_cells = num_cells_array[0]
    abscissa = s[environment, sample]
    segment = int(0)
    i = int(0)
    while i < num_segments + 1:
        if abscissa > segment_starts[i]:
            segment = i
        i += 1
    segment = wp.max(0, wp.min(segment, num_segments - 1))
    length = segment_lengths[segment]
    x = wp.float64(0.0)
    if wp.abs(length) >= epsilons[0]:
        x = wp.clamp(
            (abscissa - segment_starts[segment]) / length,
            wp.float64(0.0),
            wp.float64(1.0),
        )
    cell = int(0)
    i = int(0)
    while i < num_cells + 1:
        if x >= integration_points[segment, i]:
            cell = i
        i += 1
    cell = wp.max(0, wp.min(cell, num_cells - 1))
    x_base = integration_points[segment, cell]
    width = x - x_base
    ds = width * length
    node_item = (environment * num_segments + segment) * (num_cells + 1) + cell
    base_pose = _load_node_pose(node_pose, node_item)
    xi1 = Vec6d()
    xi2 = Vec6d()
    local = int(0)
    reference_item = segment * num_cells + cell
    while local < link_local_to_global.shape[1]:
        global_index = link_local_to_global[segment, local]
        coordinate = wp.float64(0.0)
        if global_index >= 0:
            coordinate = q[environment, global_index]
        basis_row = link_basis_rows[segment, local]
        if basis_row >= 0:
            x1 = x_base + magnus_points[0] * width
            x2 = x_base + magnus_points[1] * width
            xi1[basis_row] += (
                _basis_column_value(
                    basis_type_index,
                    basis_active_params,
                    basis_order_params,
                    link_basis_rows,
                    segment_lengths,
                    scale_rotations_array[0],
                    segment,
                    local,
                    x1,
                )
                * coordinate
            )
            xi2[basis_row] += (
                _basis_column_value(
                    basis_type_index,
                    basis_active_params,
                    basis_order_params,
                    link_basis_rows,
                    segment_lengths,
                    scale_rotations_array[0],
                    segment,
                    local,
                    x2,
                )
                * coordinate
            )
        local += 1
    row = int(0)
    while row < SPATIAL_DIM:
        xi1[row] += link_reference_z1[reference_item, row]
        xi2[row] += link_reference_z2[reference_item, row]
        row += 1
    coefficient = wp.sqrt(wp.float64(3.0)) * ds * ds / wp.float64(12.0)
    magnus = ds * wp.float64(0.5) * (xi1 + xi2) + coefficient * _ad_action(xi1, xi2)
    pose = base_pose * _relative_pose_from_magnus(magnus)
    write_gvs_sample_jacobian(
        environment,
        sample,
        segment,
        node_item,
        x_base,
        width,
        ds,
        magnus,
        xi1,
        xi2,
        pose,
        link_global_to_local,
        link_basis_rows,
        segment_lengths,
        basis_type_index,
        basis_active_params,
        basis_order_params,
        magnus_points,
        scale_rotations_array,
        node_jacobian,
        jacobians,
    )


@wp.kernel(enable_backward=False)
def gvs_samples_kernel(
    q: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    link_local_to_global: wp.array2d[wp.int32],
    link_global_to_local: wp.array2d[wp.int32],
    link_basis_rows: wp.array2d[wp.int32],
    link_reference_z1: wp.array2d[wp.float64],
    link_reference_z2: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    integration_points: wp.array2d[wp.float64],
    basis_type_index: wp.array[wp.int32],
    basis_active_params: wp.array2d[wp.float64],
    basis_order_params: wp.array2d[wp.float64],
    magnus_points: wp.array[wp.float64],
    scale_rotations_array: wp.array[wp.int32],
    epsilons: wp.array[wp.float64],
    num_cells_array: wp.array[wp.int32],
    node_pose: wp.array2d[wp.float64],
    node_jacobian: wp.array2d[wp.float64],
    poses: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Evaluate fused GVS poses and inertial Jacobians at runtime samples.

    Args:
        q: Batched active configurations ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        link_local_to_global: Padded local-to-active link-coordinate map.
        link_global_to_local: Active-to-local link-coordinate map.
        link_basis_rows: Spatial row occupied by every local basis column.
        link_reference_z1: Reference strain at first Magnus nodes.
        link_reference_z2: Reference strain at second Magnus nodes.
        segment_lengths: Physical link lengths.
        segment_starts: Cumulative link-start positions.
        integration_points: Normalized cell-boundary coordinates.
        basis_type_index: Basis-family index for every segment.
        basis_active_params: Per-row basis activation parameters.
        basis_order_params: Per-row basis-order parameters.
        magnus_points: Two normalized Gauss points within a partial cell.
        scale_rotations_array: One-entry rotational-basis scaling flag.
        epsilons: Exponential and tangent thresholds.
        num_cells_array: One-entry array containing the padded cell count.
        node_pose: Precomputed flattened node poses.
        node_jacobian: Precomputed flattened boundary body Jacobians.
        poses: Caller-owned SE(3) pose output.
        jacobians: Caller-owned inertial-Jacobian output.

    Returns:
        None. Fused outputs are written in place.
    """

    environment, sample = wp.tid()
    num_segments = segment_lengths.shape[0]
    num_cells = num_cells_array[0]
    abscissa = s[environment, sample]
    segment = int(0)
    i = int(0)
    while i < num_segments + 1:
        if abscissa > segment_starts[i]:
            segment = i
        i += 1
    segment = wp.max(0, wp.min(segment, num_segments - 1))
    length = segment_lengths[segment]
    x = wp.float64(0.0)
    if wp.abs(length) >= epsilons[0]:
        x = wp.clamp(
            (abscissa - segment_starts[segment]) / length,
            wp.float64(0.0),
            wp.float64(1.0),
        )
    cell = int(0)
    i = int(0)
    while i < num_cells + 1:
        if x >= integration_points[segment, i]:
            cell = i
        i += 1
    cell = wp.max(0, wp.min(cell, num_cells - 1))
    x_base = integration_points[segment, cell]
    width = x - x_base
    ds = width * length
    node_item = (environment * num_segments + segment) * (num_cells + 1) + cell
    base_pose = _load_node_pose(node_pose, node_item)
    magnus = Vec6d()
    xi1 = Vec6d()
    xi2 = Vec6d()
    local = int(0)
    reference_item = segment * num_cells + cell
    while local < link_local_to_global.shape[1]:
        global_index = link_local_to_global[segment, local]
        coordinate = wp.float64(0.0)
        if global_index >= 0:
            coordinate = q[environment, global_index]
        row = link_basis_rows[segment, local]
        if row >= 0:
            x1 = x_base + magnus_points[0] * width
            x2 = x_base + magnus_points[1] * width
            basis_value_z1 = _basis_column_value(
                basis_type_index,
                basis_active_params,
                basis_order_params,
                link_basis_rows,
                segment_lengths,
                scale_rotations_array[0],
                segment,
                local,
                x1,
            )
            basis_value_z2 = _basis_column_value(
                basis_type_index,
                basis_active_params,
                basis_order_params,
                link_basis_rows,
                segment_lengths,
                scale_rotations_array[0],
                segment,
                local,
                x2,
            )
            xi1[row] += basis_value_z1 * coordinate
            xi2[row] += basis_value_z2 * coordinate
        local += 1
    row = int(0)
    while row < SPATIAL_DIM:
        xi1[row] += link_reference_z1[reference_item, row]
        xi2[row] += link_reference_z2[reference_item, row]
        row += 1
    commutator_coefficient = wp.sqrt(wp.float64(3.0)) * ds * ds / wp.float64(12.0)
    magnus = ds * wp.float64(0.5) * (xi1 + xi2) + commutator_coefficient * _ad_action(
        xi1, xi2
    )
    relative = _relative_pose_from_magnus(magnus)
    pose = base_pose * relative
    row = int(0)
    while row < 4:
        column = int(0)
        while column < 4:
            poses[environment, sample, row, column] = pose[row, column]
            column += 1
        row += 1

    write_gvs_sample_jacobian(
        environment,
        sample,
        segment,
        node_item,
        x_base,
        width,
        ds,
        magnus,
        xi1,
        xi2,
        pose,
        link_global_to_local,
        link_basis_rows,
        segment_lengths,
        basis_type_index,
        basis_active_params,
        basis_order_params,
        magnus_points,
        scale_rotations_array,
        node_jacobian,
        jacobians,
    )


def launch_gvs_forward_kinematics(
    q: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    joint_adjoint: wp.array2d[wp.float64],
    cell_adjoint: wp.array2d[wp.float64],
    base_transform: wp.array2d[wp.float64],
    link_local_to_global: wp.array2d[wp.int32],
    link_basis_rows: wp.array2d[wp.int32],
    link_reference_z1: wp.array2d[wp.float64],
    link_reference_z2: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    integration_points: wp.array2d[wp.float64],
    basis_type_index: wp.array[wp.int32],
    basis_active_params: wp.array2d[wp.float64],
    basis_order_params: wp.array2d[wp.float64],
    magnus_points: wp.array[wp.float64],
    scale_rotations: wp.array[wp.int32],
    epsilons: wp.array[wp.float64],
    num_cells: wp.array[wp.int32],
    node_pose: wp.array2d[wp.float64],
    poses: wp.array4d[wp.float64],
):
    """Launch the reduced-work allocation-free GVS pose pipeline.

    Args:
        q: Batched active configurations ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        joint_adjoint: Flattened inverse adjoints for every joint.
        cell_adjoint: Flattened inverse adjoints for every full link cell.
        base_transform: Absolute robot base transform.
        link_local_to_global: Padded local-to-active link-coordinate map.
        link_basis_rows: Spatial row occupied by every local basis column.
        link_reference_z1: Reference strain at first full-cell Magnus nodes.
        link_reference_z2: Reference strain at second full-cell Magnus nodes.
        segment_lengths: Physical link lengths.
        segment_starts: Cumulative link-start positions.
        integration_points: Normalized cell-boundary coordinates.
        basis_type_index: Basis-family index for every segment.
        basis_active_params: Per-row basis activation parameters.
        basis_order_params: Per-row basis-order parameters.
        magnus_points: Two normalized Gauss points within a partial cell.
        scale_rotations: One-entry flag for rotational basis scaling.
        epsilons: Exponential and tangent thresholds.
        num_cells: One-entry array containing the padded cell count.
        node_pose: Caller-owned flattened node-pose workspace.
        poses: Caller-owned SE(3) pose output.

    Returns:
        None. Pose-only kernels are enqueued on the current Warp stream.
    """

    wp.launch(
        gvs_node_poses_kernel,
        dim=q.shape[0],
        inputs=[
            joint_adjoint,
            cell_adjoint,
            base_transform,
            segment_lengths,
            num_cells,
        ],
        outputs=[node_pose],
    )
    wp.launch(
        gvs_pose_samples_kernel,
        dim=(q.shape[0], s.shape[1]),
        inputs=[
            q,
            s,
            link_local_to_global,
            link_basis_rows,
            link_reference_z1,
            link_reference_z2,
            segment_lengths,
            segment_starts,
            integration_points,
            basis_type_index,
            basis_active_params,
            basis_order_params,
            magnus_points,
            scale_rotations,
            epsilons,
            num_cells,
            node_pose,
        ],
        outputs=[poses],
        block_dim=128,
    )


def _launch_gvs_node_states(
    q: wp.array2d[wp.float64],
    joint_adjoint: wp.array2d[wp.float64],
    joint_tangent: wp.array2d[wp.float64],
    cell_adjoint: wp.array2d[wp.float64],
    cell_tangent_local: wp.array2d[wp.float64],
    link_global_to_local: wp.array2d[wp.int32],
    base_transform: wp.array2d[wp.float64],
    num_cells: wp.array[wp.int32],
    block_dim: int,
    node_pose: wp.array2d[wp.float64],
    node_jacobian: wp.array2d[wp.float64],
) -> None:
    """Launch the device-appropriate schedule for the GVS state recurrence.

    Args:
        q: Batched active configurations, used for batch size and device.
        joint_adjoint: Flattened inverse adjoints for every joint.
        joint_tangent: Joint tangents in active coordinates.
        cell_adjoint: Flattened inverse adjoints for every full link cell.
        cell_tangent_local: Full-cell tangents in padded local coordinates.
        link_global_to_local: Active-to-local link-coordinate map.
        base_transform: Absolute robot base transform.
        num_cells: One-entry array containing the padded cell count.
        block_dim: CUDA threads in each cooperative environment block.
        node_pose: Caller-owned flattened node-pose workspace.
        node_jacobian: Caller-owned flattened body-Jacobian workspace.

    Returns:
        None. The recurrence kernel is enqueued on the current Warp stream.
    """

    inputs = [
        joint_adjoint,
        joint_tangent,
        cell_adjoint,
        cell_tangent_local,
        link_global_to_local,
        base_transform,
        num_cells,
    ]
    outputs = [node_pose, node_jacobian]
    if q.device.is_cuda:
        wp.launch_tiled(
            gvs_cooperative_node_states_kernel,
            dim=q.shape[0],
            inputs=inputs,
            outputs=outputs,
            block_dim=block_dim,
        )
    else:
        wp.launch(
            gvs_node_states_kernel,
            dim=q.shape[0],
            inputs=inputs,
            outputs=outputs,
        )


def launch_gvs_inertial_jacobians(
    q: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    joint_adjoint: wp.array2d[wp.float64],
    joint_tangent: wp.array2d[wp.float64],
    cell_adjoint: wp.array2d[wp.float64],
    cell_tangent_local: wp.array2d[wp.float64],
    base_transform: wp.array2d[wp.float64],
    link_local_to_global: wp.array2d[wp.int32],
    link_global_to_local: wp.array2d[wp.int32],
    link_basis_rows: wp.array2d[wp.int32],
    link_reference_z1: wp.array2d[wp.float64],
    link_reference_z2: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    integration_points: wp.array2d[wp.float64],
    basis_type_index: wp.array[wp.int32],
    basis_active_params: wp.array2d[wp.float64],
    basis_order_params: wp.array2d[wp.float64],
    magnus_points: wp.array[wp.float64],
    scale_rotations: wp.array[wp.int32],
    epsilons: wp.array[wp.float64],
    num_cells: wp.array[wp.int32],
    block_dim: int,
    node_pose: wp.array2d[wp.float64],
    node_jacobian: wp.array2d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Launch GVS inertial Jacobians without pose output writes.

    Args:
        q: Batched active configurations ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        joint_adjoint: Flattened inverse adjoints for every joint.
        joint_tangent: Joint tangents in active coordinates.
        cell_adjoint: Flattened inverse adjoints for every full link cell.
        cell_tangent_local: Full-cell tangents in padded local coordinates.
        base_transform: Absolute robot base transform.
        link_local_to_global: Padded local-to-active link-coordinate map.
        link_global_to_local: Active-to-local link-coordinate map.
        link_basis_rows: Spatial row occupied by every local basis column.
        link_reference_z1: Reference strain at first full-cell Magnus nodes.
        link_reference_z2: Reference strain at second full-cell Magnus nodes.
        segment_lengths: Physical link lengths.
        segment_starts: Cumulative link-start positions.
        integration_points: Normalized cell-boundary coordinates.
        basis_type_index: Basis-family index for every segment.
        basis_active_params: Per-row basis activation parameters.
        basis_order_params: Per-row basis-order parameters.
        magnus_points: Two normalized Gauss points within a partial cell.
        scale_rotations: One-entry flag for rotational basis scaling.
        epsilons: Exponential and tangent thresholds.
        num_cells: One-entry array containing the padded cell count.
        block_dim: CUDA threads in each cooperative environment block.
        node_pose: Caller-owned flattened node-pose workspace.
        node_jacobian: Caller-owned flattened body-Jacobian workspace.
        jacobians: Caller-owned inertial-Jacobian output.

    Returns:
        None. Jacobian kernels are enqueued on the current Warp stream.
    """

    _launch_gvs_node_states(
        q,
        joint_adjoint,
        joint_tangent,
        cell_adjoint,
        cell_tangent_local,
        link_global_to_local,
        base_transform,
        num_cells,
        block_dim,
        node_pose,
        node_jacobian,
    )
    wp.launch(
        gvs_jacobian_samples_kernel,
        dim=(q.shape[0], s.shape[1]),
        inputs=[
            q,
            s,
            link_local_to_global,
            link_global_to_local,
            link_basis_rows,
            link_reference_z1,
            link_reference_z2,
            segment_lengths,
            segment_starts,
            integration_points,
            basis_type_index,
            basis_active_params,
            basis_order_params,
            magnus_points,
            scale_rotations,
            epsilons,
            num_cells,
            node_pose,
            node_jacobian,
        ],
        outputs=[jacobians],
        block_dim=128,
    )


def launch_gvs_kinematics(
    q: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    joint_adjoint: wp.array2d[wp.float64],
    joint_tangent: wp.array2d[wp.float64],
    cell_adjoint: wp.array2d[wp.float64],
    cell_tangent_local: wp.array2d[wp.float64],
    base_transform: wp.array2d[wp.float64],
    link_local_to_global: wp.array2d[wp.int32],
    link_global_to_local: wp.array2d[wp.int32],
    link_basis_rows: wp.array2d[wp.int32],
    link_reference_z1: wp.array2d[wp.float64],
    link_reference_z2: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    integration_points: wp.array2d[wp.float64],
    basis_type_index: wp.array[wp.int32],
    basis_active_params: wp.array2d[wp.float64],
    basis_order_params: wp.array2d[wp.float64],
    magnus_points: wp.array[wp.float64],
    scale_rotations: wp.array[wp.int32],
    epsilons: wp.array[wp.float64],
    num_cells: wp.array[wp.int32],
    block_dim: int,
    node_pose: wp.array2d[wp.float64],
    node_jacobian: wp.array2d[wp.float64],
    poses: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Launch the allocation-free GVS node recurrence and sample stage.

    Args:
        q: Batched active configurations ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        joint_adjoint: Flattened inverse adjoints for every joint.
        joint_tangent: Joint tangents in active coordinates.
        cell_adjoint: Flattened inverse adjoints for every full link cell.
        cell_tangent_local: Full-cell tangents in padded local coordinates.
        base_transform: Absolute robot base transform.
        link_local_to_global: Padded local-to-active link-coordinate map.
        link_global_to_local: Active-to-local link-coordinate map.
        link_basis_rows: Spatial row occupied by every local basis column.
        link_reference_z1: Reference strain at first full-cell Magnus nodes.
        link_reference_z2: Reference strain at second full-cell Magnus nodes.
        segment_lengths: Physical link lengths.
        segment_starts: Cumulative link-start positions.
        integration_points: Normalized cell-boundary coordinates.
        basis_type_index: Basis-family index for every segment.
        basis_active_params: Per-row basis activation parameters.
        basis_order_params: Per-row basis-order parameters.
        magnus_points: Two normalized Gauss points within a partial cell.
        scale_rotations: One-entry flag for rotational basis scaling.
        epsilons: Exponential and tangent thresholds.
        num_cells: One-entry array containing the padded cell count.
        block_dim: CUDA threads in each cooperative environment block.
        node_pose: Caller-owned flattened node-pose workspace.
        node_jacobian: Caller-owned flattened body-Jacobian workspace.
        poses: Caller-owned SE(3) pose output.
        jacobians: Caller-owned inertial-Jacobian output.

    Returns:
        None. The two kernels are enqueued on the current Warp stream.
    """

    _launch_gvs_node_states(
        q,
        joint_adjoint,
        joint_tangent,
        cell_adjoint,
        cell_tangent_local,
        link_global_to_local,
        base_transform,
        num_cells,
        block_dim,
        node_pose,
        node_jacobian,
    )
    wp.launch(
        gvs_samples_kernel,
        dim=(q.shape[0], s.shape[1]),
        inputs=[
            q,
            s,
            link_local_to_global,
            link_global_to_local,
            link_basis_rows,
            link_reference_z1,
            link_reference_z2,
            segment_lengths,
            segment_starts,
            integration_points,
            basis_type_index,
            basis_active_params,
            basis_order_params,
            magnus_points,
            scale_rotations,
            epsilons,
            num_cells,
            node_pose,
            node_jacobian,
        ],
        outputs=[poses, jacobians],
        block_dim=128,
    )


gvs_forward_kinematics = wp.jax_callable(launch_gvs_forward_kinematics, num_outputs=2)
gvs_inertial_jacobians = wp.jax_callable(launch_gvs_inertial_jacobians, num_outputs=3)
gvs_kinematics = wp.jax_callable(launch_gvs_kinematics, num_outputs=4)
launch_forward_kinematics = launch_gvs_forward_kinematics
launch_inertial_jacobians = launch_gvs_inertial_jacobians
launch_forward_kinematics_and_jacobians = launch_gvs_kinematics


__all__ = [
    "gvs_cooperative_node_states_kernel",
    "gvs_jacobian_samples_kernel",
    "gvs_node_poses_kernel",
    "gvs_node_states_kernel",
    "gvs_pose_samples_kernel",
    "gvs_samples_kernel",
    "launch_forward_kinematics",
    "launch_forward_kinematics_and_jacobians",
    "launch_gvs_kinematics",
    "launch_gvs_forward_kinematics",
    "launch_gvs_inertial_jacobians",
    "launch_inertial_jacobians",
]
