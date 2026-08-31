# ruff: noqa: I001, UP018
"""Matrix-free GVS inertial-Jacobian products for Warp integrations."""

from __future__ import annotations

import warp as wp

from soromox.execution.warp.common.se3 import (
    Vec6d,
    _ad_action,
    _adjoint_inverse_action,
    _adjoint_inverse_transpose_action,
    _forward_coefficients,
    _left_action,
    _left_coefficients,
    _translation,
)
from soromox.execution.warp.common.spatial import (
    _flattened_matrix_action,
    _flattened_matrix_transpose_action,
    _load_spatial_vector,
    _load_spatial_vector3d,
    _rotate_spatial_from_inertial,
    _rotate_spatial_to_inertial,
    _store_spatial_vector,
)
from soromox.execution.warp.gvs.kinematics import (
    _basis_column_value,
    _load_node_pose,
    _relative_pose_from_inverse_adjoint,
    _relative_pose_from_magnus,
    _store_node_pose,
    gvs_node_poses_kernel,
)

wp.set_module_options({"enable_backward": False})

SPATIAL_DIM = 6


@wp.kernel(enable_backward=False)
def gvs_node_pose_twist_kernel(
    joint_adjoint: wp.array2d[wp.float64],
    joint_velocity: wp.array2d[wp.float64],
    cell_adjoint: wp.array2d[wp.float64],
    cell_link_velocity: wp.array2d[wp.float64],
    base_transform: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    num_cells_array: wp.array[wp.int32],
    node_pose: wp.array2d[wp.float64],
    node_twist: wp.array2d[wp.float64],
):
    """Propagate cell-boundary poses and one Jacobian direction.

    ``joint_velocity`` and ``cell_link_velocity`` are the already-contracted
    local tangent products produced by the GVS joint and cell term kernels.

    Args:
        joint_adjoint: Flattened inverse adjoints for every joint.
        joint_velocity: Joint tangent products for the active direction.
        cell_adjoint: Flattened inverse adjoints for every full link cell.
        cell_link_velocity: Cell tangent products for the active direction.
        base_transform: Absolute fixed-base transform.
        segment_lengths: Physical segment lengths.
        num_cells_array: One-entry padded cell count.
        node_pose: Cell-boundary pose output.
        node_twist: Cell-boundary spatial JVP output.

    Returns:
        None. ``node_pose`` and ``node_twist`` are written in place.
    """

    environment = wp.tid()
    num_segments = segment_lengths.shape[0]
    num_cells = num_cells_array[0]
    current_pose = wp.mat44d()
    current_twist = Vec6d()
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
        joint_twist = Vec6d()
        row = int(0)
        while row < SPATIAL_DIM:
            joint_twist[row] = joint_velocity[joint_base + row, 0]
            row += 1
        current_twist = _flattened_matrix_action(
            joint_adjoint, joint_base, current_twist + joint_twist
        )
        node_item = (environment * num_segments + segment) * (num_cells + 1)
        _store_node_pose(node_pose, node_item, current_pose)
        _store_spatial_vector(node_twist, node_item, current_twist)

        cell = int(0)
        while cell < num_cells:
            cell_base = (
                (environment * num_segments + segment) * num_cells + cell
            ) * SPATIAL_DIM
            current_pose = current_pose * _relative_pose_from_inverse_adjoint(
                cell_adjoint, cell_base
            )
            link_twist = Vec6d()
            row = int(0)
            while row < SPATIAL_DIM:
                link_twist[row] = cell_link_velocity[cell_base + row, 0]
                row += 1
            current_twist = _flattened_matrix_action(
                cell_adjoint, cell_base, current_twist + link_twist
            )
            node_item += 1
            _store_node_pose(node_pose, node_item, current_pose)
            _store_spatial_vector(node_twist, node_item, current_twist)
            cell += 1
        segment += 1


@wp.kernel(enable_backward=False)
def gvs_pose_twist_samples_kernel(
    q: wp.array2d[wp.float64],
    qd: wp.array2d[wp.float64],
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
    node_twist: wp.array2d[wp.float64],
    poses: wp.array4d[wp.float64],
    twists: wp.array3d[wp.float64],
):
    """Evaluate poses and inertial twists ``J(q, s) @ qd`` at samples.

    Args:
        q: Batched active configurations ``(E, D)``.
        qd: Batched generalized-velocity directions ``(E, D)``.
        s: Per-environment sample abscissae ``(E, N)``.
        link_local_to_global: Padded link-local to active-coordinate map.
        link_basis_rows: Spatial row occupied by every local basis column.
        link_reference_z1: Reference strain at first Magnus nodes.
        link_reference_z2: Reference strain at second Magnus nodes.
        segment_lengths: Physical segment lengths.
        segment_starts: Cumulative segment-start coordinates.
        integration_points: Normalized cell-boundary coordinates.
        basis_type_index: Basis-family index for every segment.
        basis_active_params: Per-row basis activation parameters.
        basis_order_params: Per-row basis orders.
        magnus_points: Normalized two-point Magnus coordinates.
        scale_rotations_array: One-entry rotational-basis scaling flag.
        epsilons: Exponential and tangent thresholds.
        num_cells_array: One-entry padded cell count.
        node_pose: Cell-boundary poses produced by the node recurrence.
        node_twist: Cell-boundary spatial JVPs produced by the node recurrence.
        poses: Sample pose output ``(E, N, 4, 4)``.
        twists: Inertial-frame sample JVP output ``(E, N, 6)``.

    Returns:
        None. ``poses`` and ``twists`` are written in place.
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
    xid1 = Vec6d()
    xid2 = Vec6d()
    local = int(0)
    reference_item = segment * num_cells + cell
    x1 = x_base + magnus_points[0] * width
    x2 = x_base + magnus_points[1] * width
    while local < link_local_to_global.shape[1]:
        global_index = link_local_to_global[segment, local]
        coordinate = wp.float64(0.0)
        velocity = wp.float64(0.0)
        if global_index >= 0:
            coordinate = q[environment, global_index]
            velocity = qd[environment, global_index]
        basis_row = link_basis_rows[segment, local]
        if basis_row >= 0:
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
            xi1[basis_row] += basis_value_z1 * coordinate
            xi2[basis_row] += basis_value_z2 * coordinate
            xid1[basis_row] += basis_value_z1 * velocity
            xid2[basis_row] += basis_value_z2 * velocity
        local += 1
    row = int(0)
    while row < SPATIAL_DIM:
        xi1[row] += link_reference_z1[reference_item, row]
        xi2[row] += link_reference_z2[reference_item, row]
        row += 1
    alpha = ds * wp.float64(0.5)
    coefficient = wp.sqrt(wp.float64(3.0)) * ds * ds / wp.float64(12.0)
    magnus = alpha * (xi1 + xi2) + coefficient * _ad_action(xi1, xi2)
    magnus_dot = alpha * (xid1 + xid2) + coefficient * (
        _ad_action(xid1, xi2) + _ad_action(xi1, xid2)
    )
    pose = base_pose * _relative_pose_from_magnus(magnus)
    omega = wp.vec3d(magnus[0], magnus[1], magnus[2])
    linear = wp.vec3d(magnus[3], magnus[4], magnus[5])
    angle_sq = wp.dot(omega, omega)
    coefficients = _left_coefficients(angle_sq)
    forward = _forward_coefficients(angle_sq)
    translation = _translation(omega, linear, forward)
    partial_twist = _left_action(magnus, magnus_dot, coefficients)
    body_twist = _adjoint_inverse_action(
        omega,
        translation,
        angle_sq,
        forward,
        _load_spatial_vector(node_twist, node_item) + partial_twist,
    )
    inertial_twist = _rotate_spatial_to_inertial(pose, body_twist)

    row = int(0)
    while row < 4:
        column = int(0)
        while column < 4:
            poses[environment, sample, row, column] = pose[row, column]
            column += 1
        row += 1
    row = int(0)
    while row < SPATIAL_DIM:
        twists[environment, sample, row] = inertial_twist[row]
        row += 1


@wp.kernel(enable_backward=False)
def gvs_sample_vjp_kernel(
    q: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    inertial_wrenches: wp.array3d[wp.float64],
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
    node_wrench: wp.array2d[wp.float64],
    generalized_force: wp.array2d[wp.float64],
):
    """Seed node cotangents and partial-cell generalized forces.

    Args:
        q: Batched active configurations ``(E, D)``.
        s: Per-environment sample abscissae ``(E, N)``.
        inertial_wrenches: Inertial-frame sample cotangents ``(E, N, 6)``.
        link_local_to_global: Padded link-local to active-coordinate map.
        link_basis_rows: Spatial row occupied by every local basis column.
        link_reference_z1: Reference strain at first Magnus nodes.
        link_reference_z2: Reference strain at second Magnus nodes.
        segment_lengths: Physical segment lengths.
        segment_starts: Cumulative segment-start coordinates.
        integration_points: Normalized cell-boundary coordinates.
        basis_type_index: Basis-family index for every segment.
        basis_active_params: Per-row basis activation parameters.
        basis_order_params: Per-row basis orders.
        magnus_points: Normalized two-point Magnus coordinates.
        scale_rotations_array: One-entry rotational-basis scaling flag.
        epsilons: Exponential and tangent thresholds.
        num_cells_array: One-entry padded cell count.
        node_pose: Cell-boundary poses produced by the node recurrence.
        node_wrench: Cell-boundary cotangent accumulator.
        generalized_force: Generalized-force accumulator ``(E, D)``.

    Returns:
        None. ``node_wrench`` and ``generalized_force`` are accumulated in
        place.
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
    x1 = x_base + magnus_points[0] * width
    x2 = x_base + magnus_points[1] * width
    while local < link_local_to_global.shape[1]:
        global_index = link_local_to_global[segment, local]
        coordinate = wp.float64(0.0)
        if global_index >= 0:
            coordinate = q[environment, global_index]
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
            xi1[basis_row] += value_z1 * coordinate
            xi2[basis_row] += value_z2 * coordinate
        local += 1
    row = int(0)
    while row < SPATIAL_DIM:
        xi1[row] += link_reference_z1[reference_item, row]
        xi2[row] += link_reference_z2[reference_item, row]
        row += 1
    alpha = ds * wp.float64(0.5)
    coefficient = wp.sqrt(wp.float64(3.0)) * ds * ds / wp.float64(12.0)
    magnus = alpha * (xi1 + xi2) + coefficient * _ad_action(xi1, xi2)
    pose = base_pose * _relative_pose_from_magnus(magnus)
    omega = wp.vec3d(magnus[0], magnus[1], magnus[2])
    linear = wp.vec3d(magnus[3], magnus[4], magnus[5])
    angle_sq = wp.dot(omega, omega)
    coefficients = _left_coefficients(angle_sq)
    forward = _forward_coefficients(angle_sq)
    translation = _translation(omega, linear, forward)
    body_wrench = _rotate_spatial_from_inertial(
        pose,
        _load_spatial_vector3d(inertial_wrenches, environment, sample),
    )
    source_wrench = _adjoint_inverse_transpose_action(
        omega, translation, angle_sq, forward, body_wrench
    )
    row = int(0)
    while row < SPATIAL_DIM:
        wp.atomic_add(node_wrench, node_item, row, source_wrench[row])
        row += 1

    local = int(0)
    while local < link_local_to_global.shape[1]:
        global_index = link_local_to_global[segment, local]
        basis_row = link_basis_rows[segment, local]
        if global_index >= 0 and basis_row >= 0:
            basis_z1 = Vec6d()
            basis_z2 = Vec6d()
            basis_z1[basis_row] = _basis_column_value(
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
            basis_z2[basis_row] = _basis_column_value(
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
            magnus_basis = alpha * (basis_z1 + basis_z2) + coefficient * (
                _ad_action(xi1, basis_z2) - _ad_action(xi2, basis_z1)
            )
            tangent = _left_action(magnus, magnus_basis, coefficients)
            effort = wp.float64(0.0)
            row = int(0)
            while row < SPATIAL_DIM:
                effort += tangent[row] * source_wrench[row]
                row += 1
            wp.atomic_add(generalized_force, environment, global_index, effort)
        local += 1


@wp.kernel(enable_backward=False)
def gvs_node_vjp_kernel(
    joint_adjoint: wp.array2d[wp.float64],
    joint_tangent: wp.array2d[wp.float64],
    cell_adjoint: wp.array2d[wp.float64],
    cell_tangent_local: wp.array2d[wp.float64],
    link_local_to_global: wp.array2d[wp.int32],
    segment_lengths: wp.array[wp.float64],
    num_cells_array: wp.array[wp.int32],
    node_wrench: wp.array2d[wp.float64],
    generalized_force: wp.array2d[wp.float64],
):
    """Reverse the serial node recurrence on Warp CPU.

    Args:
        joint_adjoint: Flattened inverse adjoints for every joint.
        joint_tangent: Joint tangents in active coordinates.
        cell_adjoint: Flattened inverse adjoints for every full link cell.
        cell_tangent_local: Full-cell tangents in padded local coordinates.
        link_local_to_global: Padded link-local to active-coordinate map.
        segment_lengths: Physical segment lengths.
        num_cells_array: One-entry padded cell count.
        node_wrench: Cell-boundary cotangent accumulator.
        generalized_force: Generalized-force accumulator ``(E, D)``.

    Returns:
        None. ``node_wrench`` and ``generalized_force`` are accumulated in
        place.
    """

    environment = wp.tid()
    num_segments = segment_lengths.shape[0]
    num_cells = num_cells_array[0]
    segment = num_segments - 1
    while segment >= 0:
        node_start = (environment * num_segments + segment) * (num_cells + 1)
        cell = num_cells - 1
        while cell >= 0:
            cell_base = (
                (environment * num_segments + segment) * num_cells + cell
            ) * SPATIAL_DIM
            next_item = node_start + cell + 1
            previous_item = next_item - 1
            source_wrench = _flattened_matrix_transpose_action(
                cell_adjoint,
                cell_base,
                _load_spatial_vector(node_wrench, next_item),
            )
            row = int(0)
            while row < SPATIAL_DIM:
                node_wrench[previous_item, row] += source_wrench[row]
                row += 1
            local = int(0)
            while local < link_local_to_global.shape[1]:
                global_index = link_local_to_global[segment, local]
                if global_index >= 0:
                    effort = wp.float64(0.0)
                    row = int(0)
                    while row < SPATIAL_DIM:
                        effort += (
                            cell_tangent_local[cell_base + row, local]
                            * source_wrench[row]
                        )
                        row += 1
                    generalized_force[environment, global_index] += effort
                local += 1
            cell -= 1

        joint_base = (environment * num_segments + segment) * SPATIAL_DIM
        source_wrench = _flattened_matrix_transpose_action(
            joint_adjoint,
            joint_base,
            _load_spatial_vector(node_wrench, node_start),
        )
        global_index = int(0)
        while global_index < generalized_force.shape[1]:
            effort = wp.float64(0.0)
            row = int(0)
            while row < SPATIAL_DIM:
                effort += (
                    joint_tangent[joint_base + row, global_index]
                    * source_wrench[row]
                )
                row += 1
            generalized_force[environment, global_index] += effort
            global_index += 1
        if segment > 0:
            previous_end = node_start - 1
            row = int(0)
            while row < SPATIAL_DIM:
                node_wrench[previous_end, row] += source_wrench[row]
                row += 1
        segment -= 1


@wp.kernel(enable_backward=False)
def gvs_cooperative_node_vjp_kernel(
    joint_adjoint: wp.array2d[wp.float64],
    joint_tangent: wp.array2d[wp.float64],
    cell_adjoint: wp.array2d[wp.float64],
    cell_tangent_local: wp.array2d[wp.float64],
    link_global_to_local: wp.array2d[wp.int32],
    segment_lengths: wp.array[wp.float64],
    num_cells_array: wp.array[wp.int32],
    node_wrench: wp.array2d[wp.float64],
    edge_wrench: wp.array2d[wp.float64],
    generalized_force: wp.array2d[wp.float64],
):
    """Reverse node cotangents cooperatively on CUDA.

    Args:
        joint_adjoint: Flattened inverse adjoints for every joint.
        joint_tangent: Joint tangents in active coordinates.
        cell_adjoint: Flattened inverse adjoints for every full link cell.
        cell_tangent_local: Full-cell tangents in padded local coordinates.
        link_global_to_local: Active-to-link-local coordinate map.
        segment_lengths: Physical segment lengths.
        num_cells_array: One-entry padded cell count.
        node_wrench: Cell-boundary cotangent accumulator.
        edge_wrench: Per-environment six-vector edge scratch.
        generalized_force: Generalized-force accumulator ``(E, D)``.

    Returns:
        None. ``node_wrench``, ``edge_wrench``, and ``generalized_force`` are
        updated in place.
    """

    environment, lane = wp.tid()
    num_segments = segment_lengths.shape[0]
    num_cells = num_cells_array[0]
    barrier = wp.tile_zeros(shape=(1,), dtype=wp.int32, storage="shared")
    segment = num_segments - 1
    while segment >= 0:
        node_start = (environment * num_segments + segment) * (num_cells + 1)
        cell = num_cells - 1
        while cell >= 0:
            cell_base = (
                (environment * num_segments + segment) * num_cells + cell
            ) * SPATIAL_DIM
            next_item = node_start + cell + 1
            previous_item = next_item - 1
            if lane == 0:
                source_wrench = _flattened_matrix_transpose_action(
                    cell_adjoint,
                    cell_base,
                    _load_spatial_vector(node_wrench, next_item),
                )
                row = int(0)
                while row < SPATIAL_DIM:
                    edge_wrench[environment, row] = source_wrench[row]
                    node_wrench[previous_item, row] += source_wrench[row]
                    row += 1
            wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
            global_index = lane
            while global_index < generalized_force.shape[1]:
                local = link_global_to_local[segment, global_index]
                if local >= 0:
                    effort = wp.float64(0.0)
                    row = int(0)
                    while row < SPATIAL_DIM:
                        effort += (
                            cell_tangent_local[cell_base + row, local]
                            * edge_wrench[environment, row]
                        )
                        row += 1
                    generalized_force[environment, global_index] += effort
                global_index += wp.block_dim()
            wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
            cell -= 1

        joint_base = (environment * num_segments + segment) * SPATIAL_DIM
        if lane == 0:
            source_wrench = _flattened_matrix_transpose_action(
                joint_adjoint,
                joint_base,
                _load_spatial_vector(node_wrench, node_start),
            )
            row = int(0)
            while row < SPATIAL_DIM:
                edge_wrench[environment, row] = source_wrench[row]
                if segment > 0:
                    node_wrench[node_start - 1, row] += source_wrench[row]
                row += 1
        wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
        global_index = lane
        while global_index < generalized_force.shape[1]:
            effort = wp.float64(0.0)
            row = int(0)
            while row < SPATIAL_DIM:
                effort += (
                    joint_tangent[joint_base + row, global_index]
                    * edge_wrench[environment, row]
                )
                row += 1
            generalized_force[environment, global_index] += effort
            global_index += wp.block_dim()
        wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
        segment -= 1


def launch_gvs_forward_kinematics_and_jvp(
    q: wp.array2d[wp.float64],
    qd: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    joint_adjoint: wp.array2d[wp.float64],
    joint_velocity: wp.array2d[wp.float64],
    cell_adjoint: wp.array2d[wp.float64],
    cell_link_velocity: wp.array2d[wp.float64],
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
    node_twist: wp.array2d[wp.float64],
    poses: wp.array4d[wp.float64],
    twists: wp.array3d[wp.float64],
):
    """Launch fused poses and inertial-Jacobian forward products.

    Args:
        q: Batched active configurations ``(E, D)``.
        qd: Batched generalized-velocity directions ``(E, D)``.
        s: Per-environment sample abscissae ``(E, N)``.
        joint_adjoint: Flattened inverse adjoints for every joint.
        joint_velocity: Joint tangent products for ``qd``.
        cell_adjoint: Flattened inverse adjoints for every full link cell.
        cell_link_velocity: Cell tangent products for ``qd`` before adjoint propagation.
        base_transform: Absolute fixed-base transform.
        link_local_to_global: Padded link-local to active-coordinate map.
        link_basis_rows: Spatial row occupied by every local basis column.
        link_reference_z1: Reference strain at first Magnus nodes.
        link_reference_z2: Reference strain at second Magnus nodes.
        segment_lengths: Physical segment lengths.
        segment_starts: Cumulative segment-start coordinates.
        integration_points: Normalized cell-boundary coordinates.
        basis_type_index: Basis-family index for every segment.
        basis_active_params: Per-row basis activation parameters.
        basis_order_params: Per-row basis orders.
        magnus_points: Normalized two-point Magnus coordinates.
        scale_rotations: One-entry rotational-basis scaling flag.
        epsilons: Exponential and tangent thresholds.
        num_cells: One-entry padded cell count.
        node_pose: Caller-owned node-pose workspace.
        node_twist: Caller-owned six-vector node workspace.
        poses: Caller-owned sample pose output ``(E, N, 4, 4)``.
        twists: Caller-owned output ``J(q, s) @ qd`` with shape ``(E, N, 6)``.

    Returns:
        None. The node and sample kernels are enqueued on the active Warp
        stream.
    """

    wp.launch(
        gvs_node_pose_twist_kernel,
        dim=q.shape[0],
        inputs=[
            joint_adjoint,
            joint_velocity,
            cell_adjoint,
            cell_link_velocity,
            base_transform,
            segment_lengths,
            num_cells,
        ],
        outputs=[node_pose, node_twist],
    )
    wp.launch(
        gvs_pose_twist_samples_kernel,
        dim=(q.shape[0], s.shape[1]),
        inputs=[
            q,
            qd,
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
            node_twist,
        ],
        outputs=[poses, twists],
        block_dim=128,
    )


def launch_gvs_inertial_jacobian_vjp(
    q: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    inertial_wrenches: wp.array3d[wp.float64],
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
    node_wrench: wp.array2d[wp.float64],
    edge_wrench: wp.array2d[wp.float64],
    generalized_force: wp.array2d[wp.float64],
    *,
    node_poses_ready: bool = False,
):
    """Launch the matrix-free product ``sum_s J(q, s).T @ wrench_s``.

    Args:
        q: Batched active configurations ``(E, D)``.
        s: Per-environment sample abscissae ``(E, N)``.
        inertial_wrenches: Inertial-frame sample cotangents ``(E, N, 6)``.
        joint_adjoint: Flattened inverse adjoints for every joint.
        joint_tangent: Joint tangents in active coordinates.
        cell_adjoint: Flattened inverse adjoints for every full link cell.
        cell_tangent_local: Full-cell tangents in padded local coordinates.
        base_transform: Absolute fixed-base transform.
        link_local_to_global: Padded link-local to active-coordinate map.
        link_global_to_local: Active-to-link-local coordinate map.
        link_basis_rows: Spatial row occupied by every local basis column.
        link_reference_z1: Reference strain at first Magnus nodes.
        link_reference_z2: Reference strain at second Magnus nodes.
        segment_lengths: Physical segment lengths.
        segment_starts: Cumulative segment-start coordinates.
        integration_points: Normalized cell-boundary coordinates.
        basis_type_index: Basis-family index for every segment.
        basis_active_params: Per-row basis activation parameters.
        basis_order_params: Per-row basis orders.
        magnus_points: Normalized two-point Magnus coordinates.
        scale_rotations: One-entry rotational-basis scaling flag.
        epsilons: Exponential and tangent thresholds.
        num_cells: One-entry padded cell count.
        block_dim: Cooperative CUDA lanes per environment.
        node_pose: Caller-owned node-pose workspace.
        node_wrench: Caller-owned six-vector node-cotangent workspace.
        edge_wrench: Caller-owned six-vector edge scratch per environment.
        generalized_force: Caller-owned ``(E, D)`` VJP output.
        node_poses_ready: Skip node-pose propagation when ``node_pose`` was
            populated by a preceding pose-only launch for the same state.

    Returns:
        None. The forward and reverse kernels are enqueued on the active Warp
        stream.
    """

    node_wrench.zero_()
    generalized_force.zero_()
    if not node_poses_ready:
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
        gvs_sample_vjp_kernel,
        dim=(q.shape[0], s.shape[1]),
        inputs=[
            q,
            s,
            inertial_wrenches,
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
        outputs=[node_wrench, generalized_force],
        block_dim=128,
    )
    reverse_inputs = [
        joint_adjoint,
        joint_tangent,
        cell_adjoint,
        cell_tangent_local,
    ]
    if q.device.is_cuda:
        wp.launch_tiled(
            gvs_cooperative_node_vjp_kernel,
            dim=q.shape[0],
            inputs=[
                *reverse_inputs,
                link_global_to_local,
                segment_lengths,
                num_cells,
                node_wrench,
                edge_wrench,
            ],
            outputs=[generalized_force],
            block_dim=block_dim,
        )
    else:
        wp.launch(
            gvs_node_vjp_kernel,
            dim=q.shape[0],
            inputs=[
                *reverse_inputs,
                link_local_to_global,
                segment_lengths,
                num_cells,
                node_wrench,
            ],
            outputs=[generalized_force],
        )


launch_forward_kinematics_and_jvp = launch_gvs_forward_kinematics_and_jvp
launch_inertial_jacobian_vjp = launch_gvs_inertial_jacobian_vjp


__all__ = [
    "gvs_cooperative_node_vjp_kernel",
    "gvs_node_pose_twist_kernel",
    "gvs_node_vjp_kernel",
    "gvs_pose_twist_samples_kernel",
    "gvs_sample_vjp_kernel",
    "launch_forward_kinematics_and_jvp",
    "launch_gvs_forward_kinematics_and_jvp",
    "launch_gvs_inertial_jacobian_vjp",
    "launch_inertial_jacobian_vjp",
]
