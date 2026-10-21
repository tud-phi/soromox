# ruff: noqa: I001, UP018
"""Floating-base persistent whole-chain GVS dynamics kernel.

The augmented root and accumulator kernel remains separate from
:mod:`gvs.chain` to preserve the fixed module's generated source, cache key,
and compile graph. Local joint/cell operators and spatial primitives are shared
below this entry-kernel boundary.
"""

from __future__ import annotations

import warp as wp

from soromox.execution.warp.common.floating_base import _spatial_root_jacobian_entry
from soromox.execution.warp.common.rotations import _quaternion_rotation_matrix
from soromox.execution.warp.common.se3 import Vec6d, _ad_action
from soromox.execution.warp.common.spatial import _coadjoint_wrench
from soromox.execution.warp.common.spatial_inertia import (
    _mass_jacobian_action,
    _mass_vector_action,
)
from soromox.execution.warp.common.storage import (
    _matrix_value,
    _vector_value,
    _write_matrix_value,
    _write_vector_value,
)

wp.set_module_options({"enable_backward": False})

SPATIAL_DIM = 6


@wp.func
def floating_persistent_chain_at(
    environment: int,
    lane: int,
    joint_adjoint: wp.array2d[wp.float64],
    joint_adjoint_dot: wp.array2d[wp.float64],
    joint_tangent: wp.array2d[wp.float64],
    joint_tangent_dot_qd: wp.array2d[wp.float64],
    joint_velocity: wp.array2d[wp.float64],
    cell_adjoint: wp.array2d[wp.float64],
    cell_tangent_local: wp.array2d[wp.float64],
    cell_link_velocity: wp.array2d[wp.float64],
    cell_step_velocity: wp.array2d[wp.float64],
    cell_tangent_velocity_dot: wp.array2d[wp.float64],
    global_to_local: wp.array2d[wp.int32],
    active_dofs: wp.array[wp.int32],
    base_pose: wp.array2d[wp.float64],
    velocity: wp.array2d[wp.float64],
    inertia_upper_rows: wp.array[wp.int32],
    inertia_upper_columns: wp.array[wp.int32],
    weighted_masses: wp.array2d[wp.float64],
    gravity_world: wp.array[wp.float64],
    num_cells_array: wp.array[wp.int32],
    num_quadrature_array: wp.array[wp.int32],
    lanes_per_block: wp.array[wp.int32],
    jacobian_first: wp.array2d[wp.float64],
    jacobian_dot_qd_first: wp.array2d[wp.float64],
    velocity_first: wp.array2d[wp.float64],
    gravity_first: wp.array2d[wp.float64],
    jacobian_second: wp.array2d[wp.float64],
    jacobian_dot_qd_second: wp.array2d[wp.float64],
    velocity_second: wp.array2d[wp.float64],
    gravity_second: wp.array2d[wp.float64],
    inertia: wp.array3d[wp.float64],
    coriolis_qd: wp.array2d[wp.float64],
    gravity_force: wp.array2d[wp.float64],
):
    """Traverse one augmented floating GVS chain per cooperative block.

    Explicit indices address the supplied array views. Homogeneous wrappers pass the
    full batch; packed wrappers pass one world's views and local indices.

    Joint and cell Lie operators are shared with the fixed pipeline. The
    floating entry specializes root initialization, shifted internal columns,
    augmented causal widths, and full dynamics accumulation.

    Args:
        environment: Batch row in the supplied arrays; use zero for a single-world view.
        lane: Thread index within the cooperative work item.
        joint_adjoint: Flattened joint inverse adjoints.
        joint_adjoint_dot: Flattened joint-adjoint derivatives.
        joint_tangent: Internal joint tangents.
        joint_tangent_dot_qd: Joint tangent-derivative actions.
        joint_velocity: Internal joint spatial velocities.
        cell_adjoint: Flattened cell inverse adjoints.
        cell_tangent_local: Cell tangents in padded local coordinates.
        cell_link_velocity: Internal link velocities per cell.
        cell_step_velocity: Internal Magnus-step velocities per cell.
        cell_tangent_velocity_dot: Cell tangent-derivative actions.
        global_to_local: Internal active-to-local link-coordinate map.
        active_dofs: Cumulative internal coordinate count by segment.
        base_pose: Runtime scalar-last quaternion poses with shape ``(E, 7)``.
        velocity: Total angular-first generalized velocities.
        inertia_upper_rows: Packed augmented upper-inertia row indices.
        inertia_upper_columns: Packed augmented upper-inertia column indices.
        weighted_masses: Weighted spatial inertias with shape ``(S * P, 6)``
            for diagonals or ``(S * P, 36)`` for row-major full matrices.
            P includes the unit-weight link-tip slot when present.
        gravity_world: World-frame spatial gravity.
        num_cells_array: One-entry array containing cells per segment.
        num_quadrature_array: One-entry array containing quadrature count.
        lanes_per_block: One-entry array containing cooperative lanes.
        jacobian_first: First caller-owned Jacobian workspace.
        jacobian_dot_qd_first: First derivative-action workspace.
        velocity_first: First spatial-velocity workspace.
        gravity_first: First local-gravity workspace.
        jacobian_second: Second Jacobian workspace.
        jacobian_dot_qd_second: Second derivative-action workspace.
        velocity_second: Second spatial-velocity workspace.
        gravity_second: Second local-gravity workspace.
        inertia: Batched augmented inertia output.
        coriolis_qd: Batched augmented convective-force output.
        gravity_force: Batched augmented generalized-gravity output.

    Returns:
        None. Workspaces and outputs are updated in place.
    """
    num_dofs = velocity.shape[1]
    num_cells = num_cells_array[0]
    num_segments = joint_adjoint.shape[0] // (velocity.shape[0] * SPATIAL_DIM)
    num_mass_points = weighted_masses.shape[0] // num_segments
    lane_stride = lanes_per_block[0]
    state_base_row = environment * SPATIAL_DIM
    base_dofs = int(6)
    barrier = wp.tile_zeros(shape=(1,), dtype=wp.int32, storage="shared")
    source_jacobian_velocity_shared = wp.tile_zeros(
        shape=(SPATIAL_DIM,), dtype=wp.float64, storage="shared"
    )
    transported_source_velocity_shared = wp.tile_zeros(
        shape=(SPATIAL_DIM,), dtype=wp.float64, storage="shared"
    )
    entry = lane
    while entry < SPATIAL_DIM * num_dofs:
        row = entry // num_dofs
        column = entry - row * num_dofs
        value = wp.float64(0.0)
        if column < base_dofs:
            value = _spatial_root_jacobian_entry(base_pose, environment, row, column)
        jacobian_first[state_base_row + row, column] = value
        jacobian_second[state_base_row + row, column] = wp.float64(0.0)
        entry += lane_stride
    row = lane
    while row < SPATIAL_DIM:
        rotation = _quaternion_rotation_matrix(base_pose, environment)
        root_velocity = wp.float64(0.0)
        column = int(0)
        while column < 3:
            root_velocity += (
                rotation[column, row % 3]
                * velocity[environment, (0 if row < 3 else 3) + column]
            )
            column += 1
        root_gravity = wp.float64(0.0)
        if row >= 3:
            column = int(0)
            while column < 3:
                root_gravity += rotation[column, row - 3] * gravity_world[column + 3]
                column += 1
        jacobian_dot_qd_first[state_base_row + row, 0] = wp.float64(0.0)
        jacobian_dot_qd_second[state_base_row + row, 0] = wp.float64(0.0)
        velocity_first[state_base_row + row, 0] = root_velocity
        velocity_second[state_base_row + row, 0] = wp.float64(0.0)
        gravity_first[state_base_row + row, 0] = root_gravity
        gravity_second[state_base_row + row, 0] = wp.float64(0.0)
        row += lane_stride
    entry = lane
    while entry < num_dofs * num_dofs:
        output_row = entry // num_dofs
        output_column = entry - output_row * num_dofs
        inertia[environment, output_row, output_column] = wp.float64(0.0)
        entry += lane_stride
    column = lane
    while column < num_dofs:
        coriolis_qd[environment, column] = wp.float64(0.0)
        gravity_force[environment, column] = wp.float64(0.0)
        column += lane_stride
    wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
    row = lane
    while row < SPATIAL_DIM:
        derivative = wp.float64(0.0)
        if row >= 3:
            omega = wp.vec3d(
                velocity_first[state_base_row + 0, 0],
                velocity_first[state_base_row + 1, 0],
                velocity_first[state_base_row + 2, 0],
            )
            linear = wp.vec3d(
                velocity_first[state_base_row + 3, 0],
                velocity_first[state_base_row + 4, 0],
                velocity_first[state_base_row + 5, 0],
            )
            derivative = -wp.cross(omega, linear)[row - 3]
        jacobian_dot_qd_first[state_base_row + row, 0] = derivative
        row += lane_stride
    wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
    current_is_first = bool(True)
    segment = int(0)
    while segment < num_segments:
        segment_dofs = base_dofs + active_dofs[segment]
        joint_base_row = (environment * num_segments + segment) * SPATIAL_DIM
        destination_is_first = not current_is_first
        entry = lane
        while entry < SPATIAL_DIM * segment_dofs:
            row = entry // segment_dofs
            column = entry - row * segment_dofs
            internal_column = column - base_dofs
            value = wp.float64(0.0)
            k = int(0)
            while k < SPATIAL_DIM:
                tangent_value = wp.float64(0.0)
                if internal_column >= 0:
                    tangent_value = joint_tangent[joint_base_row + k, internal_column]
                value += joint_adjoint[joint_base_row + row, k] * (
                    _matrix_value(
                        jacobian_first,
                        jacobian_second,
                        current_is_first,
                        state_base_row,
                        k,
                        column,
                    )
                    + tangent_value
                )
                k += 1
            _write_matrix_value(
                jacobian_first,
                jacobian_second,
                destination_is_first,
                state_base_row,
                row,
                column,
                value,
            )
            entry += lane_stride
        if lane_stride == 1:
            source_row = int(0)
            while source_row < SPATIAL_DIM:
                source_velocity_value = wp.float64(0.0)
                source_column = int(0)
                while source_column < segment_dofs:
                    source_velocity_value += (
                        _matrix_value(
                            jacobian_first,
                            jacobian_second,
                            current_is_first,
                            state_base_row,
                            source_row,
                            source_column,
                        )
                        * velocity[environment, source_column]
                    )
                    source_column += 1
                source_jacobian_velocity_shared[source_row] = source_velocity_value
                source_row += 1
        else:
            source_row = lane
            source_velocity_value = wp.float64(0.0)
            if source_row < SPATIAL_DIM:
                source_column = int(0)
                while source_column < segment_dofs:
                    source_velocity_value += (
                        _matrix_value(
                            jacobian_first,
                            jacobian_second,
                            current_is_first,
                            state_base_row,
                            source_row,
                            source_column,
                        )
                        * velocity[environment, source_column]
                    )
                    source_column += 1
            wp.tile_scatter_masked(
                source_jacobian_velocity_shared,
                source_row,
                source_velocity_value,
                source_row < SPATIAL_DIM,
            )
        state_row = lane
        while state_row < SPATIAL_DIM:
            derivative_value = wp.float64(0.0)
            velocity_value = wp.float64(0.0)
            gravity_value = wp.float64(0.0)
            k = int(0)
            while k < SPATIAL_DIM:
                derivative_value += (
                    joint_adjoint[joint_base_row + state_row, k]
                    * (
                        _vector_value(
                            jacobian_dot_qd_first,
                            jacobian_dot_qd_second,
                            current_is_first,
                            state_base_row,
                            k,
                        )
                        + joint_tangent_dot_qd[joint_base_row + k, 0]
                    )
                    + joint_adjoint_dot[joint_base_row + state_row, k]
                    * source_jacobian_velocity_shared[k]
                )
                velocity_value += joint_adjoint[joint_base_row + state_row, k] * (
                    _vector_value(
                        velocity_first,
                        velocity_second,
                        current_is_first,
                        state_base_row,
                        k,
                    )
                    + joint_velocity[joint_base_row + k, 0]
                )
                gravity_value += joint_adjoint[
                    joint_base_row + state_row, k
                ] * _vector_value(
                    gravity_first, gravity_second, current_is_first, state_base_row, k
                )
                k += 1
            _write_vector_value(
                jacobian_dot_qd_first,
                jacobian_dot_qd_second,
                destination_is_first,
                state_base_row,
                state_row,
                derivative_value,
            )
            _write_vector_value(
                velocity_first,
                velocity_second,
                destination_is_first,
                state_base_row,
                state_row,
                velocity_value,
            )
            _write_vector_value(
                gravity_first,
                gravity_second,
                destination_is_first,
                state_base_row,
                state_row,
                gravity_value,
            )
            state_row += lane_stride
        wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
        current_is_first = destination_is_first
        cell = int(0)
        while cell < num_cells:
            destination_is_first = not current_is_first
            cell_base_row = (
                (environment * num_segments + segment) * num_cells + cell
            ) * SPATIAL_DIM
            entry = lane
            while entry < SPATIAL_DIM * segment_dofs:
                row = entry // segment_dofs
                column = entry - row * segment_dofs
                internal_column = column - base_dofs
                local_column = int(-1)
                if internal_column >= 0:
                    local_column = global_to_local[segment, internal_column]
                value = wp.float64(0.0)
                k = int(0)
                while k < SPATIAL_DIM:
                    tangent_value = wp.float64(0.0)
                    if local_column >= 0:
                        tangent_value = cell_tangent_local[
                            cell_base_row + k, local_column
                        ]
                    value += cell_adjoint[cell_base_row + row, k] * (
                        _matrix_value(
                            jacobian_first,
                            jacobian_second,
                            current_is_first,
                            state_base_row,
                            k,
                            column,
                        )
                        + tangent_value
                    )
                    k += 1
                _write_matrix_value(
                    jacobian_first,
                    jacobian_second,
                    destination_is_first,
                    state_base_row,
                    row,
                    column,
                    value,
                )
                entry += lane_stride
            if lane_stride == 1:
                source_row = int(0)
                while source_row < SPATIAL_DIM:
                    source_velocity_value = wp.float64(0.0)
                    source_column = int(0)
                    while source_column < segment_dofs:
                        source_velocity_value += (
                            _matrix_value(
                                jacobian_first,
                                jacobian_second,
                                current_is_first,
                                state_base_row,
                                source_row,
                                source_column,
                            )
                            * velocity[environment, source_column]
                        )
                        source_column += 1
                    source_jacobian_velocity_shared[source_row] = source_velocity_value
                    source_row += 1
            else:
                source_row = lane
                source_velocity_value = wp.float64(0.0)
                if source_row < SPATIAL_DIM:
                    source_column = int(0)
                    while source_column < segment_dofs:
                        source_velocity_value += (
                            _matrix_value(
                                jacobian_first,
                                jacobian_second,
                                current_is_first,
                                state_base_row,
                                source_row,
                                source_column,
                            )
                            * velocity[environment, source_column]
                        )
                        source_column += 1
                wp.tile_scatter_masked(
                    source_jacobian_velocity_shared,
                    source_row,
                    source_velocity_value,
                    source_row < SPATIAL_DIM,
                )
            if lane_stride == 1:
                target = int(0)
                while target < SPATIAL_DIM:
                    transported_value = wp.float64(0.0)
                    k = int(0)
                    while k < SPATIAL_DIM:
                        transported_value += (
                            cell_adjoint[cell_base_row + target, k]
                            * source_jacobian_velocity_shared[k]
                        )
                        k += 1
                    transported_source_velocity_shared[target] = transported_value
                    target += 1
            else:
                target = lane
                transported_value = wp.float64(0.0)
                if target < SPATIAL_DIM:
                    k = int(0)
                    while k < SPATIAL_DIM:
                        transported_value += (
                            cell_adjoint[cell_base_row + target, k]
                            * source_jacobian_velocity_shared[k]
                        )
                        k += 1
                wp.tile_scatter_masked(
                    transported_source_velocity_shared,
                    target,
                    transported_value,
                    target < SPATIAL_DIM,
                )
            state_row = lane
            while state_row < SPATIAL_DIM:
                transported_source_velocity = Vec6d(
                    transported_source_velocity_shared[0],
                    transported_source_velocity_shared[1],
                    transported_source_velocity_shared[2],
                    transported_source_velocity_shared[3],
                    transported_source_velocity_shared[4],
                    transported_source_velocity_shared[5],
                )
                derivative_value = wp.float64(0.0)
                velocity_value = wp.float64(0.0)
                gravity_value = wp.float64(0.0)
                k = int(0)
                while k < SPATIAL_DIM:
                    adjoint_value = cell_adjoint[cell_base_row + state_row, k]
                    derivative_value += adjoint_value * (
                        _vector_value(
                            jacobian_dot_qd_first,
                            jacobian_dot_qd_second,
                            current_is_first,
                            state_base_row,
                            k,
                        )
                        + cell_tangent_velocity_dot[cell_base_row + k, 0]
                    )
                    velocity_value += adjoint_value * (
                        _vector_value(
                            velocity_first,
                            velocity_second,
                            current_is_first,
                            state_base_row,
                            k,
                        )
                        + cell_link_velocity[cell_base_row + k, 0]
                    )
                    gravity_value += adjoint_value * _vector_value(
                        gravity_first,
                        gravity_second,
                        current_is_first,
                        state_base_row,
                        k,
                    )
                    k += 1
                step = Vec6d(
                    cell_step_velocity[cell_base_row + 0, 0],
                    cell_step_velocity[cell_base_row + 1, 0],
                    cell_step_velocity[cell_base_row + 2, 0],
                    cell_step_velocity[cell_base_row + 3, 0],
                    cell_step_velocity[cell_base_row + 4, 0],
                    cell_step_velocity[cell_base_row + 5, 0],
                )
                bracket = _ad_action(step, transported_source_velocity)
                _write_vector_value(
                    jacobian_dot_qd_first,
                    jacobian_dot_qd_second,
                    destination_is_first,
                    state_base_row,
                    state_row,
                    derivative_value - bracket[state_row],
                )
                _write_vector_value(
                    velocity_first,
                    velocity_second,
                    destination_is_first,
                    state_base_row,
                    state_row,
                    velocity_value,
                )
                _write_vector_value(
                    gravity_first,
                    gravity_second,
                    destination_is_first,
                    state_base_row,
                    state_row,
                    gravity_value,
                )
                state_row += lane_stride
            wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
            current_is_first = destination_is_first

            if cell < num_mass_points:
                quadrature_item = segment * num_mass_points + cell
                num_inertia_pairs = segment_dofs * (segment_dofs + 1) // 2
                entry = lane
                while entry < num_inertia_pairs:
                    output_row = inertia_upper_rows[entry]
                    output_column = inertia_upper_columns[entry]
                    value = wp.float64(0.0)
                    row = int(0)
                    while row < SPATIAL_DIM:
                        value += _matrix_value(
                            jacobian_first,
                            jacobian_second,
                            current_is_first,
                            state_base_row,
                            row,
                            output_row,
                        ) * _mass_jacobian_action(
                            weighted_masses,
                            quadrature_item,
                            row,
                            jacobian_first,
                            jacobian_second,
                            current_is_first,
                            state_base_row,
                            output_column,
                        )
                        row += 1
                    inertia[environment, output_row, output_column] += value
                    if output_row != output_column:
                        inertia[environment, output_column, output_row] += value
                    entry += lane_stride
                column = lane
                while column < segment_dofs:
                    wrench = _coadjoint_wrench(
                        velocity_first,
                        velocity_second,
                        jacobian_dot_qd_first,
                        jacobian_dot_qd_second,
                        current_is_first,
                        state_base_row,
                        weighted_masses,
                        quadrature_item,
                    )
                    coriolis_value = wp.float64(0.0)
                    gravity_value = wp.float64(0.0)
                    row = int(0)
                    while row < SPATIAL_DIM:
                        jacobian_value = _matrix_value(
                            jacobian_first,
                            jacobian_second,
                            current_is_first,
                            state_base_row,
                            row,
                            column,
                        )
                        coriolis_value += jacobian_value * wrench[row]
                        gravity_value -= jacobian_value * _mass_vector_action(
                            weighted_masses,
                            quadrature_item,
                            row,
                            gravity_first,
                            gravity_second,
                            current_is_first,
                            state_base_row,
                        )
                        row += 1
                    coriolis_qd[environment, column] += coriolis_value
                    gravity_force[environment, column] += gravity_value
                    column += lane_stride
            wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
            cell += 1
        segment += 1


@wp.struct
class GVSFloatingChainData:
    """Floating-base GVS chain inputs, propagation workspaces and dynamics outputs.

    Array fields are non-owning views; the caller must retain their allocations
    for every launch and captured graph. Homogeneous kernels receive this struct
    directly with batched fields spanning E worlds and shared structural fields.
    Packed kernels select it from a device array with E = 1 per descriptor; each
    descriptor may have its own structural dimensions and parameter values.

    Attributes:
        joint_adjoint: Flattened joint inverse adjoints.
        joint_adjoint_dot: Flattened joint-adjoint derivatives.
        joint_tangent: Internal joint tangents.
        joint_tangent_dot_qd: Joint tangent-derivative actions.
        joint_velocity: Internal joint spatial velocities.
        cell_adjoint: Flattened cell inverse adjoints.
        cell_tangent_local: Cell tangents in padded local coordinates.
        cell_link_velocity: Internal link velocities per cell.
        cell_step_velocity: Internal Magnus-step velocities per cell.
        cell_tangent_velocity_dot: Cell tangent-derivative actions.
        global_to_local: Internal active-to-local link-coordinate map.
        active_dofs: Cumulative internal coordinate count by segment.
        base_pose: Runtime scalar-last quaternion poses with shape ``(E, 7)``.
        velocity: Total angular-first generalized velocities.
        inertia_upper_rows: Packed augmented upper-inertia row indices.
        inertia_upper_columns: Packed augmented upper-inertia column indices.
        weighted_masses: Weighted spatial inertias with shape ``(S * P, 6)``
            for diagonals or ``(S * P, 36)`` for row-major full matrices.
            P includes the unit-weight link-tip slot when present.
        gravity_world: World-frame spatial gravity.
        num_cells_array: One-entry array containing cells per segment.
        num_quadrature_array: One-entry array containing quadrature count.
        lanes_per_block: One-entry array containing cooperative lanes.
        jacobian_first: First caller-owned Jacobian workspace.
        jacobian_dot_qd_first: First derivative-action workspace.
        velocity_first: First spatial-velocity workspace.
        gravity_first: First local-gravity workspace.
        jacobian_second: Second Jacobian workspace.
        jacobian_dot_qd_second: Second derivative-action workspace.
        velocity_second: Second spatial-velocity workspace.
        gravity_second: Second local-gravity workspace.
        inertia: Batched augmented inertia output.
        coriolis_qd: Batched augmented convective-force output.
        gravity_force: Batched augmented generalized-gravity output.
    """

    joint_adjoint: wp.array2d[wp.float64]
    joint_adjoint_dot: wp.array2d[wp.float64]
    joint_tangent: wp.array2d[wp.float64]
    joint_tangent_dot_qd: wp.array2d[wp.float64]
    joint_velocity: wp.array2d[wp.float64]
    cell_adjoint: wp.array2d[wp.float64]
    cell_tangent_local: wp.array2d[wp.float64]
    cell_link_velocity: wp.array2d[wp.float64]
    cell_step_velocity: wp.array2d[wp.float64]
    cell_tangent_velocity_dot: wp.array2d[wp.float64]
    global_to_local: wp.array2d[wp.int32]
    active_dofs: wp.array[wp.int32]
    base_pose: wp.array2d[wp.float64]
    velocity: wp.array2d[wp.float64]
    inertia_upper_rows: wp.array[wp.int32]
    inertia_upper_columns: wp.array[wp.int32]
    weighted_masses: wp.array2d[wp.float64]
    gravity_world: wp.array[wp.float64]
    num_cells_array: wp.array[wp.int32]
    num_quadrature_array: wp.array[wp.int32]
    lanes_per_block: wp.array[wp.int32]
    jacobian_first: wp.array2d[wp.float64]
    jacobian_dot_qd_first: wp.array2d[wp.float64]
    velocity_first: wp.array2d[wp.float64]
    gravity_first: wp.array2d[wp.float64]
    jacobian_second: wp.array2d[wp.float64]
    jacobian_dot_qd_second: wp.array2d[wp.float64]
    velocity_second: wp.array2d[wp.float64]
    gravity_second: wp.array2d[wp.float64]
    inertia: wp.array3d[wp.float64]
    coriolis_qd: wp.array2d[wp.float64]
    gravity_force: wp.array2d[wp.float64]


@wp.kernel(enable_backward=False)
def homogeneous_floating_persistent_chain_kernel(data: GVSFloatingChainData):
    """Assemble dynamics for a homogeneous batch of floating-base chains.

    All worlds use the same structural dimensions and coordinate layout. Dimensions:
    E = worlds, B = threads per block.

    Shared across worlds: ``global_to_local``, ``active_dofs``,
    ``inertia_upper_rows``, ``inertia_upper_columns``, ``weighted_masses``,
    ``gravity_world``, ``num_cells_array``, ``num_quadrature_array``,
    ``lanes_per_block``.

    State, computed operator, workspace and output arrays carry the world axis,
    either explicitly or flattened as documented by the struct. Values may differ by
    world.

    Launch with ``wp.launch_tiled(..., dim=E, block_dim=B)``. Set
    ``lanes_per_block[0] = B``; all lanes cooperate on one world.

    Joint and cell Lie operators are shared with the fixed pipeline. The
    floating entry specializes root initialization, shifted internal columns,
    augmented causal widths, and full dynamics accumulation.

    Args:
        data: Shared GVSFloatingChainData layout referencing the complete homogeneous
            batch (E worlds). See the struct for field shapes and ownership.

    Returns:
        None. Workspaces and outputs are updated in place.
    """
    environment, lane = wp.tid()
    floating_persistent_chain_at(
        environment,
        lane,
        data.joint_adjoint,
        data.joint_adjoint_dot,
        data.joint_tangent,
        data.joint_tangent_dot_qd,
        data.joint_velocity,
        data.cell_adjoint,
        data.cell_tangent_local,
        data.cell_link_velocity,
        data.cell_step_velocity,
        data.cell_tangent_velocity_dot,
        data.global_to_local,
        data.active_dofs,
        data.base_pose,
        data.velocity,
        data.inertia_upper_rows,
        data.inertia_upper_columns,
        data.weighted_masses,
        data.gravity_world,
        data.num_cells_array,
        data.num_quadrature_array,
        data.lanes_per_block,
        data.jacobian_first,
        data.jacobian_dot_qd_first,
        data.velocity_first,
        data.gravity_first,
        data.jacobian_second,
        data.jacobian_dot_qd_second,
        data.velocity_second,
        data.gravity_second,
        data.inertia,
        data.coriolis_qd,
        data.gravity_force,
    )


@wp.kernel(enable_backward=False)
def packed_floating_persistent_chain_kernel(
    world_data: wp.array[GVSFloatingChainData],
    work_items: wp.array[wp.vec3i],
):
    """Traverse one augmented floating GVS chain per cooperative block using packed world data.

    Each selected descriptor contains one world (E = 1). Structure and numerical
    model parameters may differ between worlds. Descriptors may share allocations
    when their values match.

    Launch with ``wp.launch_tiled(..., dim=len(work_items), block_dim=B)``.
    Each block cooperates on one work item; all lanes must participate.
    Use the same block size as the arrays' ``lanes_per_block[0]``.

    Args:
        world_data: Device array of per-world data views, including caller-owned
            outputs and workspaces. Keep every referenced allocation alive.
        work_items: INT32 triples ``(world_index, local_index, 0)``. The local
            index is zero (one work item per world). The third component is reserved.

    Returns:
        None. Workspaces and outputs are updated in place.
    """
    item, lane = wp.tid()
    entry = work_items[item]
    data = world_data[entry[0]]
    floating_persistent_chain_at(
        0,
        lane,
        data.joint_adjoint,
        data.joint_adjoint_dot,
        data.joint_tangent,
        data.joint_tangent_dot_qd,
        data.joint_velocity,
        data.cell_adjoint,
        data.cell_tangent_local,
        data.cell_link_velocity,
        data.cell_step_velocity,
        data.cell_tangent_velocity_dot,
        data.global_to_local,
        data.active_dofs,
        data.base_pose,
        data.velocity,
        data.inertia_upper_rows,
        data.inertia_upper_columns,
        data.weighted_masses,
        data.gravity_world,
        data.num_cells_array,
        data.num_quadrature_array,
        data.lanes_per_block,
        data.jacobian_first,
        data.jacobian_dot_qd_first,
        data.velocity_first,
        data.gravity_first,
        data.jacobian_second,
        data.jacobian_dot_qd_second,
        data.velocity_second,
        data.gravity_second,
        data.inertia,
        data.coriolis_qd,
        data.gravity_force,
    )


def launch_homogeneous_floating_persistent_chain(
    joint_adjoint: wp.array2d[wp.float64],
    joint_adjoint_dot: wp.array2d[wp.float64],
    joint_tangent: wp.array2d[wp.float64],
    joint_tangent_dot_qd: wp.array2d[wp.float64],
    joint_velocity: wp.array2d[wp.float64],
    cell_adjoint: wp.array2d[wp.float64],
    cell_tangent_local: wp.array2d[wp.float64],
    cell_link_velocity: wp.array2d[wp.float64],
    cell_step_velocity: wp.array2d[wp.float64],
    cell_tangent_velocity_dot: wp.array2d[wp.float64],
    global_to_local: wp.array2d[wp.int32],
    active_dofs: wp.array[wp.int32],
    base_pose: wp.array2d[wp.float64],
    velocity: wp.array2d[wp.float64],
    inertia_upper_rows: wp.array[wp.int32],
    inertia_upper_columns: wp.array[wp.int32],
    weighted_masses: wp.array2d[wp.float64],
    gravity_world: wp.array[wp.float64],
    num_cells: wp.array[wp.int32],
    num_quadrature: wp.array[wp.int32],
    lanes_per_block: wp.array[wp.int32],
    block_dim: int,
    jacobian_first: wp.array2d[wp.float64],
    jacobian_dot_qd_first: wp.array2d[wp.float64],
    velocity_first: wp.array2d[wp.float64],
    gravity_first: wp.array2d[wp.float64],
    jacobian_second: wp.array2d[wp.float64],
    jacobian_dot_qd_second: wp.array2d[wp.float64],
    velocity_second: wp.array2d[wp.float64],
    gravity_second: wp.array2d[wp.float64],
    inertia: wp.array3d[wp.float64],
    coriolis_qd: wp.array2d[wp.float64],
    gravity_force: wp.array2d[wp.float64],
):
    """Launch floating persistent chain for a homogeneous batch.

    All worlds use the same structural dimensions and coordinate layout.

    Shared across worlds: ``global_to_local``, ``active_dofs``,
    ``inertia_upper_rows``, ``inertia_upper_columns``, ``weighted_masses``,
    ``gravity_world``, ``num_cells``, ``num_quadrature``, ``lanes_per_block``,
    ``block_dim``.

    State, computed operator, workspace and output arrays carry the world axis,
    either explicitly or flattened as documented in Args. Their values may differ by
    world.

    The launcher derives its work grid from these common dimensions and enqueues the
    homogeneous kernels on the current Warp stream.

    Args:
        joint_adjoint: Flattened joint inverse adjoints.
        joint_adjoint_dot: Flattened joint-adjoint derivatives.
        joint_tangent: Internal joint tangents.
        joint_tangent_dot_qd: Joint tangent-derivative actions.
        joint_velocity: Internal joint spatial velocities.
        cell_adjoint: Flattened cell inverse adjoints.
        cell_tangent_local: Cell tangents in padded local coordinates.
        cell_link_velocity: Internal link velocities per cell.
        cell_step_velocity: Internal Magnus-step velocities per cell.
        cell_tangent_velocity_dot: Cell tangent-derivative actions.
        global_to_local: Internal active-to-local link-coordinate map.
        active_dofs: Cumulative internal coordinate count by segment.
        base_pose: Batched runtime scalar-last quaternion poses.
        velocity: Batched total generalized velocities.
        inertia_upper_rows: Packed augmented upper-inertia row indices.
        inertia_upper_columns: Matching upper-inertia column indices.
        weighted_masses: Weighted spatial inertias with shape ``(S * P, 6)``
            for diagonals or ``(S * P, 36)`` for row-major full matrices.
            P includes the unit-weight link-tip slot when present.
        gravity_world: World-frame spatial gravity.
        num_cells: One-entry array containing cells per segment.
        num_quadrature: One-entry array containing quadrature points.
        lanes_per_block: One-entry array containing cooperative lanes.
        block_dim: CUDA threads per persistent block, or one on CPU.
        jacobian_first: First caller-owned Jacobian workspace.
        jacobian_dot_qd_first: First derivative-action workspace.
        velocity_first: First spatial-velocity workspace.
        gravity_first: First local-gravity workspace.
        jacobian_second: Second Jacobian workspace.
        jacobian_dot_qd_second: Second derivative-action workspace.
        velocity_second: Second spatial-velocity workspace.
        gravity_second: Second local-gravity workspace.
        inertia: Batched augmented inertia output.
        coriolis_qd: Batched augmented convective-force output.
        gravity_force: Batched augmented generalized-gravity output.

    Returns:
        None. Workspaces and outputs are updated in place.
    """
    data = GVSFloatingChainData()
    data.joint_adjoint = joint_adjoint
    data.joint_adjoint_dot = joint_adjoint_dot
    data.joint_tangent = joint_tangent
    data.joint_tangent_dot_qd = joint_tangent_dot_qd
    data.joint_velocity = joint_velocity
    data.cell_adjoint = cell_adjoint
    data.cell_tangent_local = cell_tangent_local
    data.cell_link_velocity = cell_link_velocity
    data.cell_step_velocity = cell_step_velocity
    data.cell_tangent_velocity_dot = cell_tangent_velocity_dot
    data.global_to_local = global_to_local
    data.active_dofs = active_dofs
    data.base_pose = base_pose
    data.velocity = velocity
    data.inertia_upper_rows = inertia_upper_rows
    data.inertia_upper_columns = inertia_upper_columns
    data.weighted_masses = weighted_masses
    data.gravity_world = gravity_world
    data.num_cells_array = num_cells
    data.num_quadrature_array = num_quadrature
    data.lanes_per_block = lanes_per_block
    data.jacobian_first = jacobian_first
    data.jacobian_dot_qd_first = jacobian_dot_qd_first
    data.velocity_first = velocity_first
    data.gravity_first = gravity_first
    data.jacobian_second = jacobian_second
    data.jacobian_dot_qd_second = jacobian_dot_qd_second
    data.velocity_second = velocity_second
    data.gravity_second = gravity_second
    data.inertia = inertia
    data.coriolis_qd = coriolis_qd
    data.gravity_force = gravity_force
    wp.launch_tiled(
        homogeneous_floating_persistent_chain_kernel,
        dim=velocity.shape[0],
        inputs=[data],
        block_dim=block_dim,
    )


scalable_floating_persistent_chain = wp.jax_callable(
    launch_homogeneous_floating_persistent_chain, num_outputs=11
)

__all__ = [
    "homogeneous_floating_persistent_chain_kernel",
    "launch_homogeneous_floating_persistent_chain",
]


__all__ += ["GVSFloatingChainData", "packed_floating_persistent_chain_kernel"]
