# ruff: noqa: UP018
"""Spatial inertia projection shared by fixed and floating GVS chains."""

import warp as wp

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


@wp.func
def _accumulate_mass_point(
    environment: int,
    lane: int,
    lane_stride: int,
    active_dofs: int,
    item: int,
    is_tip: bool,
    current_is_first: bool,
    weighted_masses: wp.array2d[wp.float64],
    inertia_upper_rows: wp.array[wp.int32],
    inertia_upper_columns: wp.array[wp.int32],
    jacobian_first: wp.array2d[wp.float64],
    jacobian_second: wp.array2d[wp.float64],
    jacobian_dot_qd_first: wp.array2d[wp.float64],
    jacobian_dot_qd_second: wp.array2d[wp.float64],
    velocity_first: wp.array2d[wp.float64],
    velocity_second: wp.array2d[wp.float64],
    gravity_first: wp.array2d[wp.float64],
    gravity_second: wp.array2d[wp.float64],
    inertia: wp.array3d[wp.float64],
    coriolis_qd: wp.array2d[wp.float64],
    gravity_force: wp.array2d[wp.float64],
):
    """Accumulate one weighted section or unweighted tip into M, Cv and G.

    All lanes of one chain block must call together, after state transport has
    finished. The current buffers hold J, Jdot*qdot, velocity and gravity. The
    inactive buffers are dead until the next transport overwrites them, so use
    their Jacobian, derivative-action and gravity storage for I*J, the inertial
    wrench and I*gravity. A barrier separates these writes from projection;
    the caller must synchronize again before the next transport.

    Diagonal inertia keeps its direct product, avoiding the extra I*J traffic.
    Tip zero checks use exact runtime values, so changing parameters or replaying
    a captured graph with a newly nonzero tip cannot silently drop its inertia.

    Args:
        environment: Batch row in the supplied arrays; zero for a packed world.
        lane: Thread index within the cooperative chain block.
        lane_stride: Number of active cooperative lanes; one on CPU.
        active_dofs: Number of generalized velocities active at this mass point.
        item: Row of weighted_masses for the current section or link tip.
        is_tip: Whether this is a unit-weight tip contribution eligible for
            exact zero-inertia skipping.
        current_is_first: Whether the first transport buffers hold the current
            state; the other buffers provide temporary storage.
        weighted_masses: Weighted diagonal entries with shape (items, 6), or
            full spatial inertia matrices with shape (items, 36) in row-major,
            angular-first order. Tip entries have unit weight.
        inertia_upper_rows: Row indices of the packed upper inertia triangle.
        inertia_upper_columns: Column indices of the packed upper triangle.
        jacobian_first: First Jacobian buffer, with six rows per environment.
        jacobian_second: Second Jacobian buffer; the inactive buffer stores I*J
            for full inertia until the next transport.
        jacobian_dot_qd_first: First buffer for the Jacobian-derivative action.
        jacobian_dot_qd_second: Second derivative-action buffer; the inactive
            buffer stores the inertial wrench until the next transport.
        velocity_first: First buffer of angular-first spatial velocities.
        velocity_second: Second spatial-velocity buffer.
        gravity_first: First buffer of local spatial gravity.
        gravity_second: Second local-gravity buffer; the inactive buffer stores
            weighted gravity until the next transport.
        inertia: Batched generalized inertia matrices, updated by accumulation.
        coriolis_qd: Batched convective forces, updated by accumulation.
        gravity_force: Batched generalized gravity forces, updated by accumulation.

    Returns:
        None. Accumulates into inertia, coriolis_qd, and gravity_force and
        overwrites inactive workspace buffers. An exactly zero tip leaves
        outputs and transport buffers unchanged.
    """
    state_base_row = environment * 6
    scratch_is_first = not current_is_first
    enabled = wp.tile_zeros(shape=(1,), dtype=wp.int32, storage="shared")
    include = int(1)
    if is_tip:
        if lane == 0:
            include = int(0)
            entry = int(0)
            while entry < weighted_masses.shape[1]:
                if weighted_masses[item, entry] != wp.float64(0.0):
                    include = int(1)
                entry += 1
        wp.tile_scatter_masked(enabled, 0, include, lane == 0)
        include = enabled[0]
    if include != 0:
        if weighted_masses.shape[1] == 36:
            entry = lane
            while entry < 6 * active_dofs:
                row = entry // active_dofs
                column = entry - row * active_dofs
                value = _mass_jacobian_action(
                    weighted_masses,
                    item,
                    row,
                    jacobian_first,
                    jacobian_second,
                    current_is_first,
                    state_base_row,
                    column,
                )
                _write_matrix_value(
                    jacobian_first,
                    jacobian_second,
                    scratch_is_first,
                    state_base_row,
                    row,
                    column,
                    value,
                )
                entry += lane_stride
        if lane == 0:
            wrench = _coadjoint_wrench(
                velocity_first,
                velocity_second,
                jacobian_dot_qd_first,
                jacobian_dot_qd_second,
                current_is_first,
                state_base_row,
                weighted_masses,
                item,
            )
            for row in range(6):
                _write_vector_value(
                    jacobian_dot_qd_first,
                    jacobian_dot_qd_second,
                    scratch_is_first,
                    state_base_row,
                    row,
                    wrench[row],
                )
        row = lane
        while row < 6:
            value = _mass_vector_action(
                weighted_masses,
                item,
                row,
                gravity_first,
                gravity_second,
                current_is_first,
                state_base_row,
            )
            _write_vector_value(
                gravity_first,
                gravity_second,
                scratch_is_first,
                state_base_row,
                row,
                value,
            )
            row += lane_stride
        wp.tile_scatter_masked(enabled, 0, int(1), lane == 0)
        entry = lane
        while entry < active_dofs * (active_dofs + 1) // 2:
            output_row = inertia_upper_rows[entry]
            output_column = inertia_upper_columns[entry]
            value = wp.float64(0.0)
            row = int(0)
            while row < 6:
                weighted_column = wp.float64(0.0)
                if weighted_masses.shape[1] == 6:
                    weighted_column = _mass_jacobian_action(
                        weighted_masses,
                        item,
                        row,
                        jacobian_first,
                        jacobian_second,
                        current_is_first,
                        state_base_row,
                        output_column,
                    )
                else:
                    weighted_column = _matrix_value(
                        jacobian_first,
                        jacobian_second,
                        scratch_is_first,
                        state_base_row,
                        row,
                        output_column,
                    )
                value += (
                    _matrix_value(
                        jacobian_first,
                        jacobian_second,
                        current_is_first,
                        state_base_row,
                        row,
                        output_row,
                    )
                    * weighted_column
                )
                row += 1
            inertia[environment, output_row, output_column] += value
            if output_row != output_column:
                inertia[environment, output_column, output_row] += value
            entry += lane_stride
        column = lane
        while column < active_dofs:
            coriolis_value = wp.float64(0.0)
            gravity_value = wp.float64(0.0)
            row = int(0)
            while row < 6:
                jacobian_value = _matrix_value(
                    jacobian_first,
                    jacobian_second,
                    current_is_first,
                    state_base_row,
                    row,
                    column,
                )
                coriolis_value += jacobian_value * _vector_value(
                    jacobian_dot_qd_first,
                    jacobian_dot_qd_second,
                    scratch_is_first,
                    state_base_row,
                    row,
                )
                gravity_value -= jacobian_value * _vector_value(
                    gravity_first,
                    gravity_second,
                    scratch_is_first,
                    state_base_row,
                    row,
                )
                row += 1
            coriolis_qd[environment, column] += coriolis_value
            gravity_force[environment, column] += gravity_value
            column += lane_stride
