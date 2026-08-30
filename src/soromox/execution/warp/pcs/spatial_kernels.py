# ruff: noqa: I001, UP018
"""Runtime-shaped spatial PCS dynamics kernels."""

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
from soromox.execution.warp.common.spatial import _coadjoint_wrench
from soromox.execution.warp.common.storage import (
    _matrix_value,
    _vector_value,
    _write_matrix_value,
    _write_vector_value,
)

wp.set_module_options({"enable_backward": False})

SPATIAL_DIM = 6


@wp.kernel(enable_backward=False)
def spatial_local_operators_kernel(
    q: wp.array2d[wp.float64],
    qd: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    operator_points: wp.array2d[wp.float64],
    adjoint_inverse: wp.array2d[wp.float64],
    transported_tangent: wp.array2d[wp.float64],
    local_velocity: wp.array2d[wp.float64],
    transported_tangent_dot_velocity: wp.array2d[wp.float64],
):
    """Evaluate spatial local operators for one environment-segment point.

    Args:
        q: Batched active generalized coordinates.
        qd: Batched active generalized velocities.
        active_indices: Active-coordinate index of each strain component.
        active_scales: Scale applied to each active strain component.
        reference_strain: Segment reference strains.
        operator_points: Segment-local evaluation coordinates.
        adjoint_inverse: Caller-owned flattened inverse-adjoint output.
        transported_tangent: Caller-owned flattened tangent output.
        local_velocity: Caller-owned flattened local-velocity output.
        transported_tangent_dot_velocity: Caller-owned derivative-action output.

    Returns:
        None. All results are written to caller-owned output arrays.
    """
    item = wp.tid()
    points_per_segment = operator_points.shape[1]
    points_per_environment = active_indices.shape[0] * points_per_segment
    environment = item // points_per_environment
    environment_point = item - environment * points_per_environment
    segment = environment_point // points_per_segment
    point = environment_point - segment * points_per_segment

    xi = Vec6d(
        reference_strain[segment, 0],
        reference_strain[segment, 1],
        reference_strain[segment, 2],
        reference_strain[segment, 3],
        reference_strain[segment, 4],
        reference_strain[segment, 5],
    )
    xid = Vec6d()
    local = int(0)
    while local < SPATIAL_DIM:
        global_index = active_indices[segment, local]
        if global_index >= 0:
            scale = active_scales[segment, local]
            xi[local] += scale * q[environment, global_index]
            xid[local] = scale * qd[environment, global_index]
        local += 1

    s = operator_points[segment, point]
    accumulated = s * xi
    accumulated_dot = s * xid
    angle_sq = (
        accumulated[0] * accumulated[0]
        + accumulated[1] * accumulated[1]
        + accumulated[2] * accumulated[2]
    )
    coefficients = _left_coefficients(angle_sq)
    forward = _forward_coefficients(angle_sq)
    omega = wp.vec3d(accumulated[0], accumulated[1], accumulated[2])
    linear = wp.vec3d(accumulated[3], accumulated[4], accumulated[5])
    translation = _translation(omega, linear, forward)
    output_base_row = item * SPATIAL_DIM

    row = int(0)
    while row < SPATIAL_DIM:
        column = int(0)
        while column < SPATIAL_DIM:
            adjoint_inverse[output_base_row + row, column] = _adjoint_inverse_entry(
                omega,
                translation,
                angle_sq,
                forward,
                row,
                column,
            )
            column += 1
        row += 1

    local = int(0)
    while local < SPATIAL_DIM:
        basis = Vec6d()
        basis[local] = s
        tangent_column = _left_action(accumulated, basis, coefficients)
        transported_column = _adjoint_inverse_action(
            omega,
            translation,
            angle_sq,
            forward,
            tangent_column,
        )
        row = int(0)
        while row < SPATIAL_DIM:
            transported_tangent[output_base_row + row, local] = transported_column[row]
            row += 1
        local += 1

    tangent_velocity = _left_action(accumulated, accumulated_dot, coefficients)
    transported_velocity = _adjoint_inverse_action(
        omega,
        translation,
        angle_sq,
        forward,
        tangent_velocity,
    )
    coefficient_derivatives = _left_coefficient_x_derivatives(angle_sq)
    angle_sq_dot = wp.float64(2.0) * (
        accumulated[0] * accumulated_dot[0]
        + accumulated[1] * accumulated_dot[1]
        + accumulated[2] * accumulated_dot[2]
    )
    tangent_dot_velocity = _left_total_derivative_action(
        accumulated,
        accumulated_dot,
        accumulated_dot,
        Vec6d(),
        coefficients,
        coefficient_derivatives * angle_sq_dot,
    )
    transported_tangent_dot = _adjoint_inverse_action(
        omega,
        translation,
        angle_sq,
        forward,
        tangent_dot_velocity,
    )
    row = int(0)
    while row < SPATIAL_DIM:
        local_velocity[output_base_row + row, 0] = transported_velocity[row]
        transported_tangent_dot_velocity[output_base_row + row, 0] = (
            transported_tangent_dot[row]
        )
        row += 1


def launch_spatial_local_operators(
    q: wp.array2d[wp.float64],
    qd: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    operator_points: wp.array2d[wp.float64],
    adjoint_inverse: wp.array2d[wp.float64],
    transported_tangent: wp.array2d[wp.float64],
    local_velocity: wp.array2d[wp.float64],
    transported_tangent_dot_velocity: wp.array2d[wp.float64],
):
    """Launch spatial PCS local SE(3) operator evaluation.

    Args:
        q: FP64 active coordinates with shape ``(batch_size, num_dofs)``.
        qd: FP64 active velocities with the same shape as ``q``.
        active_indices: Active-coordinate index of each spatial strain row.
        active_scales: Length-normalization scale of each strain row.
        reference_strain: Reference strain with shape ``(num_segments, 6)``.
        operator_points: Local evaluation coordinates for each segment.
        adjoint_inverse: Preallocated flattened inverse-adjoint output.
        transported_tangent: Preallocated flattened tangent output.
        local_velocity: Preallocated flattened local-velocity output.
        transported_tangent_dot_velocity: Preallocated flattened tangent
            derivative action.

    Returns:
        None. Outputs are written in place.
    """

    wp.launch(
        spatial_local_operators_kernel,
        dim=q.shape[0] * active_indices.shape[0] * operator_points.shape[1],
        inputs=[
            q,
            qd,
            active_indices,
            active_scales,
            reference_strain,
            operator_points,
        ],
        outputs=[
            adjoint_inverse,
            transported_tangent,
            local_velocity,
            transported_tangent_dot_velocity,
        ],
        block_dim=128,
    )


spatial_local_operators = wp.jax_callable(launch_spatial_local_operators, num_outputs=4)


@wp.kernel(enable_backward=False)
def spatial_persistent_chain_kernel(
    adjoint_inverse: wp.array2d[wp.float64],
    transported_tangent: wp.array2d[wp.float64],
    local_velocity: wp.array2d[wp.float64],
    transported_tangent_dot_velocity: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    active_dof_ends: wp.array[wp.int32],
    qd: wp.array2d[wp.float64],
    inertia_upper_rows: wp.array[wp.int32],
    inertia_upper_columns: wp.array[wp.int32],
    weighted_masses: wp.array2d[wp.float64],
    gravity_base: wp.array[wp.float64],
    block_dim: wp.int32,
    jacobian_first: wp.array2d[wp.float64],
    derivative_first: wp.array2d[wp.float64],
    velocity_first: wp.array2d[wp.float64],
    gravity_first: wp.array2d[wp.float64],
    jacobian_second: wp.array2d[wp.float64],
    derivative_second: wp.array2d[wp.float64],
    velocity_second: wp.array2d[wp.float64],
    gravity_second: wp.array2d[wp.float64],
    inertia: wp.array3d[wp.float64],
    coriolis_qd: wp.array2d[wp.float64],
    gravity_force: wp.array2d[wp.float64],
):
    """Traverse one spatial PCS chain per persistent cooperative block.

    Args:
        adjoint_inverse: Flattened inverse-adjoint operators.
        transported_tangent: Flattened transported local tangents.
        local_velocity: Flattened local spatial velocities.
        transported_tangent_dot_velocity: Flattened derivative actions.
        active_indices: Active-coordinate index of each strain component.
        active_scales: Scale applied to each active strain component.
        active_dof_ends: Cumulative active coordinate count by segment.
        qd: Batched generalized velocities.
        inertia_upper_rows: Packed upper-inertia row indices.
        inertia_upper_columns: Packed upper-inertia column indices.
        weighted_masses: Quadrature-weighted diagonal spatial inertias.
        gravity_base: Base-frame spatial gravity.
        block_dim: Number of active cooperative lanes.
        jacobian_first: First caller-owned Jacobian workspace.
        derivative_first: First derivative-action workspace.
        velocity_first: First spatial-velocity workspace.
        gravity_first: First local-gravity workspace.
        jacobian_second: Second Jacobian workspace.
        derivative_second: Second derivative-action workspace.
        velocity_second: Second spatial-velocity workspace.
        gravity_second: Second local-gravity workspace.
        inertia: Batched inertia output.
        coriolis_qd: Batched convective-force output.
        gravity_force: Batched generalized-gravity output.

    Returns:
        None. Workspaces and outputs are updated in place.
    """
    environment, lane = wp.tid()
    num_dofs = qd.shape[1]
    num_segments = active_indices.shape[0]
    num_quadrature = weighted_masses.shape[0] // num_segments
    points_per_segment = num_quadrature + 1
    lane_stride = block_dim
    state_base_row = environment * SPATIAL_DIM
    barrier = wp.tile_zeros(shape=(1,), dtype=wp.int32, storage="shared")
    transported_velocity_shared = wp.tile_zeros(
        shape=(SPATIAL_DIM,), dtype=wp.float64, storage="shared"
    )

    entry = lane
    while entry < SPATIAL_DIM * num_dofs:
        row = entry // num_dofs
        column = entry - row * num_dofs
        jacobian_first[state_base_row + row, column] = wp.float64(0.0)
        jacobian_second[state_base_row + row, column] = wp.float64(0.0)
        entry += lane_stride
    row = lane
    while row < SPATIAL_DIM:
        derivative_first[state_base_row + row, 0] = wp.float64(0.0)
        derivative_second[state_base_row + row, 0] = wp.float64(0.0)
        velocity_first[state_base_row + row, 0] = wp.float64(0.0)
        velocity_second[state_base_row + row, 0] = wp.float64(0.0)
        gravity_first[state_base_row + row, 0] = gravity_base[row]
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

    current_is_first = bool(True)
    segment = int(0)
    while segment < num_segments:
        active_end = active_dof_ends[segment]
        point = int(0)
        while point < points_per_segment:
            destination_is_first = not current_is_first
            operator_item = (
                environment * num_segments + segment
            ) * points_per_segment + point
            operator_base_row = operator_item * SPATIAL_DIM

            entry = lane
            while entry < SPATIAL_DIM * active_end:
                output_row = entry // active_end
                output_column = entry - output_row * active_end
                value = wp.float64(0.0)
                k = int(0)
                while k < SPATIAL_DIM:
                    value += adjoint_inverse[operator_base_row + output_row, k] * (
                        _matrix_value(
                            jacobian_first,
                            jacobian_second,
                            current_is_first,
                            state_base_row,
                            k,
                            output_column,
                        )
                    )
                    k += 1
                local = int(0)
                while local < SPATIAL_DIM:
                    if active_indices[segment, local] == output_column:
                        value += (
                            transported_tangent[operator_base_row + output_row, local]
                            * active_scales[segment, local]
                        )
                    local += 1
                _write_matrix_value(
                    jacobian_first,
                    jacobian_second,
                    destination_is_first,
                    state_base_row,
                    output_row,
                    output_column,
                    value,
                )
                entry += lane_stride
            wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)

            row = lane
            transported_velocity = wp.float64(0.0)
            gravity_value = wp.float64(0.0)
            if row < SPATIAL_DIM:
                k = int(0)
                while k < SPATIAL_DIM:
                    adjoint_value = adjoint_inverse[operator_base_row + row, k]
                    transported_velocity += adjoint_value * _vector_value(
                        velocity_first,
                        velocity_second,
                        current_is_first,
                        state_base_row,
                        k,
                    )
                    gravity_value += adjoint_value * _vector_value(
                        gravity_first,
                        gravity_second,
                        current_is_first,
                        state_base_row,
                        k,
                    )
                    k += 1
                _write_vector_value(
                    velocity_first,
                    velocity_second,
                    destination_is_first,
                    state_base_row,
                    row,
                    transported_velocity + local_velocity[operator_base_row + row, 0],
                )
                _write_vector_value(
                    gravity_first,
                    gravity_second,
                    destination_is_first,
                    state_base_row,
                    row,
                    gravity_value,
                )
            wp.tile_scatter_masked(
                transported_velocity_shared,
                row,
                transported_velocity,
                row < SPATIAL_DIM,
            )

            if row < SPATIAL_DIM:
                derivative_value = wp.float64(0.0)
                k = int(0)
                while k < SPATIAL_DIM:
                    derivative_value += adjoint_inverse[
                        operator_base_row + row, k
                    ] * _vector_value(
                        derivative_first,
                        derivative_second,
                        current_is_first,
                        state_base_row,
                        k,
                    )
                    k += 1
                local_eta = Vec6d(
                    local_velocity[operator_base_row + 0, 0],
                    local_velocity[operator_base_row + 1, 0],
                    local_velocity[operator_base_row + 2, 0],
                    local_velocity[operator_base_row + 3, 0],
                    local_velocity[operator_base_row + 4, 0],
                    local_velocity[operator_base_row + 5, 0],
                )
                transported_source = Vec6d(
                    transported_velocity_shared[0],
                    transported_velocity_shared[1],
                    transported_velocity_shared[2],
                    transported_velocity_shared[3],
                    transported_velocity_shared[4],
                    transported_velocity_shared[5],
                )
                correction = _ad_action(local_eta, transported_source)
                derivative_value += (
                    transported_tangent_dot_velocity[operator_base_row + row, 0]
                    - correction[row]
                )
                _write_vector_value(
                    derivative_first,
                    derivative_second,
                    destination_is_first,
                    state_base_row,
                    row,
                    derivative_value,
                )
            wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)

            if point < num_quadrature:
                pair = lane
                while pair < inertia_upper_rows.shape[0]:
                    output_row = inertia_upper_rows[pair]
                    output_column = inertia_upper_columns[pair]
                    if output_column < active_end:
                        value = wp.float64(0.0)
                        spatial = int(0)
                        mass_item = segment * num_quadrature + point
                        while spatial < SPATIAL_DIM:
                            value += (
                                _matrix_value(
                                    jacobian_first,
                                    jacobian_second,
                                    destination_is_first,
                                    state_base_row,
                                    spatial,
                                    output_row,
                                )
                                * weighted_masses[mass_item, spatial]
                                * _matrix_value(
                                    jacobian_first,
                                    jacobian_second,
                                    destination_is_first,
                                    state_base_row,
                                    spatial,
                                    output_column,
                                )
                            )
                            spatial += 1
                        inertia[environment, output_row, output_column] += value
                        if output_row != output_column:
                            inertia[environment, output_column, output_row] += value
                    pair += lane_stride

                column = lane
                while column < active_end:
                    mass_item = segment * num_quadrature + point
                    wrench = _coadjoint_wrench(
                        velocity_first,
                        velocity_second,
                        derivative_first,
                        derivative_second,
                        destination_is_first,
                        state_base_row,
                        weighted_masses,
                        mass_item,
                    )
                    coriolis_value = wp.float64(0.0)
                    gravity_force_value = wp.float64(0.0)
                    spatial = int(0)
                    while spatial < SPATIAL_DIM:
                        jacobian_value = _matrix_value(
                            jacobian_first,
                            jacobian_second,
                            destination_is_first,
                            state_base_row,
                            spatial,
                            column,
                        )
                        coriolis_value += jacobian_value * wrench[spatial]
                        gravity_force_value -= (
                            jacobian_value
                            * weighted_masses[mass_item, spatial]
                            * _vector_value(
                                gravity_first,
                                gravity_second,
                                destination_is_first,
                                state_base_row,
                                spatial,
                            )
                        )
                        spatial += 1
                    coriolis_qd[environment, column] += coriolis_value
                    gravity_force[environment, column] += gravity_force_value
                    column += lane_stride
                wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
            else:
                current_is_first = destination_is_first
            point += 1
        segment += 1


def launch_spatial_persistent_chain(
    adjoint_inverse: wp.array2d[wp.float64],
    transported_tangent: wp.array2d[wp.float64],
    local_velocity: wp.array2d[wp.float64],
    transported_tangent_dot_velocity: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    active_dof_ends: wp.array[wp.int32],
    qd: wp.array2d[wp.float64],
    inertia_upper_rows: wp.array[wp.int32],
    inertia_upper_columns: wp.array[wp.int32],
    weighted_masses: wp.array2d[wp.float64],
    gravity_base: wp.array[wp.float64],
    block_dim: int,
    jacobian_first: wp.array2d[wp.float64],
    derivative_first: wp.array2d[wp.float64],
    velocity_first: wp.array2d[wp.float64],
    gravity_first: wp.array2d[wp.float64],
    jacobian_second: wp.array2d[wp.float64],
    derivative_second: wp.array2d[wp.float64],
    velocity_second: wp.array2d[wp.float64],
    gravity_second: wp.array2d[wp.float64],
    inertia: wp.array3d[wp.float64],
    coriolis_qd: wp.array2d[wp.float64],
    gravity_force: wp.array2d[wp.float64],
):
    """Launch one persistent spatial PCS chain block per environment.

    Args:
        adjoint_inverse: Flattened outputs from
            :func:`launch_spatial_local_operators`.
        transported_tangent: Flattened transported local tangents.
        local_velocity: Flattened local spatial velocities.
        transported_tangent_dot_velocity: Flattened tangent derivative actions.
        active_indices: Active-coordinate index of each spatial strain row.
        active_scales: Length-normalization scale of each strain row.
        active_dof_ends: Cumulative active coordinate count after each segment.
        qd: FP64 active velocities with shape ``(batch_size, num_dofs)``.
        inertia_upper_rows: Packed upper-inertia row indices.
        inertia_upper_columns: Packed upper-inertia column indices.
        weighted_masses: Quadrature-weighted spatial mass diagonals.
        gravity_base: Spatial gravity expressed in the base frame.
        block_dim: Cooperative CUDA lanes per environment.
        jacobian_first: First caller-owned ping-pong Jacobian buffer.
        derivative_first: First ping-pong Jacobian derivative buffer.
        velocity_first: First ping-pong spatial-velocity buffer.
        gravity_first: First ping-pong local-gravity buffer.
        jacobian_second: Second ping-pong Jacobian buffer.
        derivative_second: Second ping-pong derivative buffer.
        velocity_second: Second ping-pong spatial-velocity buffer.
        gravity_second: Second ping-pong local-gravity buffer.
        inertia: Preallocated batched inertia output.
        coriolis_qd: Preallocated batched convective-force output.
        gravity_force: Preallocated batched generalized-gravity output.

    Returns:
        None. Workspaces and outputs are updated in place.
    """

    wp.launch_tiled(
        spatial_persistent_chain_kernel,
        dim=qd.shape[0],
        inputs=[
            adjoint_inverse,
            transported_tangent,
            local_velocity,
            transported_tangent_dot_velocity,
            active_indices,
            active_scales,
            active_dof_ends,
            qd,
            inertia_upper_rows,
            inertia_upper_columns,
            weighted_masses,
            gravity_base,
            block_dim,
        ],
        outputs=[
            jacobian_first,
            derivative_first,
            velocity_first,
            gravity_first,
            jacobian_second,
            derivative_second,
            velocity_second,
            gravity_second,
            inertia,
            coriolis_qd,
            gravity_force,
        ],
        block_dim=block_dim,
    )


spatial_persistent_chain = wp.jax_callable(
    launch_spatial_persistent_chain, num_outputs=11
)


__all__ = [
    "launch_spatial_local_operators",
    "launch_spatial_persistent_chain",
    "spatial_local_operators_kernel",
    "spatial_persistent_chain_kernel",
]
