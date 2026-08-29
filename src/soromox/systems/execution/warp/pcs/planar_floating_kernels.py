# ruff: noqa: I001, UP018
"""Runtime-shaped floating PlanarPCS dynamics kernels."""

from __future__ import annotations

import warp as wp

from soromox.systems.execution.warp.common.floating_base import (
    _planar_rotation_transpose_entry,
)
from soromox.systems.execution.warp.common.storage import (
    _matrix_value,
    _vector_value,
    _write_matrix_value,
    _write_vector_value,
)

wp.set_module_options({"enable_backward": False})

SPATIAL_DIM = 3

@wp.kernel(enable_backward=False)
def planar_floating_persistent_chain_kernel(
    adjoint_inverse: wp.array2d[wp.float64],
    transported_tangent: wp.array2d[wp.float64],
    local_velocity: wp.array2d[wp.float64],
    transported_tangent_dot_velocity: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    active_dof_ends: wp.array[wp.int32],
    base_pose: wp.array2d[wp.float64],
    velocity: wp.array2d[wp.float64],
    inertia_upper_rows: wp.array[wp.int32],
    inertia_upper_columns: wp.array[wp.int32],
    weighted_masses: wp.array2d[wp.float64],
    gravity_world: wp.array[wp.float64],
    block_dim: wp.int32,
    jacobian_first: wp.array2d[wp.float64],
    derivative_first: wp.array2d[wp.float64],
    gravity_first: wp.array2d[wp.float64],
    jacobian_second: wp.array2d[wp.float64],
    derivative_second: wp.array2d[wp.float64],
    gravity_second: wp.array2d[wp.float64],
    inertia: wp.array3d[wp.float64],
    coriolis_qd: wp.array2d[wp.float64],
    gravity_force: wp.array2d[wp.float64],
):
    """Traverse one augmented floating planar PCS chain per block.

    The local constant-strain operators are shared with the fixed kernel. This
    entry kernel specializes the runtime planar root, augmented causal widths,
    and full generalized-force accumulation.

    Args:
        adjoint_inverse: Flattened inverse-adjoint operators.
        transported_tangent: Flattened transported local tangents.
        local_velocity: Flattened internal local velocities.
        transported_tangent_dot_velocity: Flattened derivative actions.
        active_indices: Internal-coordinate index of each strain component.
        active_scales: Scale applied to each active strain component.
        active_dof_ends: Cumulative internal coordinate count by segment.
        base_pose: Runtime planar poses with shape ``(E, 3)``.
        velocity: Total planar generalized velocities.
        inertia_upper_rows: Packed augmented upper-inertia row indices.
        inertia_upper_columns: Packed augmented upper-inertia column indices.
        weighted_masses: Quadrature-weighted diagonal planar inertias.
        gravity_world: World-frame planar gravity.
        block_dim: Number of active cooperative lanes.
        jacobian_first: First caller-owned Jacobian workspace.
        derivative_first: First derivative-action workspace.
        gravity_first: First local-gravity workspace.
        jacobian_second: Second Jacobian workspace.
        derivative_second: Second derivative-action workspace.
        gravity_second: Second local-gravity workspace.
        inertia: Batched augmented inertia output.
        coriolis_qd: Batched augmented convective-force output.
        gravity_force: Batched augmented generalized-gravity output.

    Returns:
        None. Workspaces and outputs are updated in place.
    """
    environment, lane = wp.tid()
    num_dofs = velocity.shape[1]
    num_segments = active_indices.shape[0]
    num_quadrature = weighted_masses.shape[0] // num_segments
    points_per_segment = num_quadrature + 1
    lane_stride = block_dim
    state_base_row = environment * SPATIAL_DIM
    base_dofs = int(3)
    barrier = wp.tile_zeros(shape=(1,), dtype=wp.int32, storage="shared")
    base_velocity_shared = wp.tile_zeros(
        shape=(SPATIAL_DIM,), dtype=wp.float64, storage="shared"
    )
    transported_velocity_shared = wp.tile_zeros(
        shape=(SPATIAL_DIM,), dtype=wp.float64, storage="shared"
    )
    velocity_shared = wp.tile_zeros(
        shape=(SPATIAL_DIM,), dtype=wp.float64, storage="shared"
    )

    theta = base_pose[environment, 0]
    entry = lane
    while entry < SPATIAL_DIM * num_dofs:
        row = entry // num_dofs
        column = entry - row * num_dofs
        value = wp.float64(0.0)
        if row == 0 and column == 0:
            value = wp.float64(1.0)
        elif row >= 1 and column >= 1 and column < base_dofs:
            value = _planar_rotation_transpose_entry(theta, row - 1, column - 1)
        jacobian_first[state_base_row + row, column] = value
        jacobian_second[state_base_row + row, column] = wp.float64(0.0)
        entry += lane_stride
    row = lane
    root_velocity = wp.float64(0.0)
    if row < SPATIAL_DIM:
        column = int(0)
        while column < base_dofs:
            root_jacobian = wp.float64(0.0)
            if row == 0 and column == 0:
                root_jacobian = wp.float64(1.0)
            elif row >= 1 and column >= 1:
                root_jacobian = _planar_rotation_transpose_entry(
                    theta, row - 1, column - 1
                )
            root_velocity += root_jacobian * velocity[environment, column]
            column += 1
        root_gravity = wp.float64(0.0)
        if row >= 1:
            column = int(0)
            while column < 2:
                root_gravity += (
                    _planar_rotation_transpose_entry(theta, row - 1, column)
                    * gravity_world[column + 1]
                )
                column += 1
        derivative_first[state_base_row + row, 0] = wp.float64(0.0)
        derivative_second[state_base_row + row, 0] = wp.float64(0.0)
        gravity_first[state_base_row + row, 0] = root_gravity
        gravity_second[state_base_row + row, 0] = wp.float64(0.0)
    wp.tile_scatter_masked(
        base_velocity_shared, row, root_velocity, row < SPATIAL_DIM
    )
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
        if row == 1:
            derivative = base_velocity_shared[0] * base_velocity_shared[2]
        elif row == 2:
            derivative = -base_velocity_shared[0] * base_velocity_shared[1]
        derivative_first[state_base_row + row, 0] = derivative
        row += lane_stride
    wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)

    current_is_first = bool(True)
    segment = int(0)
    while segment < num_segments:
        active_end = base_dofs + active_dof_ends[segment]
        row = lane
        base_velocity_value = wp.float64(0.0)
        if row < SPATIAL_DIM:
            column = int(0)
            while column < active_end:
                base_velocity_value += (
                    _matrix_value(
                        jacobian_first,
                        jacobian_second,
                        current_is_first,
                        state_base_row,
                        row,
                        column,
                    )
                    * velocity[environment, column]
                )
                column += 1
        wp.tile_scatter_masked(
            base_velocity_shared,
            row,
            base_velocity_value,
            row < SPATIAL_DIM,
        )

        quadrature = int(0)
        while quadrature < num_quadrature:
            destination_is_first = not current_is_first
            operator_item = (
                environment * num_segments + segment
            ) * points_per_segment + quadrature
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
                    if active_indices[segment, local] + base_dofs == output_column:
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
            total_velocity = wp.float64(0.0)
            if row < SPATIAL_DIM:
                derivative_value = wp.float64(0.0)
                gravity_value = wp.float64(0.0)
                k = int(0)
                while k < SPATIAL_DIM:
                    adjoint_value = adjoint_inverse[operator_base_row + row, k]
                    transported_velocity += adjoint_value * base_velocity_shared[k]
                    derivative_value += adjoint_value * _vector_value(
                        derivative_first,
                        derivative_second,
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
                total_velocity = (
                    transported_velocity + local_velocity[operator_base_row + row, 0]
                )
                _write_vector_value(
                    gravity_first,
                    gravity_second,
                    destination_is_first,
                    state_base_row,
                    row,
                    gravity_value,
                )
                derivative_value += transported_tangent_dot_velocity[
                    operator_base_row + row, 0
                ]
                _write_vector_value(
                    derivative_first,
                    derivative_second,
                    destination_is_first,
                    state_base_row,
                    row,
                    derivative_value,
                )
            wp.tile_scatter_masked(
                transported_velocity_shared,
                row,
                transported_velocity,
                row < SPATIAL_DIM,
            )
            wp.tile_scatter_masked(
                velocity_shared,
                row,
                total_velocity,
                row < SPATIAL_DIM,
            )
            wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)

            if lane == 0:
                local_omega = local_velocity[operator_base_row + 0, 0]
                local_x = local_velocity[operator_base_row + 1, 0]
                local_y = local_velocity[operator_base_row + 2, 0]
                source_omega = transported_velocity_shared[0]
                source_x = transported_velocity_shared[1]
                source_y = transported_velocity_shared[2]
                _write_vector_value(
                    derivative_first,
                    derivative_second,
                    destination_is_first,
                    state_base_row,
                    1,
                    _vector_value(
                        derivative_first,
                        derivative_second,
                        destination_is_first,
                        state_base_row,
                        1,
                    )
                    - (local_y * source_omega - local_omega * source_y),
                )
                _write_vector_value(
                    derivative_first,
                    derivative_second,
                    destination_is_first,
                    state_base_row,
                    2,
                    _vector_value(
                        derivative_first,
                        derivative_second,
                        destination_is_first,
                        state_base_row,
                        2,
                    )
                    - (-local_x * source_omega + local_omega * source_x),
                )
            wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)

            pair = lane
            while pair < inertia_upper_rows.shape[0]:
                output_row = inertia_upper_rows[pair]
                output_column = inertia_upper_columns[pair]
                if output_column < active_end:
                    value = wp.float64(0.0)
                    spatial = int(0)
                    mass_item = segment * num_quadrature + quadrature
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
                mass_item = segment * num_quadrature + quadrature
                omega = velocity_shared[0]
                velocity_x = velocity_shared[1]
                velocity_y = velocity_shared[2]
                force_x = weighted_masses[mass_item, 1] * velocity_x
                force_y = weighted_masses[mass_item, 2] * velocity_y
                wrench = wp.vec3d(
                    weighted_masses[mass_item, 0]
                    * _vector_value(
                        derivative_first,
                        derivative_second,
                        destination_is_first,
                        state_base_row,
                        0,
                    )
                    + velocity_y * force_x
                    - velocity_x * force_y,
                    weighted_masses[mass_item, 1]
                    * _vector_value(
                        derivative_first,
                        derivative_second,
                        destination_is_first,
                        state_base_row,
                        1,
                    )
                    - omega * force_y,
                    weighted_masses[mass_item, 2]
                    * _vector_value(
                        derivative_first,
                        derivative_second,
                        destination_is_first,
                        state_base_row,
                        2,
                    )
                    + omega * force_x,
                )
                coriolis_value = wp.float64(0.0)
                gravity_value = wp.float64(0.0)
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
                    gravity_value -= (
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
                gravity_force[environment, column] += gravity_value
                column += lane_stride
            wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
            quadrature += 1

        destination_is_first = not current_is_first
        operator_item = (
            environment * num_segments + segment
        ) * points_per_segment + num_quadrature
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
                if active_indices[segment, local] + base_dofs == output_column:
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
        if row < SPATIAL_DIM:
            derivative_value = wp.float64(0.0)
            gravity_value = wp.float64(0.0)
            k = int(0)
            while k < SPATIAL_DIM:
                adjoint_value = adjoint_inverse[operator_base_row + row, k]
                transported_velocity += adjoint_value * base_velocity_shared[k]
                derivative_value += adjoint_value * _vector_value(
                    derivative_first,
                    derivative_second,
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
                gravity_first,
                gravity_second,
                destination_is_first,
                state_base_row,
                row,
                gravity_value,
            )
            derivative_value += transported_tangent_dot_velocity[
                operator_base_row + row, 0
            ]
            _write_vector_value(
                derivative_first,
                derivative_second,
                destination_is_first,
                state_base_row,
                row,
                derivative_value,
            )
        wp.tile_scatter_masked(
            transported_velocity_shared,
            row,
            transported_velocity,
            row < SPATIAL_DIM,
        )
        wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
        if lane == 0:
            local_omega = local_velocity[operator_base_row + 0, 0]
            local_x = local_velocity[operator_base_row + 1, 0]
            local_y = local_velocity[operator_base_row + 2, 0]
            source_omega = transported_velocity_shared[0]
            source_x = transported_velocity_shared[1]
            source_y = transported_velocity_shared[2]
            _write_vector_value(
                derivative_first,
                derivative_second,
                destination_is_first,
                state_base_row,
                1,
                _vector_value(
                    derivative_first,
                    derivative_second,
                    destination_is_first,
                    state_base_row,
                    1,
                )
                - (local_y * source_omega - local_omega * source_y),
            )
            _write_vector_value(
                derivative_first,
                derivative_second,
                destination_is_first,
                state_base_row,
                2,
                _vector_value(
                    derivative_first,
                    derivative_second,
                    destination_is_first,
                    state_base_row,
                    2,
                )
                - (-local_x * source_omega + local_omega * source_x),
            )
        wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
        current_is_first = destination_is_first
        segment += 1

def launch_planar_floating_persistent_chain(
    adjoint_inverse: wp.array2d[wp.float64],
    transported_tangent: wp.array2d[wp.float64],
    local_velocity: wp.array2d[wp.float64],
    transported_tangent_dot_velocity: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    active_dof_ends: wp.array[wp.int32],
    base_pose: wp.array2d[wp.float64],
    velocity: wp.array2d[wp.float64],
    inertia_upper_rows: wp.array[wp.int32],
    inertia_upper_columns: wp.array[wp.int32],
    weighted_masses: wp.array2d[wp.float64],
    gravity_world: wp.array[wp.float64],
    block_dim: int,
    jacobian_first: wp.array2d[wp.float64],
    derivative_first: wp.array2d[wp.float64],
    gravity_first: wp.array2d[wp.float64],
    jacobian_second: wp.array2d[wp.float64],
    derivative_second: wp.array2d[wp.float64],
    gravity_second: wp.array2d[wp.float64],
    inertia: wp.array3d[wp.float64],
    coriolis_qd: wp.array2d[wp.float64],
    gravity_force: wp.array2d[wp.float64],
):
    """Launch one persistent augmented PlanarPCS block per environment.

    Args:
        adjoint_inverse: Flattened shared local inverse-adjoint operators.
        transported_tangent: Flattened shared transported tangents.
        local_velocity: Flattened internal local velocities.
        transported_tangent_dot_velocity: Flattened derivative actions.
        active_indices: Internal-coordinate index of each strain row.
        active_scales: Scale applied to each active strain row.
        active_dof_ends: Cumulative internal coordinate count by segment.
        base_pose: Batched runtime planar poses.
        velocity: Batched total generalized velocities.
        inertia_upper_rows: Packed augmented upper-inertia row indices.
        inertia_upper_columns: Matching upper-inertia column indices.
        weighted_masses: Quadrature-weighted local inertia diagonals.
        gravity_world: World-frame planar gravity.
        block_dim: Cooperative CUDA lanes per environment.
        jacobian_first: First caller-owned Jacobian workspace.
        derivative_first: First derivative-action workspace.
        gravity_first: First local-gravity workspace.
        jacobian_second: Second Jacobian workspace.
        derivative_second: Second derivative-action workspace.
        gravity_second: Second local-gravity workspace.
        inertia: Batched augmented inertia output.
        coriolis_qd: Batched augmented convective-force output.
        gravity_force: Batched augmented gravity-force output.

    Returns:
        None. Workspaces and outputs are updated in place.
    """
    wp.launch_tiled(
        planar_floating_persistent_chain_kernel,
        dim=velocity.shape[0],
        inputs=[
            adjoint_inverse,
            transported_tangent,
            local_velocity,
            transported_tangent_dot_velocity,
            active_indices,
            active_scales,
            active_dof_ends,
            base_pose,
            velocity,
            inertia_upper_rows,
            inertia_upper_columns,
            weighted_masses,
            gravity_world,
            block_dim,
        ],
        outputs=[
            jacobian_first,
            derivative_first,
            gravity_first,
            jacobian_second,
            derivative_second,
            gravity_second,
            inertia,
            coriolis_qd,
            gravity_force,
        ],
        block_dim=block_dim,
    )

planar_floating_persistent_chain = wp.jax_callable(
    launch_planar_floating_persistent_chain, num_outputs=9
)

__all__ = [
    "launch_planar_floating_persistent_chain",
    "planar_floating_persistent_chain_kernel",
]
