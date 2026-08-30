# ruff: noqa: I001, UP018
"""Runtime-shaped floating spatial PCS dynamics kernels."""

from __future__ import annotations

import warp as wp

from soromox.execution.warp.common.floating_base import (
    _quaternion_rotation_transpose_entry,
    _spatial_root_jacobian_entry,
)
from soromox.execution.warp.common.se3 import Vec6d, _ad_action
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
def spatial_floating_persistent_chain_kernel(
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
    """Traverse one augmented floating spatial PCS chain per block.

    The local constant-strain operators are shared with the fixed kernel. This
    entry kernel specializes only the runtime root state, augmented causal
    widths, and full generalized-force accumulation.

    Args:
        adjoint_inverse: Flattened inverse-adjoint operators.
        transported_tangent: Flattened transported local tangents.
        local_velocity: Flattened internal local spatial velocities.
        transported_tangent_dot_velocity: Flattened derivative actions.
        active_indices: Internal-coordinate index of each strain component.
        active_scales: Scale applied to each active strain component.
        active_dof_ends: Cumulative internal coordinate count by segment.
        base_pose: Runtime scalar-first quaternion poses with shape ``(E, 7)``.
        velocity: Total angular-first generalized velocities.
        inertia_upper_rows: Packed augmented upper-inertia row indices.
        inertia_upper_columns: Packed augmented upper-inertia column indices.
        weighted_masses: Quadrature-weighted diagonal spatial inertias.
        gravity_world: World-frame spatial gravity.
        block_dim: Number of active cooperative lanes.
        jacobian_first: First caller-owned Jacobian workspace.
        derivative_first: First derivative-action workspace.
        velocity_first: First spatial-velocity workspace.
        gravity_first: First local-gravity workspace.
        jacobian_second: Second Jacobian workspace.
        derivative_second: Second derivative-action workspace.
        velocity_second: Second spatial-velocity workspace.
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
    base_dofs = int(6)
    barrier = wp.tile_zeros(shape=(1,), dtype=wp.int32, storage="shared")
    transported_velocity_shared = wp.tile_zeros(
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
        root_velocity = wp.float64(0.0)
        column = int(0)
        while column < base_dofs:
            root_velocity += (
                _spatial_root_jacobian_entry(base_pose, environment, row, column)
                * velocity[environment, column]
            )
            column += 1
        root_gravity = wp.float64(0.0)
        if row >= 3:
            column = int(0)
            while column < 3:
                root_gravity += (
                    _quaternion_rotation_transpose_entry(
                        base_pose, environment, row - 3, column
                    )
                    * gravity_world[column + 3]
                )
                column += 1
        derivative_first[state_base_row + row, 0] = wp.float64(0.0)
        derivative_second[state_base_row + row, 0] = wp.float64(0.0)
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
        derivative_first[state_base_row + row, 0] = derivative
        row += lane_stride
    wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)

    current_is_first = bool(True)
    segment = int(0)
    while segment < num_segments:
        active_end = base_dofs + active_dof_ends[segment]
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

def launch_spatial_floating_persistent_chain(
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
    """Launch one persistent augmented spatial PCS block per environment.

    Args:
        adjoint_inverse: Flattened shared local inverse-adjoint operators.
        transported_tangent: Flattened shared transported tangents.
        local_velocity: Flattened internal local spatial velocities.
        transported_tangent_dot_velocity: Flattened derivative actions.
        active_indices: Internal-coordinate index of each strain row.
        active_scales: Scale applied to each active strain row.
        active_dof_ends: Cumulative internal coordinate count by segment.
        base_pose: Batched runtime scalar-first quaternion poses.
        velocity: Batched total generalized velocities.
        inertia_upper_rows: Packed augmented upper-inertia row indices.
        inertia_upper_columns: Matching upper-inertia column indices.
        weighted_masses: Quadrature-weighted local inertia diagonals.
        gravity_world: World-frame spatial gravity.
        block_dim: Cooperative CUDA lanes per environment.
        jacobian_first: First caller-owned Jacobian workspace.
        derivative_first: First derivative-action workspace.
        velocity_first: First spatial-velocity workspace.
        gravity_first: First local-gravity workspace.
        jacobian_second: Second Jacobian workspace.
        derivative_second: Second derivative-action workspace.
        velocity_second: Second spatial-velocity workspace.
        gravity_second: Second local-gravity workspace.
        inertia: Batched augmented inertia output.
        coriolis_qd: Batched augmented convective-force output.
        gravity_force: Batched augmented gravity-force output.

    Returns:
        None. Workspaces and outputs are updated in place.
    """
    wp.launch_tiled(
        spatial_floating_persistent_chain_kernel,
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

spatial_floating_persistent_chain = wp.jax_callable(
    launch_spatial_floating_persistent_chain, num_outputs=11
)

__all__ = [
    "launch_spatial_floating_persistent_chain",
    "spatial_floating_persistent_chain_kernel",
]
