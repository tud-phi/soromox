# ruff: noqa: I001, UP018
"""Runtime-shaped PlanarPCS dynamics kernels."""

from __future__ import annotations

import warp as wp

from soromox.execution.warp.common.storage import (
    _matrix_value,
    _vector_value,
    _write_matrix_value,
    _write_vector_value,
)

wp.set_module_options({"enable_backward": False})

SPATIAL_DIM = 3


@wp.func
def _forward_coefficients(z: wp.float64, cutoff: wp.float64) -> wp.vec3d:
    """Evaluate stable scalar coefficients for planar exponential operators.

    Args:
        z: Integrated planar rotation.
        cutoff: Magnitude below which a polynomial expansion is used.

    Returns:
        Stable ``(sinc, cosc, tanc)`` coefficients.
    """
    x = z * z
    if wp.abs(z) <= cutoff:
        return wp.vec3d(
            wp.float64(1.0)
            + x
            * (
                -wp.float64(1.0 / 6.0)
                + x
                * (
                    wp.float64(1.0 / 120.0)
                    + x * (-wp.float64(1.0 / 5040.0) + x * wp.float64(1.0 / 362880.0))
                )
            ),
            wp.float64(0.5)
            + x
            * (
                -wp.float64(1.0 / 24.0)
                + x
                * (
                    wp.float64(1.0 / 720.0)
                    + x * (-wp.float64(1.0 / 40320.0) + x * wp.float64(1.0 / 3628800.0))
                )
            ),
            wp.float64(1.0 / 6.0)
            + x
            * (
                -wp.float64(1.0 / 120.0)
                + x
                * (
                    wp.float64(1.0 / 5040.0)
                    + x
                    * (-wp.float64(1.0 / 362880.0) + x * wp.float64(1.0 / 39916800.0))
                )
            ),
        )
    sine = wp.sin(z)
    cosine = wp.cos(z)
    return wp.vec3d(
        sine / z,
        (wp.float64(1.0) - cosine) / x,
        (z - sine) / (x * z),
    )


@wp.func
def _forward_derivatives(z: wp.float64, cutoff: wp.float64) -> wp.vec3d:
    """Evaluate derivatives of planar exponential coefficients.

    Args:
        z: Integrated planar rotation.
        cutoff: Magnitude below which a polynomial expansion is used.

    Returns:
        Derivatives of ``(sinc, cosc, tanc)`` with respect to ``z**2``.
    """
    x = z * z
    if wp.abs(z) <= cutoff:
        return wp.vec3d(
            -wp.float64(1.0 / 6.0)
            + x
            * (
                wp.float64(1.0 / 60.0)
                + x * (-wp.float64(1.0 / 1680.0) + x * wp.float64(1.0 / 90720.0))
            ),
            -wp.float64(1.0 / 24.0)
            + x
            * (
                wp.float64(1.0 / 360.0)
                + x * (-wp.float64(1.0 / 13440.0) + x * wp.float64(1.0 / 907200.0))
            ),
            -wp.float64(1.0 / 120.0)
            + x
            * (
                wp.float64(1.0 / 2520.0)
                + x * (-wp.float64(1.0 / 120960.0) + x * wp.float64(1.0 / 9979200.0))
            ),
        )
    sine = wp.sin(z)
    cosine = wp.cos(z)
    return wp.vec3d(
        (z * cosine - sine) / (wp.float64(2.0) * x * z),
        (z * sine + wp.float64(2.0) * cosine - wp.float64(2.0))
        / (wp.float64(2.0) * x * x),
        (wp.float64(3.0) * sine - wp.float64(2.0) * z - z * cosine)
        / (wp.float64(2.0) * x * x * z),
    )


@wp.func
def _planar_operators(
    xi: wp.vec3d,
    xid: wp.vec3d,
    s: wp.float64,
    global_eps: wp.float64,
    tangent_eps: wp.float64,
) -> tuple[wp.mat33d, wp.mat33d, wp.vec3d, wp.vec3d]:
    """Evaluate planar adjoint, tangent, and velocity recurrence operators.

    Args:
        xi: Constant planar strain of the current segment.
        xid: Constant planar strain rate of the current segment.
        s: Segment-local integration coordinate.
        global_eps: Small-angle tolerance for the exponential adjoint.
        tangent_eps: Small-angle tolerance for tangent derivatives.

    Returns:
        Inverse adjoint, left tangent, local velocity, and the tangent time-
        derivative action on strain rate.
    """
    z = s * xi[0]
    adjoint_cutoff = wp.max(wp.abs(s * global_eps), wp.float64(0.04964607461902946))
    adjoint_coefficients = _forward_coefficients(z, adjoint_cutoff)
    sinc_a = adjoint_coefficients[0]
    cosc_a = adjoint_coefficients[1]
    sine = wp.sin(z)
    cosine = wp.cos(z)
    q0 = s * (sinc_a * xi[2] + z * cosc_a * xi[1])
    q1 = s * (-sinc_a * xi[1] + z * cosc_a * xi[2])

    adjoint_inverse = wp.mat33d()
    adjoint_inverse[0, 0] = wp.float64(1.0)
    adjoint_inverse[1, 0] = -(cosine * q0 + sine * q1)
    adjoint_inverse[1, 1] = cosine
    adjoint_inverse[1, 2] = sine
    adjoint_inverse[2, 0] = sine * q0 - cosine * q1
    adjoint_inverse[2, 1] = -sine
    adjoint_inverse[2, 2] = cosine

    tangent_cutoff = wp.max(wp.abs(s * tangent_eps), wp.float64(0.07618835359095202))
    coefficients = _forward_coefficients(z, tangent_cutoff)
    derivatives = _forward_derivatives(z, tangent_cutoff)
    sinc = coefficients[0]
    cosc = coefficients[1]
    tanc = coefficients[2]
    accumulated_x = s * xi[1]
    accumulated_y = s * xi[2]
    lower0 = cosc * accumulated_y + z * tanc * accumulated_x
    lower1 = -cosc * accumulated_x + z * tanc * accumulated_y
    tangent = wp.mat33d()
    tangent[0, 0] = s
    tangent[1, 0] = s * lower0
    tangent[1, 1] = s * sinc
    tangent[1, 2] = -s * z * cosc
    tangent[2, 0] = s * lower1
    tangent[2, 1] = s * z * cosc
    tangent[2, 2] = s * sinc

    accumulated_dot_x = s * xid[1]
    accumulated_dot_y = s * xid[2]
    z_dot = s * xid[0]
    x = z * z
    sinc_dot = wp.float64(2.0) * z * derivatives[0] * z_dot
    cosc_dot = wp.float64(2.0) * z * derivatives[1] * z_dot
    z_cosc_dot = z_dot * (cosc + wp.float64(2.0) * x * derivatives[1])
    z_tanc = z * tanc
    z_tanc_dot = z_dot * (tanc + wp.float64(2.0) * x * derivatives[2])
    lower_dot0 = (
        cosc_dot * accumulated_y
        + cosc * accumulated_dot_y
        + z_tanc_dot * accumulated_x
        + z_tanc * accumulated_dot_x
    )
    lower_dot1 = (
        -cosc_dot * accumulated_x
        - cosc * accumulated_dot_x
        + z_tanc_dot * accumulated_y
        + z_tanc * accumulated_dot_y
    )
    tangent_dot = wp.mat33d()
    tangent_dot[1, 0] = s * lower_dot0
    tangent_dot[1, 1] = s * sinc_dot
    tangent_dot[1, 2] = -s * z_cosc_dot
    tangent_dot[2, 0] = s * lower_dot1
    tangent_dot[2, 1] = s * z_cosc_dot
    tangent_dot[2, 2] = s * sinc_dot

    transported_tangent = adjoint_inverse * tangent
    local_velocity = transported_tangent * xid
    transported_tangent_dot_velocity = adjoint_inverse * (tangent_dot * xid)
    return (
        adjoint_inverse,
        transported_tangent,
        local_velocity,
        transported_tangent_dot_velocity,
    )


@wp.kernel(enable_backward=False)
def planar_local_operators_kernel(
    q: wp.array2d[wp.float64],
    qd: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    operator_points: wp.array2d[wp.float64],
    epsilons: wp.array[wp.float64],
    adjoint_inverse: wp.array2d[wp.float64],
    transported_tangent: wp.array2d[wp.float64],
    local_velocity: wp.array2d[wp.float64],
    transported_tangent_dot_velocity: wp.array2d[wp.float64],
):
    """Evaluate planar local operators for one environment-segment point.

    Args:
        q: Batched active generalized coordinates.
        qd: Batched active generalized velocities.
        active_indices: Active-coordinate index of each strain component.
        active_scales: Scale applied to each active strain component.
        reference_strain: Segment reference strains.
        operator_points: Segment-local evaluation coordinates.
        epsilons: Exponential and tangent small-angle tolerances.
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
    xi = wp.vec3d(
        reference_strain[segment, 0],
        reference_strain[segment, 1],
        reference_strain[segment, 2],
    )
    xid = wp.vec3d()
    local = int(0)
    while local < SPATIAL_DIM:
        global_index = active_indices[segment, local]
        if global_index >= 0:
            scale = active_scales[segment, local]
            xi[local] += scale * q[environment, global_index]
            xid[local] = scale * qd[environment, global_index]
        local += 1
    (
        adjoint_inverse_value,
        transported_tangent_value,
        local_velocity_value,
        transported_tangent_dot_velocity_value,
    ) = _planar_operators(
        xi,
        xid,
        operator_points[segment, point],
        epsilons[0],
        epsilons[1],
    )
    output_base_row = item * SPATIAL_DIM
    row = int(0)
    while row < SPATIAL_DIM:
        column = int(0)
        while column < SPATIAL_DIM:
            adjoint_inverse[output_base_row + row, column] = adjoint_inverse_value[
                row, column
            ]
            transported_tangent[output_base_row + row, column] = (
                transported_tangent_value[row, column]
            )
            column += 1
        local_velocity[output_base_row + row, 0] = local_velocity_value[row]
        transported_tangent_dot_velocity[output_base_row + row, 0] = (
            transported_tangent_dot_velocity_value[row]
        )
        row += 1


def launch_planar_local_operators(
    q: wp.array2d[wp.float64],
    qd: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    operator_points: wp.array2d[wp.float64],
    epsilons: wp.array[wp.float64],
    adjoint_inverse: wp.array2d[wp.float64],
    transported_tangent: wp.array2d[wp.float64],
    local_velocity: wp.array2d[wp.float64],
    transported_tangent_dot_velocity: wp.array2d[wp.float64],
):
    """Launch PlanarPCS local SE(2) operator evaluation.

    One work item evaluates one operator point for one segment and environment.
    The caller owns every output buffer, allowing this launch to participate in
    a larger allocation-free CUDA graph.

    Args:
        q: FP64 active coordinates with shape ``(batch_size, num_dofs)``.
        qd: FP64 active velocities with the same shape as ``q``.
        active_indices: Active-coordinate index of each planar strain row.
        active_scales: Length-normalization scale of each strain row.
        reference_strain: Reference strain with shape ``(num_segments, 3)``.
        operator_points: Local evaluation coordinates for each segment.
        epsilons: Two-entry FP64 array containing exponential and tangent
            small-angle thresholds.
        adjoint_inverse: Preallocated flattened inverse-adjoint output.
        transported_tangent: Preallocated flattened tangent output.
        local_velocity: Preallocated flattened local-velocity output.
        transported_tangent_dot_velocity: Preallocated flattened tangent
            derivative action.

    Returns:
        None. Outputs are written in place.
    """

    wp.launch(
        planar_local_operators_kernel,
        dim=q.shape[0] * active_indices.shape[0] * operator_points.shape[1],
        inputs=[
            q,
            qd,
            active_indices,
            active_scales,
            reference_strain,
            operator_points,
            epsilons,
        ],
        outputs=[
            adjoint_inverse,
            transported_tangent,
            local_velocity,
            transported_tangent_dot_velocity,
        ],
        block_dim=128,
    )


planar_local_operators = wp.jax_callable(launch_planar_local_operators, num_outputs=4)


@wp.kernel(enable_backward=False)
def planar_persistent_chain_kernel(
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
    gravity_first: wp.array2d[wp.float64],
    jacobian_second: wp.array2d[wp.float64],
    derivative_second: wp.array2d[wp.float64],
    gravity_second: wp.array2d[wp.float64],
    inertia: wp.array3d[wp.float64],
    coriolis_qd: wp.array2d[wp.float64],
    gravity_force: wp.array2d[wp.float64],
):
    """Traverse one planar PCS chain per persistent cooperative block.

    Args:
        adjoint_inverse: Flattened inverse-adjoint operators.
        transported_tangent: Flattened transported local tangents.
        local_velocity: Flattened local velocities.
        transported_tangent_dot_velocity: Flattened derivative actions.
        active_indices: Active-coordinate index of each strain component.
        active_scales: Scale applied to each active strain component.
        active_dof_ends: Cumulative active coordinate count by segment.
        qd: Batched generalized velocities.
        inertia_upper_rows: Packed upper-inertia row indices.
        inertia_upper_columns: Packed upper-inertia column indices.
        weighted_masses: Quadrature-weighted diagonal planar inertias.
        gravity_base: Base-frame planar gravity.
        block_dim: Number of active cooperative lanes.
        jacobian_first: First caller-owned Jacobian workspace.
        derivative_first: First derivative-action workspace.
        gravity_first: First local-gravity workspace.
        jacobian_second: Second Jacobian workspace.
        derivative_second: Second derivative-action workspace.
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
    base_velocity_shared = wp.tile_zeros(
        shape=(SPATIAL_DIM,), dtype=wp.float64, storage="shared"
    )
    transported_velocity_shared = wp.tile_zeros(
        shape=(SPATIAL_DIM,), dtype=wp.float64, storage="shared"
    )
    velocity_shared = wp.tile_zeros(
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
                    * qd[environment, column]
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


def launch_planar_persistent_chain(
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
    gravity_first: wp.array2d[wp.float64],
    jacobian_second: wp.array2d[wp.float64],
    derivative_second: wp.array2d[wp.float64],
    gravity_second: wp.array2d[wp.float64],
    inertia: wp.array3d[wp.float64],
    coriolis_qd: wp.array2d[wp.float64],
    gravity_force: wp.array2d[wp.float64],
):
    """Launch one persistent PlanarPCS chain block per environment.

    Args:
        adjoint_inverse: Flattened outputs from
            :func:`launch_planar_local_operators`.
        transported_tangent: Flattened transported local tangents.
        local_velocity: Flattened local velocities.
        transported_tangent_dot_velocity: Flattened tangent derivative actions.
        active_indices: Active-coordinate index of each planar strain row.
        active_scales: Length-normalization scale of each strain row.
        active_dof_ends: Cumulative active coordinate count after each segment.
        qd: FP64 active velocities with shape ``(batch_size, num_dofs)``.
        inertia_upper_rows: Packed upper-inertia row indices.
        inertia_upper_columns: Packed upper-inertia column indices.
        weighted_masses: Quadrature-weighted planar mass diagonals.
        gravity_base: Planar gravity expressed in the base frame.
        block_dim: Cooperative CUDA lanes per environment.
        jacobian_first: First caller-owned ping-pong Jacobian buffer.
        derivative_first: First ping-pong Jacobian derivative buffer.
        gravity_first: First ping-pong local-gravity buffer.
        jacobian_second: Second ping-pong Jacobian buffer.
        derivative_second: Second ping-pong derivative buffer.
        gravity_second: Second ping-pong local-gravity buffer.
        inertia: Preallocated batched inertia output.
        coriolis_qd: Preallocated batched convective-force output.
        gravity_force: Preallocated batched generalized-gravity output.

    Returns:
        None. Workspaces and outputs are updated in place.
    """

    wp.launch_tiled(
        planar_persistent_chain_kernel,
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


planar_persistent_chain = wp.jax_callable(launch_planar_persistent_chain, num_outputs=9)


__all__ = [
    "launch_planar_local_operators",
    "launch_planar_persistent_chain",
    "planar_local_operators_kernel",
    "planar_persistent_chain_kernel",
]
