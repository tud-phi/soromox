# ruff: noqa: I001, UP018
"""Allocation-free fused spatial PCS Warp kinematics kernels."""

from __future__ import annotations

import warp as wp

from soromox.execution.warp.common.se3 import (
    Vec6d,
    _adjoint_inverse_action,
    _adjoint_inverse_entry,
    _forward_coefficients,
    _left_action,
    _left_coefficients,
    _rotation_entry,
    _translation,
)

wp.set_module_options({"enable_backward": False})

SPATIAL_DIM = 6
SPATIAL_KINEMATICS_BLOCK_DIM = 192


@wp.func
def spatial_pose_step_entry(
    base: wp.array4d[wp.float64],
    environment: int,
    base_index: int,
    accumulated: Vec6d,
    row: int,
    column: int,
) -> wp.float64:
    """Return one entry of an absolute pose after an SE(3) step.

    Args:
        base: Batched segment-boundary transforms.
        environment: Environment index in ``base``.
        base_index: Segment-boundary index in ``base``.
        accumulated: Integrated spatial strain for the relative step.
        row: Requested homogeneous-transform row.
        column: Requested homogeneous-transform column.

    Returns:
        Selected entry of ``base @ exp(accumulated)``.
    """

    omega = wp.vec3d(accumulated[0], accumulated[1], accumulated[2])
    linear = wp.vec3d(accumulated[3], accumulated[4], accumulated[5])
    angle_sq = wp.dot(omega, omega)
    forward = _forward_coefficients(angle_sq)
    translation = _translation(omega, linear, forward)
    value = wp.float64(0.0)
    if row < 3 and column < 3:
        k = int(0)
        while k < 3:
            value += base[environment, base_index, row, k] * _rotation_entry(
                omega, angle_sq, forward, k, column
            )
            k += 1
    elif row < 3 and column == 3:
        value = base[environment, base_index, row, 3]
        k = int(0)
        while k < 3:
            value += base[environment, base_index, row, k] * translation[k]
            k += 1
    elif row == 3 and column == 3:
        value = wp.float64(1.0)
    return value


@wp.func
def spatial_operator_entry(
    accumulated: Vec6d,
    row: int,
    column: int,
    transported: bool,
) -> wp.float64:
    """Return one inverse-adjoint or transported-tangent entry.

    Args:
        accumulated: Integrated spatial strain.
        row: Operator row.
        column: Operator column.
        transported: Select transported tangent when true and inverse adjoint
            otherwise.

    Returns:
        Requested dimension-specialized operator entry.
    """

    omega = wp.vec3d(accumulated[0], accumulated[1], accumulated[2])
    linear = wp.vec3d(accumulated[3], accumulated[4], accumulated[5])
    angle_sq = wp.dot(omega, omega)
    forward = _forward_coefficients(angle_sq)
    translation = _translation(omega, linear, forward)
    if not transported:
        return _adjoint_inverse_entry(
            omega, translation, angle_sq, forward, row, column
        )
    basis = Vec6d()
    basis[column] = wp.float64(1.0)
    tangent = _left_action(accumulated, basis, _left_coefficients(angle_sq))
    tangent = _adjoint_inverse_action(omega, translation, angle_sq, forward, tangent)
    return tangent[row]


@wp.func
def propagate_spatial_jacobian_column(
    accumulated: Vec6d,
    omega: wp.vec3d,
    translation: wp.vec3d,
    angle_sq: wp.float64,
    forward: wp.vec3d,
    coefficients: wp.vec4d,
    distance: wp.float64,
    environment: int,
    segment: int,
    column: int,
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    segment_jacobian: wp.array4d[wp.float64],
) -> Vec6d:
    """Propagate one body-Jacobian column through a constant-strain step.

    Args:
        accumulated: Integrated spatial strain for the step.
        omega: Angular part of ``accumulated``.
        translation: SE(3) exponential translation for the step.
        angle_sq: Squared norm of ``omega``.
        forward: Stable SE(3) exponential coefficients.
        coefficients: Stable left-tangent coefficients.
        distance: Physical integration distance within the segment.
        environment: Environment index.
        segment: Segment whose base Jacobian is propagated.
        column: Active generalized-coordinate column.
        active_indices: Active coordinate index for each strain row.
        active_scales: Scale applied to each active strain row.
        segment_jacobian: Precomputed segment-base body Jacobians.

    Returns:
        Propagated six-dimensional body-Jacobian column.
    """

    source = Vec6d()
    basis = Vec6d()
    row = int(0)
    while row < 6:
        source[row] = segment_jacobian[environment, segment, row, column]
        if active_indices[segment, row] == column:
            basis[row] = distance * active_scales[segment, row]
        row += 1
    source += _left_action(accumulated, basis, coefficients)
    return _adjoint_inverse_action(
        omega,
        translation,
        angle_sq,
        forward,
        source,
    )


@wp.kernel(enable_backward=False)
def spatial_pose_segment_states_kernel(
    q: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    base_transform: wp.array2d[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array4d[wp.float64],
):
    """Build spatial PCS boundary poses without Jacobian propagation.

    Args:
        q: Batched active coordinates with shape ``(E, D)``.
        active_indices: Active coordinate index for each strain row.
        active_scales: Scale applied to each active strain row.
        reference_strain: Reference strain with shape ``(S, 6)``.
        segment_lengths: Segment lengths with shape ``(S,)``.
        base_transform: Absolute base transform with shape ``(4, 4)``.
        segment_strain: Caller-owned strain workspace ``(E, S, 6)``.
        segment_pose: Caller-owned boundary-pose workspace
            ``(E, S + 1, 4, 4)``.

    Returns:
        None. Pose-only workspaces are written in place.
    """

    environment = wp.tid()
    row = int(0)
    while row < 4:
        column = int(0)
        while column < 4:
            segment_pose[environment, 0, row, column] = base_transform[row, column]
            column += 1
        row += 1
    segment = int(0)
    while segment < segment_lengths.shape[0]:
        xi = Vec6d()
        local = int(0)
        while local < 6:
            xi[local] = reference_strain[segment, local]
            active = active_indices[segment, local]
            if active >= 0:
                xi[local] += active_scales[segment, local] * q[environment, active]
            segment_strain[environment, segment, local] = xi[local]
            local += 1
        accumulated = segment_lengths[segment] * xi
        row = int(0)
        while row < 4:
            column = int(0)
            while column < 4:
                segment_pose[environment, segment + 1, row, column] = (
                    spatial_pose_step_entry(
                        segment_pose, environment, segment, accumulated, row, column
                    )
                )
                column += 1
            row += 1
        segment += 1


@wp.kernel(enable_backward=False)
def spatial_segment_states_kernel(
    q: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    base_transform: wp.array2d[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array4d[wp.float64],
    segment_jacobian: wp.array4d[wp.float64],
):
    """Build reusable spatial PCS segment-boundary states.

    Args:
        q: Batched active coordinates with shape ``(E, D)``.
        active_indices: Active coordinate index for each full strain row.
        active_scales: Scale applied to each active strain row.
        reference_strain: Reference strain with shape ``(S, 6)``.
        segment_lengths: Segment lengths with shape ``(S,)``.
        base_transform: Absolute base transform with shape ``(4, 4)``.
        segment_strain: Caller-owned strain workspace ``(E, S, 6)``.
        segment_pose: Caller-owned boundary-pose workspace
            ``(E, S + 1, 4, 4)``.
        segment_jacobian: Caller-owned body-Jacobian workspace
            ``(E, S + 1, 6, D)``.

    Returns:
        None. Workspaces are written in place.
    """

    environment = wp.tid()
    num_segments = segment_lengths.shape[0]
    num_dofs = q.shape[1]
    row = int(0)
    while row < 4:
        column = int(0)
        while column < 4:
            segment_pose[environment, 0, row, column] = base_transform[row, column]
            column += 1
        row += 1
    row = int(0)
    while row < 6:
        column = int(0)
        while column < num_dofs:
            segment_jacobian[environment, 0, row, column] = wp.float64(0.0)
            column += 1
        row += 1
    segment = int(0)
    while segment < num_segments:
        xi = Vec6d()
        local = int(0)
        while local < 6:
            xi[local] = reference_strain[segment, local]
            active = active_indices[segment, local]
            if active >= 0:
                xi[local] += active_scales[segment, local] * q[environment, active]
            segment_strain[environment, segment, local] = xi[local]
            local += 1
        accumulated = segment_lengths[segment] * xi
        omega = wp.vec3d(accumulated[0], accumulated[1], accumulated[2])
        linear = wp.vec3d(accumulated[3], accumulated[4], accumulated[5])
        angle_sq = wp.dot(omega, omega)
        forward = _forward_coefficients(angle_sq)
        translation = _translation(omega, linear, forward)
        coefficients = _left_coefficients(angle_sq)
        row = int(0)
        while row < 4:
            column = int(0)
            while column < 4:
                segment_pose[environment, segment + 1, row, column] = (
                    spatial_pose_step_entry(
                        segment_pose, environment, segment, accumulated, row, column
                    )
                )
                column += 1
            row += 1
        column = int(0)
        while column < num_dofs:
            body = propagate_spatial_jacobian_column(
                accumulated,
                omega,
                translation,
                angle_sq,
                forward,
                coefficients,
                segment_lengths[segment],
                environment,
                segment,
                column,
                active_indices,
                active_scales,
                segment_jacobian,
            )
            row = int(0)
            while row < 6:
                segment_jacobian[environment, segment + 1, row, column] = body[row]
                row += 1
            column += 1
        segment += 1


@wp.kernel(enable_backward=False)
def spatial_cooperative_segment_states_kernel(
    q: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    base_transform: wp.array2d[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array4d[wp.float64],
    segment_jacobian: wp.array4d[wp.float64],
):
    """Build spatial PCS boundary states cooperatively on CUDA.

    One persistent block owns an environment. Segments remain serial, while
    pose entries and active Jacobian columns are distributed across lanes.

    Args:
        q: Batched active coordinates with shape ``(E, D)``.
        active_indices: Active coordinate index for each strain row.
        active_scales: Scale applied to each active strain row.
        reference_strain: Reference strain with shape ``(S, 6)``.
        segment_lengths: Segment lengths with shape ``(S,)``.
        base_transform: Absolute base transform with shape ``(4, 4)``.
        segment_strain: Caller-owned strain workspace ``(E, S, 6)``.
        segment_pose: Caller-owned boundary-pose workspace
            ``(E, S + 1, 4, 4)``.
        segment_jacobian: Caller-owned body-Jacobian workspace
            ``(E, S + 1, 6, D)``.

    Returns:
        None. Workspaces are written in place.
    """

    environment, lane = wp.tid()
    num_segments = segment_lengths.shape[0]
    num_dofs = q.shape[1]
    barrier = wp.tile_zeros(shape=(1,), dtype=wp.int32, storage="shared")
    xi_shared = wp.tile_zeros(shape=(SPATIAL_DIM,), dtype=wp.float64, storage="shared")
    coefficients_shared = wp.tile_zeros(shape=(4,), dtype=wp.float64, storage="shared")
    forward_shared = wp.tile_zeros(shape=(3,), dtype=wp.float64, storage="shared")
    translation_shared = wp.tile_zeros(shape=(3,), dtype=wp.float64, storage="shared")
    angle_sq_shared = wp.tile_zeros(shape=(1,), dtype=wp.float64, storage="shared")

    entry = lane
    while entry < 16:
        row = entry // 4
        column = entry - row * 4
        segment_pose[environment, 0, row, column] = base_transform[row, column]
        entry += SPATIAL_KINEMATICS_BLOCK_DIM
    entry = lane
    while entry < SPATIAL_DIM * num_dofs:
        row = entry // num_dofs
        column = entry - row * num_dofs
        segment_jacobian[environment, 0, row, column] = wp.float64(0.0)
        entry += SPATIAL_KINEMATICS_BLOCK_DIM
    wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)

    segment = int(0)
    while segment < num_segments:
        component = lane
        xi_value = wp.float64(0.0)
        if component < SPATIAL_DIM:
            xi_value = reference_strain[segment, component]
            active = active_indices[segment, component]
            if active >= 0:
                xi_value += active_scales[segment, component] * q[environment, active]
            segment_strain[environment, segment, component] = xi_value
        wp.tile_scatter_masked(
            xi_shared,
            component,
            xi_value,
            component < SPATIAL_DIM,
        )

        length = segment_lengths[segment]
        accumulated = Vec6d(
            length * xi_shared[0],
            length * xi_shared[1],
            length * xi_shared[2],
            length * xi_shared[3],
            length * xi_shared[4],
            length * xi_shared[5],
        )
        omega = wp.vec3d(accumulated[0], accumulated[1], accumulated[2])
        linear = wp.vec3d(accumulated[3], accumulated[4], accumulated[5])

        angle_sq_value = wp.float64(0.0)
        if lane == 0:
            angle_sq_value = wp.dot(omega, omega)
        wp.tile_scatter_masked(angle_sq_shared, 0, angle_sq_value, lane == 0)

        coefficient_index = lane
        coefficient_value = wp.float64(0.0)
        if coefficient_index < 4:
            coefficient_value = _left_coefficients(angle_sq_shared[0])[
                coefficient_index
            ]
        wp.tile_scatter_masked(
            coefficients_shared,
            coefficient_index,
            coefficient_value,
            coefficient_index < 4,
        )

        forward_index = lane
        forward_value = wp.float64(0.0)
        if forward_index < 3:
            forward_value = _forward_coefficients(angle_sq_shared[0])[forward_index]
        wp.tile_scatter_masked(
            forward_shared,
            forward_index,
            forward_value,
            forward_index < 3,
        )

        translation_index = lane
        translation_value = wp.float64(0.0)
        if translation_index < 3:
            translation_value = _translation(
                omega,
                linear,
                wp.vec3d(forward_shared[0], forward_shared[1], forward_shared[2]),
            )[translation_index]
        wp.tile_scatter_masked(
            translation_shared,
            translation_index,
            translation_value,
            translation_index < 3,
        )

        entry = lane
        while entry < 16:
            row = entry // 4
            column = entry - row * 4
            value = wp.float64(0.0)
            if row < 3 and column < 3:
                k = int(0)
                while k < 3:
                    value += segment_pose[environment, segment, row, k] * (
                        _rotation_entry(
                            omega,
                            angle_sq_shared[0],
                            wp.vec3d(
                                forward_shared[0],
                                forward_shared[1],
                                forward_shared[2],
                            ),
                            k,
                            column,
                        )
                    )
                    k += 1
            elif row < 3 and column == 3:
                value = segment_pose[environment, segment, row, 3]
                k = int(0)
                while k < 3:
                    value += (
                        segment_pose[environment, segment, row, k]
                        * translation_shared[k]
                    )
                    k += 1
            elif row == 3 and column == 3:
                value = wp.float64(1.0)
            segment_pose[environment, segment + 1, row, column] = value
            entry += SPATIAL_KINEMATICS_BLOCK_DIM

        column = lane
        while column < num_dofs:
            body = propagate_spatial_jacobian_column(
                accumulated,
                omega,
                wp.vec3d(
                    translation_shared[0],
                    translation_shared[1],
                    translation_shared[2],
                ),
                angle_sq_shared[0],
                wp.vec3d(
                    forward_shared[0],
                    forward_shared[1],
                    forward_shared[2],
                ),
                wp.vec4d(
                    coefficients_shared[0],
                    coefficients_shared[1],
                    coefficients_shared[2],
                    coefficients_shared[3],
                ),
                length,
                environment,
                segment,
                column,
                active_indices,
                active_scales,
                segment_jacobian,
            )
            row = int(0)
            while row < SPATIAL_DIM:
                segment_jacobian[environment, segment + 1, row, column] = body[row]
                row += 1
            column += SPATIAL_KINEMATICS_BLOCK_DIM
        wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
        segment += 1


@wp.func
def write_spatial_sample_jacobian(
    environment: int,
    sample: int,
    segment: int,
    local_s: wp.float64,
    pose: wp.mat44d,
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_jacobian: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Write one spatial PCS inertial Jacobian from boundary states.

    Args:
        environment: Environment index.
        sample: Backbone-sample index.
        segment: Segment owning the sample.
        local_s: Sample distance from the segment base.
        pose: Absolute SE(3) pose at the sample.
        active_indices: Active coordinate index for each strain row.
        active_scales: Scale applied to each active strain row.
        segment_strain: Precomputed segment strains.
        segment_jacobian: Precomputed boundary body Jacobians.
        jacobians: Caller-owned inertial-Jacobian output.

    Returns:
        None. One output slice is written in place.
    """

    xi = Vec6d()
    k = int(0)
    while k < 6:
        xi[k] = segment_strain[environment, segment, k]
        k += 1
    accumulated = local_s * xi
    omega = wp.vec3d(accumulated[0], accumulated[1], accumulated[2])
    linear = wp.vec3d(accumulated[3], accumulated[4], accumulated[5])
    angle_sq = wp.dot(omega, omega)
    forward = _forward_coefficients(angle_sq)
    translation = _translation(omega, linear, forward)
    coefficients = _left_coefficients(angle_sq)
    column = int(0)
    while column < segment_jacobian.shape[3]:
        body = propagate_spatial_jacobian_column(
            accumulated,
            omega,
            translation,
            angle_sq,
            forward,
            coefficients,
            local_s,
            environment,
            segment,
            column,
            active_indices,
            active_scales,
            segment_jacobian,
        )
        local_row = int(0)
        while local_row < 3:
            angular = wp.float64(0.0)
            linear_output = wp.float64(0.0)
            k = int(0)
            while k < 3:
                rotation = pose[local_row, k]
                angular += rotation * body[k]
                linear_output += rotation * body[3 + k]
                k += 1
            jacobians[environment, sample, local_row, column] = angular
            jacobians[environment, sample, 3 + local_row, column] = linear_output
            local_row += 1
        column += 1


@wp.kernel(enable_backward=False)
def spatial_pose_samples_kernel(
    s: wp.array2d[wp.float64],
    segment_starts: wp.array[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array4d[wp.float64],
    poses: wp.array4d[wp.float64],
):
    """Evaluate spatial PCS poses without Jacobian work.

    Args:
        s: Per-environment abscissae with shape ``(E, N)``.
        segment_starts: Cumulative segment-start coordinates.
        segment_strain: Precomputed segment strains.
        segment_pose: Precomputed boundary poses.
        poses: Caller-owned SE(3) pose output.

    Returns:
        None. Pose outputs are written in place.
    """

    environment, sample = wp.tid()
    abscissa = s[environment, sample]
    num_segments = segment_strain.shape[1]
    segment = int(0)
    i = int(0)
    while i < num_segments + 1:
        if abscissa > segment_starts[i]:
            segment = i
        i += 1
    segment = wp.max(0, wp.min(segment, num_segments - 1))
    xi = Vec6d()
    k = int(0)
    while k < 6:
        xi[k] = segment_strain[environment, segment, k]
        k += 1
    accumulated = (abscissa - segment_starts[segment]) * xi
    row = int(0)
    while row < 4:
        column = int(0)
        while column < 4:
            poses[environment, sample, row, column] = spatial_pose_step_entry(
                segment_pose, environment, segment, accumulated, row, column
            )
            column += 1
        row += 1


@wp.kernel(enable_backward=False)
def spatial_jacobian_samples_kernel(
    s: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    segment_starts: wp.array[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array4d[wp.float64],
    segment_jacobian: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Evaluate spatial PCS inertial Jacobians without pose output writes.

    Args:
        s: Per-environment abscissae with shape ``(E, N)``.
        active_indices: Active coordinate index for each strain row.
        active_scales: Scale applied to each active strain row.
        segment_starts: Cumulative segment-start coordinates.
        segment_strain: Precomputed segment strains.
        segment_pose: Precomputed boundary poses.
        segment_jacobian: Precomputed boundary body Jacobians.
        jacobians: Caller-owned inertial-Jacobian output.

    Returns:
        None. Jacobian outputs are written in place.
    """

    environment, sample = wp.tid()
    abscissa = s[environment, sample]
    num_segments = active_indices.shape[0]
    segment = int(0)
    i = int(0)
    while i < num_segments + 1:
        if abscissa > segment_starts[i]:
            segment = i
        i += 1
    segment = wp.max(0, wp.min(segment, num_segments - 1))
    local_s = abscissa - segment_starts[segment]
    xi = Vec6d()
    k = int(0)
    while k < 6:
        xi[k] = segment_strain[environment, segment, k]
        k += 1
    accumulated = local_s * xi
    pose = wp.mat44d()
    row = int(0)
    while row < 4:
        column = int(0)
        while column < 4:
            pose[row, column] = spatial_pose_step_entry(
                segment_pose, environment, segment, accumulated, row, column
            )
            column += 1
        row += 1
    write_spatial_sample_jacobian(
        environment,
        sample,
        segment,
        local_s,
        pose,
        active_indices,
        active_scales,
        segment_strain,
        segment_jacobian,
        jacobians,
    )


@wp.kernel(enable_backward=False)
def spatial_samples_kernel(
    s: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    segment_starts: wp.array[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array4d[wp.float64],
    segment_jacobian: wp.array4d[wp.float64],
    poses: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Evaluate spatial PCS poses and inertial Jacobians at runtime samples.

    Args:
        s: Per-environment abscissae with shape ``(E, N)``.
        active_indices: Active coordinate index for each full strain row.
        active_scales: Scale applied to each active strain row.
        segment_starts: Cumulative segment-start coordinates.
        segment_strain: Precomputed segment-strain workspace.
        segment_pose: Precomputed segment-boundary poses.
        segment_jacobian: Precomputed segment-boundary body Jacobians.
        poses: Caller-owned SE(3) pose output ``(E, N, 4, 4)``.
        jacobians: Caller-owned inertial Jacobian output ``(E, N, 6, D)``.

    Returns:
        None. Outputs are written in place.
    """

    environment, sample = wp.tid()
    abscissa = s[environment, sample]
    num_segments = active_indices.shape[0]
    segment = int(0)
    i = int(0)
    while i < num_segments + 1:
        if abscissa > segment_starts[i]:
            segment = i
        i += 1
    segment = wp.max(0, wp.min(segment, num_segments - 1))
    local_s = abscissa - segment_starts[segment]
    xi = Vec6d()
    k = int(0)
    while k < 6:
        xi[k] = segment_strain[environment, segment, k]
        k += 1
    accumulated = local_s * xi
    pose = wp.mat44d()
    row = int(0)
    while row < 4:
        column = int(0)
        while column < 4:
            value = spatial_pose_step_entry(
                segment_pose, environment, segment, accumulated, row, column
            )
            pose[row, column] = value
            poses[environment, sample, row, column] = value
            column += 1
        row += 1
    write_spatial_sample_jacobian(
        environment,
        sample,
        segment,
        local_s,
        pose,
        active_indices,
        active_scales,
        segment_strain,
        segment_jacobian,
        jacobians,
    )


def launch_spatial_forward_kinematics(
    q: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    base_transform: wp.array2d[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array4d[wp.float64],
    poses: wp.array4d[wp.float64],
):
    """Launch the reduced-work spatial PCS pose pipeline.

    Args:
        q: Batched active coordinates ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        active_indices: Active coordinate indices by segment and strain row.
        active_scales: Active strain scales matching ``active_indices``.
        reference_strain: Reference strain ``(S, 6)``.
        segment_lengths: Segment lengths ``(S,)``.
        segment_starts: Cumulative segment starts ``(S + 1,)``.
        base_transform: Absolute base transform ``(4, 4)``.
        segment_strain: Caller-owned strain workspace.
        segment_pose: Caller-owned boundary-pose workspace.
        poses: Caller-owned pose output.

    Returns:
        None. Pose kernels are enqueued on the current Warp stream.
    """

    wp.launch(
        spatial_pose_segment_states_kernel,
        dim=q.shape[0],
        inputs=[
            q,
            active_indices,
            active_scales,
            reference_strain,
            segment_lengths,
            base_transform,
        ],
        outputs=[segment_strain, segment_pose],
    )
    wp.launch(
        spatial_pose_samples_kernel,
        dim=(q.shape[0], s.shape[1]),
        inputs=[s, segment_starts, segment_strain, segment_pose],
        outputs=[poses],
        block_dim=128,
    )


def launch_spatial_inertial_jacobians(
    q: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    base_transform: wp.array2d[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array4d[wp.float64],
    segment_jacobian: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Launch spatial PCS inertial Jacobians without pose output writes.

    Args:
        q: Batched active coordinates ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        active_indices: Active coordinate indices by segment and strain row.
        active_scales: Active strain scales matching ``active_indices``.
        reference_strain: Reference strain ``(S, 6)``.
        segment_lengths: Segment lengths ``(S,)``.
        segment_starts: Cumulative segment starts ``(S + 1,)``.
        base_transform: Absolute base transform ``(4, 4)``.
        segment_strain: Caller-owned strain workspace.
        segment_pose: Caller-owned boundary-pose workspace.
        segment_jacobian: Caller-owned boundary-Jacobian workspace.
        jacobians: Caller-owned inertial-Jacobian output.

    Returns:
        None. Jacobian kernels are enqueued on the current Warp stream.
    """

    wp.launch(
        spatial_segment_states_kernel,
        dim=q.shape[0],
        inputs=[
            q,
            active_indices,
            active_scales,
            reference_strain,
            segment_lengths,
            base_transform,
        ],
        outputs=[segment_strain, segment_pose, segment_jacobian],
    )
    wp.launch(
        spatial_jacobian_samples_kernel,
        dim=(q.shape[0], s.shape[1]),
        inputs=[
            s,
            active_indices,
            active_scales,
            segment_starts,
            segment_strain,
            segment_pose,
            segment_jacobian,
        ],
        outputs=[jacobians],
        block_dim=128,
    )


def launch_spatial_kinematics(
    q: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    base_transform: wp.array2d[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array4d[wp.float64],
    segment_jacobian: wp.array4d[wp.float64],
    poses: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Launch the allocation-free fused spatial PCS pipeline.

    Args:
        q: Batched active coordinates ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        active_indices: Active coordinate indices by segment and strain row.
        active_scales: Active strain scales matching ``active_indices``.
        reference_strain: Reference strain ``(S, 6)``.
        segment_lengths: Segment lengths ``(S,)``.
        segment_starts: Cumulative segment starts ``(S + 1,)``.
        base_transform: Absolute base transform ``(4, 4)``.
        segment_strain: Caller-owned strain workspace.
        segment_pose: Caller-owned boundary-pose workspace.
        segment_jacobian: Caller-owned boundary-Jacobian workspace.
        poses: Caller-owned pose output.
        jacobians: Caller-owned inertial-Jacobian output.

    Returns:
        None. The two kernels are enqueued on the current Warp stream.
    """

    wp.launch(
        spatial_segment_states_kernel,
        dim=q.shape[0],
        inputs=[
            q,
            active_indices,
            active_scales,
            reference_strain,
            segment_lengths,
            base_transform,
        ],
        outputs=[segment_strain, segment_pose, segment_jacobian],
    )
    wp.launch(
        spatial_samples_kernel,
        dim=(q.shape[0], s.shape[1]),
        inputs=[
            s,
            active_indices,
            active_scales,
            segment_starts,
            segment_strain,
            segment_pose,
            segment_jacobian,
        ],
        outputs=[poses, jacobians],
        block_dim=128,
    )


def launch_spatial_cooperative_inertial_jacobians(
    q: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    base_transform: wp.array2d[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array4d[wp.float64],
    segment_jacobian: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Launch cooperative spatial PCS inertial Jacobians on CUDA.

    Args:
        q: Batched active coordinates ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        active_indices: Active coordinate indices by segment and strain row.
        active_scales: Active strain scales matching ``active_indices``.
        reference_strain: Reference strain ``(S, 6)``.
        segment_lengths: Segment lengths ``(S,)``.
        segment_starts: Cumulative segment starts ``(S + 1,)``.
        base_transform: Absolute base transform ``(4, 4)``.
        segment_strain: Caller-owned strain workspace.
        segment_pose: Caller-owned boundary-pose workspace.
        segment_jacobian: Caller-owned boundary-Jacobian workspace.
        jacobians: Caller-owned inertial-Jacobian output.

    Returns:
        None. Kernels are enqueued on the current Warp stream.
    """

    wp.launch_tiled(
        spatial_cooperative_segment_states_kernel,
        dim=q.shape[0],
        inputs=[
            q,
            active_indices,
            active_scales,
            reference_strain,
            segment_lengths,
            base_transform,
        ],
        outputs=[segment_strain, segment_pose, segment_jacobian],
        block_dim=SPATIAL_KINEMATICS_BLOCK_DIM,
    )
    wp.launch(
        spatial_jacobian_samples_kernel,
        dim=(q.shape[0], s.shape[1]),
        inputs=[
            s,
            active_indices,
            active_scales,
            segment_starts,
            segment_strain,
            segment_pose,
            segment_jacobian,
        ],
        outputs=[jacobians],
        block_dim=128,
    )


def launch_spatial_cooperative_kinematics(
    q: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    base_transform: wp.array2d[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array4d[wp.float64],
    segment_jacobian: wp.array4d[wp.float64],
    poses: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Launch cooperative fused spatial PCS poses and Jacobians on CUDA.

    Args:
        q: Batched active coordinates ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        active_indices: Active coordinate indices by segment and strain row.
        active_scales: Active strain scales matching ``active_indices``.
        reference_strain: Reference strain ``(S, 6)``.
        segment_lengths: Segment lengths ``(S,)``.
        segment_starts: Cumulative segment starts ``(S + 1,)``.
        base_transform: Absolute base transform ``(4, 4)``.
        segment_strain: Caller-owned strain workspace.
        segment_pose: Caller-owned boundary-pose workspace.
        segment_jacobian: Caller-owned boundary-Jacobian workspace.
        poses: Caller-owned pose output.
        jacobians: Caller-owned inertial-Jacobian output.

    Returns:
        None. Kernels are enqueued on the current Warp stream.
    """

    wp.launch_tiled(
        spatial_cooperative_segment_states_kernel,
        dim=q.shape[0],
        inputs=[
            q,
            active_indices,
            active_scales,
            reference_strain,
            segment_lengths,
            base_transform,
        ],
        outputs=[segment_strain, segment_pose, segment_jacobian],
        block_dim=SPATIAL_KINEMATICS_BLOCK_DIM,
    )
    wp.launch(
        spatial_samples_kernel,
        dim=(q.shape[0], s.shape[1]),
        inputs=[
            s,
            active_indices,
            active_scales,
            segment_starts,
            segment_strain,
            segment_pose,
            segment_jacobian,
        ],
        outputs=[poses, jacobians],
        block_dim=128,
    )


spatial_forward_kinematics = wp.jax_callable(
    launch_spatial_forward_kinematics, num_outputs=3
)
spatial_inertial_jacobians = wp.jax_callable(
    launch_spatial_inertial_jacobians, num_outputs=4
)
spatial_kinematics = wp.jax_callable(launch_spatial_kinematics, num_outputs=5)
spatial_cooperative_inertial_jacobians = wp.jax_callable(
    launch_spatial_cooperative_inertial_jacobians, num_outputs=4
)
spatial_cooperative_kinematics = wp.jax_callable(
    launch_spatial_cooperative_kinematics, num_outputs=5
)


__all__ = [
    "launch_spatial_forward_kinematics",
    "launch_spatial_cooperative_inertial_jacobians",
    "launch_spatial_cooperative_kinematics",
    "launch_spatial_inertial_jacobians",
    "launch_spatial_kinematics",
    "spatial_jacobian_samples_kernel",
    "spatial_cooperative_segment_states_kernel",
    "spatial_operator_entry",
    "spatial_pose_samples_kernel",
    "spatial_pose_segment_states_kernel",
    "spatial_pose_step_entry",
    "spatial_samples_kernel",
    "spatial_segment_states_kernel",
    "write_spatial_sample_jacobian",
]
