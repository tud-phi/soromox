# ruff: noqa: I001, UP018
"""Matrix-free spatial PCS inertial-Jacobian products for Warp integrations."""

from __future__ import annotations

import warp as wp

from soromox.execution.warp.common.se3 import (
    Vec6d,
    _adjoint_inverse_action,
    _adjoint_inverse_entry,
    _adjoint_inverse_transpose_action,
    _forward_coefficients,
    _left_action,
    _left_coefficients,
    _translation,
)
from soromox.execution.warp.common.spatial import (
    _load_spatial_vector3d,
    _rotate_spatial_from_inertial,
    _rotate_spatial_to_inertial,
)
from soromox.execution.warp.pcs.spatial_kinematics import (
    spatial_pose_segment_states_kernel,
    spatial_pose_step_entry,
)

wp.set_module_options({"enable_backward": False})

SPATIAL_DIM = 6


@wp.kernel(enable_backward=False)
def spatial_segment_pose_twist_kernel(
    q: wp.array2d[wp.float64],
    qd: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    base_transform: wp.array2d[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array4d[wp.float64],
    segment_twist: wp.array3d[wp.float64],
):
    """Propagate spatial PCS boundary poses and one Jacobian direction.

    Args:
        q: Batched active coordinates ``(E, D)``.
        qd: Batched generalized-velocity directions ``(E, D)``.
        active_indices: Active coordinate index for each strain row.
        active_scales: Scale applied to each active strain row.
        reference_strain: Reference strain ``(S, 6)``.
        segment_lengths: Physical segment lengths ``(S,)``.
        base_transform: Absolute fixed-base transform ``(4, 4)``.
        segment_strain: Segment-strain output ``(E, S, 6)``.
        segment_pose: Boundary-pose output ``(E, S + 1, 4, 4)``.
        segment_twist: Boundary body-JVP output ``(E, S + 1, 6)``.

    Returns:
        None. Boundary states are written in place.
    """

    environment = wp.tid()
    row = int(0)
    while row < 4:
        column = int(0)
        while column < 4:
            segment_pose[environment, 0, row, column] = base_transform[row, column]
            column += 1
        row += 1
    row = int(0)
    while row < SPATIAL_DIM:
        segment_twist[environment, 0, row] = wp.float64(0.0)
        row += 1

    segment = int(0)
    while segment < segment_lengths.shape[0]:
        xi = Vec6d()
        xid = Vec6d()
        local = int(0)
        while local < SPATIAL_DIM:
            xi[local] = reference_strain[segment, local]
            active = active_indices[segment, local]
            if active >= 0:
                scale = active_scales[segment, local]
                xi[local] += scale * q[environment, active]
                xid[local] = scale * qd[environment, active]
            segment_strain[environment, segment, local] = xi[local]
            local += 1
        length = segment_lengths[segment]
        accumulated = length * xi
        accumulated_dot = length * xid
        omega = wp.vec3d(accumulated[0], accumulated[1], accumulated[2])
        linear = wp.vec3d(accumulated[3], accumulated[4], accumulated[5])
        angle_sq = wp.dot(omega, omega)
        forward = _forward_coefficients(angle_sq)
        translation = _translation(omega, linear, forward)
        partial_twist = _left_action(
            accumulated,
            accumulated_dot,
            _left_coefficients(angle_sq),
        )
        boundary_twist = _adjoint_inverse_action(
            omega,
            translation,
            angle_sq,
            forward,
            _load_spatial_vector3d(segment_twist, environment, segment) + partial_twist,
        )
        row = int(0)
        while row < 4:
            column = int(0)
            while column < 4:
                segment_pose[environment, segment + 1, row, column] = (
                    spatial_pose_step_entry(
                        segment_pose,
                        environment,
                        segment,
                        accumulated,
                        row,
                        column,
                    )
                )
                column += 1
            row += 1
        row = int(0)
        while row < SPATIAL_DIM:
            segment_twist[environment, segment + 1, row] = boundary_twist[row]
            row += 1
        segment += 1


@wp.kernel(enable_backward=False)
def spatial_pose_twist_samples_kernel(
    qd: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    segment_starts: wp.array[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array4d[wp.float64],
    segment_twist: wp.array3d[wp.float64],
    poses: wp.array4d[wp.float64],
    twists: wp.array3d[wp.float64],
):
    """Evaluate spatial PCS poses and inertial JVPs at runtime samples.

    Args:
        qd: Batched generalized-velocity directions ``(E, D)``.
        s: Per-environment sample abscissae ``(E, N)``.
        active_indices: Active coordinate index for each strain row.
        active_scales: Scale applied to each active strain row.
        segment_starts: Cumulative segment-start coordinates ``(S + 1,)``.
        segment_strain: Precomputed segment strains ``(E, S, 6)``.
        segment_pose: Precomputed boundary poses ``(E, S + 1, 4, 4)``.
        segment_twist: Precomputed boundary body JVPs ``(E, S + 1, 6)``.
        poses: Sample-pose output ``(E, N, 4, 4)``.
        twists: Inertial sample-JVP output ``(E, N, 6)``.

    Returns:
        None. ``poses`` and ``twists`` are written in place.
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
    xid = Vec6d()
    row = int(0)
    while row < SPATIAL_DIM:
        xi[row] = segment_strain[environment, segment, row]
        active = active_indices[segment, row]
        if active >= 0:
            xid[row] = active_scales[segment, row] * qd[environment, active]
        row += 1
    accumulated = local_s * xi
    accumulated_dot = local_s * xid
    omega = wp.vec3d(accumulated[0], accumulated[1], accumulated[2])
    linear = wp.vec3d(accumulated[3], accumulated[4], accumulated[5])
    angle_sq = wp.dot(omega, omega)
    forward = _forward_coefficients(angle_sq)
    translation = _translation(omega, linear, forward)
    partial_twist = _left_action(
        accumulated,
        accumulated_dot,
        _left_coefficients(angle_sq),
    )
    body_twist = _adjoint_inverse_action(
        omega,
        translation,
        angle_sq,
        forward,
        _load_spatial_vector3d(segment_twist, environment, segment) + partial_twist,
    )
    pose = wp.mat44d()
    row = int(0)
    while row < 4:
        column = int(0)
        while column < 4:
            value = spatial_pose_step_entry(
                segment_pose,
                environment,
                segment,
                accumulated,
                row,
                column,
            )
            pose[row, column] = value
            poses[environment, sample, row, column] = value
            column += 1
        row += 1
    inertial_twist = _rotate_spatial_to_inertial(pose, body_twist)
    row = int(0)
    while row < SPATIAL_DIM:
        twists[environment, sample, row] = inertial_twist[row]
        row += 1


@wp.kernel(enable_backward=False)
def spatial_sample_vjp_kernel(
    s: wp.array2d[wp.float64],
    inertial_wrenches: wp.array3d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    segment_starts: wp.array[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array4d[wp.float64],
    segment_wrench: wp.array3d[wp.float64],
    generalized_force: wp.array2d[wp.float64],
):
    """Seed spatial PCS boundary cotangents and partial-segment efforts.

    Args:
        s: Per-environment sample abscissae ``(E, N)``.
        inertial_wrenches: Inertial sample cotangents ``(E, N, 6)``.
        active_indices: Active coordinate index for each strain row.
        active_scales: Scale applied to each active strain row.
        segment_starts: Cumulative segment-start coordinates ``(S + 1,)``.
        segment_strain: Precomputed segment strains ``(E, S, 6)``.
        segment_pose: Precomputed boundary poses ``(E, S + 1, 4, 4)``.
        segment_wrench: Boundary body-cotangent accumulator ``(E, S + 1, 6)``.
        generalized_force: Generalized-force accumulator ``(E, D)``.

    Returns:
        None. ``segment_wrench`` and ``generalized_force`` are accumulated in
        place.
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
    row = int(0)
    while row < SPATIAL_DIM:
        xi[row] = segment_strain[environment, segment, row]
        row += 1
    accumulated = local_s * xi
    omega = wp.vec3d(accumulated[0], accumulated[1], accumulated[2])
    linear = wp.vec3d(accumulated[3], accumulated[4], accumulated[5])
    angle_sq = wp.dot(omega, omega)
    coefficients = _left_coefficients(angle_sq)
    forward = _forward_coefficients(angle_sq)
    translation = _translation(omega, linear, forward)
    pose = wp.mat44d()
    row = int(0)
    while row < 4:
        column = int(0)
        while column < 4:
            pose[row, column] = spatial_pose_step_entry(
                segment_pose,
                environment,
                segment,
                accumulated,
                row,
                column,
            )
            column += 1
        row += 1
    body_wrench = _rotate_spatial_from_inertial(
        pose,
        _load_spatial_vector3d(inertial_wrenches, environment, sample),
    )
    source_wrench = _adjoint_inverse_transpose_action(
        omega,
        translation,
        angle_sq,
        forward,
        body_wrench,
    )
    row = int(0)
    while row < SPATIAL_DIM:
        wp.atomic_add(segment_wrench, environment, segment, row, source_wrench[row])
        active = active_indices[segment, row]
        if active >= 0:
            basis = Vec6d()
            basis[row] = local_s * active_scales[segment, row]
            tangent = _left_action(accumulated, basis, coefficients)
            effort = wp.float64(0.0)
            k = int(0)
            while k < SPATIAL_DIM:
                effort += tangent[k] * source_wrench[k]
                k += 1
            wp.atomic_add(generalized_force, environment, active, effort)
        row += 1


@wp.kernel(enable_backward=False)
def spatial_segment_vjp_kernel(
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_wrench: wp.array3d[wp.float64],
    generalized_force: wp.array2d[wp.float64],
):
    """Reverse the spatial PCS boundary recurrence.

    Args:
        active_indices: Active coordinate index for each strain row.
        active_scales: Scale applied to each active strain row.
        segment_lengths: Physical segment lengths ``(S,)``.
        segment_strain: Precomputed segment strains ``(E, S, 6)``.
        segment_wrench: Boundary body-cotangent accumulator ``(E, S + 1, 6)``.
        generalized_force: Generalized-force accumulator ``(E, D)``.

    Returns:
        None. ``segment_wrench`` and ``generalized_force`` are accumulated in
        place.
    """

    environment = wp.tid()
    segment = segment_lengths.shape[0] - 1
    while segment >= 0:
        xi = Vec6d()
        row = int(0)
        while row < SPATIAL_DIM:
            xi[row] = segment_strain[environment, segment, row]
            row += 1
        length = segment_lengths[segment]
        accumulated = length * xi
        omega = wp.vec3d(accumulated[0], accumulated[1], accumulated[2])
        linear = wp.vec3d(accumulated[3], accumulated[4], accumulated[5])
        angle_sq = wp.dot(omega, omega)
        coefficients = _left_coefficients(angle_sq)
        forward = _forward_coefficients(angle_sq)
        translation = _translation(omega, linear, forward)
        source_wrench = _adjoint_inverse_transpose_action(
            omega,
            translation,
            angle_sq,
            forward,
            _load_spatial_vector3d(segment_wrench, environment, segment + 1),
        )
        row = int(0)
        while row < SPATIAL_DIM:
            segment_wrench[environment, segment, row] += source_wrench[row]
            active = active_indices[segment, row]
            if active >= 0:
                basis = Vec6d()
                basis[row] = length * active_scales[segment, row]
                tangent = _left_action(accumulated, basis, coefficients)
                effort = wp.float64(0.0)
                k = int(0)
                while k < SPATIAL_DIM:
                    effort += tangent[k] * source_wrench[k]
                    k += 1
                generalized_force[environment, active] += effort
            row += 1
        segment -= 1


@wp.kernel(enable_backward=False)
def spatial_cooperative_segment_vjp_kernel(
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_wrench: wp.array3d[wp.float64],
    generalized_force: wp.array2d[wp.float64],
):
    """Reverse the spatial PCS boundary recurrence cooperatively on CUDA.

    Args:
        active_indices: Active coordinate index for each strain row.
        active_scales: Scale applied to each active strain row.
        segment_lengths: Physical segment lengths ``(S,)``.
        segment_strain: Precomputed segment strains ``(E, S, 6)``.
        segment_wrench: Boundary body-cotangent accumulator ``(E, S + 1, 6)``.
        generalized_force: Generalized-force accumulator ``(E, D)``.

    Returns:
        None. The six spatial rows of each environment are processed by one
        cooperative CUDA block.
    """

    environment, lane = wp.tid()
    source_wrench = wp.tile_zeros(
        shape=(SPATIAL_DIM,), dtype=wp.float64, storage="shared"
    )
    barrier = wp.tile_zeros(shape=(1,), dtype=wp.int32, storage="shared")
    segment = segment_lengths.shape[0] - 1
    while segment >= 0:
        xi = Vec6d()
        row = int(0)
        while row < SPATIAL_DIM:
            xi[row] = segment_strain[environment, segment, row]
            row += 1
        length = segment_lengths[segment]
        accumulated = length * xi
        omega = wp.vec3d(accumulated[0], accumulated[1], accumulated[2])
        linear = wp.vec3d(accumulated[3], accumulated[4], accumulated[5])
        angle_sq = wp.dot(omega, omega)
        coefficients = _left_coefficients(angle_sq)
        forward = _forward_coefficients(angle_sq)
        translation = _translation(omega, linear, forward)

        source_value = wp.float64(0.0)
        if lane < SPATIAL_DIM:
            row = int(0)
            while row < SPATIAL_DIM:
                source_value += (
                    _adjoint_inverse_entry(
                        omega,
                        translation,
                        angle_sq,
                        forward,
                        row,
                        lane,
                    )
                    * segment_wrench[environment, segment + 1, row]
                )
                row += 1
        wp.tile_scatter_masked(
            source_wrench,
            lane,
            source_value,
            lane < SPATIAL_DIM,
        )
        wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)

        if lane < SPATIAL_DIM:
            segment_wrench[environment, segment, lane] += source_wrench[lane]
            active = active_indices[segment, lane]
            if active >= 0:
                basis = Vec6d()
                basis[lane] = length * active_scales[segment, lane]
                tangent = _left_action(accumulated, basis, coefficients)
                effort = wp.float64(0.0)
                row = int(0)
                while row < SPATIAL_DIM:
                    effort += tangent[row] * source_wrench[row]
                    row += 1
                generalized_force[environment, active] += effort
        wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
        segment -= 1


def launch_spatial_forward_kinematics_and_jvp(
    q: wp.array2d[wp.float64],
    qd: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    base_transform: wp.array2d[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array4d[wp.float64],
    segment_twist: wp.array3d[wp.float64],
    poses: wp.array4d[wp.float64],
    twists: wp.array3d[wp.float64],
):
    """Launch spatial PCS poses and one inertial-Jacobian JVP.

    Args:
        q: Batched active coordinates ``(E, D)``.
        qd: Batched generalized-velocity directions ``(E, D)``.
        s: Per-environment sample abscissae ``(E, N)``.
        active_indices: Active coordinate index for each strain row.
        active_scales: Scale applied to each active strain row.
        reference_strain: Reference strain ``(S, 6)``.
        segment_lengths: Physical segment lengths ``(S,)``.
        segment_starts: Cumulative segment starts ``(S + 1,)``.
        base_transform: Absolute fixed-base transform ``(4, 4)``.
        segment_strain: Caller-owned segment-strain workspace.
        segment_pose: Caller-owned boundary-pose workspace.
        segment_twist: Caller-owned boundary-JVP workspace.
        poses: Caller-owned sample-pose output ``(E, N, 4, 4)``.
        twists: Caller-owned inertial-JVP output ``(E, N, 6)``.

    Returns:
        None. The boundary and sample kernels are enqueued on the current Warp
        stream.
    """

    wp.launch(
        spatial_segment_pose_twist_kernel,
        dim=q.shape[0],
        inputs=[
            q,
            qd,
            active_indices,
            active_scales,
            reference_strain,
            segment_lengths,
            base_transform,
        ],
        outputs=[segment_strain, segment_pose, segment_twist],
    )
    wp.launch(
        spatial_pose_twist_samples_kernel,
        dim=(q.shape[0], s.shape[1]),
        inputs=[
            qd,
            s,
            active_indices,
            active_scales,
            segment_starts,
            segment_strain,
            segment_pose,
            segment_twist,
        ],
        outputs=[poses, twists],
        block_dim=128,
    )


def launch_spatial_inertial_jacobian_vjp(
    q: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    inertial_wrenches: wp.array3d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    base_transform: wp.array2d[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array4d[wp.float64],
    segment_wrench: wp.array3d[wp.float64],
    generalized_force: wp.array2d[wp.float64],
    *,
    segment_states_ready: bool = False,
):
    """Launch ``sum_s J(q, s).T @ wrench_s`` for spatial PCS.

    Args:
        q: Batched active coordinates ``(E, D)``.
        s: Per-environment sample abscissae ``(E, N)``.
        inertial_wrenches: Inertial sample cotangents ``(E, N, 6)``.
        active_indices: Active coordinate index for each strain row.
        active_scales: Scale applied to each active strain row.
        reference_strain: Reference strain ``(S, 6)``.
        segment_lengths: Physical segment lengths ``(S,)``.
        segment_starts: Cumulative segment starts ``(S + 1,)``.
        base_transform: Absolute fixed-base transform ``(4, 4)``.
        segment_strain: Caller-owned segment-strain workspace.
        segment_pose: Caller-owned boundary-pose workspace.
        segment_wrench: Caller-owned boundary-cotangent workspace.
        generalized_force: Caller-owned VJP output ``(E, D)``.
        segment_states_ready: Skip pose-only propagation when the strain and
            pose workspaces already describe ``q``.

    Returns:
        None. The forward and reverse kernels are enqueued on the current Warp
        stream.
    """

    segment_wrench.zero_()
    generalized_force.zero_()
    if not segment_states_ready:
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
        spatial_sample_vjp_kernel,
        dim=(q.shape[0], s.shape[1]),
        inputs=[
            s,
            inertial_wrenches,
            active_indices,
            active_scales,
            segment_starts,
            segment_strain,
            segment_pose,
        ],
        outputs=[segment_wrench, generalized_force],
        block_dim=128,
    )
    reverse_inputs = [
        active_indices,
        active_scales,
        segment_lengths,
        segment_strain,
        segment_wrench,
    ]
    if q.device.is_cuda:
        wp.launch_tiled(
            spatial_cooperative_segment_vjp_kernel,
            dim=q.shape[0],
            inputs=reverse_inputs,
            outputs=[generalized_force],
            block_dim=32,
        )
    else:
        wp.launch(
            spatial_segment_vjp_kernel,
            dim=q.shape[0],
            inputs=reverse_inputs,
            outputs=[generalized_force],
        )


__all__ = [
    "launch_spatial_forward_kinematics_and_jvp",
    "launch_spatial_inertial_jacobian_vjp",
    "spatial_cooperative_segment_vjp_kernel",
    "spatial_pose_twist_samples_kernel",
    "spatial_sample_vjp_kernel",
    "spatial_segment_pose_twist_kernel",
    "spatial_segment_vjp_kernel",
]
