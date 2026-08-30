# ruff: noqa: I001, UP018
"""Allocation-free fused PlanarPCS Warp kinematics kernels."""

from __future__ import annotations

import warp as wp

from soromox.execution.warp.pcs.planar_kernels import (
    _forward_coefficients,
    _planar_operators,
)

wp.set_module_options({"enable_backward": False})


@wp.func
def planar_pose_step(
    theta: wp.float64,
    x: wp.float64,
    y: wp.float64,
    xi: wp.vec3d,
    arc_length: wp.float64,
    eps: wp.float64,
) -> wp.vec3d:
    """Integrate one constant-strain SE(2) pose step.

    Args:
        theta: Absolute orientation at the start of the step.
        x: Absolute horizontal position at the start of the step.
        y: Absolute vertical position at the start of the step.
        xi: Constant planar strain in angular-first coordinates.
        arc_length: Local integration distance.
        eps: Requested small-angle threshold.

    Returns:
        Absolute planar pose ``[theta, x, y]`` after the step.
    """

    z = arc_length * xi[0]
    cutoff = wp.max(wp.abs(arc_length * eps), wp.float64(0.04964607461902946))
    coefficients = _forward_coefficients(z, cutoff)
    vx = arc_length * xi[1]
    vy = arc_length * xi[2]
    local_x = coefficients[0] * vx - z * coefficients[1] * vy
    local_y = z * coefficients[1] * vx + coefficients[0] * vy
    cosine = wp.cos(theta)
    sine = wp.sin(theta)
    return wp.vec3d(
        theta + z,
        x + cosine * local_x - sine * local_y,
        y + sine * local_x + cosine * local_y,
    )


@wp.kernel(enable_backward=False)
def planar_pose_segment_states_kernel(
    q: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    base_pose: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array3d[wp.float64],
):
    """Build PlanarPCS segment states without Jacobian propagation.

    Args:
        q: Batched active coordinates with shape ``(E, D)``.
        active_indices: Active coordinate index for each full strain row.
        active_scales: Scale applied to each active strain row.
        reference_strain: Reference strain with shape ``(S, 3)``.
        segment_lengths: Segment lengths with shape ``(S,)``.
        base_pose: Absolute planar base pose with shape ``(3,)``.
        epsilons: Exponential and tangent thresholds with shape ``(2,)``.
        segment_strain: Caller-owned strain workspace ``(E, S, 3)``.
        segment_pose: Caller-owned boundary-pose workspace ``(E, S + 1, 3)``.

    Returns:
        None. Pose-only workspaces are written in place.
    """

    environment = wp.tid()
    segment_pose[environment, 0, 0] = base_pose[0]
    segment_pose[environment, 0, 1] = base_pose[1]
    segment_pose[environment, 0, 2] = base_pose[2]
    segment = int(0)
    while segment < segment_lengths.shape[0]:
        xi = wp.vec3d(
            reference_strain[segment, 0],
            reference_strain[segment, 1],
            reference_strain[segment, 2],
        )
        local = int(0)
        while local < 3:
            active = active_indices[segment, local]
            if active >= 0:
                xi[local] += active_scales[segment, local] * q[environment, active]
            segment_strain[environment, segment, local] = xi[local]
            local += 1
        pose = planar_pose_step(
            segment_pose[environment, segment, 0],
            segment_pose[environment, segment, 1],
            segment_pose[environment, segment, 2],
            xi,
            segment_lengths[segment],
            epsilons[0],
        )
        row = int(0)
        while row < 3:
            segment_pose[environment, segment + 1, row] = pose[row]
            row += 1
        segment += 1


@wp.kernel(enable_backward=False)
def planar_segment_states_kernel(
    q: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    base_pose: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array3d[wp.float64],
    segment_jacobian: wp.array4d[wp.float64],
):
    """Build reusable PlanarPCS segment-boundary pose/Jacobian states.

    Args:
        q: Batched active coordinates with shape ``(E, D)``.
        active_indices: Active coordinate index for each full strain row.
        active_scales: Scale applied to each active strain row.
        reference_strain: Reference strain with shape ``(S, 3)``.
        segment_lengths: Segment lengths with shape ``(S,)``.
        base_pose: Absolute planar base pose with shape ``(3,)``.
        epsilons: Exponential and tangent thresholds with shape ``(2,)``.
        segment_strain: Caller-owned strain workspace ``(E, S, 3)``.
        segment_pose: Caller-owned boundary-pose workspace ``(E, S + 1, 3)``.
        segment_jacobian: Caller-owned body-Jacobian workspace
            ``(E, S + 1, 3, D)``.

    Returns:
        None. Workspaces are written in place.
    """

    environment = wp.tid()
    num_segments = segment_lengths.shape[0]
    num_dofs = q.shape[1]
    segment_pose[environment, 0, 0] = base_pose[0]
    segment_pose[environment, 0, 1] = base_pose[1]
    segment_pose[environment, 0, 2] = base_pose[2]
    row = int(0)
    while row < 3:
        column = int(0)
        while column < num_dofs:
            segment_jacobian[environment, 0, row, column] = wp.float64(0.0)
            column += 1
        row += 1

    segment = int(0)
    while segment < num_segments:
        xi = wp.vec3d(
            reference_strain[segment, 0],
            reference_strain[segment, 1],
            reference_strain[segment, 2],
        )
        local = int(0)
        while local < 3:
            active = active_indices[segment, local]
            if active >= 0:
                xi[local] += active_scales[segment, local] * q[environment, active]
            segment_strain[environment, segment, local] = xi[local]
            local += 1
        length = segment_lengths[segment]
        pose = planar_pose_step(
            segment_pose[environment, segment, 0],
            segment_pose[environment, segment, 1],
            segment_pose[environment, segment, 2],
            xi,
            length,
            epsilons[0],
        )
        row = int(0)
        while row < 3:
            segment_pose[environment, segment + 1, row] = pose[row]
            row += 1
        adjoint_inverse, transported_tangent, _, _ = _planar_operators(
            xi, wp.vec3d(), length, epsilons[0], epsilons[1]
        )
        row = int(0)
        while row < 3:
            column = int(0)
            while column < num_dofs:
                value = wp.float64(0.0)
                k = int(0)
                while k < 3:
                    value += (
                        adjoint_inverse[row, k]
                        * segment_jacobian[environment, segment, k, column]
                    )
                    if active_indices[segment, k] == column:
                        value += transported_tangent[row, k] * active_scales[segment, k]
                    k += 1
                segment_jacobian[environment, segment + 1, row, column] = value
                column += 1
            row += 1
        segment += 1


@wp.func
def write_planar_sample_jacobian(
    environment: int,
    sample: int,
    segment: int,
    local_s: wp.float64,
    pose: wp.vec3d,
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    epsilons: wp.array[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_jacobian: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Write one PlanarPCS inertial Jacobian from boundary states.

    Args:
        environment: Environment index.
        sample: Backbone-sample index.
        segment: Segment owning the sample.
        local_s: Sample distance from the segment base.
        pose: Absolute pose at the sample.
        active_indices: Active coordinate index for each strain row.
        active_scales: Scale applied to each active strain row.
        epsilons: Exponential and tangent thresholds.
        segment_strain: Precomputed segment strains.
        segment_jacobian: Precomputed boundary body Jacobians.
        jacobians: Caller-owned inertial-Jacobian output.

    Returns:
        None. One output slice is written in place.
    """

    xi = wp.vec3d(
        segment_strain[environment, segment, 0],
        segment_strain[environment, segment, 1],
        segment_strain[environment, segment, 2],
    )
    adjoint_inverse, transported_tangent, _, _ = _planar_operators(
        xi, wp.vec3d(), local_s, epsilons[0], epsilons[1]
    )
    cosine = wp.cos(pose[0])
    sine = wp.sin(pose[0])
    row = int(0)
    while row < 3:
        column = int(0)
        while column < segment_jacobian.shape[3]:
            body = wp.vec3d()
            body_row = int(0)
            while body_row < 3:
                value = wp.float64(0.0)
                k = int(0)
                while k < 3:
                    value += (
                        adjoint_inverse[body_row, k]
                        * segment_jacobian[environment, segment, k, column]
                    )
                    if active_indices[segment, k] == column:
                        value += (
                            transported_tangent[body_row, k] * active_scales[segment, k]
                        )
                    k += 1
                body[body_row] = value
                body_row += 1
            output = body[0]
            if row == 1:
                output = cosine * body[1] - sine * body[2]
            elif row == 2:
                output = sine * body[1] + cosine * body[2]
            jacobians[environment, sample, row, column] = output
            column += 1
        row += 1


@wp.kernel(enable_backward=False)
def planar_pose_samples_kernel(
    s: wp.array2d[wp.float64],
    segment_starts: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array3d[wp.float64],
    poses: wp.array3d[wp.float64],
):
    """Evaluate PlanarPCS poses without Jacobian work.

    Args:
        s: Per-environment abscissae with shape ``(E, N)``.
        segment_starts: Cumulative segment-start coordinates.
        epsilons: Exponential and tangent thresholds.
        segment_strain: Precomputed segment strains.
        segment_pose: Precomputed boundary poses.
        poses: Caller-owned planar-pose output.

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
    xi = wp.vec3d(
        segment_strain[environment, segment, 0],
        segment_strain[environment, segment, 1],
        segment_strain[environment, segment, 2],
    )
    pose = planar_pose_step(
        segment_pose[environment, segment, 0],
        segment_pose[environment, segment, 1],
        segment_pose[environment, segment, 2],
        xi,
        abscissa - segment_starts[segment],
        epsilons[0],
    )
    row = int(0)
    while row < 3:
        poses[environment, sample, row] = pose[row]
        row += 1


@wp.kernel(enable_backward=False)
def planar_jacobian_samples_kernel(
    s: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    segment_starts: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array3d[wp.float64],
    segment_jacobian: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Evaluate PlanarPCS inertial Jacobians without pose output writes.

    Args:
        s: Per-environment abscissae with shape ``(E, N)``.
        active_indices: Active coordinate index for each strain row.
        active_scales: Scale applied to each active strain row.
        segment_starts: Cumulative segment-start coordinates.
        epsilons: Exponential and tangent thresholds.
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
    xi = wp.vec3d(
        segment_strain[environment, segment, 0],
        segment_strain[environment, segment, 1],
        segment_strain[environment, segment, 2],
    )
    pose = planar_pose_step(
        segment_pose[environment, segment, 0],
        segment_pose[environment, segment, 1],
        segment_pose[environment, segment, 2],
        xi,
        local_s,
        epsilons[0],
    )
    write_planar_sample_jacobian(
        environment,
        sample,
        segment,
        local_s,
        pose,
        active_indices,
        active_scales,
        epsilons,
        segment_strain,
        segment_jacobian,
        jacobians,
    )


@wp.kernel(enable_backward=False)
def planar_samples_kernel(
    s: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    segment_starts: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array3d[wp.float64],
    segment_jacobian: wp.array4d[wp.float64],
    poses: wp.array3d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Evaluate PlanarPCS poses and inertial Jacobians at runtime samples.

    Args:
        s: Per-environment abscissae with shape ``(E, N)``.
        active_indices: Active coordinate index for each full strain row.
        active_scales: Scale applied to each active strain row.
        segment_starts: Cumulative segment-start coordinates.
        epsilons: Exponential and tangent thresholds.
        segment_strain: Precomputed segment-strain workspace.
        segment_pose: Precomputed segment-boundary poses.
        segment_jacobian: Precomputed segment-boundary body Jacobians.
        poses: Caller-owned planar pose output ``(E, N, 3)``.
        jacobians: Caller-owned inertial Jacobian output ``(E, N, 3, D)``.

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
    xi = wp.vec3d(
        segment_strain[environment, segment, 0],
        segment_strain[environment, segment, 1],
        segment_strain[environment, segment, 2],
    )
    pose = planar_pose_step(
        segment_pose[environment, segment, 0],
        segment_pose[environment, segment, 1],
        segment_pose[environment, segment, 2],
        xi,
        local_s,
        epsilons[0],
    )
    row = int(0)
    while row < 3:
        poses[environment, sample, row] = pose[row]
        row += 1
    write_planar_sample_jacobian(
        environment,
        sample,
        segment,
        local_s,
        pose,
        active_indices,
        active_scales,
        epsilons,
        segment_strain,
        segment_jacobian,
        jacobians,
    )


def launch_planar_forward_kinematics(
    q: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    base_pose: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array3d[wp.float64],
    poses: wp.array3d[wp.float64],
):
    """Launch the reduced-work PlanarPCS pose pipeline.

    Args:
        q: Batched active coordinates ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        active_indices: Active coordinate indices by segment and strain row.
        active_scales: Active strain scales matching ``active_indices``.
        reference_strain: Reference strain ``(S, 3)``.
        segment_lengths: Segment lengths ``(S,)``.
        segment_starts: Cumulative segment starts ``(S + 1,)``.
        base_pose: Absolute planar base pose ``(3,)``.
        epsilons: Exponential and tangent thresholds ``(2,)``.
        segment_strain: Caller-owned strain workspace.
        segment_pose: Caller-owned boundary-pose workspace.
        poses: Caller-owned pose output.

    Returns:
        None. Pose kernels are enqueued on the current Warp stream.
    """

    wp.launch(
        planar_pose_segment_states_kernel,
        dim=q.shape[0],
        inputs=[
            q,
            active_indices,
            active_scales,
            reference_strain,
            segment_lengths,
            base_pose,
            epsilons,
        ],
        outputs=[segment_strain, segment_pose],
    )
    wp.launch(
        planar_pose_samples_kernel,
        dim=(q.shape[0], s.shape[1]),
        inputs=[s, segment_starts, epsilons, segment_strain, segment_pose],
        outputs=[poses],
        block_dim=128,
    )


def launch_planar_inertial_jacobians(
    q: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    base_pose: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array3d[wp.float64],
    segment_jacobian: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Launch PlanarPCS inertial Jacobians without pose output writes.

    Args:
        q: Batched active coordinates ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        active_indices: Active coordinate indices by segment and strain row.
        active_scales: Active strain scales matching ``active_indices``.
        reference_strain: Reference strain ``(S, 3)``.
        segment_lengths: Segment lengths ``(S,)``.
        segment_starts: Cumulative segment starts ``(S + 1,)``.
        base_pose: Absolute planar base pose ``(3,)``.
        epsilons: Exponential and tangent thresholds ``(2,)``.
        segment_strain: Caller-owned strain workspace.
        segment_pose: Caller-owned boundary-pose workspace.
        segment_jacobian: Caller-owned boundary-Jacobian workspace.
        jacobians: Caller-owned inertial-Jacobian output.

    Returns:
        None. Jacobian kernels are enqueued on the current Warp stream.
    """

    wp.launch(
        planar_segment_states_kernel,
        dim=q.shape[0],
        inputs=[
            q,
            active_indices,
            active_scales,
            reference_strain,
            segment_lengths,
            base_pose,
            epsilons,
        ],
        outputs=[segment_strain, segment_pose, segment_jacobian],
    )
    wp.launch(
        planar_jacobian_samples_kernel,
        dim=(q.shape[0], s.shape[1]),
        inputs=[
            s,
            active_indices,
            active_scales,
            segment_starts,
            epsilons,
            segment_strain,
            segment_pose,
            segment_jacobian,
        ],
        outputs=[jacobians],
        block_dim=128,
    )


def launch_planar_kinematics(
    q: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    base_pose: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array3d[wp.float64],
    segment_jacobian: wp.array4d[wp.float64],
    poses: wp.array3d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Launch the allocation-free fused PlanarPCS pipeline.

    Args:
        q: Batched active coordinates ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        active_indices: Active coordinate indices by segment and strain row.
        active_scales: Active strain scales matching ``active_indices``.
        reference_strain: Reference strain ``(S, 3)``.
        segment_lengths: Segment lengths ``(S,)``.
        segment_starts: Cumulative segment starts ``(S + 1,)``.
        base_pose: Absolute planar base pose ``(3,)``.
        epsilons: Exponential and tangent thresholds ``(2,)``.
        segment_strain: Caller-owned strain workspace.
        segment_pose: Caller-owned boundary-pose workspace.
        segment_jacobian: Caller-owned boundary-Jacobian workspace.
        poses: Caller-owned pose output.
        jacobians: Caller-owned inertial-Jacobian output.

    Returns:
        None. The two kernels are enqueued on the current Warp stream.
    """

    wp.launch(
        planar_segment_states_kernel,
        dim=q.shape[0],
        inputs=[
            q,
            active_indices,
            active_scales,
            reference_strain,
            segment_lengths,
            base_pose,
            epsilons,
        ],
        outputs=[segment_strain, segment_pose, segment_jacobian],
    )
    wp.launch(
        planar_samples_kernel,
        dim=(q.shape[0], s.shape[1]),
        inputs=[
            s,
            active_indices,
            active_scales,
            segment_starts,
            epsilons,
            segment_strain,
            segment_pose,
            segment_jacobian,
        ],
        outputs=[poses, jacobians],
        block_dim=128,
    )


planar_forward_kinematics = wp.jax_callable(
    launch_planar_forward_kinematics, num_outputs=3
)
planar_inertial_jacobians = wp.jax_callable(
    launch_planar_inertial_jacobians, num_outputs=4
)
planar_kinematics = wp.jax_callable(launch_planar_kinematics, num_outputs=5)


__all__ = [
    "launch_planar_kinematics",
    "launch_planar_forward_kinematics",
    "launch_planar_inertial_jacobians",
    "planar_jacobian_samples_kernel",
    "planar_pose_step",
    "planar_pose_samples_kernel",
    "planar_pose_segment_states_kernel",
    "planar_samples_kernel",
    "planar_segment_states_kernel",
    "write_planar_sample_jacobian",
]
