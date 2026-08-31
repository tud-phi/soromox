# ruff: noqa: I001, UP018
"""Matrix-free PlanarPCS inertial-Jacobian products for Warp integrations."""

from __future__ import annotations

import warp as wp

from soromox.execution.warp.pcs.planar_kernels import _planar_operators
from soromox.execution.warp.pcs.planar_kinematics import (
    planar_pose_segment_states_kernel,
    planar_pose_step,
)

wp.set_module_options({"enable_backward": False})

PLANAR_DIM = 3


@wp.kernel(enable_backward=False)
def planar_segment_pose_twist_kernel(
    q: wp.array2d[wp.float64],
    qd: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    base_pose: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array3d[wp.float64],
    segment_twist: wp.array3d[wp.float64],
):
    """Propagate PlanarPCS boundary poses and one Jacobian direction.

    Args:
        q: Batched active coordinates ``(E, D)``.
        qd: Batched generalized-velocity directions ``(E, D)``.
        active_indices: Active coordinate index for each strain row.
        active_scales: Scale applied to each active strain row.
        reference_strain: Reference strain ``(S, 3)``.
        segment_lengths: Physical segment lengths ``(S,)``.
        base_pose: Absolute fixed-base pose ``(3,)``.
        epsilons: Exponential and tangent thresholds ``(2,)``.
        segment_strain: Segment-strain output ``(E, S, 3)``.
        segment_pose: Boundary-pose output ``(E, S + 1, 3)``.
        segment_twist: Boundary body-JVP output ``(E, S + 1, 3)``.

    Returns:
        None. Boundary states are written in place.
    """

    environment = wp.tid()
    row = int(0)
    while row < PLANAR_DIM:
        segment_pose[environment, 0, row] = base_pose[row]
        segment_twist[environment, 0, row] = wp.float64(0.0)
        row += 1

    segment = int(0)
    while segment < segment_lengths.shape[0]:
        xi = wp.vec3d(
            reference_strain[segment, 0],
            reference_strain[segment, 1],
            reference_strain[segment, 2],
        )
        xid = wp.vec3d()
        row = int(0)
        while row < PLANAR_DIM:
            active = active_indices[segment, row]
            if active >= 0:
                scale = active_scales[segment, row]
                xi[row] += scale * q[environment, active]
                xid[row] = scale * qd[environment, active]
            segment_strain[environment, segment, row] = xi[row]
            row += 1
        length = segment_lengths[segment]
        pose = planar_pose_step(
            segment_pose[environment, segment, 0],
            segment_pose[environment, segment, 1],
            segment_pose[environment, segment, 2],
            xi,
            length,
            epsilons[0],
        )
        adjoint_inverse, transported_tangent, local_velocity, tangent_dot_velocity = (
            _planar_operators(
                xi,
                xid,
                length,
                epsilons[0],
                epsilons[1],
            )
        )
        boundary_twist = (
            adjoint_inverse
            * wp.vec3d(
                segment_twist[environment, segment, 0],
                segment_twist[environment, segment, 1],
                segment_twist[environment, segment, 2],
            )
            + local_velocity
        )
        row = int(0)
        while row < PLANAR_DIM:
            segment_pose[environment, segment + 1, row] = pose[row]
            segment_twist[environment, segment + 1, row] = boundary_twist[row]
            row += 1
        segment += 1


@wp.kernel(enable_backward=False)
def planar_pose_twist_samples_kernel(
    qd: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    segment_starts: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array3d[wp.float64],
    segment_twist: wp.array3d[wp.float64],
    poses: wp.array3d[wp.float64],
    twists: wp.array3d[wp.float64],
):
    """Evaluate PlanarPCS poses and inertial JVPs at runtime samples.

    Args:
        qd: Batched generalized-velocity directions ``(E, D)``.
        s: Per-environment sample abscissae ``(E, N)``.
        active_indices: Active coordinate index for each strain row.
        active_scales: Scale applied to each active strain row.
        segment_starts: Cumulative segment-start coordinates ``(S + 1,)``.
        epsilons: Exponential and tangent thresholds ``(2,)``.
        segment_strain: Precomputed segment strains ``(E, S, 3)``.
        segment_pose: Precomputed boundary poses ``(E, S + 1, 3)``.
        segment_twist: Precomputed boundary body JVPs ``(E, S + 1, 3)``.
        poses: Sample-pose output ``(E, N, 3)``.
        twists: Inertial sample-JVP output ``(E, N, 3)``.

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
    xi = wp.vec3d(
        segment_strain[environment, segment, 0],
        segment_strain[environment, segment, 1],
        segment_strain[environment, segment, 2],
    )
    xid = wp.vec3d()
    row = int(0)
    while row < PLANAR_DIM:
        active = active_indices[segment, row]
        if active >= 0:
            xid[row] = active_scales[segment, row] * qd[environment, active]
        row += 1
    pose = planar_pose_step(
        segment_pose[environment, segment, 0],
        segment_pose[environment, segment, 1],
        segment_pose[environment, segment, 2],
        xi,
        local_s,
        epsilons[0],
    )
    adjoint_inverse, transported_tangent, local_velocity, tangent_dot_velocity = (
        _planar_operators(
            xi,
            xid,
            local_s,
            epsilons[0],
            epsilons[1],
        )
    )
    body_twist = (
        adjoint_inverse
        * wp.vec3d(
            segment_twist[environment, segment, 0],
            segment_twist[environment, segment, 1],
            segment_twist[environment, segment, 2],
        )
        + local_velocity
    )
    cosine = wp.cos(pose[0])
    sine = wp.sin(pose[0])
    inertial_twist = wp.vec3d(
        body_twist[0],
        cosine * body_twist[1] - sine * body_twist[2],
        sine * body_twist[1] + cosine * body_twist[2],
    )
    row = int(0)
    while row < PLANAR_DIM:
        poses[environment, sample, row] = pose[row]
        twists[environment, sample, row] = inertial_twist[row]
        row += 1


@wp.kernel(enable_backward=False)
def planar_sample_vjp_kernel(
    s: wp.array2d[wp.float64],
    inertial_wrenches: wp.array3d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    segment_starts: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array3d[wp.float64],
    segment_wrench: wp.array3d[wp.float64],
    generalized_force: wp.array2d[wp.float64],
):
    """Seed PlanarPCS boundary cotangents and partial-segment efforts.

    Args:
        s: Per-environment sample abscissae ``(E, N)``.
        inertial_wrenches: Inertial sample cotangents ``(E, N, 3)``.
        active_indices: Active coordinate index for each strain row.
        active_scales: Scale applied to each active strain row.
        segment_starts: Cumulative segment-start coordinates ``(S + 1,)``.
        epsilons: Exponential and tangent thresholds ``(2,)``.
        segment_strain: Precomputed segment strains ``(E, S, 3)``.
        segment_pose: Precomputed boundary poses ``(E, S + 1, 3)``.
        segment_wrench: Boundary body-cotangent accumulator ``(E, S + 1, 3)``.
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
    adjoint_inverse, transported_tangent, local_velocity, tangent_dot_velocity = (
        _planar_operators(
            xi,
            wp.vec3d(),
            local_s,
            epsilons[0],
            epsilons[1],
        )
    )
    cosine = wp.cos(pose[0])
    sine = wp.sin(pose[0])
    body_wrench = wp.vec3d(
        inertial_wrenches[environment, sample, 0],
        cosine * inertial_wrenches[environment, sample, 1]
        + sine * inertial_wrenches[environment, sample, 2],
        -sine * inertial_wrenches[environment, sample, 1]
        + cosine * inertial_wrenches[environment, sample, 2],
    )
    row = int(0)
    while row < PLANAR_DIM:
        source = wp.float64(0.0)
        effort = wp.float64(0.0)
        k = int(0)
        while k < PLANAR_DIM:
            source += adjoint_inverse[k, row] * body_wrench[k]
            effort += transported_tangent[k, row] * body_wrench[k]
            k += 1
        wp.atomic_add(segment_wrench, environment, segment, row, source)
        active = active_indices[segment, row]
        if active >= 0:
            wp.atomic_add(
                generalized_force,
                environment,
                active,
                active_scales[segment, row] * effort,
            )
        row += 1


@wp.kernel(enable_backward=False)
def planar_segment_vjp_kernel(
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_wrench: wp.array3d[wp.float64],
    generalized_force: wp.array2d[wp.float64],
):
    """Reverse the PlanarPCS boundary recurrence.

    Args:
        active_indices: Active coordinate index for each strain row.
        active_scales: Scale applied to each active strain row.
        segment_lengths: Physical segment lengths ``(S,)``.
        epsilons: Exponential and tangent thresholds ``(2,)``.
        segment_strain: Precomputed segment strains ``(E, S, 3)``.
        segment_wrench: Boundary body-cotangent accumulator ``(E, S + 1, 3)``.
        generalized_force: Generalized-force accumulator ``(E, D)``.

    Returns:
        None. ``segment_wrench`` and ``generalized_force`` are accumulated in
        place.
    """

    environment = wp.tid()
    segment = segment_lengths.shape[0] - 1
    while segment >= 0:
        xi = wp.vec3d(
            segment_strain[environment, segment, 0],
            segment_strain[environment, segment, 1],
            segment_strain[environment, segment, 2],
        )
        adjoint_inverse, transported_tangent, local_velocity, tangent_dot_velocity = (
            _planar_operators(
                xi,
                wp.vec3d(),
                segment_lengths[segment],
                epsilons[0],
                epsilons[1],
            )
        )
        next_wrench = wp.vec3d(
            segment_wrench[environment, segment + 1, 0],
            segment_wrench[environment, segment + 1, 1],
            segment_wrench[environment, segment + 1, 2],
        )
        row = int(0)
        while row < PLANAR_DIM:
            source = wp.float64(0.0)
            effort = wp.float64(0.0)
            k = int(0)
            while k < PLANAR_DIM:
                source += adjoint_inverse[k, row] * next_wrench[k]
                effort += transported_tangent[k, row] * next_wrench[k]
                k += 1
            segment_wrench[environment, segment, row] += source
            active = active_indices[segment, row]
            if active >= 0:
                generalized_force[environment, active] += (
                    active_scales[segment, row] * effort
                )
            row += 1
        segment -= 1


def launch_planar_forward_kinematics_and_jvp(
    q: wp.array2d[wp.float64],
    qd: wp.array2d[wp.float64],
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
    segment_twist: wp.array3d[wp.float64],
    poses: wp.array3d[wp.float64],
    twists: wp.array3d[wp.float64],
):
    """Launch PlanarPCS poses and one inertial-Jacobian JVP.

    Args:
        q: Batched active coordinates ``(E, D)``.
        qd: Batched generalized-velocity directions ``(E, D)``.
        s: Per-environment sample abscissae ``(E, N)``.
        active_indices: Active coordinate index for each strain row.
        active_scales: Scale applied to each active strain row.
        reference_strain: Reference strain ``(S, 3)``.
        segment_lengths: Physical segment lengths ``(S,)``.
        segment_starts: Cumulative segment starts ``(S + 1,)``.
        base_pose: Absolute fixed-base pose ``(3,)``.
        epsilons: Exponential and tangent thresholds ``(2,)``.
        segment_strain: Caller-owned segment-strain workspace.
        segment_pose: Caller-owned boundary-pose workspace.
        segment_twist: Caller-owned boundary-JVP workspace.
        poses: Caller-owned sample-pose output ``(E, N, 3)``.
        twists: Caller-owned inertial-JVP output ``(E, N, 3)``.

    Returns:
        None. The boundary and sample kernels are enqueued on the current Warp
        stream.
    """

    wp.launch(
        planar_segment_pose_twist_kernel,
        dim=q.shape[0],
        inputs=[
            q,
            qd,
            active_indices,
            active_scales,
            reference_strain,
            segment_lengths,
            base_pose,
            epsilons,
        ],
        outputs=[segment_strain, segment_pose, segment_twist],
    )
    wp.launch(
        planar_pose_twist_samples_kernel,
        dim=(q.shape[0], s.shape[1]),
        inputs=[
            qd,
            s,
            active_indices,
            active_scales,
            segment_starts,
            epsilons,
            segment_strain,
            segment_pose,
            segment_twist,
        ],
        outputs=[poses, twists],
        block_dim=128,
    )


def launch_planar_inertial_jacobian_vjp(
    q: wp.array2d[wp.float64],
    s: wp.array2d[wp.float64],
    inertial_wrenches: wp.array3d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    segment_lengths: wp.array[wp.float64],
    segment_starts: wp.array[wp.float64],
    base_pose: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    segment_strain: wp.array3d[wp.float64],
    segment_pose: wp.array3d[wp.float64],
    segment_wrench: wp.array3d[wp.float64],
    generalized_force: wp.array2d[wp.float64],
    *,
    segment_states_ready: bool = False,
):
    """Launch ``sum_s J(q, s).T @ wrench_s`` for PlanarPCS.

    Args:
        q: Batched active coordinates ``(E, D)``.
        s: Per-environment sample abscissae ``(E, N)``.
        inertial_wrenches: Inertial sample cotangents ``(E, N, 3)``.
        active_indices: Active coordinate index for each strain row.
        active_scales: Scale applied to each active strain row.
        reference_strain: Reference strain ``(S, 3)``.
        segment_lengths: Physical segment lengths ``(S,)``.
        segment_starts: Cumulative segment starts ``(S + 1,)``.
        base_pose: Absolute fixed-base pose ``(3,)``.
        epsilons: Exponential and tangent thresholds ``(2,)``.
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
        planar_sample_vjp_kernel,
        dim=(q.shape[0], s.shape[1]),
        inputs=[
            s,
            inertial_wrenches,
            active_indices,
            active_scales,
            segment_starts,
            epsilons,
            segment_strain,
            segment_pose,
        ],
        outputs=[segment_wrench, generalized_force],
        block_dim=128,
    )
    wp.launch(
        planar_segment_vjp_kernel,
        dim=q.shape[0],
        inputs=[
            active_indices,
            active_scales,
            segment_lengths,
            epsilons,
            segment_strain,
            segment_wrench,
        ],
        outputs=[generalized_force],
    )


__all__ = [
    "launch_planar_forward_kinematics_and_jvp",
    "launch_planar_inertial_jacobian_vjp",
    "planar_pose_twist_samples_kernel",
    "planar_sample_vjp_kernel",
    "planar_segment_pose_twist_kernel",
    "planar_segment_vjp_kernel",
]
