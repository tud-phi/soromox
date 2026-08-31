# ruff: noqa: I001, UP018
"""Warp-native runtime-base composition for continuum kinematics."""

from __future__ import annotations

import warp as wp

from soromox.execution.warp.common.floating_base import (
    _quaternion_rotation_transpose_entry,
)

wp.set_module_options({"enable_backward": False})


@wp.func
def _spatial_base_rotation_entry(
    base_pose: wp.array2d[wp.float64], environment: int, row: int, column: int
) -> wp.float64:
    """Return one normalized runtime base-rotation entry."""
    return _quaternion_rotation_transpose_entry(
        base_pose, environment, column, row
    )


@wp.kernel(enable_backward=False)
def _compose_spatial_poses_kernel(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    output: wp.array4d[wp.float64],
):
    environment, sample = wp.tid()
    row = int(0)
    while row < 3:
        column = int(0)
        while column < 3:
            value = wp.float64(0.0)
            inner = int(0)
            while inner < 3:
                value += _spatial_base_rotation_entry(
                    base_pose, environment, row, inner
                ) * relative_pose[environment, sample, inner, column]
                inner += 1
            output[environment, sample, row, column] = value
            column += 1
        position = base_pose[environment, 4 + row]
        inner = int(0)
        while inner < 3:
            position += _spatial_base_rotation_entry(
                base_pose, environment, row, inner
            ) * relative_pose[environment, sample, inner, 3]
            inner += 1
        output[environment, sample, row, 3] = position
        row += 1
    column = int(0)
    while column < 3:
        output[environment, sample, 3, column] = wp.float64(0.0)
        column += 1
    output[environment, sample, 3, 3] = wp.float64(1.0)


@wp.kernel(enable_backward=False)
def _compose_spatial_jacobians_kernel(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    internal: wp.array4d[wp.float64],
    output: wp.array4d[wp.float64],
):
    environment, sample = wp.tid()
    rx = wp.float64(0.0)
    ry = wp.float64(0.0)
    rz = wp.float64(0.0)
    inner = int(0)
    while inner < 3:
        rx += _spatial_base_rotation_entry(
            base_pose, environment, 0, inner
        ) * relative_pose[environment, sample, inner, 3]
        ry += _spatial_base_rotation_entry(
            base_pose, environment, 1, inner
        ) * relative_pose[environment, sample, inner, 3]
        rz += _spatial_base_rotation_entry(
            base_pose, environment, 2, inner
        ) * relative_pose[environment, sample, inner, 3]
        inner += 1

    row = int(0)
    while row < 6:
        column = int(0)
        while column < 6:
            value = wp.float64(0.0)
            if (
                (row < 3 and column < 3 or row >= 3 and column >= 3)
                and row == column
            ):
                value = wp.float64(1.0)
            elif row == 3 and column == 1:
                value = rz
            elif row == 3 and column == 2:
                value = -ry
            elif row == 4 and column == 0:
                value = -rz
            elif row == 4 and column == 2:
                value = rx
            elif row == 5 and column == 0:
                value = ry
            elif row == 5 and column == 1:
                value = -rx
            output[environment, sample, row, column] = value
            column += 1
        internal_column = int(0)
        while internal_column < internal.shape[3]:
            value = wp.float64(0.0)
            inner = int(0)
            while inner < 3:
                value += _spatial_base_rotation_entry(
                    base_pose, environment, row % 3, inner
                ) * internal[
                    environment,
                    sample,
                    (0 if row < 3 else 3) + inner,
                    internal_column,
                ]
                inner += 1
            output[environment, sample, row, 6 + internal_column] = value
            internal_column += 1
        row += 1


@wp.kernel(enable_backward=False)
def _compose_spatial_poses_and_twists_kernel(
    base_pose: wp.array2d[wp.float64],
    base_velocity: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    internal_twist: wp.array3d[wp.float64],
    poses: wp.array4d[wp.float64],
    twists: wp.array3d[wp.float64],
):
    """Compose floating poses and one augmented Jacobian direction.

    Args:
        base_pose: Runtime scalar-last base poses ``(E, 7)``.
        base_velocity: Runtime generalized velocities whose first six entries
            are the angular-linear base direction.
        relative_pose: Fixed-base relative sample poses ``(E, N, 4, 4)``.
        internal_twist: Relative inertial JVPs ``(E, N, 6)``.
        poses: Absolute pose output ``(E, N, 4, 4)``.
        twists: Absolute inertial JVP output ``(E, N, 6)``.

    Returns:
        None. ``poses`` and ``twists`` are written in place.
    """

    environment, sample = wp.tid()
    rx = wp.float64(0.0)
    ry = wp.float64(0.0)
    rz = wp.float64(0.0)
    inner = int(0)
    while inner < 3:
        rx += _spatial_base_rotation_entry(
            base_pose, environment, 0, inner
        ) * relative_pose[environment, sample, inner, 3]
        ry += _spatial_base_rotation_entry(
            base_pose, environment, 1, inner
        ) * relative_pose[environment, sample, inner, 3]
        rz += _spatial_base_rotation_entry(
            base_pose, environment, 2, inner
        ) * relative_pose[environment, sample, inner, 3]
        inner += 1
    radius = wp.vec3d(rx, ry, rz)
    base_angular = wp.vec3d(
        base_velocity[environment, 0],
        base_velocity[environment, 1],
        base_velocity[environment, 2],
    )
    base_linear = wp.vec3d(
        base_velocity[environment, 3],
        base_velocity[environment, 4],
        base_velocity[environment, 5],
    )

    row = int(0)
    while row < 3:
        column = int(0)
        while column < 3:
            value = wp.float64(0.0)
            inner = int(0)
            while inner < 3:
                value += _spatial_base_rotation_entry(
                    base_pose, environment, row, inner
                ) * relative_pose[environment, sample, inner, column]
                inner += 1
            poses[environment, sample, row, column] = value
            column += 1
        poses[environment, sample, row, 3] = base_pose[environment, 4 + row] + radius[row]
        row += 1
    column = int(0)
    while column < 3:
        poses[environment, sample, 3, column] = wp.float64(0.0)
        column += 1
    poses[environment, sample, 3, 3] = wp.float64(1.0)

    point_velocity = base_linear + wp.cross(base_angular, radius)
    row = int(0)
    while row < 3:
        angular = base_angular[row]
        linear = point_velocity[row]
        inner = int(0)
        while inner < 3:
            rotation = _spatial_base_rotation_entry(
                base_pose, environment, row, inner
            )
            angular += rotation * internal_twist[environment, sample, inner]
            linear += rotation * internal_twist[environment, sample, 3 + inner]
            inner += 1
        twists[environment, sample, row] = angular
        twists[environment, sample, 3 + row] = linear
        row += 1


@wp.kernel(enable_backward=False)
def _compose_spatial_wrench_vjp_kernel(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    inertial_wrenches: wp.array3d[wp.float64],
    relative_wrenches: wp.array3d[wp.float64],
    base_generalized_force: wp.array2d[wp.float64],
):
    """Pull absolute sample wrenches through floating-base composition.

    Args:
        base_pose: Runtime scalar-last base poses ``(E, 7)``.
        relative_pose: Fixed-base relative sample poses ``(E, N, 4, 4)``.
        inertial_wrenches: Absolute inertial sample cotangents ``(E, N, 6)``.
        relative_wrenches: Rotated cotangent output for the internal VJP.
        base_generalized_force: Base effort accumulator ``(E, 6)``.

    Returns:
        None. ``relative_wrenches`` is written and ``base_generalized_force``
        is accumulated in place.
    """

    environment, sample = wp.tid()
    radius = wp.vec3d()
    row = int(0)
    while row < 3:
        inner = int(0)
        while inner < 3:
            radius[row] += _spatial_base_rotation_entry(
                base_pose, environment, row, inner
            ) * relative_pose[environment, sample, inner, 3]
            inner += 1
        row += 1
    moment = wp.vec3d(
        inertial_wrenches[environment, sample, 0],
        inertial_wrenches[environment, sample, 1],
        inertial_wrenches[environment, sample, 2],
    )
    force = wp.vec3d(
        inertial_wrenches[environment, sample, 3],
        inertial_wrenches[environment, sample, 4],
        inertial_wrenches[environment, sample, 5],
    )
    base_moment = moment + wp.cross(radius, force)
    row = int(0)
    while row < 3:
        relative_moment = wp.float64(0.0)
        relative_force = wp.float64(0.0)
        inner = int(0)
        while inner < 3:
            rotation = _spatial_base_rotation_entry(
                base_pose, environment, inner, row
            )
            relative_moment += rotation * moment[inner]
            relative_force += rotation * force[inner]
            inner += 1
        relative_wrenches[environment, sample, row] = relative_moment
        relative_wrenches[environment, sample, 3 + row] = relative_force
        wp.atomic_add(base_generalized_force, environment, row, base_moment[row])
        wp.atomic_add(base_generalized_force, environment, 3 + row, force[row])
        row += 1


@wp.kernel(enable_backward=False)
def _assemble_floating_vjp_kernel(
    base_generalized_force: wp.array2d[wp.float64],
    internal_generalized_force: wp.array2d[wp.float64],
    generalized_force: wp.array2d[wp.float64],
):
    """Assemble base and internal blocks of an augmented floating VJP.

    Args:
        base_generalized_force: Base effort ``(E, B)``.
        internal_generalized_force: Internal-coordinate effort ``(E, D)``.
        generalized_force: Augmented effort output ``(E, B + D)``.

    Returns:
        None. ``generalized_force`` is written in place.
    """

    environment, column = wp.tid()
    base_dimension = base_generalized_force.shape[1]
    if column < base_dimension:
        generalized_force[environment, column] = base_generalized_force[
            environment, column
        ]
    else:
        generalized_force[environment, column] = internal_generalized_force[
            environment, column - base_dimension
        ]


@wp.kernel(enable_backward=False)
def _compose_planar_poses_and_twists_kernel(
    base_pose: wp.array2d[wp.float64],
    base_velocity: wp.array2d[wp.float64],
    relative_pose: wp.array3d[wp.float64],
    internal_twist: wp.array3d[wp.float64],
    poses: wp.array3d[wp.float64],
    twists: wp.array3d[wp.float64],
):
    """Compose planar floating poses and one augmented Jacobian direction.

    Args:
        base_pose: Runtime planar base poses ``(E, 3)``.
        base_velocity: Runtime generalized velocities whose first three entries
            are the angular-linear base direction.
        relative_pose: Fixed-base relative sample poses ``(E, N, 3)``.
        internal_twist: Relative inertial JVPs ``(E, N, 3)``.
        poses: Absolute pose output ``(E, N, 3)``.
        twists: Absolute inertial JVP output ``(E, N, 3)``.

    Returns:
        None. ``poses`` and ``twists`` are written in place.
    """

    environment, sample = wp.tid()
    theta = base_pose[environment, 0]
    cosine = wp.cos(theta)
    sine = wp.sin(theta)
    relative_x = relative_pose[environment, sample, 1]
    relative_y = relative_pose[environment, sample, 2]
    radius_x = cosine * relative_x - sine * relative_y
    radius_y = sine * relative_x + cosine * relative_y
    angular = base_velocity[environment, 0]
    poses[environment, sample, 0] = theta + relative_pose[environment, sample, 0]
    poses[environment, sample, 1] = base_pose[environment, 1] + radius_x
    poses[environment, sample, 2] = base_pose[environment, 2] + radius_y
    twists[environment, sample, 0] = (
        angular + internal_twist[environment, sample, 0]
    )
    twists[environment, sample, 1] = (
        base_velocity[environment, 1]
        - angular * radius_y
        + cosine * internal_twist[environment, sample, 1]
        - sine * internal_twist[environment, sample, 2]
    )
    twists[environment, sample, 2] = (
        base_velocity[environment, 2]
        + angular * radius_x
        + sine * internal_twist[environment, sample, 1]
        + cosine * internal_twist[environment, sample, 2]
    )


@wp.kernel(enable_backward=False)
def _compose_planar_wrench_vjp_kernel(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array3d[wp.float64],
    inertial_wrenches: wp.array3d[wp.float64],
    relative_wrenches: wp.array3d[wp.float64],
    base_generalized_force: wp.array2d[wp.float64],
):
    """Pull planar sample wrenches through floating-base composition.

    Args:
        base_pose: Runtime planar base poses ``(E, 3)``.
        relative_pose: Fixed-base relative sample poses ``(E, N, 3)``.
        inertial_wrenches: Absolute inertial sample cotangents ``(E, N, 3)``.
        relative_wrenches: Rotated cotangent output for the internal VJP.
        base_generalized_force: Base effort accumulator ``(E, 3)``.

    Returns:
        None. ``relative_wrenches`` is written and ``base_generalized_force``
        is accumulated in place.
    """

    environment, sample = wp.tid()
    theta = base_pose[environment, 0]
    cosine = wp.cos(theta)
    sine = wp.sin(theta)
    relative_x = relative_pose[environment, sample, 1]
    relative_y = relative_pose[environment, sample, 2]
    radius_x = cosine * relative_x - sine * relative_y
    radius_y = sine * relative_x + cosine * relative_y
    moment = inertial_wrenches[environment, sample, 0]
    force_x = inertial_wrenches[environment, sample, 1]
    force_y = inertial_wrenches[environment, sample, 2]
    relative_wrenches[environment, sample, 0] = moment
    relative_wrenches[environment, sample, 1] = cosine * force_x + sine * force_y
    relative_wrenches[environment, sample, 2] = -sine * force_x + cosine * force_y
    wp.atomic_add(
        base_generalized_force,
        environment,
        0,
        moment + radius_x * force_y - radius_y * force_x,
    )
    wp.atomic_add(base_generalized_force, environment, 1, force_x)
    wp.atomic_add(base_generalized_force, environment, 2, force_y)


@wp.kernel(enable_backward=False)
def _compose_planar_poses_kernel(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array3d[wp.float64],
    output: wp.array3d[wp.float64],
):
    environment, sample = wp.tid()
    theta = base_pose[environment, 0]
    cosine = wp.cos(theta)
    sine = wp.sin(theta)
    x = relative_pose[environment, sample, 1]
    y = relative_pose[environment, sample, 2]
    output[environment, sample, 0] = theta + relative_pose[environment, sample, 0]
    output[environment, sample, 1] = (
        base_pose[environment, 1] + cosine * x - sine * y
    )
    output[environment, sample, 2] = (
        base_pose[environment, 2] + sine * x + cosine * y
    )


@wp.kernel(enable_backward=False)
def _compose_planar_jacobians_kernel(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array3d[wp.float64],
    internal: wp.array4d[wp.float64],
    output: wp.array4d[wp.float64],
):
    environment, sample = wp.tid()
    theta = base_pose[environment, 0]
    cosine = wp.cos(theta)
    sine = wp.sin(theta)
    x_relative = relative_pose[environment, sample, 1]
    y_relative = relative_pose[environment, sample, 2]
    rx = cosine * x_relative - sine * y_relative
    ry = sine * x_relative + cosine * y_relative

    output[environment, sample, 0, 0] = wp.float64(1.0)
    output[environment, sample, 0, 1] = wp.float64(0.0)
    output[environment, sample, 0, 2] = wp.float64(0.0)
    output[environment, sample, 1, 0] = -ry
    output[environment, sample, 1, 1] = wp.float64(1.0)
    output[environment, sample, 1, 2] = wp.float64(0.0)
    output[environment, sample, 2, 0] = rx
    output[environment, sample, 2, 1] = wp.float64(0.0)
    output[environment, sample, 2, 2] = wp.float64(1.0)

    column = int(0)
    while column < internal.shape[3]:
        output[environment, sample, 0, 3 + column] = internal[
            environment, sample, 0, column
        ]
        output[environment, sample, 1, 3 + column] = (
            cosine * internal[environment, sample, 1, column]
            - sine * internal[environment, sample, 2, column]
        )
        output[environment, sample, 2, 3 + column] = (
            sine * internal[environment, sample, 1, column]
            + cosine * internal[environment, sample, 2, column]
        )
        column += 1


def launch_spatial_floating_pose_composition(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    poses: wp.array4d[wp.float64],
):
    """Compose batched spatial runtime-base and relative poses."""
    wp.launch(
        _compose_spatial_poses_kernel,
        dim=(relative_pose.shape[0], relative_pose.shape[1]),
        inputs=[base_pose, relative_pose],
        outputs=[poses],
    )


def launch_spatial_floating_jacobian_composition(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    internal: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Build batched spatial base columns and rotate internal columns."""
    wp.launch(
        _compose_spatial_jacobians_kernel,
        dim=(relative_pose.shape[0], relative_pose.shape[1]),
        inputs=[base_pose, relative_pose, internal],
        outputs=[jacobians],
    )


def launch_spatial_floating_kinematics_composition(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    internal: wp.array4d[wp.float64],
    poses: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Compose spatial poses and augmented Jacobians in one callable."""
    launch_spatial_floating_pose_composition(base_pose, relative_pose, poses)
    launch_spatial_floating_jacobian_composition(
        base_pose, relative_pose, internal, jacobians
    )


def launch_spatial_floating_kinematics_jvp_composition(
    base_pose: wp.array2d[wp.float64],
    base_velocity: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    internal_twist: wp.array3d[wp.float64],
    poses: wp.array4d[wp.float64],
    twists: wp.array3d[wp.float64],
):
    """Compose spatial poses and ``J @ qd`` without an augmented Jacobian.

    Args:
        base_pose: Runtime scalar-last base poses ``(E, 7)``.
        base_velocity: Runtime base velocities beginning in angular-linear order.
        relative_pose: Fixed-base relative sample poses ``(E, N, 4, 4)``.
        internal_twist: Relative inertial JVPs ``(E, N, 6)``.
        poses: Caller-owned absolute pose output.
        twists: Caller-owned absolute inertial JVP output.

    Returns:
        None. The composition kernel is enqueued on the active Warp stream.
    """

    wp.launch(
        _compose_spatial_poses_and_twists_kernel,
        dim=(relative_pose.shape[0], relative_pose.shape[1]),
        inputs=[base_pose, base_velocity, relative_pose, internal_twist],
        outputs=[poses, twists],
    )


def launch_spatial_floating_vjp_composition(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    inertial_wrenches: wp.array3d[wp.float64],
    relative_wrenches: wp.array3d[wp.float64],
    base_generalized_force: wp.array2d[wp.float64],
):
    """Split an augmented floating-base VJP into base and internal seeds.

    Args:
        base_pose: Runtime scalar-last base poses ``(E, 7)``.
        relative_pose: Fixed-base relative sample poses ``(E, N, 4, 4)``.
        inertial_wrenches: Absolute inertial sample cotangents ``(E, N, 6)``.
        relative_wrenches: Caller-owned rotated cotangents for the internal VJP.
        base_generalized_force: Caller-owned base effort output ``(E, 6)``.

    Returns:
        None. The composition kernel is enqueued on the active Warp stream.
    """

    base_generalized_force.zero_()
    wp.launch(
        _compose_spatial_wrench_vjp_kernel,
        dim=(relative_pose.shape[0], relative_pose.shape[1]),
        inputs=[base_pose, relative_pose, inertial_wrenches],
        outputs=[relative_wrenches, base_generalized_force],
    )


def launch_spatial_floating_vjp_assembly(
    base_generalized_force: wp.array2d[wp.float64],
    internal_generalized_force: wp.array2d[wp.float64],
    generalized_force: wp.array2d[wp.float64],
):
    """Assemble a spatial floating-base VJP without host-side concatenation.

    Args:
        base_generalized_force: Base effort in angular-linear order ``(E, 6)``.
        internal_generalized_force: Internal-coordinate effort ``(E, D)``.
        generalized_force: Caller-owned augmented output ``(E, 6 + D)``.

    Returns:
        None. The assembly kernel is enqueued on the active Warp stream.
    """

    wp.launch(
        _assemble_floating_vjp_kernel,
        dim=generalized_force.shape,
        inputs=[base_generalized_force, internal_generalized_force],
        outputs=[generalized_force],
    )


def launch_planar_floating_kinematics_jvp_composition(
    base_pose: wp.array2d[wp.float64],
    base_velocity: wp.array2d[wp.float64],
    relative_pose: wp.array3d[wp.float64],
    internal_twist: wp.array3d[wp.float64],
    poses: wp.array3d[wp.float64],
    twists: wp.array3d[wp.float64],
):
    """Compose planar poses and ``J @ qd`` without an augmented Jacobian.

    Args:
        base_pose: Runtime planar base poses ``(E, 3)``.
        base_velocity: Runtime generalized velocities beginning in
            angular-linear order.
        relative_pose: Fixed-base relative sample poses ``(E, N, 3)``.
        internal_twist: Relative inertial JVPs ``(E, N, 3)``.
        poses: Caller-owned absolute pose output.
        twists: Caller-owned absolute inertial JVP output.

    Returns:
        None. The composition kernel is enqueued on the current Warp stream.
    """

    wp.launch(
        _compose_planar_poses_and_twists_kernel,
        dim=(relative_pose.shape[0], relative_pose.shape[1]),
        inputs=[base_pose, base_velocity, relative_pose, internal_twist],
        outputs=[poses, twists],
    )


def launch_planar_floating_vjp_composition(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array3d[wp.float64],
    inertial_wrenches: wp.array3d[wp.float64],
    relative_wrenches: wp.array3d[wp.float64],
    base_generalized_force: wp.array2d[wp.float64],
):
    """Split a planar floating-base VJP into base and internal seeds.

    Args:
        base_pose: Runtime planar base poses ``(E, 3)``.
        relative_pose: Fixed-base relative sample poses ``(E, N, 3)``.
        inertial_wrenches: Absolute inertial sample cotangents ``(E, N, 3)``.
        relative_wrenches: Caller-owned rotated cotangents for the internal VJP.
        base_generalized_force: Caller-owned base effort output ``(E, 3)``.

    Returns:
        None. The composition kernel is enqueued on the current Warp stream.
    """

    base_generalized_force.zero_()
    wp.launch(
        _compose_planar_wrench_vjp_kernel,
        dim=(relative_pose.shape[0], relative_pose.shape[1]),
        inputs=[base_pose, relative_pose, inertial_wrenches],
        outputs=[relative_wrenches, base_generalized_force],
    )


def launch_planar_floating_vjp_assembly(
    base_generalized_force: wp.array2d[wp.float64],
    internal_generalized_force: wp.array2d[wp.float64],
    generalized_force: wp.array2d[wp.float64],
):
    """Assemble a planar floating-base VJP without host-side concatenation.

    Args:
        base_generalized_force: Base effort in angular-linear order ``(E, 3)``.
        internal_generalized_force: Internal-coordinate effort ``(E, D)``.
        generalized_force: Caller-owned augmented output ``(E, 3 + D)``.

    Returns:
        None. The assembly kernel is enqueued on the current Warp stream.
    """

    wp.launch(
        _assemble_floating_vjp_kernel,
        dim=generalized_force.shape,
        inputs=[base_generalized_force, internal_generalized_force],
        outputs=[generalized_force],
    )


def launch_planar_floating_pose_composition(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array3d[wp.float64],
    poses: wp.array3d[wp.float64],
):
    """Compose batched planar runtime-base and relative poses."""
    wp.launch(
        _compose_planar_poses_kernel,
        dim=(relative_pose.shape[0], relative_pose.shape[1]),
        inputs=[base_pose, relative_pose],
        outputs=[poses],
    )


def launch_planar_floating_jacobian_composition(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array3d[wp.float64],
    internal: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Build batched planar base columns and rotate internal columns."""
    wp.launch(
        _compose_planar_jacobians_kernel,
        dim=(relative_pose.shape[0], relative_pose.shape[1]),
        inputs=[base_pose, relative_pose, internal],
        outputs=[jacobians],
    )


def launch_planar_floating_kinematics_composition(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array3d[wp.float64],
    internal: wp.array4d[wp.float64],
    poses: wp.array3d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Compose planar poses and augmented Jacobians in one callable."""
    launch_planar_floating_pose_composition(base_pose, relative_pose, poses)
    launch_planar_floating_jacobian_composition(
        base_pose, relative_pose, internal, jacobians
    )


spatial_floating_pose_composition = wp.jax_callable(
    launch_spatial_floating_pose_composition, num_outputs=1
)
spatial_floating_jacobian_composition = wp.jax_callable(
    launch_spatial_floating_jacobian_composition, num_outputs=1
)
spatial_floating_kinematics_composition = wp.jax_callable(
    launch_spatial_floating_kinematics_composition, num_outputs=2
)
planar_floating_pose_composition = wp.jax_callable(
    launch_planar_floating_pose_composition, num_outputs=1
)
planar_floating_jacobian_composition = wp.jax_callable(
    launch_planar_floating_jacobian_composition, num_outputs=1
)
planar_floating_kinematics_composition = wp.jax_callable(
    launch_planar_floating_kinematics_composition, num_outputs=2
)


__all__ = [
    "launch_planar_floating_jacobian_composition",
    "launch_planar_floating_kinematics_composition",
    "launch_planar_floating_kinematics_jvp_composition",
    "launch_planar_floating_pose_composition",
    "launch_planar_floating_vjp_assembly",
    "launch_planar_floating_vjp_composition",
    "launch_spatial_floating_jacobian_composition",
    "launch_spatial_floating_kinematics_composition",
    "launch_spatial_floating_kinematics_jvp_composition",
    "launch_spatial_floating_pose_composition",
    "launch_spatial_floating_vjp_composition",
    "launch_spatial_floating_vjp_assembly",
    "planar_floating_jacobian_composition",
    "planar_floating_kinematics_composition",
    "planar_floating_pose_composition",
    "spatial_floating_jacobian_composition",
    "spatial_floating_kinematics_composition",
    "spatial_floating_pose_composition",
]
