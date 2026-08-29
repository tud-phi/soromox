# ruff: noqa: I001, UP018
"""Warp-native runtime-base composition for continuum kinematics."""

from __future__ import annotations

import warp as wp

from soromox.systems.execution.warp.common.floating_base import (
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
    "launch_planar_floating_pose_composition",
    "launch_spatial_floating_jacobian_composition",
    "launch_spatial_floating_kinematics_composition",
    "launch_spatial_floating_pose_composition",
    "planar_floating_jacobian_composition",
    "planar_floating_kinematics_composition",
    "planar_floating_pose_composition",
    "spatial_floating_jacobian_composition",
    "spatial_floating_kinematics_composition",
    "spatial_floating_pose_composition",
]
