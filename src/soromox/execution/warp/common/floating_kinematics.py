# ruff: noqa: I001, UP018
"""Warp-native runtime-base composition for continuum kinematics."""

from __future__ import annotations

import warp as wp

from soromox.execution.warp.common.rotations import (
    _quaternion_rotation_matrix,
    _quaternion_rotation_transpose_entry,
)
from soromox.execution.warp.common.so2 import (
    _rotate_vector,
    _rotation_matrix,
)

wp.set_module_options({"enable_backward": False})


@wp.func
def _compose_spatial_poses_at(
    environment: int,
    sample: int,
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    poses: wp.array4d[wp.float64],
):
    """Compose batched spatial base poses and base-relative transforms.

    Args:
        base_pose: Scalar-last base poses with shape ``(E, 7)``.
        relative_pose: Base-relative transforms with shape ``(E, N, 4, 4)``.
        poses: Inertial-frame transform output with shape ``(E, N, 4, 4)``.

    Returns:
        None. One transform in ``poses`` is written per thread.
    """
    base_rotation = _quaternion_rotation_matrix(base_pose, environment)
    row = int(0)
    while row < 3:
        column = int(0)
        while column < 3:
            value = wp.float64(0.0)
            inner = int(0)
            while inner < 3:
                value += (
                    base_rotation[row, inner]
                    * relative_pose[environment, sample, inner, column]
                )
                inner += 1
            poses[environment, sample, row, column] = value
            column += 1
        position = base_pose[environment, 4 + row]
        inner = int(0)
        while inner < 3:
            position += (
                base_rotation[row, inner] * relative_pose[environment, sample, inner, 3]
            )
            inner += 1
        poses[environment, sample, row, 3] = position
        row += 1
    column = int(0)
    while column < 3:
        poses[environment, sample, 3, column] = wp.float64(0.0)
        column += 1
    poses[environment, sample, 3, 3] = wp.float64(1.0)


@wp.kernel(enable_backward=False)
def _compose_spatial_poses_kernel(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    poses: wp.array4d[wp.float64],
):
    """Compose batched spatial base poses and base-relative transforms.

    Args:
        base_pose: Scalar-last base poses with shape ``(E, 7)``.
        relative_pose: Base-relative transforms with shape ``(E, N, 4, 4)``.
        poses: Inertial-frame transform output with shape ``(E, N, 4, 4)``.

    Returns:
        None. One transform in ``poses`` is written per thread.
    """
    environment, sample = wp.tid()
    _compose_spatial_poses_at(environment, sample, base_pose, relative_pose, poses)


@wp.struct
class ComposeSpatialPosesArrays:
    base_pose: wp.array2d[wp.float64]
    relative_pose: wp.array4d[wp.float64]
    poses: wp.array4d[wp.float64]


@wp.kernel(enable_backward=False)
def packed_compose_spatial_poses_kernel(
    worlds: wp.array(dtype=ComposeSpatialPosesArrays), work: wp.array(dtype=wp.vec3i)
):
    item = wp.tid()
    entry = work[item]
    d = worlds[entry[0]]
    _compose_spatial_poses_at(0, entry[1], d.base_pose, d.relative_pose, d.poses)


@wp.func
def _compose_spatial_jacobians_at(
    environment: int,
    sample: int,
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    jacobians_internal: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Compose spatial floating-base Jacobians on one thread per sample.

    Args:
        base_pose: Scalar-last base poses with shape ``(E, 7)``.
        relative_pose: Base-relative transforms with shape ``(E, N, 4, 4)``.
        jacobians_internal: Base-frame internal Jacobians with shape
            ``(E, N, 6, D)``.
        jacobians: Inertial-frame augmented output with shape
            ``(E, N, 6, D + 6)``.

    Returns:
        None. One augmented Jacobian in ``jacobians`` is written per thread.
    """
    base_rotation = _quaternion_rotation_matrix(base_pose, environment)
    offset_x = wp.float64(0.0)
    offset_y = wp.float64(0.0)
    offset_z = wp.float64(0.0)
    inner = int(0)
    while inner < 3:
        offset_x += (
            base_rotation[0, inner] * relative_pose[environment, sample, inner, 3]
        )
        offset_y += (
            base_rotation[1, inner] * relative_pose[environment, sample, inner, 3]
        )
        offset_z += (
            base_rotation[2, inner] * relative_pose[environment, sample, inner, 3]
        )
        inner += 1
    row = int(0)
    while row < 6:
        column = int(0)
        while column < 6:
            value = wp.float64(0.0)
            if (row < 3 and column < 3 or (row >= 3 and column >= 3)) and row == column:
                value = wp.float64(1.0)
            elif row == 3 and column == 1:
                value = offset_z
            elif row == 3 and column == 2:
                value = -offset_y
            elif row == 4 and column == 0:
                value = -offset_z
            elif row == 4 and column == 2:
                value = offset_x
            elif row == 5 and column == 0:
                value = offset_y
            elif row == 5 and column == 1:
                value = -offset_x
            jacobians[environment, sample, row, column] = value
            column += 1
        internal_column = int(0)
        while internal_column < jacobians_internal.shape[3]:
            value = wp.float64(0.0)
            inner = int(0)
            while inner < 3:
                value += (
                    base_rotation[row % 3, inner]
                    * jacobians_internal[
                        environment,
                        sample,
                        (0 if row < 3 else 3) + inner,
                        internal_column,
                    ]
                )
                inner += 1
            jacobians[environment, sample, row, 6 + internal_column] = value
            internal_column += 1
        row += 1


@wp.kernel(enable_backward=False)
def _compose_spatial_jacobians_kernel(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    jacobians_internal: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Compose spatial floating-base Jacobians on one thread per sample.

    Args:
        base_pose: Scalar-last base poses with shape ``(E, 7)``.
        relative_pose: Base-relative transforms with shape ``(E, N, 4, 4)``.
        jacobians_internal: Base-frame internal Jacobians with shape
            ``(E, N, 6, D)``.
        jacobians: Inertial-frame augmented output with shape
            ``(E, N, 6, D + 6)``.

    Returns:
        None. One augmented Jacobian in ``jacobians`` is written per thread.
    """
    environment, sample = wp.tid()
    _compose_spatial_jacobians_at(
        environment, sample, base_pose, relative_pose, jacobians_internal, jacobians
    )


@wp.struct
class ComposeSpatialJacobiansArrays:
    base_pose: wp.array2d[wp.float64]
    relative_pose: wp.array4d[wp.float64]
    jacobians_internal: wp.array4d[wp.float64]
    jacobians: wp.array4d[wp.float64]


@wp.kernel(enable_backward=False)
def packed_compose_spatial_jacobians_kernel(
    worlds: wp.array(dtype=ComposeSpatialJacobiansArrays),
    work: wp.array(dtype=wp.vec3i),
):
    item = wp.tid()
    entry = work[item]
    d = worlds[entry[0]]
    _compose_spatial_jacobians_at(
        0, entry[1], d.base_pose, d.relative_pose, d.jacobians_internal, d.jacobians
    )


@wp.func
def _compose_spatial_jacobians_tiled_at(
    work_item: int,
    lane: int,
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    jacobians_internal: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Compose one floating spatial Jacobian cooperatively per sample.

    Args:
        base_pose: Batched scalar-last quaternion poses with shape ``(E, 7)``.
        relative_pose: Base-relative transforms with shape ``(E, N, 4, 4)``.
        jacobians_internal: Base-frame internal Jacobians with shape
            ``(E, N, 6, D)``.
        jacobians: Inertial-frame augmented Jacobians with shape
            ``(E, N, 6, D + 6)``.

    Returns:
        None. ``jacobians`` is updated in place by one cooperative block for
        each environment-sample pair.
    """
    sample_count = relative_pose.shape[1]
    environment = work_item // sample_count
    sample = work_item - environment * sample_count
    base_rotation = wp.tile_zeros(shape=(9,), dtype=wp.float64, storage="shared")
    world_offset = wp.tile_zeros(shape=(3,), dtype=wp.float64, storage="shared")
    rotation_entry = wp.float64(0.0)
    if lane < 9:
        rotation_row = lane // 3
        rotation_column = lane - 3 * rotation_row
        rotation_entry = _quaternion_rotation_transpose_entry(
            base_pose, environment, rotation_column, rotation_row
        )
    wp.tile_scatter_masked(base_rotation, lane, rotation_entry, lane < 9)
    world_offset_entry = wp.float64(0.0)
    if lane < 3:
        inner = int(0)
        while inner < 3:
            world_offset_entry += (
                base_rotation[3 * lane + inner]
                * relative_pose[environment, sample, inner, 3]
            )
            inner += 1
    wp.tile_scatter_masked(world_offset, lane, world_offset_entry, lane < 3)
    entry = lane
    column_count = jacobians.shape[3]
    while entry < 6 * column_count:
        row = entry // column_count
        column = entry - row * column_count
        value = wp.float64(0.0)
        if column < 6:
            if row == column and (row < 3 and column < 3 or (row >= 3 and column >= 3)):
                value = wp.float64(1.0)
            elif row == 3 and column == 1:
                value = world_offset[2]
            elif row == 3 and column == 2:
                value = -world_offset[1]
            elif row == 4 and column == 0:
                value = -world_offset[2]
            elif row == 4 and column == 2:
                value = world_offset[0]
            elif row == 5 and column == 0:
                value = world_offset[1]
            elif row == 5 and column == 1:
                value = -world_offset[0]
        else:
            internal_column = column - 6
            inner = int(0)
            while inner < 3:
                value += (
                    base_rotation[3 * (row % 3) + inner]
                    * jacobians_internal[
                        environment,
                        sample,
                        (0 if row < 3 else 3) + inner,
                        internal_column,
                    ]
                )
                inner += 1
        jacobians[environment, sample, row, column] = value
        entry += wp.block_dim()


@wp.kernel(enable_backward=False)
def _compose_spatial_jacobians_tiled_kernel(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    jacobians_internal: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Compose one floating spatial Jacobian cooperatively per sample.

    Args:
        base_pose: Batched scalar-last quaternion poses with shape ``(E, 7)``.
        relative_pose: Base-relative transforms with shape ``(E, N, 4, 4)``.
        jacobians_internal: Base-frame internal Jacobians with shape
            ``(E, N, 6, D)``.
        jacobians: Inertial-frame augmented Jacobians with shape
            ``(E, N, 6, D + 6)``.

    Returns:
        None. ``jacobians`` is updated in place by one cooperative block for
        each environment-sample pair.
    """
    work_item, lane = wp.tid()
    _compose_spatial_jacobians_tiled_at(
        work_item, lane, base_pose, relative_pose, jacobians_internal, jacobians
    )


@wp.struct
class ComposeSpatialJacobiansTiledArrays:
    base_pose: wp.array2d[wp.float64]
    relative_pose: wp.array4d[wp.float64]
    jacobians_internal: wp.array4d[wp.float64]
    jacobians: wp.array4d[wp.float64]


@wp.kernel(enable_backward=False)
def packed_compose_spatial_jacobians_tiled_kernel(
    worlds: wp.array(dtype=ComposeSpatialJacobiansTiledArrays),
    work: wp.array(dtype=wp.vec3i),
):
    item, lane = wp.tid()
    entry = work[item]
    d = worlds[entry[0]]
    _compose_spatial_jacobians_tiled_at(
        entry[1], lane, d.base_pose, d.relative_pose, d.jacobians_internal, d.jacobians
    )


@wp.func
def _compose_spatial_poses_and_twists_at(
    environment: int,
    sample: int,
    base_pose: wp.array2d[wp.float64],
    qd: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    twists_internal: wp.array3d[wp.float64],
    poses: wp.array4d[wp.float64],
    twists: wp.array3d[wp.float64],
):
    """Compose floating poses and one augmented Jacobian direction.

    Args:
        base_pose: Runtime scalar-last base poses ``(E, 7)``.
        qd: Batched generalized-velocity directions. The first six entries are
            the angular-linear world-frame base direction.
        relative_pose: Fixed-base relative sample poses ``(E, N, 4, 4)``.
        twists_internal: Base-frame internal-Jacobian products ``(E, N, 6)``.
        poses: Absolute pose output ``(E, N, 4, 4)``.
        twists: Absolute inertial JVP output ``(E, N, 6)``.

    Returns:
        None. ``poses`` and ``twists`` are written in place.
    """
    base_rotation = _quaternion_rotation_matrix(base_pose, environment)
    offset_x = wp.float64(0.0)
    offset_y = wp.float64(0.0)
    offset_z = wp.float64(0.0)
    inner = int(0)
    while inner < 3:
        offset_x += (
            base_rotation[0, inner] * relative_pose[environment, sample, inner, 3]
        )
        offset_y += (
            base_rotation[1, inner] * relative_pose[environment, sample, inner, 3]
        )
        offset_z += (
            base_rotation[2, inner] * relative_pose[environment, sample, inner, 3]
        )
        inner += 1
    world_offset = wp.vec3d(offset_x, offset_y, offset_z)
    base_angular_velocity = wp.vec3d(
        qd[environment, 0], qd[environment, 1], qd[environment, 2]
    )
    base_linear_velocity = wp.vec3d(
        qd[environment, 3], qd[environment, 4], qd[environment, 5]
    )
    row = int(0)
    while row < 3:
        column = int(0)
        while column < 3:
            value = wp.float64(0.0)
            inner = int(0)
            while inner < 3:
                value += (
                    base_rotation[row, inner]
                    * relative_pose[environment, sample, inner, column]
                )
                inner += 1
            poses[environment, sample, row, column] = value
            column += 1
        poses[environment, sample, row, 3] = (
            base_pose[environment, 4 + row] + world_offset[row]
        )
        row += 1
    column = int(0)
    while column < 3:
        poses[environment, sample, 3, column] = wp.float64(0.0)
        column += 1
    poses[environment, sample, 3, 3] = wp.float64(1.0)
    base_point_velocity = base_linear_velocity + wp.cross(
        base_angular_velocity, world_offset
    )
    row = int(0)
    while row < 3:
        angular_velocity = base_angular_velocity[row]
        linear_velocity = base_point_velocity[row]
        inner = int(0)
        while inner < 3:
            rotation_entry = base_rotation[row, inner]
            angular_velocity += (
                rotation_entry * twists_internal[environment, sample, inner]
            )
            linear_velocity += (
                rotation_entry * twists_internal[environment, sample, 3 + inner]
            )
            inner += 1
        twists[environment, sample, row] = angular_velocity
        twists[environment, sample, 3 + row] = linear_velocity
        row += 1


@wp.kernel(enable_backward=False)
def _compose_spatial_poses_and_twists_kernel(
    base_pose: wp.array2d[wp.float64],
    qd: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    twists_internal: wp.array3d[wp.float64],
    poses: wp.array4d[wp.float64],
    twists: wp.array3d[wp.float64],
):
    """Compose floating poses and one augmented Jacobian direction.

    Args:
        base_pose: Runtime scalar-last base poses ``(E, 7)``.
        qd: Batched generalized-velocity directions. The first six entries are
            the angular-linear world-frame base direction.
        relative_pose: Fixed-base relative sample poses ``(E, N, 4, 4)``.
        twists_internal: Base-frame internal-Jacobian products ``(E, N, 6)``.
        poses: Absolute pose output ``(E, N, 4, 4)``.
        twists: Absolute inertial JVP output ``(E, N, 6)``.

    Returns:
        None. ``poses`` and ``twists`` are written in place.
    """
    environment, sample = wp.tid()
    _compose_spatial_poses_and_twists_at(
        environment,
        sample,
        base_pose,
        qd,
        relative_pose,
        twists_internal,
        poses,
        twists,
    )


@wp.struct
class ComposeSpatialPosesAndTwistsArrays:
    base_pose: wp.array2d[wp.float64]
    qd: wp.array2d[wp.float64]
    relative_pose: wp.array4d[wp.float64]
    twists_internal: wp.array3d[wp.float64]
    poses: wp.array4d[wp.float64]
    twists: wp.array3d[wp.float64]


@wp.kernel(enable_backward=False)
def packed_compose_spatial_poses_and_twists_kernel(
    worlds: wp.array(dtype=ComposeSpatialPosesAndTwistsArrays),
    work: wp.array(dtype=wp.vec3i),
):
    item = wp.tid()
    entry = work[item]
    d = worlds[entry[0]]
    _compose_spatial_poses_and_twists_at(
        0,
        entry[1],
        d.base_pose,
        d.qd,
        d.relative_pose,
        d.twists_internal,
        d.poses,
        d.twists,
    )


@wp.func
def _compose_spatial_wrench_vjp_at(
    environment: int,
    sample: int,
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
        relative_wrenches: Rotated cotangent output for the internal VJP with
            shape ``(E, N, 6)``.
        base_generalized_force: Base effort accumulator ``(E, 6)``.

    Returns:
        None. ``relative_wrenches`` is written and ``base_generalized_force``
        is accumulated in place.
    """
    base_rotation = _quaternion_rotation_matrix(base_pose, environment)
    world_offset = wp.vec3d()
    row = int(0)
    while row < 3:
        inner = int(0)
        while inner < 3:
            world_offset[row] += (
                base_rotation[row, inner] * relative_pose[environment, sample, inner, 3]
            )
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
    base_moment = moment + wp.cross(world_offset, force)
    row = int(0)
    while row < 3:
        relative_moment = wp.float64(0.0)
        relative_force = wp.float64(0.0)
        inner = int(0)
        while inner < 3:
            rotation_entry = base_rotation[inner, row]
            relative_moment += rotation_entry * moment[inner]
            relative_force += rotation_entry * force[inner]
            inner += 1
        relative_wrenches[environment, sample, row] = relative_moment
        relative_wrenches[environment, sample, 3 + row] = relative_force
        wp.atomic_add(base_generalized_force, environment, row, base_moment[row])
        wp.atomic_add(base_generalized_force, environment, 3 + row, force[row])
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
        relative_wrenches: Rotated cotangent output for the internal VJP with
            shape ``(E, N, 6)``.
        base_generalized_force: Base effort accumulator ``(E, 6)``.

    Returns:
        None. ``relative_wrenches`` is written and ``base_generalized_force``
        is accumulated in place.
    """
    environment, sample = wp.tid()
    _compose_spatial_wrench_vjp_at(
        environment,
        sample,
        base_pose,
        relative_pose,
        inertial_wrenches,
        relative_wrenches,
        base_generalized_force,
    )


@wp.struct
class ComposeSpatialWrenchVjpArrays:
    base_pose: wp.array2d[wp.float64]
    relative_pose: wp.array4d[wp.float64]
    inertial_wrenches: wp.array3d[wp.float64]
    relative_wrenches: wp.array3d[wp.float64]
    base_generalized_force: wp.array2d[wp.float64]


@wp.kernel(enable_backward=False)
def packed_compose_spatial_wrench_vjp_kernel(
    worlds: wp.array(dtype=ComposeSpatialWrenchVjpArrays),
    work: wp.array(dtype=wp.vec3i),
):
    item = wp.tid()
    entry = work[item]
    d = worlds[entry[0]]
    _compose_spatial_wrench_vjp_at(
        0,
        entry[1],
        d.base_pose,
        d.relative_pose,
        d.inertial_wrenches,
        d.relative_wrenches,
        d.base_generalized_force,
    )


@wp.func
def _assemble_floating_vjp_at(
    environment: int,
    column: int,
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
    _assemble_floating_vjp_at(
        environment,
        column,
        base_generalized_force,
        internal_generalized_force,
        generalized_force,
    )


@wp.struct
class AssembleFloatingVjpArrays:
    base_generalized_force: wp.array2d[wp.float64]
    internal_generalized_force: wp.array2d[wp.float64]
    generalized_force: wp.array2d[wp.float64]


@wp.kernel(enable_backward=False)
def packed_assemble_floating_vjp_kernel(
    worlds: wp.array(dtype=AssembleFloatingVjpArrays), work: wp.array(dtype=wp.vec3i)
):
    item = wp.tid()
    entry = work[item]
    d = worlds[entry[0]]
    _assemble_floating_vjp_at(
        0,
        entry[1],
        d.base_generalized_force,
        d.internal_generalized_force,
        d.generalized_force,
    )


@wp.kernel(enable_backward=False)
def _compose_planar_poses_and_twists_kernel(
    base_pose: wp.array2d[wp.float64],
    qd: wp.array2d[wp.float64],
    relative_pose: wp.array3d[wp.float64],
    twists_internal: wp.array3d[wp.float64],
    poses: wp.array3d[wp.float64],
    twists: wp.array3d[wp.float64],
):
    """Compose planar floating poses and one augmented Jacobian direction.

    Args:
        base_pose: Runtime planar base poses ``(E, 3)``.
        qd: Batched generalized-velocity directions. The first three entries
            are the angular-linear world-frame base direction.
        relative_pose: Fixed-base relative sample poses ``(E, N, 3)``.
        twists_internal: Base-frame internal-Jacobian products ``(E, N, 3)``.
        poses: Absolute pose output ``(E, N, 3)``.
        twists: Absolute inertial JVP output ``(E, N, 3)``.

    Returns:
        None. ``poses`` and ``twists`` are written in place.
    """

    environment, sample = wp.tid()
    theta = base_pose[environment, 0]
    base_rotation = _rotation_matrix(theta)
    world_offset = base_rotation * wp.vec2d(
        relative_pose[environment, sample, 1],
        relative_pose[environment, sample, 2],
    )
    internal_linear_velocity = base_rotation * wp.vec2d(
        twists_internal[environment, sample, 1],
        twists_internal[environment, sample, 2],
    )
    base_angular_velocity = qd[environment, 0]
    poses[environment, sample, 0] = theta + relative_pose[environment, sample, 0]
    poses[environment, sample, 1] = base_pose[environment, 1] + world_offset[0]
    poses[environment, sample, 2] = base_pose[environment, 2] + world_offset[1]
    twists[environment, sample, 0] = (
        base_angular_velocity + twists_internal[environment, sample, 0]
    )
    twists[environment, sample, 1] = (
        qd[environment, 1]
        - base_angular_velocity * world_offset[1]
        + internal_linear_velocity[0]
    )
    twists[environment, sample, 2] = (
        qd[environment, 2]
        + base_angular_velocity * world_offset[0]
        + internal_linear_velocity[1]
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
        relative_wrenches: Rotated cotangent output for the internal VJP with
            shape ``(E, N, 3)``.
        base_generalized_force: Base effort accumulator ``(E, 3)``.

    Returns:
        None. ``relative_wrenches`` is written and ``base_generalized_force``
        is accumulated in place.
    """

    environment, sample = wp.tid()
    theta = base_pose[environment, 0]
    base_rotation = _rotation_matrix(theta)
    world_offset = base_rotation * wp.vec2d(
        relative_pose[environment, sample, 1],
        relative_pose[environment, sample, 2],
    )
    moment = inertial_wrenches[environment, sample, 0]
    force_x = inertial_wrenches[environment, sample, 1]
    force_y = inertial_wrenches[environment, sample, 2]
    relative_force = wp.vec2d(
        base_rotation[0, 0] * force_x + base_rotation[1, 0] * force_y,
        base_rotation[0, 1] * force_x + base_rotation[1, 1] * force_y,
    )
    relative_wrenches[environment, sample, 0] = moment
    relative_wrenches[environment, sample, 1] = relative_force[0]
    relative_wrenches[environment, sample, 2] = relative_force[1]
    wp.atomic_add(
        base_generalized_force,
        environment,
        0,
        moment + world_offset[0] * force_y - world_offset[1] * force_x,
    )
    wp.atomic_add(base_generalized_force, environment, 1, force_x)
    wp.atomic_add(base_generalized_force, environment, 2, force_y)


@wp.kernel(enable_backward=False)
def _compose_planar_poses_kernel(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array3d[wp.float64],
    poses: wp.array3d[wp.float64],
):
    """Compose batched planar base poses and base-relative poses.

    Args:
        base_pose: Planar base poses with shape ``(E, 3)``.
        relative_pose: Base-relative poses with shape ``(E, N, 3)``.
        poses: Inertial-frame pose output with shape ``(E, N, 3)``.

    Returns:
        None. One pose in ``poses`` is written per thread.
    """
    environment, sample = wp.tid()
    theta = base_pose[environment, 0]
    world_offset = _rotate_vector(
        theta,
        wp.vec2d(
            relative_pose[environment, sample, 1],
            relative_pose[environment, sample, 2],
        ),
    )
    poses[environment, sample, 0] = theta + relative_pose[environment, sample, 0]
    poses[environment, sample, 1] = base_pose[environment, 1] + world_offset[0]
    poses[environment, sample, 2] = base_pose[environment, 2] + world_offset[1]


@wp.kernel(enable_backward=False)
def _compose_planar_jacobians_kernel(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array3d[wp.float64],
    jacobians_internal: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Compose planar floating-base Jacobians on one thread per sample.

    Args:
        base_pose: Planar base poses with shape ``(E, 3)``.
        relative_pose: Base-relative poses with shape ``(E, N, 3)``.
        jacobians_internal: Base-frame internal Jacobians with shape
            ``(E, N, 3, D)``.
        jacobians: Inertial-frame augmented output with shape
            ``(E, N, 3, D + 3)``.

    Returns:
        None. One augmented Jacobian in ``jacobians`` is written per thread.
    """
    environment, sample = wp.tid()
    theta = base_pose[environment, 0]
    base_rotation = _rotation_matrix(theta)
    world_offset = base_rotation * wp.vec2d(
        relative_pose[environment, sample, 1],
        relative_pose[environment, sample, 2],
    )

    jacobians[environment, sample, 0, 0] = wp.float64(1.0)
    jacobians[environment, sample, 0, 1] = wp.float64(0.0)
    jacobians[environment, sample, 0, 2] = wp.float64(0.0)
    jacobians[environment, sample, 1, 0] = -world_offset[1]
    jacobians[environment, sample, 1, 1] = wp.float64(1.0)
    jacobians[environment, sample, 1, 2] = wp.float64(0.0)
    jacobians[environment, sample, 2, 0] = world_offset[0]
    jacobians[environment, sample, 2, 1] = wp.float64(0.0)
    jacobians[environment, sample, 2, 2] = wp.float64(1.0)

    column = int(0)
    while column < jacobians_internal.shape[3]:
        jacobians[environment, sample, 0, 3 + column] = jacobians_internal[
            environment, sample, 0, column
        ]
        jacobian_linear_world = base_rotation * wp.vec2d(
            jacobians_internal[environment, sample, 1, column],
            jacobians_internal[environment, sample, 2, column],
        )
        jacobians[environment, sample, 1, 3 + column] = jacobian_linear_world[0]
        jacobians[environment, sample, 2, 3 + column] = jacobian_linear_world[1]
        column += 1


def launch_spatial_floating_pose_composition(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    poses: wp.array4d[wp.float64],
):
    """Compose batched spatial runtime-base and relative poses.

    Args:
        base_pose: Scalar-last base poses with shape ``(E, 7)``.
        relative_pose: Base-relative transforms with shape ``(E, N, 4, 4)``.
        poses: Caller-owned inertial-frame transforms with shape
            ``(E, N, 4, 4)``.

    Returns:
        None. The pose kernel is enqueued on the active Warp stream.
    """
    wp.launch(
        _compose_spatial_poses_kernel,
        dim=(relative_pose.shape[0], relative_pose.shape[1]),
        inputs=[base_pose, relative_pose],
        outputs=[poses],
    )


def launch_spatial_floating_jacobian_composition(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    jacobians_internal: wp.array4d[wp.float64],
    block_dim: int,
    jacobians: wp.array4d[wp.float64],
):
    """Build batched spatial base columns and rotate internal columns.

    Args:
        base_pose: Batched scalar-last quaternion poses with shape ``(E, 7)``.
        relative_pose: Base-relative transforms with shape ``(E, N, 4, 4)``.
        jacobians_internal: Base-frame internal Jacobians with shape
            ``(E, N, 6, D)``.
        block_dim: Model-configured CUDA threads per cooperative block.
        jacobians: Inertial-frame augmented output with shape
            ``(E, N, 6, D + 6)``.

    Returns:
        None. ``jacobians`` is updated in place.
    """
    if base_pose.device.is_cuda:
        wp.launch_tiled(
            _compose_spatial_jacobians_tiled_kernel,
            dim=relative_pose.shape[0] * relative_pose.shape[1],
            inputs=[base_pose, relative_pose, jacobians_internal],
            outputs=[jacobians],
            block_dim=block_dim,
        )
    else:
        wp.launch(
            _compose_spatial_jacobians_kernel,
            dim=(relative_pose.shape[0], relative_pose.shape[1]),
            inputs=[base_pose, relative_pose, jacobians_internal],
            outputs=[jacobians],
        )


def launch_spatial_floating_kinematics_composition(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    jacobians_internal: wp.array4d[wp.float64],
    block_dim: int,
    poses: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Compose spatial poses and augmented Jacobians in one callable.

    Args:
        base_pose: Scalar-last base poses with shape ``(E, 7)``.
        relative_pose: Base-relative transforms with shape ``(E, N, 4, 4)``.
        jacobians_internal: Base-frame internal Jacobians with shape
            ``(E, N, 6, D)``.
        block_dim: Model-configured CUDA threads per cooperative block.
        poses: Caller-owned inertial-frame transforms with shape
            ``(E, N, 4, 4)``.
        jacobians: Caller-owned inertial-frame augmented Jacobians with shape
            ``(E, N, 6, D + 6)``.

    Returns:
        None. The pose and Jacobian kernels are enqueued on the active Warp
        stream.
    """
    launch_spatial_floating_pose_composition(base_pose, relative_pose, poses)
    launch_spatial_floating_jacobian_composition(
        base_pose, relative_pose, jacobians_internal, block_dim, jacobians
    )


def launch_spatial_floating_kinematics_jvp_composition(
    base_pose: wp.array2d[wp.float64],
    qd: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    twists_internal: wp.array3d[wp.float64],
    poses: wp.array4d[wp.float64],
    twists: wp.array3d[wp.float64],
):
    """Compose spatial poses and ``J @ qd`` without an augmented Jacobian.

    Args:
        base_pose: Runtime scalar-last base poses ``(E, 7)``.
        qd: Batched generalized-velocity directions. The first six entries are
            the angular-linear world-frame base direction.
        relative_pose: Fixed-base relative sample poses ``(E, N, 4, 4)``.
        twists_internal: Base-frame internal-Jacobian products ``(E, N, 6)``.
        poses: Caller-owned inertial-frame transforms ``(E, N, 4, 4)``.
        twists: Caller-owned inertial-frame JVP output ``(E, N, 6)``.

    Returns:
        None. The composition kernel is enqueued on the active Warp stream.
    """

    wp.launch(
        _compose_spatial_poses_and_twists_kernel,
        dim=(relative_pose.shape[0], relative_pose.shape[1]),
        inputs=[base_pose, qd, relative_pose, twists_internal],
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
        relative_wrenches: Caller-owned rotated cotangents for the internal VJP
            with shape ``(E, N, 6)``.
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
    qd: wp.array2d[wp.float64],
    relative_pose: wp.array3d[wp.float64],
    twists_internal: wp.array3d[wp.float64],
    poses: wp.array3d[wp.float64],
    twists: wp.array3d[wp.float64],
):
    """Compose planar poses and ``J @ qd`` without an augmented Jacobian.

    Args:
        base_pose: Runtime planar base poses ``(E, 3)``.
        qd: Batched generalized-velocity directions. The first three entries
            are the angular-linear world-frame base direction.
        relative_pose: Fixed-base relative sample poses ``(E, N, 3)``.
        twists_internal: Base-frame internal-Jacobian products ``(E, N, 3)``.
        poses: Caller-owned inertial-frame poses ``(E, N, 3)``.
        twists: Caller-owned inertial-frame JVP output ``(E, N, 3)``.

    Returns:
        None. The composition kernel is enqueued on the current Warp stream.
    """

    wp.launch(
        _compose_planar_poses_and_twists_kernel,
        dim=(relative_pose.shape[0], relative_pose.shape[1]),
        inputs=[base_pose, qd, relative_pose, twists_internal],
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
        relative_wrenches: Caller-owned rotated cotangents for the internal VJP
            with shape ``(E, N, 3)``.
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
    """Compose batched planar runtime-base and relative poses.

    Args:
        base_pose: Planar base poses with shape ``(E, 3)``.
        relative_pose: Base-relative poses with shape ``(E, N, 3)``.
        poses: Caller-owned inertial-frame poses with shape ``(E, N, 3)``.

    Returns:
        None. The pose kernel is enqueued on the active Warp stream.
    """
    wp.launch(
        _compose_planar_poses_kernel,
        dim=(relative_pose.shape[0], relative_pose.shape[1]),
        inputs=[base_pose, relative_pose],
        outputs=[poses],
    )


def launch_planar_floating_jacobian_composition(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array3d[wp.float64],
    jacobians_internal: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Build batched planar base columns and rotate internal columns.

    Args:
        base_pose: Planar base poses with shape ``(E, 3)``.
        relative_pose: Base-relative poses with shape ``(E, N, 3)``.
        jacobians_internal: Base-frame internal Jacobians with shape
            ``(E, N, 3, D)``.
        jacobians: Caller-owned inertial-frame augmented Jacobians with shape
            ``(E, N, 3, D + 3)``.

    Returns:
        None. The Jacobian kernel is enqueued on the active Warp stream.
    """
    wp.launch(
        _compose_planar_jacobians_kernel,
        dim=(relative_pose.shape[0], relative_pose.shape[1]),
        inputs=[base_pose, relative_pose, jacobians_internal],
        outputs=[jacobians],
    )


def launch_planar_floating_kinematics_composition(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array3d[wp.float64],
    jacobians_internal: wp.array4d[wp.float64],
    poses: wp.array3d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Compose planar poses and augmented Jacobians in one callable.

    Args:
        base_pose: Planar base poses with shape ``(E, 3)``.
        relative_pose: Base-relative poses with shape ``(E, N, 3)``.
        jacobians_internal: Base-frame internal Jacobians with shape
            ``(E, N, 3, D)``.
        poses: Caller-owned inertial-frame poses with shape ``(E, N, 3)``.
        jacobians: Caller-owned inertial-frame augmented Jacobians with shape
            ``(E, N, 3, D + 3)``.

    Returns:
        None. The pose and Jacobian kernels are enqueued on the active Warp
        stream.
    """
    launch_planar_floating_pose_composition(base_pose, relative_pose, poses)
    launch_planar_floating_jacobian_composition(
        base_pose, relative_pose, jacobians_internal, jacobians
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


__all__ += [
    "ComposeSpatialPosesArrays",
    "packed_compose_spatial_poses_kernel",
    "ComposeSpatialJacobiansArrays",
    "packed_compose_spatial_jacobians_kernel",
    "ComposeSpatialJacobiansTiledArrays",
    "packed_compose_spatial_jacobians_tiled_kernel",
    "ComposeSpatialPosesAndTwistsArrays",
    "packed_compose_spatial_poses_and_twists_kernel",
    "ComposeSpatialWrenchVjpArrays",
    "packed_compose_spatial_wrench_vjp_kernel",
    "AssembleFloatingVjpArrays",
    "packed_assemble_floating_vjp_kernel",
]
