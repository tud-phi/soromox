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

    Explicit indices address the supplied array views. Homogeneous wrappers pass the
    full batch; packed wrappers pass one world's views and local indices.

    Args:
        environment: Batch row in the supplied arrays; use zero for a single-world view.
        sample: Sample index within the selected environment.
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
def homogeneous_compose_spatial_poses_kernel(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    poses: wp.array4d[wp.float64],
):
    """Compose floating-base poses for a homogeneous batch.

    All worlds use the same structural dimensions and coordinate layout. Dimensions:
    E = worlds, N = samples.

    All array arguments carry a world axis and their values may differ by world.
    Sample and coordinate counts are common to the batch.

    Launch with ``dim=(E, N)``: one thread per world/sample pair.

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
class SpatialPoseCompositionData:
    """Base poses, relative poses and composed spatial pose outputs for one world.

    Array fields are non-owning views; the caller must retain their allocations
    for every launch and captured graph. Batched fields use one environment
    (``E = 1``); structural fields use this world's own dimensions.

    Attributes:
        base_pose: Scalar-last base poses with shape ``(E, 7)``.
        relative_pose: Base-relative transforms with shape ``(E, N, 4, 4)``.
        poses: Inertial-frame transform output with shape ``(E, N, 4, 4)``.
    """

    base_pose: wp.array2d[wp.float64]
    relative_pose: wp.array4d[wp.float64]
    poses: wp.array4d[wp.float64]


@wp.kernel(enable_backward=False)
def packed_compose_spatial_poses_kernel(
    world_data: wp.array(dtype=SpatialPoseCompositionData),
    work_items: wp.array(dtype=wp.vec3i),
):
    """Compose batched spatial base poses and base-relative transforms using packed world data.

    Each selected descriptor contains one world (E = 1). Structure and numerical
    model parameters may differ between worlds. Descriptors may share allocations
    when their values match.

    Launch with ``dim=len(work_items)``; each thread handles one work item.

    Args:
        world_data: Device array of per-world data views, including caller-owned
            outputs and workspaces. Keep every referenced allocation alive.
        work_items: INT32 triples ``(world_index, local_index, 0)``. The local
            index is local sample. The third component is reserved.

    Returns:
        None. One transform in ``poses`` is written per thread.
    """
    item = wp.tid()
    entry = work_items[item]
    data = world_data[entry[0]]
    _compose_spatial_poses_at(
        0, entry[1], data.base_pose, data.relative_pose, data.poses
    )


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

    Explicit indices address the supplied array views. Homogeneous wrappers pass the
    full batch; packed wrappers pass one world's views and local indices.

    Args:
        environment: Batch row in the supplied arrays; use zero for a single-world view.
        sample: Sample index within the selected environment.
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
def homogeneous_compose_spatial_jacobians_kernel(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    jacobians_internal: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Compose floating-base Jacobians for a homogeneous batch.

    All worlds use the same structural dimensions and coordinate layout. Dimensions:
    E = worlds, N = samples, D = coordinates.

    All array arguments carry a world axis and their values may differ by world.
    Sample and coordinate counts are common to the batch.

    Launch with ``dim=(E, N)``: one thread per world/sample pair.

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
class SpatialJacobianCompositionData:
    """Inputs and outputs for floating-base spatial Jacobian composition.

    Array fields are non-owning views; the caller must retain their allocations
    for every launch and captured graph. Batched fields use one environment
    (``E = 1``); structural fields use this world's own dimensions.

    Attributes:
        base_pose: Batched scalar-last quaternion poses with shape ``(E, 7)``.
        relative_pose: Base-relative transforms with shape ``(E, N, 4, 4)``.
        jacobians_internal: Base-frame internal Jacobians with shape ``(E, N, 6, D)``.
        jacobians: Inertial-frame augmented Jacobians with shape ``(E, N, 6, D + 6)``.
    """

    base_pose: wp.array2d[wp.float64]
    relative_pose: wp.array4d[wp.float64]
    jacobians_internal: wp.array4d[wp.float64]
    jacobians: wp.array4d[wp.float64]


@wp.kernel(enable_backward=False)
def packed_compose_spatial_jacobians_kernel(
    world_data: wp.array(dtype=SpatialJacobianCompositionData),
    work_items: wp.array(dtype=wp.vec3i),
):
    """Compose spatial floating-base Jacobians on one thread per sample using packed world data.

    Each selected descriptor contains one world (E = 1). Structure and numerical
    model parameters may differ between worlds. Descriptors may share allocations
    when their values match.

    Launch with ``dim=len(work_items)``; each thread handles one work item.

    Args:
        world_data: Device array of per-world data views, including caller-owned
            outputs and workspaces. Keep every referenced allocation alive.
        work_items: INT32 triples ``(world_index, local_index, 0)``. The local
            index is local sample. The third component is reserved.

    Returns:
        None. One augmented Jacobian in ``jacobians`` is written per thread.
    """
    item = wp.tid()
    entry = work_items[item]
    data = world_data[entry[0]]
    _compose_spatial_jacobians_at(
        0,
        entry[1],
        data.base_pose,
        data.relative_pose,
        data.jacobians_internal,
        data.jacobians,
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

    Explicit indices address the supplied array views. Homogeneous wrappers pass the
    full batch; packed wrappers pass one world's views and local indices.

    Args:
        work_item: Flattened environment-sample index in the supplied arrays.
        lane: Thread index within the cooperative work item.
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
def homogeneous_compose_spatial_jacobians_tiled_kernel(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    jacobians_internal: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Compose homogeneous floating-base Jacobians cooperatively.

    All worlds use the same structural dimensions and coordinate layout. Dimensions:
    E = worlds, N = samples, D = coordinates, B = threads per block.

    All array arguments carry a world axis and their values may differ by world.
    Sample and coordinate counts are common to the batch.

    Launch with ``wp.launch_tiled(..., dim=E*N, block_dim=B)``. Each block
    cooperates on one flattened world/sample index.

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


@wp.kernel(enable_backward=False)
def packed_compose_spatial_jacobians_tiled_kernel(
    world_data: wp.array(dtype=SpatialJacobianCompositionData),
    work_items: wp.array(dtype=wp.vec3i),
):
    """Compose one floating spatial Jacobian cooperatively per sample using packed world data.

    Each selected descriptor contains one world (E = 1). Structure and numerical
    model parameters may differ between worlds. Descriptors may share allocations
    when their values match.

    Launch with ``wp.launch_tiled(..., dim=len(work_items), block_dim=B)``.
    Each block cooperates on one work item; all lanes must participate.
    Use at least 32 threads per CUDA block.

    Args:
        world_data: Device array of per-world data views, including caller-owned
            outputs and workspaces. Keep every referenced allocation alive.
        work_items: INT32 triples ``(world_index, local_index, 0)``. The local
            index is local sample. The third component is reserved.

    Returns:
        None. ``jacobians`` is updated in place by one cooperative block for
        each environment-sample pair.
    """
    item, lane = wp.tid()
    entry = work_items[item]
    data = world_data[entry[0]]
    _compose_spatial_jacobians_tiled_at(
        entry[1],
        lane,
        data.base_pose,
        data.relative_pose,
        data.jacobians_internal,
        data.jacobians,
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

    Explicit indices address the supplied array views. Homogeneous wrappers pass the
    full batch; packed wrappers pass one world's views and local indices.

    Args:
        environment: Batch row in the supplied arrays; use zero for a single-world view.
        sample: Sample index within the selected environment.
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
def homogeneous_compose_spatial_poses_and_twists_kernel(
    base_pose: wp.array2d[wp.float64],
    qd: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    twists_internal: wp.array3d[wp.float64],
    poses: wp.array4d[wp.float64],
    twists: wp.array3d[wp.float64],
):
    """Compose floating-base poses and twists for a homogeneous batch.

    All worlds use the same structural dimensions and coordinate layout. Dimensions:
    E = worlds, N = samples.

    All array arguments carry a world axis and their values may differ by world.
    Sample and coordinate counts are common to the batch.

    Launch with ``dim=(E, N)``: one thread per world/sample pair.

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
class SpatialPoseTwistCompositionData:
    """Inputs and outputs for floating-base spatial pose and twist composition.

    Array fields are non-owning views; the caller must retain their allocations
    for every launch and captured graph. Batched fields use one environment
    (``E = 1``); structural fields use this world's own dimensions.

    Attributes:
        base_pose: Runtime scalar-last base poses ``(E, 7)``.
        qd: Batched generalized-velocity directions. The first six entries are the
            angular-linear world-frame base direction.
        relative_pose: Fixed-base relative sample poses ``(E, N, 4, 4)``.
        twists_internal: Base-frame internal-Jacobian products ``(E, N, 6)``.
        poses: Absolute pose output ``(E, N, 4, 4)``.
        twists: Absolute inertial JVP output ``(E, N, 6)``.
    """

    base_pose: wp.array2d[wp.float64]
    qd: wp.array2d[wp.float64]
    relative_pose: wp.array4d[wp.float64]
    twists_internal: wp.array3d[wp.float64]
    poses: wp.array4d[wp.float64]
    twists: wp.array3d[wp.float64]


@wp.kernel(enable_backward=False)
def packed_compose_spatial_poses_and_twists_kernel(
    world_data: wp.array(dtype=SpatialPoseTwistCompositionData),
    work_items: wp.array(dtype=wp.vec3i),
):
    """Compose floating poses and one augmented Jacobian direction using packed world data.

    Each selected descriptor contains one world (E = 1). Structure and numerical
    model parameters may differ between worlds. Descriptors may share allocations
    when their values match.

    Launch with ``dim=len(work_items)``; each thread handles one work item.

    Args:
        world_data: Device array of per-world data views, including caller-owned
            outputs and workspaces. Keep every referenced allocation alive.
        work_items: INT32 triples ``(world_index, local_index, 0)``. The local
            index is local sample. The third component is reserved.

    Returns:
        None. ``poses`` and ``twists`` are written in place.
    """
    item = wp.tid()
    entry = work_items[item]
    data = world_data[entry[0]]
    _compose_spatial_poses_and_twists_at(
        0,
        entry[1],
        data.base_pose,
        data.qd,
        data.relative_pose,
        data.twists_internal,
        data.poses,
        data.twists,
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

    Explicit indices address the supplied array views. Homogeneous wrappers pass the
    full batch; packed wrappers pass one world's views and local indices.

    Args:
        environment: Batch row in the supplied arrays; use zero for a single-world view.
        sample: Sample index within the selected environment.
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
def homogeneous_compose_spatial_wrench_vjp_kernel(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    inertial_wrenches: wp.array3d[wp.float64],
    relative_wrenches: wp.array3d[wp.float64],
    base_generalized_force: wp.array2d[wp.float64],
):
    """Project floating-base sample wrenches for a homogeneous batch.

    All worlds use the same structural dimensions and coordinate layout. Dimensions:
    E = worlds, N = samples.

    All array arguments carry a world axis and their values may differ by world.
    Sample and coordinate counts are common to the batch.

    Launch with ``dim=(E, N)``: one thread per world/sample pair.

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
class SpatialWrenchProjectionData:
    """World-frame wrenches and base-frame wrench projection buffers.

    Array fields are non-owning views; the caller must retain their allocations
    for every launch and captured graph. Batched fields use one environment
    (``E = 1``); structural fields use this world's own dimensions.

    Attributes:
        base_pose: Runtime scalar-last base poses ``(E, 7)``.
        relative_pose: Fixed-base relative sample poses ``(E, N, 4, 4)``.
        inertial_wrenches: Absolute inertial sample cotangents ``(E, N, 6)``.
        relative_wrenches: Rotated cotangent output for the internal VJP with shape
            ``(E, N, 6)``.
        base_generalized_force: Base effort accumulator ``(E, 6)``.
    """

    base_pose: wp.array2d[wp.float64]
    relative_pose: wp.array4d[wp.float64]
    inertial_wrenches: wp.array3d[wp.float64]
    relative_wrenches: wp.array3d[wp.float64]
    base_generalized_force: wp.array2d[wp.float64]


@wp.kernel(enable_backward=False)
def packed_compose_spatial_wrench_vjp_kernel(
    world_data: wp.array(dtype=SpatialWrenchProjectionData),
    work_items: wp.array(dtype=wp.vec3i),
):
    """Pull absolute sample wrenches through floating-base composition using packed world data.

    Each selected descriptor contains one world (E = 1). Structure and numerical
    model parameters may differ between worlds. Descriptors may share allocations
    when their values match.

    Launch with ``dim=len(work_items)``; each thread handles one work item.

    Zero ``base_generalized_force`` before accumulating a new set of sample
    wrenches.

    Args:
        world_data: Device array of per-world data views, including caller-owned
            outputs and workspaces. Keep every referenced allocation alive.
        work_items: INT32 triples ``(world_index, local_index, 0)``. The local
            index is local sample. The third component is reserved.

    Returns:
        None. ``relative_wrenches`` is written and ``base_generalized_force``
        is accumulated in place.
    """
    item = wp.tid()
    entry = work_items[item]
    data = world_data[entry[0]]
    _compose_spatial_wrench_vjp_at(
        0,
        entry[1],
        data.base_pose,
        data.relative_pose,
        data.inertial_wrenches,
        data.relative_wrenches,
        data.base_generalized_force,
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

    Explicit indices address the supplied array views. Homogeneous wrappers pass the
    full batch; packed wrappers pass one world's views and local indices.

    Args:
        environment: Batch row in the supplied arrays; use zero for a single-world view.
        column: Generalized-velocity column within the selected environment.
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
def homogeneous_assemble_floating_vjp_kernel(
    base_generalized_force: wp.array2d[wp.float64],
    internal_generalized_force: wp.array2d[wp.float64],
    generalized_force: wp.array2d[wp.float64],
):
    """Assemble floating-base generalized forces for a homogeneous batch.

    All worlds use the same structural dimensions and coordinate layout. Dimensions:
    E = worlds, D = coordinates, B = threads per block.

    All array arguments carry a world axis and their values may differ by world.
    Sample and coordinate counts are common to the batch.

    Launch with ``dim=(E, D + 6)``: one thread writes one world/augmented-coordinate
    entry.

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
class FloatingBaseForceData:
    """Base and internal generalized-force buffers for one floating world.

    Array fields are non-owning views; the caller must retain their allocations
    for every launch and captured graph. Batched fields use one environment
    (``E = 1``); structural fields use this world's own dimensions.

    Attributes:
        base_generalized_force: Base effort ``(E, B)``.
        internal_generalized_force: Internal-coordinate effort ``(E, D)``.
        generalized_force: Augmented effort output ``(E, B + D)``.
    """

    base_generalized_force: wp.array2d[wp.float64]
    internal_generalized_force: wp.array2d[wp.float64]
    generalized_force: wp.array2d[wp.float64]


@wp.kernel(enable_backward=False)
def packed_assemble_floating_vjp_kernel(
    world_data: wp.array(dtype=FloatingBaseForceData),
    work_items: wp.array(dtype=wp.vec3i),
):
    """Assemble base and internal blocks of an augmented floating VJP using packed world data.

    Each selected descriptor contains one world (E = 1). Structure and numerical
    model parameters may differ between worlds. Descriptors may share allocations
    when their values match.

    Launch with ``dim=len(work_items)``; each thread handles one work item.

    Args:
        world_data: Device array of per-world data views, including caller-owned
            outputs and workspaces. Keep every referenced allocation alive.
        work_items: INT32 triples ``(world_index, local_index, 0)``. The local
            index is generalized-velocity column. The third component is reserved.

    Returns:
        None. ``generalized_force`` is written in place.
    """
    item = wp.tid()
    entry = work_items[item]
    data = world_data[entry[0]]
    _assemble_floating_vjp_at(
        0,
        entry[1],
        data.base_generalized_force,
        data.internal_generalized_force,
        data.generalized_force,
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


def launch_homogeneous_spatial_floating_pose_composition(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    poses: wp.array4d[wp.float64],
):
    """Compose batched spatial runtime-base and relative poses in a homogeneous batch.

    All worlds use the same structural dimensions and coordinate layout. Dimensions:
    E = worlds, N = samples.

    All array arguments carry a world axis and their values may differ by world.
    Sample and coordinate counts are common to the batch.

    The launcher derives its work grid from these common dimensions and enqueues the
    homogeneous kernels on the current Warp stream.

    Args:
        base_pose: Scalar-last base poses with shape ``(E, 7)``.
        relative_pose: Base-relative transforms with shape ``(E, N, 4, 4)``.
        poses: Caller-owned inertial-frame transforms with shape
            ``(E, N, 4, 4)``.

    Returns:
        None. The pose kernel is enqueued on the active Warp stream.
    """
    wp.launch(
        homogeneous_compose_spatial_poses_kernel,
        dim=(relative_pose.shape[0], relative_pose.shape[1]),
        inputs=[base_pose, relative_pose],
        outputs=[poses],
    )


def launch_homogeneous_spatial_floating_jacobian_composition(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    jacobians_internal: wp.array4d[wp.float64],
    block_dim: int,
    jacobians: wp.array4d[wp.float64],
):
    """Launch spatial floating jacobian composition for a homogeneous batch.

    All worlds use the same structural dimensions and coordinate layout. Dimensions:
    E = worlds, N = samples, D = coordinates.

    Shared across worlds: ``block_dim``.

    All array arguments carry a world axis and their values may differ by world.
    Sample and coordinate counts are common to the batch.

    The launcher derives its work grid from these common dimensions and enqueues the
    homogeneous kernels on the current Warp stream.

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
            homogeneous_compose_spatial_jacobians_tiled_kernel,
            dim=relative_pose.shape[0] * relative_pose.shape[1],
            inputs=[base_pose, relative_pose, jacobians_internal],
            outputs=[jacobians],
            block_dim=block_dim,
        )
    else:
        wp.launch(
            homogeneous_compose_spatial_jacobians_kernel,
            dim=(relative_pose.shape[0], relative_pose.shape[1]),
            inputs=[base_pose, relative_pose, jacobians_internal],
            outputs=[jacobians],
        )


def launch_homogeneous_spatial_floating_kinematics_composition(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    jacobians_internal: wp.array4d[wp.float64],
    block_dim: int,
    poses: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Launch spatial floating kinematics composition for a homogeneous batch.

    All worlds use the same structural dimensions and coordinate layout. Dimensions:
    E = worlds, N = samples, D = coordinates.

    Shared across worlds: ``block_dim``.

    All array arguments carry a world axis and their values may differ by world.
    Sample and coordinate counts are common to the batch.

    The launcher derives its work grid from these common dimensions and enqueues the
    homogeneous kernels on the current Warp stream.

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
    launch_homogeneous_spatial_floating_pose_composition(
        base_pose, relative_pose, poses
    )
    launch_homogeneous_spatial_floating_jacobian_composition(
        base_pose, relative_pose, jacobians_internal, block_dim, jacobians
    )


def launch_homogeneous_spatial_floating_kinematics_jvp_composition(
    base_pose: wp.array2d[wp.float64],
    qd: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    twists_internal: wp.array3d[wp.float64],
    poses: wp.array4d[wp.float64],
    twists: wp.array3d[wp.float64],
):
    """Launch spatial floating kinematics jvp composition for a homogeneous batch.

    All worlds use the same structural dimensions and coordinate layout. Dimensions:
    E = worlds, N = samples.

    All array arguments carry a world axis and their values may differ by world.
    Sample and coordinate counts are common to the batch.

    The launcher derives its work grid from these common dimensions and enqueues the
    homogeneous kernels on the current Warp stream.

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
        homogeneous_compose_spatial_poses_and_twists_kernel,
        dim=(relative_pose.shape[0], relative_pose.shape[1]),
        inputs=[base_pose, qd, relative_pose, twists_internal],
        outputs=[poses, twists],
    )


def launch_homogeneous_spatial_floating_vjp_composition(
    base_pose: wp.array2d[wp.float64],
    relative_pose: wp.array4d[wp.float64],
    inertial_wrenches: wp.array3d[wp.float64],
    relative_wrenches: wp.array3d[wp.float64],
    base_generalized_force: wp.array2d[wp.float64],
):
    """Launch spatial floating vjp composition for a homogeneous batch.

    All worlds use the same structural dimensions and coordinate layout. Dimensions:
    E = worlds, N = samples.

    All array arguments carry a world axis and their values may differ by world.
    Sample and coordinate counts are common to the batch.

    The launcher derives its work grid from these common dimensions and enqueues the
    homogeneous kernels on the current Warp stream.

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
        homogeneous_compose_spatial_wrench_vjp_kernel,
        dim=(relative_pose.shape[0], relative_pose.shape[1]),
        inputs=[base_pose, relative_pose, inertial_wrenches],
        outputs=[relative_wrenches, base_generalized_force],
    )


def launch_homogeneous_spatial_floating_vjp_assembly(
    base_generalized_force: wp.array2d[wp.float64],
    internal_generalized_force: wp.array2d[wp.float64],
    generalized_force: wp.array2d[wp.float64],
):
    """Launch spatial floating vjp assembly for a homogeneous batch.

    All worlds use the same structural dimensions and coordinate layout. Dimensions:
    E = worlds, D = coordinates.

    All array arguments carry a world axis and their values may differ by world.
    Sample and coordinate counts are common to the batch.

    The launcher derives its work grid from these common dimensions and enqueues the
    homogeneous kernels on the current Warp stream.

    Args:
        base_generalized_force: Base effort in angular-linear order ``(E, 6)``.
        internal_generalized_force: Internal-coordinate effort ``(E, D)``.
        generalized_force: Caller-owned augmented output ``(E, 6 + D)``.

    Returns:
        None. The assembly kernel is enqueued on the active Warp stream.
    """

    wp.launch(
        homogeneous_assemble_floating_vjp_kernel,
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
        homogeneous_assemble_floating_vjp_kernel,
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
    launch_homogeneous_spatial_floating_pose_composition, num_outputs=1
)
spatial_floating_jacobian_composition = wp.jax_callable(
    launch_homogeneous_spatial_floating_jacobian_composition, num_outputs=1
)
spatial_floating_kinematics_composition = wp.jax_callable(
    launch_homogeneous_spatial_floating_kinematics_composition, num_outputs=2
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
    "launch_homogeneous_spatial_floating_jacobian_composition",
    "launch_homogeneous_spatial_floating_kinematics_composition",
    "launch_homogeneous_spatial_floating_kinematics_jvp_composition",
    "launch_homogeneous_spatial_floating_pose_composition",
    "launch_homogeneous_spatial_floating_vjp_composition",
    "launch_homogeneous_spatial_floating_vjp_assembly",
    "planar_floating_jacobian_composition",
    "planar_floating_kinematics_composition",
    "planar_floating_pose_composition",
    "spatial_floating_jacobian_composition",
    "spatial_floating_kinematics_composition",
    "spatial_floating_pose_composition",
]


__all__ += [
    "SpatialPoseCompositionData",
    "packed_compose_spatial_poses_kernel",
    "SpatialJacobianCompositionData",
    "packed_compose_spatial_jacobians_kernel",
    "packed_compose_spatial_jacobians_tiled_kernel",
    "SpatialPoseTwistCompositionData",
    "packed_compose_spatial_poses_and_twists_kernel",
    "SpatialWrenchProjectionData",
    "packed_compose_spatial_wrench_vjp_kernel",
    "FloatingBaseForceData",
    "packed_assemble_floating_vjp_kernel",
]

__all__ += [
    "homogeneous_compose_spatial_poses_kernel",
    "homogeneous_compose_spatial_jacobians_kernel",
    "homogeneous_compose_spatial_jacobians_tiled_kernel",
    "homogeneous_compose_spatial_poses_and_twists_kernel",
    "homogeneous_compose_spatial_wrench_vjp_kernel",
    "homogeneous_assemble_floating_vjp_kernel",
]
