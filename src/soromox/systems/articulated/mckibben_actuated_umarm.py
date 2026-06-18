"""McKibben-actuated UMArm articulated system.

This module implements the UMArm as a SoRoMoX articulated system with
McKibben pressure actuation. The UMArm is a pneumatically driven rigid-soft
hybrid arm introduced by Zuo, Han, Li, Jamal, and Bruder in "UMArm:
Untethered, Modular, Portable, Soft Pneumatic Arm", arXiv:2505.11476,
https://doi.org/10.48550/arXiv.2505.11476.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
from jax import Array
from jax import numpy as jnp

from soromox.rendering.actuators import ActuatorVisualLayer
from soromox.systems.articulated.articulated_soft_robot import ArticulatedSoftRobot
from soromox.systems.articulated.params import McKibbenActuatedUMArmParams

_ACTUATION_SIGN = 1.0


def _rot_x(angle: Array) -> Array:
    """Return the active 3D rotation matrix about the x-axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        Rotation matrix with shape ``(3, 3)``.
    """
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    z = jnp.zeros_like(angle)
    o = jnp.ones_like(angle)
    return jnp.stack(
        [
            jnp.stack([o, z, z]),
            jnp.stack([z, c, -s]),
            jnp.stack([z, s, c]),
        ]
    )


def _rot_y(angle: Array) -> Array:
    """Return the active 3D rotation matrix about the y-axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        Rotation matrix with shape ``(3, 3)``.
    """
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    z = jnp.zeros_like(angle)
    o = jnp.ones_like(angle)
    return jnp.stack(
        [
            jnp.stack([c, z, s]),
            jnp.stack([z, o, z]),
            jnp.stack([-s, z, c]),
        ]
    )


def _drot_x(angle: Array) -> Array:
    """Return the derivative of the active x-axis rotation matrix.

    This is the closed-form derivative ``d(_rot_x(angle)) / d angle`` used by
    the McKibben length Jacobian. The derivative follows the same active
    rotation convention as :func:`_rot_x`.

    Args:
        angle: Rotation angle in radians.

    Returns:
        Rotation-matrix derivative with shape ``(3, 3)``.
    """
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    z = jnp.zeros_like(angle)
    return jnp.stack(
        [
            jnp.stack([z, z, z]),
            jnp.stack([z, -s, -c]),
            jnp.stack([z, c, -s]),
        ]
    )


def _drot_y(angle: Array) -> Array:
    """Return the derivative of the active y-axis rotation matrix.

    This is the closed-form derivative ``d(_rot_y(angle)) / d angle`` used by
    the McKibben length Jacobian. The derivative follows the same active
    rotation convention as :func:`_rot_y`.

    Args:
        angle: Rotation angle in radians.

    Returns:
        Rotation-matrix derivative with shape ``(3, 3)``.
    """
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    z = jnp.zeros_like(angle)
    return jnp.stack(
        [
            jnp.stack([-s, z, c]),
            jnp.stack([z, z, z]),
            jnp.stack([-c, z, -s]),
        ]
    )


class McKibbenActuatedUMArm(ArticulatedSoftRobot):
    """Spatial UMArm with rigid links, universal joints, and McKibben actuation.

    The arm is represented as a 12-DOF serial articulated chain, where each
    universal joint contributes two one-DOF revolute joints. McKibben pressures
    are mapped to generalized joint torques by differentiating the
    pressure-normalized actuator energy with respect to joint configuration.

    References:
        Zuo, R., Han, D. H., Li, R., Jamal, S., & Bruder, D. (2025). UMArm:
        Untethered, Modular, Portable, Soft Pneumatic Arm. arXiv:2505.11476.
        https://doi.org/10.48550/arXiv.2505.11476

    Attributes:
        params: Typed cached parameters used to construct the model.
        joint_armature: Per-joint rotor armature added to the dense inertia
            matrix, shape ``(num_dofs,)``.
        mckibben_moving_base_transform: Per-U-joint transform from the fixed
            actuator-side frame to the moving ring frame at zero joint
            displacement, shape ``(num_mckibben_groups, 4, 4)``.
        mckibben_moving_points: Moving-ring actuator attachment points expressed
            in the moving ring frames, shape ``(num_mckibben_groups, 4, 3)``.
        mckibben_fixed_points: Fixed actuator attachment points expressed in
            each group-local frame, shape ``(num_mckibben_groups, 4, 3)``.
        mckibben_reference_length: Geometric actuator lengths at zero joint
            displacement, shape ``(num_mckibben_groups, 4)``.
        mckibben_length_offset: Neutral effective McKibben lengths, shape
            ``(num_mckibben_groups, 4)``.
        mckibben_thread_length: McKibben thread lengths ``B``, shape
            ``(num_mckibben_groups, 4)``.
        mckibben_num_turns: McKibben fiber turn counts ``N``, shape
            ``(num_mckibben_groups, 4)``.
        mckibben_joint_pair_indices: Joint-coordinate pairs driving each
            U-joint actuator group, shape ``(num_mckibben_groups, 2)``.
        num_segments: Number of physical UMArm segments.
        num_mckibben_groups: Number of U-joint actuator groups.
    """

    params: McKibbenActuatedUMArmParams
    joint_armature: Array
    mckibben_moving_base_transform: Array
    mckibben_moving_points: Array
    mckibben_fixed_points: Array
    mckibben_reference_length: Array
    mckibben_length_offset: Array
    mckibben_thread_length: Array
    mckibben_num_turns: Array
    mckibben_joint_pair_indices: Array
    num_segments: int = eqx.field(static=True)
    num_mckibben_groups: int = eqx.field(static=True)

    def __init__(self, params: McKibbenActuatedUMArmParams, **kwargs: Any) -> None:
        """Initialize the UMArm from typed cached parameters.

        Args:
            params: UMArm dynamic and McKibben actuation parameters.
            **kwargs: Additional keyword arguments forwarded to
                :class:`ArticulatedSoftRobot`.

        Raises:
            TypeError: If ``params`` is not a
                :class:`McKibbenActuatedUMArmParams` instance.
        """
        if not isinstance(params, McKibbenActuatedUMArmParams):
            raise TypeError("params must be a McKibbenActuatedUMArmParams instance.")
        super().__init__(params, **kwargs)

        self.joint_armature = jnp.asarray(params.joint_armature)
        self.mckibben_moving_base_transform = jnp.asarray(
            params.mckibben_moving_base_transform
        )
        self.mckibben_moving_points = jnp.asarray(params.mckibben_moving_points)
        self.mckibben_fixed_points = jnp.asarray(params.mckibben_fixed_points)
        self.mckibben_reference_length = jnp.asarray(params.mckibben_reference_length)
        self.mckibben_length_offset = jnp.asarray(params.mckibben_length_offset)
        self.mckibben_thread_length = jnp.asarray(params.mckibben_thread_length)
        self.mckibben_num_turns = jnp.asarray(params.mckibben_num_turns)
        self.mckibben_joint_pair_indices = jnp.asarray(
            params.mckibben_joint_pair_indices, dtype=jnp.int32
        )

        self.num_mckibben_groups = int(self.mckibben_thread_length.shape[0])
        self.num_segments = self.num_mckibben_groups // 2
        self.num_actuators = int(self.mckibben_thread_length.size)

    @classmethod
    def from_cached_parameters(
        cls, path: str | Path, **kwargs: Any
    ) -> McKibbenActuatedUMArm:
        """Build the UMArm from an explicit cached parameter file.

        Args:
            path: Path to a UMArm ``.npz`` parameter file.
            **kwargs: Additional keyword arguments forwarded to the constructor.

        Returns:
            UMArm system initialized from cached parameters.
        """
        return cls(McKibbenActuatedUMArmParams.from_cached_npz(path), **kwargs)

    def _actuator_points_local_group(
        self, q: Array, group_index: Array
    ) -> tuple[Array, Array]:
        """Compute group-local actuator endpoints for one McKibben group.

        The returned coordinates are expressed in the group's local kinematic
        frame. This local frame is sufficient for actuator lengths because both
        endpoints are represented in the same frame. Rendering transforms these
        endpoints to the world frame separately via
        :meth:`mckibben_actuator_segments`.

        Args:
            q: Joint coordinates with shape ``(num_dofs,)``.
            group_index: Scalar index of the McKibben U-joint group.

        Returns:
            Tuple ``(moving, fixed)``. Each array has shape ``(4, 3)`` and stores
            the four moving or fixed actuator attachment points for the group.
        """
        moving_base_transform = self.mckibben_moving_base_transform[group_index]
        joint_pair = self.mckibben_joint_pair_indices[group_index]
        rotation = (
            moving_base_transform[:3, :3]
            @ _rot_x(q[joint_pair[0]])
            @ _rot_y(q[joint_pair[1]])
        )
        moving = self.mckibben_moving_points[group_index] @ rotation.T
        moving = moving + moving_base_transform[:3, 3]
        fixed = self.mckibben_fixed_points[group_index]
        return moving, fixed

    @eqx.filter_jit
    def _mckibben_actuator_segments_local(self, q: Array) -> Array:
        """Return group-local McKibben actuator endpoint segments.

        Args:
            q: Joint coordinates with shape ``(num_dofs,)``.

        Returns:
            Local endpoint segments with shape
            ``(num_mckibben_groups, 4, 2, 3)``. Axis 2 stores
            ``[fixed_point, moving_point]``.
        """
        if self.num_actuators == 0:
            return jnp.zeros(
                (self.num_mckibben_groups, self.mckibben_moving_points.shape[1], 2, 3),
                dtype=q.dtype,
            )
        group_indices = jnp.arange(self.num_mckibben_groups)
        moving, fixed = jax.vmap(lambda idx: self._actuator_points_local_group(q, idx))(
            group_indices
        )
        return jnp.stack([fixed, moving], axis=2)

    def _mckibben_group_frames(self, q: Array) -> Array:
        """Return world frames for all McKibben actuator groups.

        The cached McKibben attachment geometry is local to each physical UMArm
        segment. Upper U-joint groups are expressed in the pre-upper-joint frame,
        while lower U-joint groups are expressed in the rod frame after the
        upper U-joint pair. These frames are obtained from the articulated FK
        recursion and are used only for visualization/world-space geometry, not
        for the local actuator length calculation.

        Args:
            q: Joint coordinates with shape ``(num_dofs,)``.

        Returns:
            SE(3) transforms with shape ``(num_mckibben_groups, 4, 4)``.
        """
        if self.num_mckibben_groups == 0:
            return jnp.zeros((0, 4, 4), dtype=q.dtype)
        joint_frames, link_frames, _, _ = self._kinematic_frames(q)
        group_indices = jnp.arange(self.num_mckibben_groups)
        pair_starts = self.mckibben_joint_pair_indices[:, 0]
        use_link_frame = (group_indices % 2).astype(bool)
        upper_frames = joint_frames[pair_starts]
        lower_frames = link_frames[jnp.maximum(pair_starts - 1, 0)]
        return jnp.where(use_link_frame[:, None, None], lower_frames, upper_frames)

    @eqx.filter_jit
    def mckibben_actuator_segments(self, q: Array) -> Array:
        """Return world-space actuator endpoint segments for rendering.

        Args:
            q: Joint coordinates with shape ``(num_dofs,)``.

        Returns:
            World-space actuator endpoint segments with shape
            ``(num_actuators, 2, 3)``. Axis 1 stores
            ``[fixed_point, moving_point]`` for each actuator.
        """
        if self.num_actuators == 0:
            return jnp.zeros((0, 2, 3), dtype=q.dtype)
        local_segments = self._mckibben_actuator_segments_local(q)
        frames = self._mckibben_group_frames(q)
        rotations = frames[:, :3, :3]
        translations = frames[:, :3, 3]
        world_segments = (
            jnp.einsum("gij,gapj->gapi", rotations, local_segments)
            + translations[:, None, None, :]
        )
        return world_segments.reshape((self.num_actuators, 2, 3))

    def actuator_visual_layers(
        self,
        q: Array,
        s_points: Array,
        *,
        actuator_inputs: Array | None = None,
    ) -> tuple[ActuatorVisualLayer, ...]:
        """Return renderer-facing McKibben muscle segments."""
        del s_points
        if self.num_actuators == 0:
            return ()
        scalar_fields = {}
        if actuator_inputs is not None:
            pressures = jnp.asarray(actuator_inputs).reshape((self.num_actuators,))
            scalar_fields["pressure"] = pressures
            scalar_fields["force"] = self.actuator_forces(q, pressures)
        return (
            ActuatorVisualLayer(
                name="mckibben_muscles",
                kind="muscle",
                points=self.mckibben_actuator_segments(q),
                scalar_fields=scalar_fields,
            ),
        )

    @eqx.filter_jit
    def effective_lengths(self, q: Array) -> Array:
        """Return effective McKibben lengths.

        The effective length for actuator ``i`` is
        ``||moving_i(q) - fixed_i|| - reference_length_i + length_offset_i``.
        The computation is carried out in each group-local frame, so rigid
        world-frame transforms do not affect the length.

        Args:
            q: Joint coordinates with shape ``(num_dofs,)``.

        Returns:
            Effective lengths with shape ``(num_actuators,)``.
        """
        if self.num_actuators == 0:
            return jnp.zeros((0,), dtype=q.dtype)
        segments = self._mckibben_actuator_segments_local(q)
        current_length = jnp.linalg.norm(
            segments[:, :, 1, :] - segments[:, :, 0, :], axis=-1
        )
        effective_length = (
            current_length
            - self.mckibben_reference_length
            + self.mckibben_length_offset
        )
        return effective_length.reshape((self.num_actuators,))

    def _potential_per_pressure(self, q: Array) -> Array:
        """Return per-actuator McKibben potential energy per unit pressure.

        Args:
            q: Joint coordinates with shape ``(num_dofs,)``.

        Returns:
            Pressure-normalized actuator potential values with shape
            ``(num_actuators,)``.
        """
        if self.num_actuators == 0:
            return jnp.zeros((0,), dtype=q.dtype)
        length = self.effective_lengths(q).reshape(self.mckibben_thread_length.shape)
        energy = (
            (self.mckibben_thread_length**2 - length**2)
            * length
            / (4.0 * math.pi * self.mckibben_num_turns**2)
        )
        return energy.reshape((self.num_actuators,))

    def _length_jacobian_local_group(
        self, q: Array, group_index: Array
    ) -> tuple[Array, Array]:
        """Return effective lengths and local two-joint length derivatives.

        Args:
            q: Joint coordinates with shape ``(num_dofs,)``.
            group_index: Scalar index of the McKibben U-joint group.

        Returns:
            Tuple ``(effective_length, dlength_dq_pair)``. The length array has
            shape ``(4,)`` and the Jacobian has shape ``(4, 2)``. The two
            Jacobian columns correspond to this group's joint-coordinate pair.
        """
        moving_base_transform = self.mckibben_moving_base_transform[group_index]
        joint_pair = self.mckibben_joint_pair_indices[group_index]
        qx = q[joint_pair[0]]
        qy = q[joint_pair[1]]

        base_rotation = moving_base_transform[:3, :3]
        rotation_x = _rot_x(qx)
        rotation_y = _rot_y(qy)
        rotation = base_rotation @ rotation_x @ rotation_y
        rotation_dqx = base_rotation @ _drot_x(qx) @ rotation_y
        rotation_dqy = base_rotation @ rotation_x @ _drot_y(qy)

        moving_points = self.mckibben_moving_points[group_index]
        moving = moving_points @ rotation.T + moving_base_transform[:3, 3]
        fixed = self.mckibben_fixed_points[group_index]
        displacement = moving - fixed
        current_length = jnp.linalg.norm(displacement, axis=-1)

        direction = displacement / current_length[:, None]
        moving_dqx = moving_points @ rotation_dqx.T
        moving_dqy = moving_points @ rotation_dqy.T
        dlength_dqx = jnp.sum(direction * moving_dqx, axis=-1)
        dlength_dqy = jnp.sum(direction * moving_dqy, axis=-1)
        dlength_dq_pair = jnp.stack([dlength_dqx, dlength_dqy], axis=-1)

        effective_length = (
            current_length
            - self.mckibben_reference_length[group_index]
            + self.mckibben_length_offset[group_index]
        )
        return effective_length, dlength_dq_pair

    def _actuation_matrix_group(self, q: Array, group_index: Array) -> Array:
        """Return one McKibben U-joint group's contribution to ``A(q)``.

        The group-local block is computed in closed form as
        ``dU_i / dl_i * dl_i / dq_pair`` for the four actuators attached to the
        selected universal-joint pair, where ``U_i`` is the pressure-normalized
        McKibben potential. The resulting ``(2, 4)`` block is scattered into the
        full pressure-to-generalized-torque matrix; all rows and columns outside
        this group's joint pair and actuators are zero.

        Args:
            q: Joint coordinates with shape ``(num_dofs,)``.
            group_index: Scalar index of the McKibben U-joint group.

        Returns:
            Sparse group contribution to the actuation matrix, with shape
            ``(num_dofs, num_actuators)``.
        """
        effective_length, dlength_dq_pair = self._length_jacobian_local_group(
            q, group_index
        )
        potential_force_per_pressure = (
            self.mckibben_thread_length[group_index] ** 2
            - 3.0 * effective_length**2
        ) / (4.0 * math.pi * self.mckibben_num_turns[group_index] ** 2)
        local_block = potential_force_per_pressure[:, None] * dlength_dq_pair

        joint_pair = self.mckibben_joint_pair_indices[group_index]
        actuator_indices = (
            group_index * self.mckibben_moving_points.shape[1]
            + jnp.arange(self.mckibben_moving_points.shape[1])
        )
        return jnp.zeros((self.num_dofs, self.num_actuators), dtype=q.dtype).at[
            joint_pair[:, None], actuator_indices[None, :]
        ].set(local_block.T)

    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """Return the pressure-to-joint-torque map ``A(q)``.

        Args:
            q: Joint coordinates with shape ``(num_dofs,)``.

        Returns:
            Actuation matrix with shape ``(num_dofs, num_actuators)`` such that
            ``tau = A(q) @ pressures``.
        """
        if self.num_actuators == 0:
            return jnp.zeros((self.num_dofs, 0), dtype=q.dtype)
        group_indices = jnp.arange(self.num_mckibben_groups)
        group_matrices = jax.vmap(lambda idx: self._actuation_matrix_group(q, idx))(
            group_indices
        )
        return _ACTUATION_SIGN * jnp.sum(group_matrices, axis=0)

    @eqx.filter_jit
    def actuator_forces(self, q: Array, pressures: Array) -> Array:
        """Return per-actuator McKibben axial forces.

        Args:
            q: Joint coordinates with shape ``(num_dofs,)``.
            pressures: Actuator pressures in Pa with shape ``(num_actuators,)``.

        Returns:
            McKibben actuator forces with shape ``(num_actuators,)``. The sign
            convention matches the cached UMArm/MuJoCo force model.
        """
        if self.num_actuators == 0:
            return jnp.zeros((0,), dtype=q.dtype)
        length = self.effective_lengths(q).reshape(self.mckibben_thread_length.shape)
        pressure = pressures.reshape(length.shape)
        force = (
            pressure
            * (self.mckibben_thread_length**2 - 3.0 * length**2)
            / (4.0 * math.pi * self.mckibben_num_turns**2)
        )
        return force.reshape((self.num_actuators,))

    @eqx.filter_jit
    def inertia_matrix(self, q: Array) -> Array:
        """Return generalized inertia including rotor armature.

        Args:
            q: Joint coordinates with shape ``(num_dofs,)``.

        Returns:
            Dense generalized inertia matrix with shape
            ``(num_dofs, num_dofs)``.
        """
        return super().inertia_matrix(q) + jnp.diag(self.joint_armature)

    def _joint_armature(self, q: Array) -> Array:
        """Return rotor armature for the inherited ABA forward dynamics."""
        del q
        return self.joint_armature
