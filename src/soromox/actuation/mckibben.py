"""McKibben pneumatic-muscle transmission for spatial articulated robots.

This module's geometry is intentionally specific to grouped muscles spanning
two-axis articulated joints. It requires a host that provides spatial
kinematic frames; it is not a continuum-rod transmission. Continuum models can
use threadlike muscle or equivalent pressure-chamber coordinates, while a
future continuum McKibben model may reuse the constitutive equation with a
different geometry contract.
"""

from __future__ import annotations

import math
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax import numpy as jnp

from soromox.systems.params import BaseSystemParams, _contains_tracer
from soromox.utils.geometry.rotations import (
    principal_axis_rotation_matrix,
    principal_axis_rotation_matrix_derivative,
)
from soromox.utils.numerics import safe_norm, safe_normalize

from .core import Actuator, ActuatorMetadata, DirectEffort, Transmission


class ArticulatedMcKibbenTransmissionParams(BaseSystemParams):
    """Attachment geometry and constitutive parameters for McKibben groups."""

    moving_base_transform: Array
    moving_points: Array
    fixed_points: Array
    reference_length: Array
    length_offset: Array
    thread_length: Array
    num_turns: Array
    joint_pair_indices: Array

    def __check_init__(self) -> None:
        self.validate_for_update()

    @property
    def group_shape(self) -> tuple[int, int]:
        return tuple(jnp.asarray(self.thread_length).shape)  # type: ignore[return-value]

    @property
    def num_groups(self) -> int:
        return self.group_shape[0]

    @property
    def num_channels(self) -> int:
        return self.group_shape[0] * self.group_shape[1]

    def validate_structure(self) -> None:
        shape = jnp.asarray(self.thread_length).shape
        if len(shape) != 2:
            raise ValueError(
                "thread_length must have shape "
                f"(num_groups, num_actuators_per_group), got {shape}."
            )
        expected = {
            "moving_base_transform": (shape[0], 4, 4),
            "moving_points": (*shape, 3),
            "fixed_points": (*shape, 3),
            "reference_length": shape,
            "length_offset": shape,
            "num_turns": shape,
            "joint_pair_indices": (shape[0], 2),
        }
        for name, expected_shape in expected.items():
            value = jnp.asarray(getattr(self, name))
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {value.shape}."
                )

    def validate_values(self) -> None:
        """Validate eager McKibben topology values."""
        shape = jnp.asarray(self.thread_length).shape
        joint_pair_indices = jnp.asarray(self.joint_pair_indices)
        if shape[0] > 0 and bool(
            jnp.any(joint_pair_indices[:, 0] == joint_pair_indices[:, 1])
        ):
            raise ValueError(
                "joint_pair_indices must contain two distinct DOF indices per group."
            )

    def assert_same_topology(
        self, other: ArticulatedMcKibbenTransmissionParams
    ) -> None:
        if not isinstance(other, ArticulatedMcKibbenTransmissionParams):
            raise ValueError("Changing transmission type requires reconstruction.")
        if other.group_shape != self.group_shape:
            raise ValueError(
                "Changing McKibben group topology requires reconstruction."
            )
        if not _contains_tracer(other.joint_pair_indices) and not np.array_equal(
            np.asarray(other.joint_pair_indices),
            np.asarray(self.joint_pair_indices),
        ):
            raise ValueError(
                "Changing McKibben joint-pair topology requires reconstruction."
            )

    @classmethod
    def from_cached_npz(cls, path: str | Path) -> ArticulatedMcKibbenTransmissionParams:
        """Load McKibben transmission fields from an existing UMArm cache."""
        with np.load(path) as data:

            def read(name: str, *legacy_names: str) -> Array:
                for key in (name, *legacy_names):
                    if key in data:
                        return jnp.asarray(data[key])
                candidates = ", ".join((name, *legacy_names))
                raise KeyError(f"UMArm parameter file is missing one of: {candidates}.")

            return cls(
                moving_base_transform=read(
                    "mckibben_moving_base_transform", "mck_moving_base_transform"
                ),
                moving_points=read("mckibben_moving_points", "mck_moving_points"),
                fixed_points=read("mckibben_fixed_points", "mck_fixed_points"),
                reference_length=read("mckibben_reference_length", "mck_base_length"),
                length_offset=read("mckibben_length_offset", "mck_length_offset"),
                thread_length=read("mckibben_thread_length", "mck_b_thread"),
                num_turns=read("mckibben_num_turns", "mck_n_turns"),
                joint_pair_indices=jnp.asarray(
                    read("mckibben_joint_pair_indices", "mck_q_pair_index"),
                    dtype=jnp.int32,
                ),
            )


class ArticulatedMcKibbenTransmission(Transmission):
    """Pressure-work transmission using McKibben volume coordinates.

    The transmission owns fixed and moving attachment geometry for grouped
    universal joints on a spatial articulated host. It computes effective
    muscle lengths, the volume-like coordinate work-conjugate to pressure, its
    analytic moment matrix, and renderer-facing endpoint segments without
    embedding those mechanics in a particular articulated robot class.
    """

    params: ArticulatedMcKibbenTransmissionParams

    def __init__(self, params: ArticulatedMcKibbenTransmissionParams) -> None:
        if not isinstance(params, ArticulatedMcKibbenTransmissionParams):
            raise TypeError("params must be ArticulatedMcKibbenTransmissionParams.")
        params.validate_for_update()
        self.params = params

    @property
    def num_channels(self) -> int:
        return self.params.num_channels

    def _actuator_points_local_group(
        self, q: Array, group_index: Array
    ) -> tuple[Array, Array]:
        """Compute moving and fixed attachment points for one local group.

        Both endpoint families are expressed in the group's local kinematic
        frame, which is sufficient for length evaluation and avoids introducing
        world-frame transformations into the constitutive mechanics.
        """
        transform = self.params.moving_base_transform[group_index]
        joint_pair = self.params.joint_pair_indices[group_index]
        rotation = (
            transform[:3, :3]
            @ principal_axis_rotation_matrix(q[joint_pair[0]], "x")
            @ principal_axis_rotation_matrix(q[joint_pair[1]], "y")
        )
        moving = self.params.moving_points[group_index] @ rotation.T
        moving = moving + transform[:3, 3]
        return moving, self.params.fixed_points[group_index]

    def local_segments(self, q: Array) -> Array:
        """Return group-local fixed/moving endpoint segments.

        Returns:
            Array with shape ``(num_groups, actuators_per_group, 2, 3)``.
            Endpoint axis 2 is ordered as ``[fixed, moving]``.
        """
        if self.num_channels == 0:
            return jnp.zeros(
                (self.params.num_groups, self.params.group_shape[1], 2, 3),
                dtype=q.dtype,
            )
        indices = jnp.arange(self.params.num_groups)
        moving, fixed = jax.vmap(
            lambda index: self._actuator_points_local_group(q, index)
        )(indices)
        return jnp.stack([fixed, moving], axis=2)

    def _group_frames(self, robot, q: Array) -> Array:
        """Return world frames for all McKibben actuator groups.

        Upper groups use the pre-upper-joint frame; lower groups use the rod
        frame following the upper joint pair, matching the cached UMArm geometry.
        """
        if self.params.num_groups == 0:
            return jnp.zeros((0, 4, 4), dtype=q.dtype)
        joint_frames, link_frames, _, _ = robot._kinematic_frames(q)
        group_indices = jnp.arange(self.params.num_groups)
        pair_starts = self.params.joint_pair_indices[:, 0]
        use_link_frame = (group_indices % 2).astype(bool)
        upper_frames = joint_frames[pair_starts]
        lower_frames = link_frames[jnp.maximum(pair_starts - 1, 0)]
        return jnp.where(use_link_frame[:, None, None], lower_frames, upper_frames)

    def segments(self, robot, q: Array) -> Array:
        """Return world-space actuator endpoint segments for rendering.

        Returns:
            Array with shape ``(num_channels, 2, 3)`` whose endpoint axis is
            ordered as ``[fixed, moving]``.
        """
        if self.num_channels == 0:
            return jnp.zeros((0, 2, 3), dtype=q.dtype)
        local_segments = self.local_segments(q)
        frames = self._group_frames(robot, q)
        world_segments = (
            jnp.einsum("gij,gapj->gapi", frames[:, :3, :3], local_segments)
            + frames[:, :3, 3][:, None, None, :]
        )
        return world_segments.reshape((self.num_channels, 2, 3))

    def effective_lengths(self, q: Array) -> Array:
        """Return effective McKibben muscle lengths.

        For actuator ``i``, the effective length is its current geometric
        endpoint distance minus ``reference_length[i]`` plus
        ``length_offset[i]``.
        """
        if self.num_channels == 0:
            return jnp.zeros((0,), dtype=q.dtype)
        segments = self.local_segments(q)
        current_length = safe_norm(segments[:, :, 1, :] - segments[:, :, 0, :], axis=-1)
        length = (
            current_length - self.params.reference_length + self.params.length_offset
        )
        return length.reshape((self.num_channels,))

    def coordinates(self, robot, q: Array) -> Array:
        """Return pressure-work coordinates for all McKibben muscles.

        The coordinate is ``(B**2 - length**2) * length / (4*pi*N**2)`` and has
        volume units, making pressure its work-conjugate effort.
        """
        del robot
        if self.num_channels == 0:
            return jnp.zeros((0,), dtype=q.dtype)
        length = self.effective_lengths(q).reshape(self.params.group_shape)
        coordinate = (
            (self.params.thread_length**2 - length**2)
            * length
            / (4.0 * math.pi * self.params.num_turns**2)
        )
        return coordinate.reshape((self.num_channels,))

    def _length_jacobian_local_group(
        self, q: Array, group_index: Array
    ) -> tuple[Array, Array]:
        """Return effective lengths and two-joint derivatives for one group.

        Returns:
            A tuple containing group effective lengths with shape
            ``(actuators_per_group,)`` and their local Jacobian with shape
            ``(actuators_per_group, 2)``.
        """
        transform = self.params.moving_base_transform[group_index]
        joint_pair = self.params.joint_pair_indices[group_index]
        qx, qy = q[joint_pair[0]], q[joint_pair[1]]
        base_rotation = transform[:3, :3]
        rotation_x = principal_axis_rotation_matrix(qx, "x")
        rotation_y = principal_axis_rotation_matrix(qy, "y")
        rotation = base_rotation @ rotation_x @ rotation_y
        rotation_dqx = (
            base_rotation
            @ principal_axis_rotation_matrix_derivative(qx, "x")
            @ rotation_y
        )
        rotation_dqy = (
            base_rotation
            @ rotation_x
            @ principal_axis_rotation_matrix_derivative(qy, "y")
        )
        moving_points = self.params.moving_points[group_index]
        moving = moving_points @ rotation.T + transform[:3, 3]
        displacement = moving - self.params.fixed_points[group_index]
        current_length = safe_norm(displacement, axis=-1)
        direction = safe_normalize(displacement, axis=-1)
        dlength_dqx = jnp.sum(direction * (moving_points @ rotation_dqx.T), axis=-1)
        dlength_dqy = jnp.sum(direction * (moving_points @ rotation_dqy.T), axis=-1)
        effective_length = (
            current_length
            - self.params.reference_length[group_index]
            + self.params.length_offset[group_index]
        )
        return effective_length, jnp.stack([dlength_dqx, dlength_dqy], axis=-1)

    def _moment_matrix_group(self, robot, q: Array, group_index: Array) -> Array:
        """Scatter one group's analytic pressure-to-generalized-force block.

        The local block is ``d(y_a)/d(length) * d(length)/d(q_pair)`` and is
        scattered into the full robot/channel matrix without changing the
        cached actuator ordering.
        """
        length, length_jacobian = self._length_jacobian_local_group(q, group_index)
        force_per_pressure = (
            self.params.thread_length[group_index] ** 2 - 3.0 * length**2
        ) / (4.0 * math.pi * self.params.num_turns[group_index] ** 2)
        local_block = force_per_pressure[:, None] * length_jacobian
        joint_pair = self.params.joint_pair_indices[group_index]
        actuator_indices = group_index * self.params.group_shape[1] + jnp.arange(
            self.params.group_shape[1]
        )
        return (
            jnp.zeros((robot.num_dofs, self.num_channels), dtype=q.dtype)
            .at[joint_pair[:, None], actuator_indices[None, :]]
            .set(local_block.T)
        )

    def moment_matrix(self, robot, q: Array) -> Array:
        """Return the analytic pressure-to-generalized-force matrix ``A(q)``."""
        if self.num_channels == 0:
            return jnp.zeros((robot.num_dofs, 0), dtype=q.dtype)
        indices = jnp.arange(self.params.num_groups)
        matrices = jax.vmap(lambda index: self._moment_matrix_group(robot, q, index))(
            indices
        )
        return jnp.sum(matrices, axis=0)

    def axial_forces(self, q: Array, pressures: Array) -> Array:
        """Return per-muscle axial forces for the supplied pressures.

        The sign convention and constitutive equation match the cached UMArm
        model and its MuJoCo implementation.
        """
        if self.num_channels == 0:
            return jnp.zeros((0,), dtype=q.dtype)
        length = self.effective_lengths(q).reshape(self.params.group_shape)
        pressure = jnp.asarray(pressures).reshape(self.params.group_shape)
        force = (
            pressure
            * (self.params.thread_length**2 - 3.0 * length**2)
            / (4.0 * math.pi * self.params.num_turns**2)
        )
        return force.reshape((self.num_channels,))

    def with_params(
        self, params: ArticulatedMcKibbenTransmissionParams
    ) -> ArticulatedMcKibbenTransmission:
        self.params.assert_same_topology(params)
        params.validate_for_update()
        params = eqx.tree_at(
            lambda current: current.joint_pair_indices,
            params,
            self.params.joint_pair_indices,
        )
        return ArticulatedMcKibbenTransmission(params)


class ArticulatedMcKibbenActuatorParams(BaseSystemParams):
    """McKibben transmission and pressure bounds."""

    transmission: ArticulatedMcKibbenTransmissionParams
    lower_bounds: Array
    upper_bounds: Array

    def __check_init__(self) -> None:
        self.validate_for_update()

    def validate_structure(self) -> None:
        """Validate McKibben actuator parameter structure."""
        self.transmission.validate_structure()
        count = self.transmission.num_channels
        for name in ("lower_bounds", "upper_bounds"):
            if jnp.asarray(getattr(self, name)).shape != (count,):
                raise ValueError(f"{name} must have shape ({count},).")

    def validate_values(self) -> None:
        """Validate eager nested McKibben transmission values."""
        self.transmission.validate_values()


class ArticulatedMcKibbenActuator(Actuator):
    """Direct-pressure McKibben muscle actuator.

    Controls are pressures in pascals. The transmission supplies equivalent
    volume coordinates, while diagnostic helpers expose effective lengths,
    endpoint segments, and signed axial forces. Installation requires a spatial
    articulated robot implementing the kinematic-frame contract; continuum
    hosts must use an appropriate continuum transmission instead.
    """

    _params: ArticulatedMcKibbenActuatorParams
    _transmission: ArticulatedMcKibbenTransmission
    _effort_model: DirectEffort
    _metadata: ActuatorMetadata
    name: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        params: ArticulatedMcKibbenActuatorParams,
        labels: tuple[str, ...] | None = None,
        name: str = "mckibben_muscles",
    ) -> None:
        if not isinstance(params, ArticulatedMcKibbenActuatorParams):
            raise TypeError("params must be ArticulatedMcKibbenActuatorParams.")
        params.validate_for_update()
        count = params.transmission.num_channels
        if labels is None:
            labels = tuple(f"mckibben_pressure_{index}" for index in range(count))
        if len(labels) != count:
            raise ValueError("labels must contain one entry per McKibben muscle.")
        self._params = params
        self._transmission = ArticulatedMcKibbenTransmission(params.transmission)
        self._effort_model = DirectEffort()
        self._metadata = ActuatorMetadata(
            labels=labels,
            units="Pa",
            kind="muscle",
            lower_bounds=params.lower_bounds,
            upper_bounds=params.upper_bounds,
        )
        self.name = name

    @classmethod
    def from_cached_npz(cls, path: str | Path) -> ArticulatedMcKibbenActuator:
        transmission = ArticulatedMcKibbenTransmissionParams.from_cached_npz(path)
        count = transmission.num_channels
        return cls(
            params=ArticulatedMcKibbenActuatorParams(
                transmission=transmission,
                lower_bounds=jnp.zeros((count,)),
                upper_bounds=jnp.full((count,), jnp.inf),
            )
        )

    @classmethod
    def empty(cls, actuators_per_group: int = 4) -> ArticulatedMcKibbenActuator:
        """Construct a zero-channel actuator with fixed empty group topology."""
        group_shape = (0, int(actuators_per_group))
        transmission = ArticulatedMcKibbenTransmissionParams(
            moving_base_transform=jnp.zeros((0, 4, 4)),
            moving_points=jnp.zeros((*group_shape, 3)),
            fixed_points=jnp.zeros((*group_shape, 3)),
            reference_length=jnp.zeros(group_shape),
            length_offset=jnp.zeros(group_shape),
            thread_length=jnp.zeros(group_shape),
            num_turns=jnp.zeros(group_shape),
            joint_pair_indices=jnp.zeros((0, 2), dtype=jnp.int32),
        )
        return cls(
            params=ArticulatedMcKibbenActuatorParams(
                transmission=transmission,
                lower_bounds=jnp.zeros((0,)),
                upper_bounds=jnp.zeros((0,)),
            )
        )

    @property
    def params(self) -> ArticulatedMcKibbenActuatorParams:
        return self._params

    @property
    def transmission(self) -> ArticulatedMcKibbenTransmission:
        return self._transmission

    @property
    def effort_model(self) -> DirectEffort:
        return self._effort_model

    @property
    def metadata(self) -> ActuatorMetadata:
        return self._metadata

    def validate_structure_for_robot(self, robot) -> None:
        if not callable(getattr(robot, "_kinematic_frames", None)):
            raise TypeError(
                "ArticulatedMcKibbenActuator requires a spatial articulated host "
                "implementing the _kinematic_frames geometry contract."
            )

    def validate_values_for_robot(self, robot) -> None:
        indices = self.params.transmission.joint_pair_indices
        if indices.shape[0] > 0 and (
            np.any(np.asarray(indices) < 0)
            or np.any(np.asarray(indices) >= robot.num_internal_dofs)
        ):
            raise ValueError("joint_pair_indices must reference valid robot DOFs.")

    def _robot_value_validation_tree(self):
        return self.params.transmission.joint_pair_indices

    def with_params(self, params: BaseSystemParams) -> ArticulatedMcKibbenActuator:
        if not isinstance(params, ArticulatedMcKibbenActuatorParams):
            raise TypeError("params must be ArticulatedMcKibbenActuatorParams.")
        self.params.transmission.assert_same_topology(params.transmission)
        params.validate_for_update()
        params = eqx.tree_at(
            lambda current: current.transmission.joint_pair_indices,
            params,
            self.params.transmission.joint_pair_indices,
        )
        return ArticulatedMcKibbenActuator(
            params=params, labels=self.metadata.labels, name=self.name
        )

    def local_segments(self, q: Array) -> Array:
        return self.transmission.local_segments(q)

    def segments(self, robot, q: Array) -> Array:
        return self.transmission.segments(robot, q)

    def effective_lengths(self, q: Array) -> Array:
        return self.transmission.effective_lengths(q)

    def axial_forces(self, q: Array, pressures: Array) -> Array:
        return self.transmission.axial_forces(q, pressures)


__all__ = [
    "ArticulatedMcKibbenActuator",
    "ArticulatedMcKibbenActuatorParams",
    "ArticulatedMcKibbenTransmission",
    "ArticulatedMcKibbenTransmissionParams",
]
