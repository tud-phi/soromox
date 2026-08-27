"""Composable actuation primitives.

The actuation model follows a work-conjugate formulation. A transmission
defines actuator coordinates ``y_a(q)`` and their moment matrix
``A(q) = dy_a/dq.T``. An effort model maps user controls to efforts conjugate to
those coordinates, and generalized actuation forces are ``A(q) @ effort``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar, Literal, final

import equinox as eqx
from jax import Array, ensure_compile_time_eval
from jax import numpy as jnp

from soromox.systems.params import BaseSystemParams, _contains_tracer

if TYPE_CHECKING:
    from soromox.systems.soft_robot import SoftRobot

ActuatorKind = Literal[
    "identity", "tendon", "push_rod", "muscle", "pneumatic", "generic"
]


class ActuatorMetadata(eqx.Module):
    """Ordered labels, units, limits, and semantics for input channels."""

    labels: tuple[str, ...] = eqx.field(static=True)
    units: tuple[str, ...] = eqx.field(static=True)
    kind: ActuatorKind = eqx.field(static=True)
    lower_bounds: Array
    upper_bounds: Array

    def __init__(
        self,
        *,
        labels: tuple[str, ...],
        units: str | tuple[str, ...],
        kind: ActuatorKind = "generic",
        lower_bounds: Array | None = None,
        upper_bounds: Array | None = None,
    ) -> None:
        count = len(labels)
        if isinstance(units, str):
            units = (units,) * count
        if len(units) != count:
            raise ValueError("units must contain one entry per actuator channel.")
        self.labels = tuple(labels)
        self.units = tuple(units)
        self.kind = kind
        self.lower_bounds = (
            jnp.full((count,), -jnp.inf)
            if lower_bounds is None
            else jnp.asarray(lower_bounds)
        )
        self.upper_bounds = (
            jnp.full((count,), jnp.inf)
            if upper_bounds is None
            else jnp.asarray(upper_bounds)
        )
        if self.lower_bounds.shape != (count,):
            raise ValueError("lower_bounds must have shape (num_channels,).")
        if self.upper_bounds.shape != (count,):
            raise ValueError("upper_bounds must have shape (num_channels,).")


class Transmission(eqx.Module):
    """Map robot configuration to work-conjugate actuator coordinates."""

    @property
    @abstractmethod
    def num_channels(self) -> int:
        """Number of scalar transmission coordinates."""
        ...

    @abstractmethod
    def coordinates(self, robot: SoftRobot, q: Array) -> Array:
        """Return transmission coordinates ``y_a(q)``."""
        ...

    @abstractmethod
    def moment_matrix(self, robot: SoftRobot, q: Array) -> Array:
        """Return ``dy_a/dq.T`` with shape ``(num_dofs, num_channels)``."""
        ...

    def velocities(self, robot: SoftRobot, q: Array, qd: Array) -> Array:
        """Return transmission-coordinate velocities."""
        return self.moment_matrix(robot, q).T @ qd


class EffortModel(eqx.Module):
    """Map controls to efforts work-conjugate to transmission coordinates."""

    requires_coordinate: ClassVar[bool] = True
    requires_velocity: ClassVar[bool] = True

    @abstractmethod
    def effort(
        self,
        control: Array,
        coordinate: Array | None,
        velocity: Array | None,
    ) -> Array:
        """Return actuator effort for the current control and transmission state."""
        ...


class DirectEffort(EffortModel):
    """Stateless effort model for which effort is exactly the user control."""

    requires_coordinate: ClassVar[bool] = False
    requires_velocity: ClassVar[bool] = False

    def effort(
        self,
        control: Array,
        coordinate: Array | None,
        velocity: Array | None,
    ) -> Array:
        del coordinate, velocity
        return control


class PassiveElement(eqx.Module):
    """Independent passive contribution to robot mechanics."""

    @property
    @abstractmethod
    def params(self) -> BaseSystemParams: ...

    @abstractmethod
    def with_params(self, params: BaseSystemParams) -> PassiveElement: ...

    def update_params(self, **updates) -> PassiveElement:
        return self.with_params(self.params.replace(**updates))

    def validate_structure_for_robot(self, robot: SoftRobot) -> None:
        """Validate tracer-safe component compatibility with ``robot``."""
        del robot

    def validate_values_for_robot(self, robot: SoftRobot) -> None:
        """Validate concrete component values against ``robot``."""
        del robot

    def _robot_value_validation_tree(self):
        """Return the leaves used by concrete robot-compatibility checks."""
        return self.params

    @final
    def validate_for_robot(self, robot: SoftRobot) -> None:
        """Validate robot compatibility without concretizing dynamic leaves."""
        self.validate_structure_for_robot(robot)
        if _contains_tracer(self._robot_value_validation_tree()):
            return
        with ensure_compile_time_eval():
            self.validate_values_for_robot(robot)

    def elastic_force(self, robot: SoftRobot, q: Array) -> Array:
        return jnp.zeros((robot.num_dofs,), dtype=q.dtype)

    def damping_matrix(self, robot: SoftRobot, q: Array) -> Array:
        return jnp.zeros((robot.num_dofs, robot.num_dofs), dtype=q.dtype)

    def elastic_energy(self, robot: SoftRobot, q: Array) -> Array:
        del robot
        return jnp.zeros((), dtype=q.dtype)


class Actuator(eqx.Module):
    """A transmission paired with an effort law and ordered channel metadata."""

    @property
    @abstractmethod
    def transmission(self) -> Transmission: ...

    @property
    @abstractmethod
    def effort_model(self) -> EffortModel: ...

    @property
    @abstractmethod
    def params(self) -> BaseSystemParams: ...

    @property
    @abstractmethod
    def metadata(self) -> ActuatorMetadata: ...

    @property
    def num_channels(self) -> int:
        return self.transmission.num_channels

    def coordinates(self, robot: SoftRobot, q: Array) -> Array:
        return self.transmission.coordinates(robot, q)

    def velocities(self, robot: SoftRobot, q: Array, qd: Array) -> Array:
        return self.transmission.velocities(robot, q, qd)

    def efforts(
        self,
        robot: SoftRobot,
        q: Array,
        control: Array,
        qd: Array | None = None,
        *,
        transmission_matrix: Array | None = None,
    ) -> Array:
        """Map controls to work-conjugate efforts for this actuator.

        Only transmission coordinates and velocities required by the installed
        effort model are evaluated. A precomputed transmission matrix may be
        supplied to reuse it when evaluating actuator-coordinate velocities.

        Args:
            robot: Soft robot on which the actuator is installed.
            q: Generalized coordinates of shape ``(robot.num_dofs,)``.
            control: Controls for this actuator with shape
                ``(self.num_channels,)``.
            qd: Optional generalized velocities of shape
                ``(robot.num_dofs,)``. If omitted for a velocity-dependent
                effort model, zero generalized velocity is used.
            transmission_matrix: Optional precomputed transmission matrix with
                shape ``(robot.num_dofs, self.num_channels)``.

        Returns:
            e: Work-conjugate actuator efforts with shape
                ``(self.num_channels,)``.

        Raises:
            ValueError: If ``transmission_matrix`` has an incompatible shape.
        """
        if transmission_matrix is not None and transmission_matrix.shape != (
            robot.num_dofs,
            self.num_channels,
        ):
            raise ValueError(
                "transmission_matrix must have shape "
                f"({robot.num_dofs}, {self.num_channels}), got "
                f"{transmission_matrix.shape}."
            )

        coordinate = None
        if self.effort_model.requires_coordinate:
            coordinate = self.coordinates(robot, q)

        velocity = None
        if self.effort_model.requires_velocity:
            if qd is None:
                qd = jnp.zeros_like(q)
            if transmission_matrix is None:
                velocity = self.velocities(robot, q, qd)
            else:
                velocity = transmission_matrix.T @ qd

        return self.effort_model.effort(control, coordinate, velocity)

    def validate_structure_for_robot(self, robot: SoftRobot) -> None:
        """Validate tracer-safe component compatibility with ``robot``."""
        del robot

    def validate_values_for_robot(self, robot: SoftRobot) -> None:
        """Validate concrete component values against ``robot``."""
        del robot

    def _robot_value_validation_tree(self):
        """Return the leaves used by concrete robot-compatibility checks."""
        return self.params

    @final
    def validate_for_robot(self, robot: SoftRobot) -> None:
        """Validate robot compatibility without concretizing dynamic leaves."""
        self.validate_structure_for_robot(robot)
        if _contains_tracer(self._robot_value_validation_tree()):
            return
        with ensure_compile_time_eval():
            self.validate_values_for_robot(robot)

    @abstractmethod
    def with_params(self, params: BaseSystemParams) -> Actuator: ...

    def update_params(self, **updates) -> Actuator:
        return self.with_params(self.params.replace(**updates))


class IdentityActuatorParams(BaseSystemParams):
    """Empty immutable parameter object for identity actuation."""


class IdentityTransmission(Transmission):
    """Identity generalized-coordinate transmission."""

    _num_channels: int = eqx.field(static=True)

    def __init__(self, num_channels: int) -> None:
        if num_channels < 0:
            raise ValueError("num_channels must be non-negative.")
        self._num_channels = int(num_channels)

    @property
    def num_channels(self) -> int:
        return self._num_channels

    def coordinates(self, robot: SoftRobot, q: Array) -> Array:
        del robot
        return q

    def moment_matrix(self, robot: SoftRobot, q: Array) -> Array:
        del robot
        return jnp.eye(self.num_channels, dtype=q.dtype)


class IdentityActuator(Actuator):
    """Direct generalized-coordinate actuation used by default."""

    _transmission: IdentityTransmission
    _effort_model: DirectEffort
    _params: IdentityActuatorParams
    _metadata: ActuatorMetadata

    def __init__(self, num_channels: int) -> None:
        self._transmission = IdentityTransmission(num_channels)
        self._effort_model = DirectEffort()
        self._params = IdentityActuatorParams()
        self._metadata = ActuatorMetadata(
            labels=tuple(f"generalized_effort_{idx}" for idx in range(num_channels)),
            units="generalized effort",
            kind="identity",
        )

    @property
    def transmission(self) -> IdentityTransmission:
        return self._transmission

    @property
    def effort_model(self) -> DirectEffort:
        return self._effort_model

    @property
    def params(self) -> IdentityActuatorParams:
        return self._params

    @property
    def metadata(self) -> ActuatorMetadata:
        return self._metadata

    def validate_structure_for_robot(self, robot: SoftRobot) -> None:
        if self.num_channels != robot.num_dofs:
            raise ValueError(
                "IdentityActuator must have one channel per robot degree of "
                f"freedom; expected {robot.num_dofs}, got {self.num_channels}."
            )

    def with_params(self, params: BaseSystemParams) -> IdentityActuator:
        if not isinstance(params, IdentityActuatorParams):
            raise TypeError("params must be IdentityActuatorParams.")
        return self


__all__ = [
    "Actuator",
    "ActuatorKind",
    "ActuatorMetadata",
    "DirectEffort",
    "EffortModel",
    "IdentityActuator",
    "IdentityActuatorParams",
    "IdentityTransmission",
    "PassiveElement",
    "Transmission",
]
