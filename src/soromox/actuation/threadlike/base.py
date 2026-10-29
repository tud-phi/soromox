"""Strategy contract shared by fixed and configuration-dependent routing."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import equinox as eqx
from jax import Array

from soromox.systems.params import BaseSystemParams

if TYPE_CHECKING:
    from soromox.actuation.friction import ThreadlikeFriction
    from soromox.systems.soft_robot import SoftRobot


class BaseThreadlikeRouting(eqx.Module):
    """Supply physical path geometry independently of actuator sign and scale.

    Transmissions delegate to this contract rather than robot-specific hooks.
    Path lengths are in metres. Velocity Jacobians have ``num_velocities``
    columns, including zero floating-base columns for internal paths.
    Strategies retain their own geometry, solver settings, and parameter types.
    """

    params: BaseSystemParams

    @property
    @abstractmethod
    def num_paths(self) -> int:
        """Number of physical filaments, each corresponding to one channel."""
        ...

    @abstractmethod
    def path_lengths(self, robot: SoftRobot, q: Array) -> Array:
        """Return physical lengths, shape ``(num_paths,)``."""
        ...

    @abstractmethod
    def path_velocity_jacobian(self, robot: SoftRobot, q: Array) -> Array:
        """Map generalized velocity to length rates, shape ``(paths, velocities)``."""
        ...

    @abstractmethod
    def path_positions(self, robot: SoftRobot, q: Array, s: Array) -> Array:
        """Sample paths at a backbone station, shape ``(paths, 2 or 3)``.

        Stations outside a path clamp to its endpoints. Guided strategies
        parameterize each span by its backbone interval; at a moving joint,
        the station selects the upstream guide. Use their span sampling API
        to represent both guides and the connecting cable explicitly.
        """
        ...

    def validate_friction(self, friction: ThreadlikeFriction) -> None:
        """Reject effort loss unless the strategy implements a friction model."""
        if not friction.is_frictionless:
            raise ValueError(
                f"{type(self).__name__} supports only frictionless transmission."
            )

    def path_force_matrix(
        self, robot: SoftRobot, q: Array, *, friction: ThreadlikeFriction
    ) -> Array:
        """Map length-conjugate efforts to generalized forces before scaling.

        Positive efforts favor lengthening. A tendon transmission supplies a
        negative coordinate scale. Frictionless paths obey virtual work.
        """
        self.validate_friction(friction)
        return self.path_velocity_jacobian(robot, q).T

    @abstractmethod
    def with_params(self, params: BaseSystemParams) -> BaseThreadlikeRouting:
        """Replace numerical geometry while preserving topology and settings."""
        ...

    def update_params(self, **updates) -> BaseThreadlikeRouting:
        """Replace selected geometry parameters using the strategy's validation."""
        return self.with_params(self.params.replace(**updates))

    @abstractmethod
    def validate_structure_for_robot(self, robot: SoftRobot) -> None:
        """Validate host compatibility without reading traced parameter values."""
        ...

    @abstractmethod
    def validate_values_for_robot(self, robot: SoftRobot) -> None:
        """Validate concrete geometry against the host."""
        ...
