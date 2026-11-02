"""Strategy contract shared by fixed and configuration-dependent routing."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import equinox as eqx
from jax import Array

from soromox.systems.params import BaseSystemParams

if TYPE_CHECKING:
    from soromox.actuation.friction import ThreadlikeFriction
    from soromox.systems.soft_robot import SoftRobot


class BaseThreadlikeRouting(eqx.Module):
    """Supply physical path geometry independently of actuator sign and scale.

    Transmissions delegate to this contract rather than robot-specific hooks.
    Path lengths are in metres. Velocity Jacobians include zero floating-base
    columns for internal paths. Each strategy retains its geometry, solver
    settings, and parameter type.

    Attributes:
        params: Numerical geometry and static topology of the routing strategy.
    """

    params: BaseSystemParams

    @property
    @abstractmethod
    def num_paths(self) -> int:
        """Return the number of physical filaments in the routing family.

        Returns:
            Number of paths, each corresponding to one actuator channel.
        """
        ...

    @abstractmethod
    def path_lengths(self, robot: SoftRobot, q: Array) -> Array:
        """Evaluate the physical length of every routed path.

        Args:
            robot: Continuum host supplying the strategy's required kinematics.
            q: Configuration with shape ``(robot.num_coordinates,)``.

        Returns:
            Physical lengths in metres with shape ``(self.num_paths,)``, before
            applying actuator coordinate signs or scales.
        """
        ...

    @abstractmethod
    def path_velocity_jacobian(self, robot: SoftRobot, q: Array) -> Array:
        """Map generalized velocities to physical path-length rates.

        Args:
            robot: Continuum host supplying the strategy's required kinematics.
            q: Configuration with shape ``(robot.num_coordinates,)``.

        Returns:
            Jacobian with shape ``(self.num_paths, robot.num_velocities)``.
            Multiplication by generalized velocity gives length rates in metres
            per second. Internal paths have zero floating-base columns.
        """
        ...

    @abstractmethod
    def path_positions(self, robot: SoftRobot, q: Array, s: Array) -> Array:
        """Sample each path at a reference backbone station.

        Guided strategies parameterize each span by its backbone interval.
        At a moving joint, the station selects the upstream guide; use the
        strategy's span sampling API to represent both guides and their cable
        connection explicitly.

        Args:
            robot: Continuum host defining world-frame path positions.
            q: Configuration with shape ``(robot.num_coordinates,)``.
            s: Scalar reference backbone station in metres, clamped to each
                path's endpoints.

        Returns:
            World positions in metres with shape ``(self.num_paths, 2)`` for
            planar hosts or ``(self.num_paths, 3)`` for spatial hosts.
        """
        ...

    def validate_friction(self, friction: ThreadlikeFriction) -> None:
        """Check whether this strategy supports the supplied friction model.

        Args:
            friction: Runtime effort-loss model to pair with the routing.

        Raises:
            ValueError: The model is not frictionless. Strategies implementing
                friction override this check with their own requirements.
        """
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

        Args:
            robot: Continuum host receiving the generalized forces.
            q: Configuration with shape ``(robot.num_coordinates,)``.
            friction: Runtime effort-loss model supported by this strategy.

        Returns:
            Force map with shape ``(robot.num_velocities, self.num_paths)``.
            The default implementation returns the transpose of the physical
            length-rate Jacobian.

        Raises:
            ValueError: The strategy does not support the friction model.
        """
        self.validate_friction(friction)
        return self.path_velocity_jacobian(robot, q).T

    @abstractmethod
    def with_params(self, params: BaseSystemParams) -> BaseThreadlikeRouting:
        """Replace numerical geometry while preserving topology and settings.

        Args:
            params: Replacement parameters of the strategy's existing type,
                with the same path count and connectivity.

        Returns:
            Routing with updated parameter leaves and preserved runtime
            callbacks, discretization, and solver settings where applicable.

        Raises:
            TypeError: The replacement parameter type is incompatible.
            ValueError: Parameters are invalid or change routing topology.
        """
        ...

    def update_params(self, **updates: Any) -> BaseThreadlikeRouting:
        """Replace selected geometry parameters through validated replacement.

        Args:
            **updates: Parameter field names and replacement values accepted by
                this strategy's parameter object.

        Returns:
            Routing with the requested numerical changes and unchanged topology.

        Raises:
            TypeError: A field name or replacement type is unsupported.
            ValueError: A replacement is invalid or changes routing topology.
        """
        return self.with_params(self.params.replace(**updates))

    @abstractmethod
    def validate_structure_for_robot(self, robot: SoftRobot) -> None:
        """Check host compatibility without reading traced parameter values.

        Args:
            robot: Host whose kinematic interface and segment topology must
                support this routing strategy.

        Raises:
            TypeError: The host lacks a required interface.
            ValueError: Routing shapes or connectivity are incompatible with
                the host.
        """
        ...

    @abstractmethod
    def validate_values_for_robot(self, robot: SoftRobot) -> None:
        """Check concrete routing geometry against the host.

        Args:
            robot: Host supplying physical geometry. Call this check with
                concrete numerical values, after structure validation.

        Raises:
            ValueError: Numerical geometry violates a strategy or host
                restriction, such as nonplanar offsets on a planar robot.
        """
        ...
