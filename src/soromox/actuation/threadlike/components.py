"""Transmissions, active actuators, and passive mechanics for all routing strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import equinox as eqx
from jax import Array
from jax import numpy as jnp

from soromox.systems.params import BaseSystemParams

from ..core import (
    Actuator,
    ActuatorMetadata,
    DirectEffort,
    PassiveElement,
    Transmission,
)
from ..friction import (
    BaseThreadlikeFrictionParams,
    FrictionlessParams,
    ThreadlikeFriction,
)
from .base import BaseThreadlikeRouting
from .fixed_routing import (
    BaseThreadlikeRoutingParams,
    LinearThreadlikeRoutingParams,
    ThreadlikeRouting,
)
from .routing import RoutingSpanParams
from .shortest_path import ShortestPathRouting, ShortestPathRoutingParams
from .straight_span import StraightSpanRouting

if TYPE_CHECKING:
    from soromox.systems.soft_robot import SoftRobot

ThreadlikeKind = Literal["tendon", "push_rod", "muscle", "pneumatic", "generic"]


def _channel_array(value: Any, count: int, name: str) -> Array:
    """Broadcast a scalar or validate one value per actuation channel.

    Args:
        value: Scalar or one-dimensional array of channel values.
        count: Required number of actuator channels.
        name: Parameter name used in validation messages.

    Returns:
        Array with shape ``(count,)`` and the input value's numerical dtype.

    Raises:
        ValueError: A nonscalar input does not have shape ``(count,)``.
    """
    array = jnp.asarray(value)
    if array.ndim == 0:
        array = jnp.full((count,), array)
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},), got {array.shape}.")
    return array


def _normalize_friction(friction: Any) -> ThreadlikeFriction | None:
    """Validate an optional runtime threadlike-friction object.

    Args:
        friction: Runtime friction model, or ``None`` for the default model.

    Returns:
        The supplied friction model or ``None``.

    Raises:
        TypeError: A non-None value is not a ``ThreadlikeFriction`` instance.
    """
    if friction is not None and not isinstance(friction, ThreadlikeFriction):
        raise TypeError(
            "friction must be a ThreadlikeFriction instance, got "
            f"{type(friction).__name__}."
        )
    return friction


# Numerical parameter types supported by the built-in strategy constructors.
ThreadlikeRoutingParameters = (
    BaseThreadlikeRoutingParams | RoutingSpanParams | ShortestPathRoutingParams
)


def _normalize_routing(
    routing: BaseThreadlikeRouting | ThreadlikeRoutingParameters,
) -> BaseThreadlikeRouting:
    """Keep runtime strategies or construct a built-in strategy from parameters.

    Args:
        routing: Runtime routing strategy, or parameters for fixed linear,
            straight-span, or shortest-path routing. Runtime objects preserve
            custom callbacks and solver settings.

    Returns:
        Routing strategy. Parameter-only inputs use the corresponding built-in
        constructor and its default algorithm settings.

    Raises:
        TypeError: The input is unsupported, or custom fixed-routing parameters
            are supplied without their runtime callbacks.
    """
    if isinstance(routing, BaseThreadlikeRouting):
        return routing
    if isinstance(routing, LinearThreadlikeRoutingParams):
        return ThreadlikeRouting(params=routing)
    if isinstance(routing, RoutingSpanParams):
        return StraightSpanRouting(routing)
    if isinstance(routing, ShortestPathRoutingParams):
        return ShortestPathRouting(routing)
    if isinstance(routing, BaseThreadlikeRoutingParams):
        raise TypeError(
            "custom routing params must be wrapped in ThreadlikeRouting with "
            "offset_fn and derivative_fn."
        )
    raise TypeError(
        "routing must be a BaseThreadlikeRouting or built-in routing parameters."
    )


class ThreadlikeTransmissionParams(BaseSystemParams):
    """Routing, work-coordinate scale, and friction parameters per path."""

    routing: ThreadlikeRoutingParameters
    coordinate_scale: Array
    friction: BaseThreadlikeFrictionParams = eqx.field(
        default_factory=FrictionlessParams
    )

    def __check_init__(self) -> None:
        """Validate newly constructed transmission parameters."""
        self.validate_for_update()

    def validate_structure(self) -> None:
        """Validate threadlike transmission parameter structure."""
        self.routing.validate_structure()
        scale = jnp.asarray(self.coordinate_scale)
        if scale.shape != (self.routing.num_paths,):
            raise ValueError(
                "coordinate_scale must have shape "
                f"({self.routing.num_paths},), got {scale.shape}."
            )
        if not isinstance(self.friction, BaseThreadlikeFrictionParams):
            raise TypeError(
                "friction must be a BaseThreadlikeFrictionParams instance, got "
                f"{type(self.friction).__name__}."
            )
        self.friction.validate_structure()
        self.friction.validate_structure_compatibility(self.routing.num_paths)

    def validate_values(self) -> None:
        """Validate eager nested routing and friction values."""
        self.routing.validate_values()
        self.friction.validate_values()


class ThreadlikeTransmission(Transmission):
    """Map scaled path lengths and anchor efforts to robot coordinates."""

    params: ThreadlikeTransmissionParams
    routing: BaseThreadlikeRouting
    friction: ThreadlikeFriction

    def __init__(
        self,
        params: ThreadlikeTransmissionParams,
        *,
        routing: BaseThreadlikeRouting | None = None,
        friction: ThreadlikeFriction | None = None,
    ) -> None:
        """Construct a runtime transmission from matching parameter objects.

        Args:
            params: Nested routing, coordinate-scale, and friction parameters.
            routing: Runtime strategy, including any custom callbacks or solver
                settings. Omission reconstructs the built-in strategy from its
                parameter type using default settings.
            friction: Runtime effort-ratio law. Frictionless behavior is
                reconstructed automatically when this argument is omitted.
        """
        if not isinstance(params, ThreadlikeTransmissionParams):
            raise TypeError("params must be ThreadlikeTransmissionParams.")
        params.validate_for_update()
        self.params = params
        if routing is None:
            routing = _normalize_routing(params.routing)
        else:
            routing = _normalize_routing(routing).with_params(params.routing)
        self.routing = routing
        if friction is None:
            if not isinstance(params.friction, FrictionlessParams):
                raise TypeError(
                    "friction params require their matching "
                    "ThreadlikeFriction runtime object."
                )
            friction = ThreadlikeFriction()
        else:
            friction = friction.with_params(params.friction)
        self.friction = friction

    @property
    def num_channels(self) -> int:
        """Return the number of independently routed paths."""
        return self.routing.num_paths

    def coordinates(self, robot: SoftRobot, q: Array) -> Array:
        """Return scaled routed-path lengths.

        Args:
            robot: Continuum robot hosting the routing.
            q: Generalized coordinates with shape
                ``(robot.num_coordinates,)``.

        Returns:
            y_a: Work-conjugate coordinates with shape
                ``(self.num_channels,)``.
        """
        return self.params.coordinate_scale * self.path_lengths(robot, q)

    def coordinate_jacobian(self, robot: SoftRobot, q: Array) -> Array:
        """Return the scaled path-coordinate Jacobian.

        The Jacobian maps generalized velocity to actuator-coordinate velocity
        as ``yd_a = J_a(q) @ qd``.

        Args:
            robot: Continuum robot hosting the routing.
            q: Generalized coordinates with shape
                ``(robot.num_coordinates,)``.

        Returns:
            J_a: Coordinate Jacobian with shape
                ``(self.num_channels, robot.num_velocities)``.
        """
        raw_jacobian = self.routing.path_velocity_jacobian(robot, q)
        return self.params.coordinate_scale[:, None] * raw_jacobian

    def actuation_matrix(self, robot: SoftRobot, q: Array) -> Array:
        """Return the anchor-effort to generalized-force map.

        Friction weights each local contribution by its surviving effort ratio,
        so this matrix need not equal :meth:`coordinate_jacobian` transposed.

        Args:
            robot: Continuum robot hosting the routing.
            q: Generalized coordinates with shape
                ``(robot.num_coordinates,)``.

        Returns:
            A: Actuation matrix with shape
                ``(robot.num_velocities, self.num_channels)``.
        """
        raw_matrix = self.routing.path_force_matrix(robot, q, friction=self.friction)
        return raw_matrix * self.params.coordinate_scale[None, :]

    def path_lengths(self, robot: SoftRobot, q: Array) -> Array:
        """Return unscaled physical routed-path lengths.

        Args:
            robot: Continuum robot hosting the routing.
            q: Generalized coordinates with shape
                ``(robot.num_coordinates,)``.

        Returns:
            Path lengths with shape ``(self.num_channels,)``.
        """
        return self.routing.path_lengths(robot, q)

    def path_length_jacobian(self, robot: SoftRobot, q: Array) -> Array:
        """Return the unscaled physical path-length Jacobian.

        This Jacobian maps generalized velocity to physical path-length rate.

        Args:
            robot: Continuum robot hosting the routing.
            q: Generalized coordinates with shape
                ``(robot.num_coordinates,)``.

        Returns:
            J_l: Path-length Jacobian with shape
                ``(self.num_channels, robot.num_velocities)``.
        """
        return self.routing.path_velocity_jacobian(robot, q)

    def path_velocities(self, robot: SoftRobot, q: Array, qd: Array) -> Array:
        """Return unscaled physical path-length rates.

        Args:
            robot: Continuum robot hosting the routing.
            q: Generalized coordinates with shape
                ``(robot.num_coordinates,)``.
            qd: Generalized velocities with shape ``(robot.num_velocities,)``.

        Returns:
            Path-length rates with shape ``(self.num_channels,)``.
        """
        return self.path_length_jacobian(robot, q) @ qd

    def path_poses(self, robot: SoftRobot, q: Array, s: Array) -> Array:
        """Return routed-path positions at a backbone abscissa.

        Args:
            robot: Continuum robot hosting the routing.
            q: Generalized coordinates with shape
                ``(robot.num_coordinates,)``.
            s: Scalar backbone abscissa.

        Returns:
            Path positions with shape ``(self.num_channels, 2)`` for planar
            hosts or ``(self.num_channels, 3)`` for spatial hosts.
        """
        return self.routing.path_positions(robot, q, s)

    def with_params(
        self, params: ThreadlikeTransmissionParams
    ) -> ThreadlikeTransmission:
        """Return a copy carrying replacement transmission parameter leaves."""
        if not isinstance(params, ThreadlikeTransmissionParams):
            raise TypeError("params must be ThreadlikeTransmissionParams.")
        self.params.routing.assert_same_topology(params.routing)
        params.validate_for_update()
        return ThreadlikeTransmission(
            params, routing=self.routing, friction=self.friction
        )

    def update_params(self, **updates: Any) -> ThreadlikeTransmission:
        """Return a copy with replaced transmission parameter leaves."""
        return self.with_params(self.params.replace(**updates))


class ThreadlikeActuatorParams(BaseSystemParams):
    """Physical and control parameters for a threadlike actuator family."""

    transmission: ThreadlikeTransmissionParams
    lower_bounds: Array
    upper_bounds: Array

    def __check_init__(self) -> None:
        """Validate newly constructed threadlike-actuator parameters."""
        self.validate_for_update()

    def validate_structure(self) -> None:
        """Validate threadlike actuator parameter structure."""
        self.transmission.validate_structure()
        count = self.transmission.routing.num_paths
        for name in ("lower_bounds", "upper_bounds"):
            value = jnp.asarray(getattr(self, name))
            if value.shape != (count,):
                raise ValueError(
                    f"{name} must have shape ({count},), got {value.shape}."
                )

    def validate_values(self) -> None:
        """Validate eager nested threadlike transmission values."""
        self.transmission.validate_values()


class ThreadlikeActuator(Actuator):
    """Vectorized threadlike actuator with modality-specific presets."""

    _params: ThreadlikeActuatorParams
    _transmission: ThreadlikeTransmission
    _effort_model: DirectEffort
    _metadata: ActuatorMetadata
    name: str = eqx.field(static=True)
    kind: ThreadlikeKind = eqx.field(static=True)

    def __init__(
        self,
        *,
        params: ThreadlikeActuatorParams,
        routing: BaseThreadlikeRouting | None = None,
        friction: ThreadlikeFriction | None = None,
        labels: tuple[str, ...],
        units: str | tuple[str, ...],
        name: str,
        kind: ThreadlikeKind,
    ) -> None:
        """Construct a vectorized threadlike actuator.

        Prefer the modality-specific constructors such as :meth:`tendons` or
        :meth:`push_rods` when their effort sign conventions apply.
        """
        params.validate_for_update()
        count = params.transmission.routing.num_paths
        if len(labels) != count:
            raise ValueError("labels must contain one entry per threadlike path.")
        self._params = params
        self._transmission = ThreadlikeTransmission(
            params.transmission, routing=routing, friction=friction
        )
        self._effort_model = DirectEffort()
        self._metadata = ActuatorMetadata(
            labels=labels,
            units=units,
            kind=kind,
            lower_bounds=params.lower_bounds,
            upper_bounds=params.upper_bounds,
        )
        self.name = name
        self.kind = kind

    @staticmethod
    def _normalize_routing(
        routings: BaseThreadlikeRouting | ThreadlikeRoutingParameters,
    ) -> BaseThreadlikeRouting:
        """Normalize a routing strategy or its built-in parameters.

        Args:
            routings: Runtime strategy or parameters for a built-in strategy.

        Returns:
            Runtime routing object, preserving callbacks and solver settings
            when an existing strategy is supplied.

        Raises:
            TypeError: The input is unsupported or requires custom callbacks.
        """
        return _normalize_routing(routings)

    @classmethod
    def _preset(
        cls,
        routings: BaseThreadlikeRouting | ThreadlikeRoutingParameters,
        *,
        coordinate_scale: Array,
        kind: ThreadlikeKind,
        name: str,
        unit: str,
        label_prefix: str,
        lower_bounds: Array | float = 0.0,
        upper_bounds: Array | float = jnp.inf,
        labels: tuple[str, ...] | None = None,
        friction: ThreadlikeFriction | None = None,
    ) -> ThreadlikeActuator:
        """Construct a modality preset with a prescribed coordinate scale."""
        routing = cls._normalize_routing(routings)
        friction = _normalize_friction(friction)
        count = routing.num_paths
        params = ThreadlikeActuatorParams(
            transmission=ThreadlikeTransmissionParams(
                routing=routing.params,
                coordinate_scale=_channel_array(
                    coordinate_scale, count, "coordinate_scale"
                ),
                friction=(
                    friction.params if friction is not None else FrictionlessParams()
                ),
            ),
            lower_bounds=_channel_array(lower_bounds, count, "lower_bounds"),
            upper_bounds=_channel_array(upper_bounds, count, "upper_bounds"),
        )
        if labels is None:
            labels = tuple(f"{label_prefix}_{index}" for index in range(count))
        return cls(
            params=params,
            routing=routing,
            friction=friction,
            labels=labels,
            units=unit,
            name=name,
            kind=kind,
        )

    @classmethod
    def tendons(
        cls,
        routings: BaseThreadlikeRouting | ThreadlikeRoutingParameters,
        *,
        labels: tuple[str, ...] | None = None,
        friction: ThreadlikeFriction | None = None,
    ) -> ThreadlikeActuator:
        """Construct tension-only tendons with shortening-positive coordinates.

        Args:
            routings: Fixed, straight-span, or shortest-path routing geometry.
            labels: Optional label for each routed path.
            friction: Optional active effort-loss model.

        Returns:
            Vectorized tendon actuator with one channel per path.
        """
        routing = cls._normalize_routing(routings)
        return cls._preset(
            routing,
            coordinate_scale=-jnp.ones((routing.num_paths,)),
            kind="tendon",
            name="tendons",
            unit="N",
            label_prefix="tendon_tension",
            labels=labels,
            friction=friction,
        )

    @classmethod
    def push_rods(
        cls,
        routings: BaseThreadlikeRouting | ThreadlikeRoutingParameters,
        *,
        labels: tuple[str, ...] | None = None,
        friction: ThreadlikeFriction | None = None,
    ) -> ThreadlikeActuator:
        """Construct compression push rods with lengthening-positive coordinates.

        Args:
            routings: Fixed, straight-span, or shortest-path routing geometry.
            labels: Optional label for each routed path.
            friction: Optional active effort-loss model.

        Returns:
            Vectorized push-rod actuator with one channel per path.
        """
        routing = cls._normalize_routing(routings)
        return cls._preset(
            routing,
            coordinate_scale=jnp.ones((routing.num_paths,)),
            kind="push_rod",
            name="push_rods",
            unit="N",
            label_prefix="rod_compression",
            labels=labels,
            friction=friction,
        )

    @classmethod
    def muscles(
        cls,
        routings: BaseThreadlikeRouting | ThreadlikeRoutingParameters,
        *,
        labels: tuple[str, ...] | None = None,
        friction: ThreadlikeFriction | None = None,
    ) -> ThreadlikeActuator:
        """Construct tensile muscles with shortening-positive coordinates.

        Args:
            routings: Fixed, straight-span, or shortest-path routing geometry.
            labels: Optional label for each routed path.
            friction: Optional active effort-loss model.

        Returns:
            Vectorized muscle actuator with one channel per path.
        """
        routing = cls._normalize_routing(routings)
        return cls._preset(
            routing,
            coordinate_scale=-jnp.ones((routing.num_paths,)),
            kind="muscle",
            name="muscles",
            unit="N",
            label_prefix="muscle_tension",
            labels=labels,
            friction=friction,
        )

    @classmethod
    def pressure_chambers(
        cls,
        routings: BaseThreadlikeRouting | ThreadlikeRoutingParameters,
        *,
        effective_areas: Array,
        labels: tuple[str, ...] | None = None,
        friction: ThreadlikeFriction | None = None,
    ) -> ThreadlikeActuator:
        """Construct pressure chambers scaled by their effective areas.

        Args:
            routings: Fixed, straight-span, or shortest-path routing geometry.
            effective_areas: Effective area of each chamber with shape
                ``(num_paths,)``.
            labels: Optional label for each routed path.
            friction: Optional active effort-loss model.

        Returns:
            Vectorized pressure-chamber actuator with one channel per path.
        """
        routing = cls._normalize_routing(routings)
        return cls._preset(
            routing,
            coordinate_scale=_channel_array(
                effective_areas, routing.num_paths, "effective_areas"
            ),
            kind="pneumatic",
            name="pressure_chambers",
            unit="Pa",
            label_prefix="chamber_pressure",
            labels=labels,
            friction=friction,
        )

    @property
    def params(self) -> ThreadlikeActuatorParams:
        """Return the actuator parameter tree."""
        return self._params

    @property
    def transmission(self) -> ThreadlikeTransmission:
        """Return the threadlike transmission."""
        return self._transmission

    @property
    def effort_model(self) -> DirectEffort:
        """Return the direct commanded-effort model."""
        return self._effort_model

    @property
    def metadata(self) -> ActuatorMetadata:
        """Return channel labels, units, kind, and bounds."""
        return self._metadata

    def with_params(self, params: BaseSystemParams) -> ThreadlikeActuator:
        """Return a copy carrying replacement actuator parameter leaves.

        Args:
            params: Replacement threadlike-actuator parameters.

        Returns:
            Actuator with updated differentiable parameter leaves.

        Raises:
            TypeError: If ``params`` has the wrong parameter type.
            ValueError: If ``params`` changes the routing topology.
        """
        if not isinstance(params, ThreadlikeActuatorParams):
            raise TypeError("params must be ThreadlikeActuatorParams.")
        self.params.transmission.routing.assert_same_topology(
            params.transmission.routing
        )
        params.validate_for_update()
        return ThreadlikeActuator(
            params=params,
            routing=self.transmission.routing,
            friction=self.transmission.friction,
            labels=self.metadata.labels,
            units=self.metadata.units,
            name=self.name,
            kind=self.kind,
        )

    def validate_structure_for_robot(self, robot: SoftRobot) -> None:
        """Validate routing and friction structure for the host robot."""
        self.transmission.routing.validate_structure_for_robot(robot)
        self.transmission.routing.validate_friction(self.transmission.friction)

    def validate_values_for_robot(self, robot: SoftRobot) -> None:
        """Validate concrete routing and friction values for the host robot."""
        self.transmission.routing.validate_values_for_robot(robot)

    def _robot_value_validation_tree(self):
        """Return routing and friction leaves for robot-specific validation."""
        return (
            self.transmission.routing.params,
            self.params.transmission.friction,
        )

    def path_lengths(self, robot: SoftRobot, q: Array) -> Array:
        """Return physical routed-path lengths.

        Args:
            robot: Continuum robot hosting the routing.
            q: Generalized coordinates with shape
                ``(robot.num_coordinates,)``.

        Returns:
            Path lengths with shape ``(self.num_channels,)``.
        """
        return self.transmission.path_lengths(robot, q)

    def path_velocities(self, robot: SoftRobot, q: Array, qd: Array) -> Array:
        """Return physical routed-path length rates.

        Args:
            robot: Continuum robot hosting the routing.
            q: Generalized coordinates with shape
                ``(robot.num_coordinates,)``.
            qd: Generalized velocities with shape ``(robot.num_velocities,)``.

        Returns:
            Path-length rates with shape ``(self.num_channels,)``.
        """
        return self.transmission.path_velocities(robot, q, qd)

    def path_poses(self, robot: SoftRobot, q: Array, s: Array) -> Array:
        """Return routed-path positions at a backbone abscissa.

        Args:
            robot: Continuum robot hosting the routing.
            q: Generalized coordinates with shape
                ``(robot.num_coordinates,)``.
            s: Scalar backbone abscissa.

        Returns:
            Path positions with shape ``(self.num_channels, 2)`` for planar
            hosts or ``(self.num_channels, 3)`` for spatial hosts.
        """
        return self.transmission.path_poses(robot, q, s)


class ThreadlikeImpedanceParams(BaseSystemParams):
    """Spring-damper parameters for passive routed paths."""

    routing: ThreadlikeRoutingParameters
    stiffness: Array
    damping: Array
    rest_length: Array

    def __check_init__(self) -> None:
        """Validate newly constructed impedance parameters."""
        self.validate_for_update()

    def validate_structure(self) -> None:
        """Validate threadlike impedance parameter structure."""
        self.routing.validate_structure()
        count = self.routing.num_paths
        for name in ("stiffness", "damping", "rest_length"):
            value = jnp.asarray(getattr(self, name))
            if value.shape != (count,):
                raise ValueError(
                    f"{name} must have shape ({count},), got {value.shape}."
                )

    def validate_values(self) -> None:
        """Validate eager nested threadlike routing values."""
        self.routing.validate_values()


class ThreadlikeImpedance(PassiveElement):
    """Passive spring-damper mechanics along threadlike paths."""

    _params: ThreadlikeImpedanceParams
    routing: BaseThreadlikeRouting
    name: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        routing: BaseThreadlikeRouting | ThreadlikeRoutingParameters,
        stiffness: Array,
        damping: Array,
        rest_length: Array,
        name: str = "passive_threadlike",
    ) -> None:
        """Construct passive spring-damper paths.

        Args:
            routing: Fixed, straight-span, or shortest-path routing geometry.
            stiffness: One axial stiffness per path.
            damping: One axial damping coefficient per path.
            rest_length: One undeformed length per path.
            name: Human-readable passive-element name.
        """
        routing = ThreadlikeActuator._normalize_routing(routing)
        count = routing.num_paths
        self._params = ThreadlikeImpedanceParams(
            routing=routing.params,
            stiffness=_channel_array(stiffness, count, "stiffness"),
            damping=_channel_array(damping, count, "damping"),
            rest_length=_channel_array(rest_length, count, "rest_length"),
        )
        self.routing = routing
        self.name = name

    @classmethod
    def _from_params(
        cls,
        params: ThreadlikeImpedanceParams,
        *,
        name: str,
        routing: BaseThreadlikeRouting,
    ) -> ThreadlikeImpedance:
        """Reconstruct an impedance while preserving its runtime routing law."""
        return cls(
            routing=routing.with_params(params.routing),
            stiffness=params.stiffness,
            damping=params.damping,
            rest_length=params.rest_length,
            name=name,
        )

    @property
    def params(self) -> ThreadlikeImpedanceParams:
        """Return the passive element parameters."""
        return self._params

    def with_params(self, params: BaseSystemParams) -> ThreadlikeImpedance:
        """Return a copy carrying replacement impedance parameters.

        Args:
            params: Replacement threadlike-impedance parameters.

        Returns:
            Passive element with updated differentiable parameter leaves.

        Raises:
            TypeError: If ``params`` has the wrong parameter type.
            ValueError: If ``params`` changes the routing topology.
        """
        if not isinstance(params, ThreadlikeImpedanceParams):
            raise TypeError("params must be ThreadlikeImpedanceParams.")
        self.params.routing.assert_same_topology(params.routing)
        params.validate_for_update()
        return self._from_params(params, name=self.name, routing=self.routing)

    def validate_structure_for_robot(self, robot: SoftRobot) -> None:
        """Validate routing structure for the host robot."""
        self.routing.validate_structure_for_robot(robot)

    def validate_values_for_robot(self, robot: SoftRobot) -> None:
        """Validate concrete routing values for the host robot."""
        self.routing.validate_values_for_robot(robot)

    def _robot_value_validation_tree(self):
        """Return routing leaves used by robot-specific eager validation."""
        return self.routing.params

    def path_lengths(self, robot: SoftRobot, q: Array) -> Array:
        """Return the raw lengths of the passive routed paths.

        Args:
            robot: Continuum robot hosting the routing.
            q: Generalized coordinates with shape
                ``(robot.num_coordinates,)``.

        Returns:
            Path lengths with shape ``(self.routing.num_paths,)``.
        """
        return self.routing.path_lengths(robot, q)

    def path_length_jacobian(self, robot: SoftRobot, q: Array) -> Array:
        """Return the Jacobian of passive physical path lengths.

        The Jacobian maps generalized velocity to passive path-length rate.

        Args:
            robot: Continuum robot hosting the routing.
            q: Generalized coordinates with shape
                ``(robot.num_coordinates,)``.

        Returns:
            J_l: Path-length Jacobian with shape
                ``(self.routing.num_paths, robot.num_velocities)``.
        """
        return self.routing.path_velocity_jacobian(robot, q)

    def path_velocities(self, robot: SoftRobot, q: Array, qd: Array) -> Array:
        """Return the raw length rates of the passive routed paths.

        Args:
            robot: Continuum robot hosting the routing.
            q: Generalized coordinates with shape
                ``(robot.num_coordinates,)``.
            qd: Generalized velocities with shape ``(robot.num_velocities,)``.

        Returns:
            Path-length rates with shape ``(self.routing.num_paths,)``.
        """
        return self.path_length_jacobian(robot, q) @ qd

    def path_poses(self, robot: SoftRobot, q: Array, s: Array) -> Array:
        """Return passive routed-path positions at a backbone abscissa.

        Args:
            robot: Continuum robot hosting the routing.
            q: Generalized coordinates with shape
                ``(robot.num_coordinates,)``.
            s: Scalar backbone abscissa.

        Returns:
            Path positions with shape ``(self.routing.num_paths, 2)`` for
            planar hosts or ``(self.routing.num_paths, 3)`` for spatial hosts.
        """
        return self.routing.path_positions(robot, q, s)

    def elastic_force(self, robot: SoftRobot, q: Array) -> Array:
        """Return conservative generalized spring force from the paths."""
        lengths = self.path_lengths(robot, q)
        jacobian = self.path_length_jacobian(robot, q)
        return jacobian.T @ (
            self.params.stiffness * (lengths - self.params.rest_length)
        )

    def damping_matrix(self, robot: SoftRobot, q: Array) -> Array:
        """Return generalized viscous damping from the routed paths."""
        jacobian = self.path_length_jacobian(robot, q)
        return jacobian.T @ (self.params.damping[:, None] * jacobian)

    def elastic_energy(self, robot: SoftRobot, q: Array) -> Array:
        """Return elastic energy stored in the routed springs."""
        delta = self.path_lengths(robot, q) - self.params.rest_length
        return 0.5 * jnp.sum(self.params.stiffness * delta**2)


__all__ = [
    "ThreadlikeActuator",
    "ThreadlikeActuatorParams",
    "ThreadlikeImpedance",
    "ThreadlikeImpedanceParams",
    "ThreadlikeTransmission",
    "ThreadlikeTransmissionParams",
]
