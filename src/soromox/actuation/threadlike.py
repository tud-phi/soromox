"""Fixed material-frame routing for threadlike continuum actuators."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import equinox as eqx
from jax import Array, vmap
from jax import numpy as jnp

from soromox.systems.params import BaseSystemParams

if TYPE_CHECKING:
    from soromox.systems.soft_robot import SoftRobot

from .core import (
    Actuator,
    ActuatorMetadata,
    DirectEffort,
    PassiveElement,
    Transmission,
)
from .friction import (
    BaseThreadlikeFrictionParams,
    FrictionlessParams,
    ThreadlikeFriction,
    _validate_friction_model,
)

ThreadlikeKind = Literal["tendon", "push_rod", "muscle", "pneumatic", "generic"]


def _validate_routing_structure_for_robot(routing: ThreadlikeRouting, robot) -> None:
    """Validate routing topology and tracer-safe host compatibility."""
    required_hooks = (
        "_threadlike_path_lengths",
        "_threadlike_path_length_jacobian",
        "_threadlike_actuation_matrix",
        "_threadlike_path_positions",
    )
    if not all(callable(getattr(robot, name, None)) for name in required_hooks):
        raise TypeError(
            "Threadlike components require a continuum host implementing the "
            "threadlike routing hooks, such as PCS, PlanarPCS, or GVS."
        )
    if not hasattr(robot, "num_segments"):
        raise TypeError(
            "Threadlike components require continuum segment topology through "
            "num_segments."
        )
    routing.params.validate_structure_for_robot(robot.num_segments)
    if not robot.is_planar:
        return
    if not hasattr(robot, "L_cum"):
        raise TypeError(
            "Planar threadlike routing validation requires cumulative segment "
            "coordinates through L_cum."
        )
    samples = jnp.linspace(0.0, robot.L_cum[-1], 17)
    offsets = vmap(
        lambda path_params: vmap(lambda s: routing.offset(path_params, s))(samples)
    )(routing.params)
    if offsets.shape != (routing.num_paths, samples.shape[0], 3):
        raise ValueError(
            "ThreadlikeRouting.offset_fn must return material-frame [x, y, z] "
            "offsets with final shape (3,)."
        )


def _validate_routing_values_for_robot(routing: ThreadlikeRouting, robot) -> None:
    """Validate concrete routing values and host-specific geometry."""
    routing.params.validate_values()
    if not robot.is_planar:
        return
    samples = jnp.linspace(0.0, robot.L_cum[-1], 17)
    offsets = vmap(
        lambda path_params: vmap(lambda s: routing.offset(path_params, s))(samples)
    )(routing.params)
    leaves_planar_frame = jnp.any(offsets[..., 0] != 0.0) | jnp.any(
        offsets[..., 2] != 0.0
    )
    if bool(leaves_planar_frame):
        raise ValueError(
            "PlanarPCS threadlike routing must remain in the local-y direction; "
            "its sampled local x and z offsets must be zero."
        )


def _index_tuple(value: int | tuple[int, ...] | list[int] | Array) -> tuple[int, ...]:
    """Normalize one or more segment indices to an immutable tuple."""
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, tuple):
        return tuple(int(index) for index in value)
    if isinstance(value, list):
        return tuple(int(index) for index in value)
    array = jnp.asarray(value)
    if array.ndim == 0:
        return (int(array.item()),)
    if array.ndim != 1:
        raise ValueError("segment indices must be scalar or one-dimensional.")
    return tuple(int(index) for index in array.tolist())


def _channel_array(value: Any, count: int, name: str) -> Array:
    """Broadcast a scalar or validate one value per actuation channel."""
    array = jnp.asarray(value)
    if array.ndim == 0:
        array = jnp.full((count,), array)
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},), got {array.shape}.")
    return array


def _normalize_friction(friction: Any) -> ThreadlikeFriction | None:
    """Validate an optional runtime threadlike-friction object."""
    if friction is not None and not isinstance(friction, ThreadlikeFriction):
        raise TypeError(
            "friction must be a ThreadlikeFriction instance, got "
            f"{type(friction).__name__}."
        )
    return friction


def _path_vector_array(value: Any, name: str, *, count: int | None = None) -> Array:
    """Normalize material-frame vectors to shape ``(num_paths, 3)``."""
    array = jnp.asarray(value)
    if array.shape == (3,):
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(
            f"{name} must have shape (3,) or (num_paths, 3), got {array.shape}."
        )
    if count is not None:
        if array.shape == (1, 3) and count != 1:
            array = jnp.broadcast_to(array, (count, 3))
        if array.shape != (count, 3):
            raise ValueError(f"{name} must have shape ({count}, 3), got {array.shape}.")
    return array


class BaseThreadlikeRoutingParams(BaseSystemParams):
    """Base PyTree contract for vectorized threadlike routing parameters."""

    @property
    def num_paths(self) -> int:
        """Return the number of routed paths represented by the parameters."""
        raise NotImplementedError

    @property
    def start_segment_index_array(self) -> Array:
        """Return the first active segment of every routed path.

        Returns:
            Start indices with shape ``(self.num_paths,)``.
        """
        return jnp.asarray(self.start_segment_index, dtype=jnp.int32)

    @property
    def end_segment_index_array(self) -> Array:
        """Return the last active segment of every routed path.

        Returns:
            End indices with shape ``(self.num_paths,)``.
        """
        return jnp.asarray(self.end_segment_index, dtype=jnp.int32)

    def validate_structure_for_robot(self, num_segments: int) -> None:
        """Validate path spans against a host robot's segment count.

        Args:
            num_segments: Number of continuum body segments in the host robot.

        Raises:
            ValueError: If a path span falls outside the host robot or its start
                index follows its end index.
        """
        self.validate_structure()
        for start, end in zip(self.start_segment_index, self.end_segment_index):
            if start < 0 or start > end or end >= num_segments:
                raise ValueError(
                    "each routing span must satisfy 0 <= start <= end < "
                    f"num_segments; got ({start}, {end}) for {num_segments} segments."
                )

    def assert_same_topology(self, other: BaseThreadlikeRoutingParams) -> None:
        """Reject a replacement that changes routing type, count, or spans.

        Args:
            other: Candidate replacement routing parameters.

        Raises:
            ValueError: If ``other`` changes the parameter type, number of
                paths, or segment spans.
        """
        if type(other) is not type(self):
            raise ValueError("Changing routing type requires reconstruction.")
        if other.num_paths != self.num_paths:
            raise ValueError("Changing the routing count requires reconstruction.")
        if (
            other.start_segment_index != self.start_segment_index
            or other.end_segment_index != self.end_segment_index
        ):
            raise ValueError("Changing routing segment spans requires reconstruction.")

    def replace(self, **updates: Any) -> BaseThreadlikeRoutingParams:
        """Replace numeric fields while keeping routing topology immutable."""
        for name in ("start_segment_index", "end_segment_index"):
            if name in updates:
                replacement = _index_tuple(updates.pop(name))
                if replacement != getattr(self, name):
                    raise ValueError(
                        f"Changing {name} changes routing topology; reconstruct "
                        "the routing."
                    )
        return super().replace(**updates)


class LinearThreadlikeRoutingParams(BaseThreadlikeRoutingParams):
    """Batched linear material-frame routing and contiguous segment spans.

    ``intercept`` and ``slope`` have shape ``(num_paths, 3)`` in the local
    material frame. Because SoRoMoX aligns the backbone with the local x-axis,
    their x-components must be zero; the remaining components describe the
    path within each material-frame cross-section.
    """

    intercept: Array = eqx.field(
        converter=lambda value: _path_vector_array(value, "intercept")
    )
    slope: Array = eqx.field(converter=lambda value: _path_vector_array(value, "slope"))
    start_segment_index: tuple[int, ...] = eqx.field(
        static=True, converter=_index_tuple
    )
    end_segment_index: tuple[int, ...] = eqx.field(static=True, converter=_index_tuple)

    def __check_init__(self) -> None:
        """Validate newly constructed linear-routing parameters."""
        self.validate_for_update()

    @property
    def num_paths(self) -> int:
        """Return the number of linear paths."""
        return int(jnp.asarray(self.intercept).shape[0])

    @classmethod
    def empty(cls) -> LinearThreadlikeRoutingParams:
        """Construct an empty linear-routing parameter batch."""
        empty = jnp.zeros((0, 3), dtype=jnp.float64)
        return cls(
            intercept=empty,
            slope=empty,
            start_segment_index=(),
            end_segment_index=(),
        )

    def validate_structure(self) -> None:
        """Validate linear-routing array shapes and segment-span topology."""
        intercept = jnp.asarray(self.intercept)
        slope = jnp.asarray(self.slope)
        if intercept.ndim != 2 or intercept.shape[1:] != (3,):
            raise ValueError("intercept must have shape (num_paths, 3).")
        if slope.shape != intercept.shape:
            raise ValueError(
                f"slope must have shape {intercept.shape}, got {slope.shape}."
            )
        if len(self.start_segment_index) != self.num_paths:
            raise ValueError("start_segment_index must contain one entry per path.")
        if len(self.end_segment_index) != self.num_paths:
            raise ValueError("end_segment_index must contain one entry per path.")
        for start, end in zip(self.start_segment_index, self.end_segment_index):
            if start < 0 or end < 0:
                raise ValueError("routing segment indices must be non-negative.")
            if start > end:
                raise ValueError(
                    "each threadlike routing must have start_segment_index <= "
                    "end_segment_index."
                )

    def validate_values(self) -> None:
        """Validate eager linear-routing values."""
        intercept = jnp.asarray(self.intercept)
        slope = jnp.asarray(self.slope)
        if bool(jnp.any(intercept[:, 0] != 0.0)):
            raise ValueError(
                "intercept local-x components must be zero because the backbone "
                "is aligned with the local x-axis."
            )
        if bool(jnp.any(slope[:, 0] != 0.0)):
            raise ValueError(
                "slope local-x components must be zero because the backbone is "
                "aligned with the local x-axis."
            )


def linear_threadlike_routing(params: LinearThreadlikeRoutingParams, s: Array) -> Array:
    """Evaluate linear material-frame routing offsets.

    Args:
        params: Linear routing parameters for one path or a path batch.
        s: Scalar backbone abscissa.

    Returns:
        Material-frame offsets ``params.intercept + params.slope * s`` with
        shape ``(3,)`` or ``(num_paths, 3)``.
    """
    return jnp.asarray(params.intercept) + jnp.asarray(params.slope) * jnp.asarray(s)


def linear_threadlike_routing_derivative(
    params: LinearThreadlikeRoutingParams, s: Array
) -> Array:
    """Evaluate the first derivative of a linear routing offset.

    Args:
        params: Linear routing parameters for one path or a path batch.
        s: Scalar backbone abscissa, used to preserve dtype and tracing.

    Returns:
        Constant slopes with shape ``(3,)`` or ``(num_paths, 3)``.
    """
    return jnp.asarray(params.slope) + jnp.zeros_like(
        jnp.asarray(params.slope)
    ) * jnp.asarray(s)


def linear_threadlike_routing_second_derivative(
    params: LinearThreadlikeRoutingParams, s: Array
) -> Array:
    """Return the zero second derivative of a linear routing offset.

    Args:
        params: Linear routing parameters for one path or a path batch.
        s: Scalar backbone abscissa, used to preserve output dtype and tracing.

    Returns:
        Zeros with the same shape as ``params.slope``.
    """
    return jnp.zeros_like(jnp.asarray(params.slope)) * jnp.asarray(s)


class ThreadlikeRouting(eqx.Module):
    """Runtime routing object for a batched fixed material-frame path family."""

    params: BaseThreadlikeRoutingParams
    offset_fn: Callable = eqx.field(static=True, default=linear_threadlike_routing)
    derivative_fn: Callable = eqx.field(
        static=True, default=linear_threadlike_routing_derivative
    )
    second_derivative_fn: Callable | None = eqx.field(static=True, default=None)

    @classmethod
    def linear(
        cls,
        *,
        intercept: Array,
        slope: Array | float = 0.0,
        start_segment_index: int | tuple[int, ...] | Array = 0,
        end_segment_index: int | tuple[int, ...] | Array = 0,
    ) -> ThreadlikeRouting:
        """Construct linear cross-section paths in the local material frame.

        The final array axis is ``[x, y, z]``. Local x follows the backbone, so
        the x-components of ``intercept`` and ``slope`` must both be zero.

        Args:
            intercept: Offset at zero backbone abscissa with shape ``(3,)`` or
                ``(num_paths, 3)``.
            slope: Constant offset derivative, either scalar or shaped ``(3,)``
                or ``(num_paths, 3)``.
            start_segment_index: First segment traversed by each path.
            end_segment_index: Last segment traversed by each path.

        Returns:
            Vectorized linear routing family.
        """
        intercept = _path_vector_array(intercept, "intercept")
        count = int(intercept.shape[0])
        slope_array = jnp.asarray(slope)
        if slope_array.ndim == 0:
            slope_array = jnp.full(intercept.shape, slope_array)
        else:
            slope_array = _path_vector_array(slope_array, "slope", count=count)
        starts = _index_tuple(start_segment_index)
        ends = _index_tuple(end_segment_index)
        if len(starts) == 1 and count != 1:
            starts = starts * count
        if len(ends) == 1 and count != 1:
            ends = ends * count
        params = LinearThreadlikeRoutingParams(
            intercept=intercept,
            slope=slope_array,
            start_segment_index=starts,
            end_segment_index=ends,
        )
        return cls(
            params=params,
            second_derivative_fn=linear_threadlike_routing_second_derivative,
        )

    @property
    def num_paths(self) -> int:
        """Return the number of paths represented by the routing family."""
        return self.params.num_paths

    @property
    def has_analytical_second_derivative(self) -> bool:
        """Return whether the routing supplies the derivative needed for turn.

        Returns:
            Whether an explicit second-derivative function is installed or all
            routing callbacks are the built-in linear functions.
        """
        return self.second_derivative_fn is not None or (
            isinstance(self.params, LinearThreadlikeRoutingParams)
            and self.offset_fn is linear_threadlike_routing
            and self.derivative_fn is linear_threadlike_routing_derivative
        )

    def offset(self, path_params: BaseThreadlikeRoutingParams, s: Array) -> Array:
        """Evaluate one path's material-frame offset.

        Args:
            path_params: Parameters of one routed path.
            s: Scalar backbone abscissa.

        Returns:
            Material-frame offset with shape ``(3,)``.
        """
        return self.offset_fn(path_params, s)

    def derivative(self, path_params: BaseThreadlikeRoutingParams, s: Array) -> Array:
        """Evaluate one path's material-frame offset derivative.

        Args:
            path_params: Parameters of one routed path.
            s: Scalar backbone abscissa.

        Returns:
            First backbone-abscissa derivative with shape ``(3,)``.
        """
        return self.derivative_fn(path_params, s)

    def second_derivative(
        self, path_params: BaseThreadlikeRoutingParams, s: Array
    ) -> Array:
        """Evaluate one path's material-frame offset second derivative.

        Args:
            path_params: Parameters of one routed path.
            s: Scalar backbone abscissa.

        Returns:
            Second backbone-abscissa derivative with shape ``(3,)``.

        Raises:
            ValueError: If a custom routing does not provide an analytical
                ``second_derivative_fn``.
        """
        if self.second_derivative_fn is None:
            if (
                isinstance(path_params, LinearThreadlikeRoutingParams)
                and self.offset_fn is linear_threadlike_routing
                and self.derivative_fn is linear_threadlike_routing_derivative
            ):
                return linear_threadlike_routing_second_derivative(path_params, s)
            raise ValueError(
                "custom threadlike routing requires second_derivative_fn for "
                "friction calculations."
            )
        return self.second_derivative_fn(path_params, s)

    def with_params(self, params: BaseThreadlikeRoutingParams) -> ThreadlikeRouting:
        """Return a routing with new values and unchanged runtime functions."""
        if not isinstance(params, BaseThreadlikeRoutingParams):
            raise TypeError("params must be BaseThreadlikeRoutingParams.")
        self.params.assert_same_topology(params)
        params.validate_for_update()
        return ThreadlikeRouting(
            params=params,
            offset_fn=self.offset_fn,
            derivative_fn=self.derivative_fn,
            second_derivative_fn=self.second_derivative_fn,
        )

    def update_params(self, **updates: Any) -> ThreadlikeRouting:
        """Return a routing with selected parameter leaves replaced."""
        return self.with_params(self.params.replace(**updates))


class ThreadlikeTransmissionParams(BaseSystemParams):
    """Routing, work-coordinate scale, and friction parameters per path."""

    routing: BaseThreadlikeRoutingParams
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
    routing: ThreadlikeRouting
    friction: ThreadlikeFriction

    def __init__(
        self,
        params: ThreadlikeTransmissionParams,
        *,
        routing: ThreadlikeRouting | None = None,
        friction: ThreadlikeFriction | None = None,
    ) -> None:
        """Construct a runtime transmission from matching parameter objects.

        Args:
            params: Nested routing, coordinate-scale, and friction parameters.
            routing: Runtime routing functions. Linear routing is reconstructed
                automatically when this argument is omitted.
            friction: Runtime effort-ratio law. Frictionless behavior is
                reconstructed automatically when this argument is omitted.
        """
        if not isinstance(params, ThreadlikeTransmissionParams):
            raise TypeError("params must be ThreadlikeTransmissionParams.")
        params.validate_for_update()
        self.params = params
        if routing is None:
            if not isinstance(params.routing, LinearThreadlikeRoutingParams):
                raise TypeError(
                    "custom routing params require their matching "
                    "ThreadlikeRouting runtime object."
                )
            routing = ThreadlikeRouting(params=params.routing)
        else:
            routing = routing.with_params(params.routing)
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
        return self.params.coordinate_scale * robot._threadlike_path_lengths(
            q, self.routing
        )

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
        raw_jacobian = robot._threadlike_path_length_jacobian(q, self.routing)
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
        raw_matrix = robot._threadlike_actuation_matrix(
            q, self.routing, friction=self.friction
        )
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
        return robot._threadlike_path_lengths(q, self.routing)

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
        return robot._threadlike_path_length_jacobian(q, self.routing)

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
        return robot._threadlike_path_positions(q, s, self.routing)

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
        routing: ThreadlikeRouting | None = None,
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
        routings: ThreadlikeRouting | BaseThreadlikeRoutingParams,
    ) -> ThreadlikeRouting:
        """Normalize routing parameters to their runtime routing object."""
        if isinstance(routings, ThreadlikeRouting):
            return routings
        if isinstance(routings, LinearThreadlikeRoutingParams):
            return ThreadlikeRouting(params=routings)
        if isinstance(routings, BaseThreadlikeRoutingParams):
            raise TypeError(
                "custom routing params must be wrapped in ThreadlikeRouting with "
                "offset_fn and derivative_fn."
            )
        raise TypeError(
            "routings must be ThreadlikeRouting or LinearThreadlikeRoutingParams."
        )

    @classmethod
    def _preset(
        cls,
        routings: ThreadlikeRouting | BaseThreadlikeRoutingParams,
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
        routings: ThreadlikeRouting | BaseThreadlikeRoutingParams,
        *,
        labels: tuple[str, ...] | None = None,
        friction: ThreadlikeFriction | None = None,
    ) -> ThreadlikeActuator:
        """Construct tension-only tendons with shortening-positive coordinates.

        Args:
            routings: Fixed material-frame routing geometry.
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
        routings: ThreadlikeRouting | BaseThreadlikeRoutingParams,
        *,
        labels: tuple[str, ...] | None = None,
        friction: ThreadlikeFriction | None = None,
    ) -> ThreadlikeActuator:
        """Construct compression push rods with lengthening-positive coordinates.

        Args:
            routings: Fixed material-frame routing geometry.
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
        routings: ThreadlikeRouting | BaseThreadlikeRoutingParams,
        *,
        labels: tuple[str, ...] | None = None,
        friction: ThreadlikeFriction | None = None,
    ) -> ThreadlikeActuator:
        """Construct tensile muscles with shortening-positive coordinates.

        Args:
            routings: Fixed material-frame routing geometry.
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
        routings: ThreadlikeRouting | BaseThreadlikeRoutingParams,
        *,
        effective_areas: Array,
        labels: tuple[str, ...] | None = None,
        friction: ThreadlikeFriction | None = None,
    ) -> ThreadlikeActuator:
        """Construct pressure chambers scaled by their effective areas.

        Args:
            routings: Fixed material-frame routing geometry.
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
        _validate_routing_structure_for_robot(self.transmission.routing, robot)
        _validate_friction_model(self.transmission.friction, self.transmission.routing)

    def validate_values_for_robot(self, robot: SoftRobot) -> None:
        """Validate concrete routing and friction values for the host robot."""
        _validate_routing_values_for_robot(self.transmission.routing, robot)

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

    routing: BaseThreadlikeRoutingParams
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
    """Passive spring-damper mechanics along fixed threadlike paths."""

    _params: ThreadlikeImpedanceParams
    routing: ThreadlikeRouting
    name: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        routing: ThreadlikeRouting | BaseThreadlikeRoutingParams,
        stiffness: Array,
        damping: Array,
        rest_length: Array,
        name: str = "passive_threadlike",
    ) -> None:
        """Construct passive spring-damper paths.

        Args:
            routing: Fixed material-frame routing geometry.
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
        routing: ThreadlikeRouting,
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
        _validate_routing_structure_for_robot(self.routing, robot)

    def validate_values_for_robot(self, robot: SoftRobot) -> None:
        """Validate concrete routing values for the host robot."""
        _validate_routing_values_for_robot(self.routing, robot)

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
        return robot._threadlike_path_lengths(q, self.routing)

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
        return robot._threadlike_path_length_jacobian(q, self.routing)

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
        return robot._threadlike_path_positions(q, s, self.routing)

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
    "BaseThreadlikeRoutingParams",
    "LinearThreadlikeRoutingParams",
    "ThreadlikeActuator",
    "ThreadlikeActuatorParams",
    "ThreadlikeImpedance",
    "ThreadlikeImpedanceParams",
    "ThreadlikeRouting",
    "ThreadlikeTransmission",
    "ThreadlikeTransmissionParams",
    "linear_threadlike_routing",
    "linear_threadlike_routing_derivative",
    "linear_threadlike_routing_second_derivative",
]
