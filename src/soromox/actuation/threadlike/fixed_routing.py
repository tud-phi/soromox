"""Fixed material-frame routing for threadlike continuum actuators."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import equinox as eqx
from jax import Array, vmap
from jax import numpy as jnp

from soromox.systems.params import BaseSystemParams

if TYPE_CHECKING:
    from soromox.systems.soft_robot import SoftRobot

from ..friction import ThreadlikeFriction, _validate_friction_model
from .base import BaseThreadlikeRouting


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


class ThreadlikeRouting(BaseThreadlikeRouting):
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

    def validate_structure_for_robot(self, robot: SoftRobot) -> None:
        """Validate the fixed path topology and host hooks.

        Args:
            robot: Continuum host supplying fixed-routing integration hooks and
                segment topology.

        Raises:
            TypeError: The host lacks the required integration or segment data.
            ValueError: A path span exceeds the host or a routing callback
                returns offsets with an incompatible shape.
        """
        _validate_routing_structure_for_robot(self, robot)

    def validate_values_for_robot(self, robot: SoftRobot) -> None:
        """Validate material offsets, including the planar restriction.

        Args:
            robot: Continuum host with concrete segment lengths and material
                frame conventions.

        Raises:
            ValueError: Routing parameters are invalid or sampled offsets leave
                the local-y direction required by a planar host.
        """
        _validate_routing_values_for_robot(self, robot)

    def validate_friction(self, friction: ThreadlikeFriction) -> None:
        """Validate the derivatives required by a fixed-path friction model.

        Args:
            friction: Runtime effort-loss model paired with this routing.

        Raises:
            ValueError: Nonzero friction requires an analytical second
                derivative that the routing does not provide.
        """
        _validate_friction_model(friction, self)

    def path_lengths(self, robot: SoftRobot, q: Array) -> Array:
        """Return physical path lengths, independent of rigid base motion.

        Args:
            robot: Continuum host supplying fixed-routing length integration.
            q: Configuration with shape ``(robot.num_coordinates,)``.

        Returns:
            Physical lengths in metres with shape ``(self.num_paths,)``.
                Actuator coordinate signs and scales are not applied.
        """
        _, internal = robot.split_configuration(q)
        return robot._threadlike_path_lengths(internal, self)

    def path_velocity_jacobian(self, robot: SoftRobot, q: Array) -> Array:
        """Return length-rate derivatives with zero floating-base columns.

        Args:
            robot: Continuum host supplying fixed-routing length derivatives.
            q: Configuration with shape ``(robot.num_coordinates,)``.

        Returns:
            Jacobian with shape ``(self.num_paths, robot.num_velocities)``.
                Multiplication by generalized velocity gives physical path
                length rates in metres per second.
        """
        _, internal = robot.split_configuration(q)
        jacobian = robot._threadlike_path_length_jacobian(internal, self)
        return jnp.pad(jacobian, ((0, 0), (robot.num_base_velocities, 0)))

    def path_force_matrix(
        self, robot: SoftRobot, q: Array, *, friction: ThreadlikeFriction
    ) -> Array:
        """Integrate length-conjugate efforts with optional fixed-path friction.

        Args:
            robot: Continuum host supplying the fixed-routing force integral.
            q: Configuration with shape ``(robot.num_coordinates,)``.
            friction: Runtime effort-loss model supported by this routing.

        Returns:
            Force map with shape ``(robot.num_velocities, self.num_paths)``.
                Floating-base rows are zero; actuator signs and scales have
                not been applied.

        Raises:
            ValueError: The routing lacks the analytical derivatives required
                by the friction model.
        """
        self.validate_friction(friction)
        _, internal = robot.split_configuration(q)
        matrix = robot._threadlike_actuation_matrix(internal, self, friction=friction)
        return jnp.pad(matrix, ((robot.num_base_velocities, 0), (0, 0)))

    def path_positions(self, robot: SoftRobot, q: Array, s: Array) -> Array:
        """Return world positions with stations clamped to each path's ends.

        Args:
            robot: Continuum host supplying fixed-routing world positions.
            q: Configuration with shape ``(robot.num_coordinates,)``, including
                the runtime base pose for floating hosts.
            s: Scalar reference backbone station in metres.

        Returns:
            World positions in metres with shape ``(self.num_paths, 2)`` for
                planar hosts or ``(self.num_paths, 3)`` for spatial hosts.
        """
        if robot.floating_base:
            base, q = robot.split_configuration(q)
            robot = robot._fixed_base_robot_at_pose(base)
        return robot._threadlike_path_positions(q, s, self)


__all__ = [
    "BaseThreadlikeRoutingParams",
    "LinearThreadlikeRoutingParams",
    "ThreadlikeRouting",
    "linear_threadlike_routing",
    "linear_threadlike_routing_derivative",
    "linear_threadlike_routing_second_derivative",
]
