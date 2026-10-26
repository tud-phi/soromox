"""Renderer-side adapters for semantic actuator geometry."""

from __future__ import annotations

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax import numpy as jnp

from soromox.actuation.mckibben import ArticulatedMcKibbenActuator
from soromox.actuation.threadlike import (
    ConfigurationDependentThreadlikeRouting,
    RoutingSolution,
    ThreadlikeActuator,
    ThreadlikeImpedance,
)
from soromox.systems.soft_robot import SoftRobot

from .actuators import ActuatorVisualLayer, TrajectoryActuatorVisualLayer


def actuator_visual_layers(
    robot: SoftRobot,
    q: Array,
    s_points: Array,
    *,
    actuator_inputs: Array | None = None,
) -> tuple[ActuatorVisualLayer, ...]:
    """Return renderer-facing geometry for installed actuation components.

    The returned layers describe semantic geometry and optional scalar fields
    for supported active actuators and passive elements. Actuation objects
    provide path geometry and semantic kinds, while visual styling such as
    colors, radii, line widths, and scalar colormaps remains the renderer's
    responsibility. Unsupported components do not produce a layer.

    Args:
        robot: Soft robot containing the installed actuation components.
        q: Generalized coordinates of shape ``(robot.num_coordinates,)``.
        s_points: Backbone sample coordinates with shape ``(num_points,)``
            used for routed continuum geometry.
        actuator_inputs: Optional actuator inputs of shape
            ``(robot.num_actuators,)``. When supplied, supported adapters attach
            input-dependent scalar fields such as pressure or axial force.

    Returns:
        layers: Tuple of semantic actuator visual layers. The tuple is empty
            when no installed component has a renderer adapter.
    """
    layers: list[ActuatorVisualLayer] = []
    start = 0
    for actuator in robot.actuators:
        stop = start + actuator.num_channels
        inputs = (
            None
            if actuator_inputs is None
            else jnp.asarray(actuator_inputs)[start:stop]
        )
        if isinstance(actuator, ThreadlikeActuator):
            points = jnp.stack(
                [actuator.path_poses(robot, q, s) for s in jnp.asarray(s_points)],
                axis=0,
            ).transpose(1, 0, 2)
            scalar_fields = {}
            if inputs is not None:
                scalar_fields["input"] = inputs
            layers.append(
                ActuatorVisualLayer(
                    name=actuator.name,
                    kind=actuator.kind,
                    points=points,
                    scalar_fields=scalar_fields,
                )
            )
        elif isinstance(actuator, ArticulatedMcKibbenActuator):
            if actuator.num_channels == 0:
                start = stop
                continue
            scalar_fields = {}
            if inputs is not None:
                scalar_fields["pressure"] = inputs
                scalar_fields["force"] = actuator.axial_forces(q, inputs)
            layers.append(
                ActuatorVisualLayer(
                    name=actuator.name,
                    kind="muscle",
                    points=actuator.segments(robot, q),
                    scalar_fields=scalar_fields,
                )
            )
        start = stop

    for element in robot.passive_elements:
        if isinstance(element, ThreadlikeImpedance):
            points = jnp.stack(
                [
                    robot._threadlike_path_positions(q, s, element.routing)
                    for s in jnp.asarray(s_points)
                ],
                axis=0,
            ).transpose(1, 0, 2)
            layers.append(
                ActuatorVisualLayer(
                    name=element.name,
                    kind="generic",
                    points=points,
                )
            )

    return tuple(layers)


def vectorized_actuator_visual_layers_trajectory(
    robot: SoftRobot,
    q_ts: Array,
    s_points: Array,
    *,
    actuator_inputs: Array | None = None,
) -> tuple[TrajectoryActuatorVisualLayer, ...]:
    """Vectorize installed actuator geometry over robots and timesteps."""
    q_ts = jnp.asarray(q_ts)
    if q_ts.ndim != 3:
        raise ValueError(f"q_ts must have shape (N, T, DOF), got {q_ts.shape}")

    inputs_ts = None
    if actuator_inputs is not None:
        inputs = jnp.asarray(actuator_inputs)
        num_robots, num_steps = q_ts.shape[:2]
        if inputs.ndim >= 3 and inputs.shape[:2] == (num_robots, num_steps):
            inputs_ts = inputs
        elif num_robots == 1 and inputs.ndim >= 2 and inputs.shape[0] == num_steps:
            inputs_ts = inputs[None, ...]
        elif inputs.ndim >= 2 and inputs.shape[0] == num_robots:
            inputs_ts = jnp.broadcast_to(
                inputs[:, None, ...],
                (num_robots, num_steps, *inputs.shape[1:]),
            )
        elif num_robots == 1:
            inputs_ts = jnp.broadcast_to(
                inputs[None, None, ...],
                (1, num_steps, *inputs.shape),
            )
        else:
            raise ValueError(
                "actuator_inputs must have leading (N, T) or N dimensions "
                "matching q_ts; got "
                f"{inputs.shape} for q_ts {q_ts.shape}"
            )

    s_points = jnp.asarray(s_points)

    def vectorize_q(fn):
        return jax.jit(jax.vmap(jax.vmap(fn)))(q_ts)

    layers: list[TrajectoryActuatorVisualLayer] = []
    start = 0
    for actuator in robot.actuators:
        stop = start + actuator.num_channels
        inputs = None if inputs_ts is None else inputs_ts[..., start:stop]
        if isinstance(actuator, ThreadlikeActuator):

            def threadlike_points(q, current_actuator=actuator):
                return jax.vmap(lambda s: current_actuator.path_poses(robot, q, s))(
                    s_points
                ).transpose(1, 0, 2)

            scalar_fields = {} if inputs is None else {"input": inputs}
            layers.append(
                TrajectoryActuatorVisualLayer(
                    name=actuator.name,
                    kind=actuator.kind,
                    points=vectorize_q(threadlike_points),
                    scalar_fields=scalar_fields,
                )
            )
        elif isinstance(actuator, ArticulatedMcKibbenActuator):
            if actuator.num_channels == 0:
                start = stop
                continue
            scalar_fields = {}
            if inputs is not None:
                scalar_fields["pressure"] = inputs

                def axial_forces(q, u, current_actuator=actuator):
                    return current_actuator.axial_forces(q, u)

                scalar_fields["force"] = jax.jit(
                    jax.vmap(
                        jax.vmap(
                            axial_forces,
                            in_axes=(0, 0),
                        ),
                        in_axes=(0, 0),
                    )
                )(q_ts, inputs)

            def muscle_segments(q, current_actuator=actuator):
                return current_actuator.segments(robot, q)

            layers.append(
                TrajectoryActuatorVisualLayer(
                    name=actuator.name,
                    kind="muscle",
                    points=vectorize_q(muscle_segments),
                    scalar_fields=scalar_fields,
                )
            )
        start = stop

    for element in robot.passive_elements:
        if isinstance(element, ThreadlikeImpedance):

            def passive_points(q, current_element=element):
                return jax.vmap(
                    lambda s: robot._threadlike_path_positions(
                        q, s, current_element.routing
                    )
                )(s_points).transpose(1, 0, 2)

            layers.append(
                TrajectoryActuatorVisualLayer(
                    name=element.name,
                    kind="generic",
                    points=vectorize_q(passive_points),
                )
            )

    return tuple(layers)


@eqx.filter_jit
def _sample_routing_paths(robot, q, routing, solution, num_points):
    """Sample each physical path, including straight connections between guides."""
    fractions = jnp.linspace(0.0, 1.0, num_points)
    spans = tuple(
        routing.span_positions(robot, q, solution, index, fractions)
        for index in range(routing.spans.num_spans)
    )
    return tuple(
        jnp.concatenate(
            [
                points
                for index, points in enumerate(spans)
                if routing.spans.span_path_index[index] == path
            ],
            axis=0,
        )[None, ...]
        for path in range(routing.spans.num_paths)
    )


@eqx.filter_jit
def _solve_visual_routing(robot, q, routing):
    """Solve once per configuration when no existing solution was supplied."""
    return routing.solve_routing(robot, q)


def routing_visual_layers(
    robot: SoftRobot,
    q: Array,
    routing: ConfigurationDependentThreadlikeRouting,
    *,
    solution: RoutingSolution | None = None,
    num_points: int = 80,
) -> tuple[ActuatorVisualLayer, ...]:
    """Sample configuration-dependent routing into world-space tendon polylines.

    Args:
        robot: Host used to solve the routing.
        q: Configuration, shape ``(robot.num_coordinates,)``.
        routing: Shortest-path or free-space routing, including guide connectivity.
        solution: Existing solution for exactly this robot, routing, and configuration.
            If supplied, no routing optimization or solve is performed. Callers must
            ensure it matches ``q``; solutions do not store their input configuration.
        num_points: Samples per span, including both endpoints; at least two.

    Returns:
        One tendon layer per physical path, named ``routing/path_<index>``. Keeping
        paths separate supports unequal span counts without padding geometry.
        Consecutive span endpoints form the straight cable across their joint guides.
        Positions have shape ``(1, samples_in_path, 3)`` and are in metres.
        Use existing layer styling to display other filament roles.

    Raises:
        ValueError: Invalid sample count, incompatible solution topology, or an
            unconverged/nonfinite route. Invalid routes are not displayed as valid.
    """
    if not isinstance(routing, ConfigurationDependentThreadlikeRouting):
        raise TypeError("routing must be configuration-dependent threadlike routing.")
    if not isinstance(num_points, int) or num_points < 2:
        raise ValueError("num_points must be an integer of at least two.")
    q = jnp.asarray(q)
    if q.shape != (robot.num_coordinates,):
        raise ValueError("q must have shape (robot.num_coordinates,).")
    if solution is None:
        solution = _solve_visual_routing(robot, q, routing)
    spans_match = len(solution.spans) == routing.spans.num_spans
    paths_match = solution.path_lengths.shape == (routing.spans.num_paths,)
    if not spans_match or not paths_match:
        raise ValueError("routing_solution must match the routing's spans and paths.")
    if not bool(solution.converged):
        raise ValueError("Cannot render an unconverged routing solution.")
    paths = _sample_routing_paths(robot, q, routing, solution, num_points)
    if any(not np.all(np.isfinite(points)) for points in paths):
        raise ValueError("Cannot render nonfinite routing geometry.")
    return tuple(
        ActuatorVisualLayer(name=f"routing/path_{index}", kind="tendon", points=points)
        for index, points in enumerate(paths)
    )


def _reshape_routing_solution(solution, leading_shape):
    """Normalize optional singleton batch/time axes without broadcasting solutions.

    ``leading_shape`` is the configuration's leading shape after renderer input
    normalization. Only insertion/removal of singleton axes is permitted. Every
    array leaf must carry the same leading axes as ``path_lengths``.
    """
    if solution is None:
        return None
    source = solution.path_lengths.shape[:-1]
    if tuple(n for n in source if n != 1) != tuple(n for n in leading_shape if n != 1):
        raise ValueError(
            f"routing_solution leading shape {source} must match configurations "
            f"{leading_shape}; solutions are not broadcast across configurations."
        )

    def reshape(value):
        if value.shape[: len(source)] != source:
            raise ValueError(
                "Every routing_solution array must share its batch/time axes."
            )
        return value.reshape((*leading_shape, *value.shape[len(source) :]))

    return jax.tree.map(reshape, solution)


def _routing_solution_at(solution, index):
    """Select a configuration or frame from a normalized solution PyTree."""
    return (
        None if solution is None else jax.tree.map(lambda value: value[index], solution)
    )


__all__ = [
    "actuator_visual_layers",
    "routing_visual_layers",
    "vectorized_actuator_visual_layers_trajectory",
]
