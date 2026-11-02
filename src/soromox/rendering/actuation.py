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
    ThreadlikeRouting,
)
from soromox.actuation.threadlike.fixed_routing import (
    _validate_routing_structure_for_robot,
    _validate_routing_values_for_robot,
)
from soromox.systems.soft_robot import SoftRobot

from .actuators import ActuatorVisualLayer, TrajectoryActuatorVisualLayer


def actuator_visual_layers(
    robot: SoftRobot,
    q: Array,
    s_points: Array,
    *,
    actuator_inputs: Array | None = None,
    routing: ThreadlikeRouting | ConfigurationDependentThreadlikeRouting | None = None,
    routing_solution: RoutingSolution | None = None,
) -> tuple[ActuatorVisualLayer, ...]:
    """Return geometry for installed components and optional resolved routing.

    The returned layers describe semantic geometry and optional scalar fields
    for supported active actuators and passive elements. Actuation objects
    provide path geometry and semantic kinds, while visual styling such as
    colors, radii, line widths, and scalar colormaps remains the renderer's
    responsibility. Unsupported components do not produce a layer.

    Args:
        robot: Soft robot containing the installed actuation components.
        q: Generalized coordinates of shape ``(robot.num_coordinates,)``.
        s_points: Backbone sample coordinates with shape ``(num_points,)``
            used for fixed routing. For configuration-dependent routing, only
            the sample count is used, uniformly within each span.
        actuator_inputs: Optional actuator inputs of shape
            ``(robot.num_actuators,)``. When supplied, supported adapters attach
            input-dependent scalar fields such as pressure or axial force.
        routing: Optional fixed or configuration-dependent routing alongside
            installed components, using the renderer's tendon styling.
        routing_solution: Solution for exactly this robot, routing, and ``q``.
            Supplied solutions are sampled without solving. Otherwise routing is
            solved once. Callers must ensure correspondence; solutions do not
            store their input configuration. Must be ``None`` for fixed routing,
            which is evaluated directly.

    Returns:
        Tuple of semantic layers in world coordinates (metres), before display
        offsets. Installed layers have shape ``(channels, samples, dim)``.
        Optional routing adds one layer per physical path, named
        ``routing/path_<index>``, with shape ``(1, samples_in_path, dim)``.
        Fixed routing uses the host dimension; other strategies use 3.
        Consecutive spans connect their guides with straight lines; separate
        layers allow unequal span counts. Empty if no geometry is available.

    Raises:
        TypeError: The optional routing has an unsupported type.
        ValueError: A solution is supplied without routing, or routing has an
            invalid configuration, fewer than two samples per span, incompatible
            solution topology, an unconverged solution, or nonfinite geometry.
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
            if isinstance(
                actuator.transmission.routing, ConfigurationDependentThreadlikeRouting
            ):
                resolved = _routing_visual_layers(
                    robot, q, actuator.transmission.routing, s_points=s_points
                )
                layers.extend(
                    ActuatorVisualLayer(
                        name=f"{actuator.name}/path_{index}",
                        kind=actuator.kind,
                        points=layer.points,
                        scalar_fields={}
                        if inputs is None
                        else {"input": inputs[index : index + 1]},
                    )
                    for index, layer in enumerate(resolved)
                )
                start = stop
                continue
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
            if isinstance(element.routing, ConfigurationDependentThreadlikeRouting):
                layers.extend(
                    ActuatorVisualLayer(
                        name=f"{element.name}/path_{index}",
                        kind="generic",
                        points=layer.points,
                    )
                    for index, layer in enumerate(
                        _routing_visual_layers(
                            robot, q, element.routing, s_points=s_points
                        )
                    )
                )
                continue
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

    return tuple(layers) + _routing_visual_layers(
        robot, q, routing, solution=routing_solution, s_points=s_points
    )


def vectorized_actuator_visual_layers_trajectory(
    robot: SoftRobot,
    q_ts: Array,
    s_points: Array,
    *,
    actuator_inputs: Array | None = None,
) -> tuple[TrajectoryActuatorVisualLayer, ...]:
    """Vectorize installed actuator geometry over robots and timesteps.

    Args:
        robot: Soft robot containing installed actuation components.
        q_ts: Configurations, shape ``(N, T, robot.num_coordinates)``.
        s_points: Backbone sample coordinates, shape ``(num_points,)``.
        actuator_inputs: Optional inputs, shape ``(N, T, channels)`` or
            ``(N, channels)`` for constant inputs. For ``N=1``, also accepts
            ``(T, channels)`` or a constant channel vector.

    Returns:
        Semantic layers with points shaped ``(N, T, channels, samples, dim)``
        in world coordinates (metres), before display offsets. Only installed
        components are evaluated by this vectorized path.

    Raises:
        ValueError: Configurations or actuator inputs have incompatible axes.
    """
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
            if isinstance(
                actuator.transmission.routing, ConfigurationDependentThreadlikeRouting
            ):
                paths = vectorize_q(
                    lambda q, routing=actuator.transmission.routing: (
                        _installed_routing_paths(robot, q, routing, s_points.shape[0])
                    )
                )
                for index, points in enumerate(paths):
                    if not np.all(np.isfinite(points)):
                        raise ValueError(
                            "Cannot render an unconverged or nonfinite routing solution."
                        )
                    layers.append(
                        TrajectoryActuatorVisualLayer(
                            name=f"{actuator.name}/path_{index}",
                            kind=actuator.kind,
                            points=points,
                            scalar_fields={}
                            if inputs is None
                            else {"input": inputs[..., index : index + 1]},
                        )
                    )
                start = stop
                continue

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
            if isinstance(element.routing, ConfigurationDependentThreadlikeRouting):
                paths = vectorize_q(
                    lambda q, routing=element.routing: _installed_routing_paths(
                        robot, q, routing, s_points.shape[0]
                    )
                )
                for index, points in enumerate(paths):
                    if not np.all(np.isfinite(points)):
                        raise ValueError(
                            "Cannot render an unconverged or nonfinite routing solution."
                        )
                    layers.append(
                        TrajectoryActuatorVisualLayer(
                            name=f"{element.name}/path_{index}",
                            kind="generic",
                            points=points,
                        )
                    )
                continue

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
def _sample_fixed_routing_paths(robot, q, routing, s_points):
    """Evaluate fixed routing with JAX over stations and configuration axes.

    Args:
        robot: Continuum host defining world-frame path positions.
        q: Configurations with shape ``(..., robot.num_coordinates)``.
        routing: Fixed material-frame routing family.
        s_points: Backbone coordinates with shape ``(num_points,)``.

    Returns:
        World positions in metres, shape ``(..., paths, num_points, dim)``.
        Coordinates outside a path's segment range clamp to its endpoints,
        following the same host hook as installed threadlike actuators.
    """

    def sample(configuration):
        return jax.vmap(
            lambda s: robot._threadlike_path_positions(configuration, s, routing)
        )(s_points).transpose(1, 0, 2)

    for _ in q.shape[:-1]:
        sample = jax.vmap(sample)
    return sample(q)


def _fixed_routing_points(robot, q, routing, s_points, solution):
    """Validate and sample a standalone fixed route, including batches.

    Args:
        robot: Continuum host defining the route geometry.
        q: Configurations with shape ``(..., robot.num_coordinates)``.
        routing: Fixed material-frame routing family.
        s_points: Backbone coordinates with shape ``(num_points,)``.
        solution: Must be ``None``; fixed routing has no solution object.

    Returns:
        World positions in metres, shape ``(..., paths, num_points, dim)``.

    Raises:
        ValueError: A solution is supplied, input shapes or route parameters are
            invalid, fewer than two samples are supplied, or geometry is nonfinite.
        TypeError: The robot does not support fixed threadlike routing.
    """
    if solution is not None:
        raise ValueError("routing_solution must be None for fixed ThreadlikeRouting.")
    q, s_points = jnp.asarray(q), jnp.asarray(s_points)
    if q.ndim < 1 or q.shape[-1] != robot.num_coordinates:
        raise ValueError("q must have shape (..., robot.num_coordinates).")
    if s_points.ndim != 1 or s_points.shape[0] < 2:
        raise ValueError("s_points must be a vector with at least two samples.")
    _validate_routing_structure_for_robot(routing, robot)
    _validate_routing_values_for_robot(routing, robot)
    points = _sample_fixed_routing_paths(robot, q, routing, s_points)
    if not np.all(np.isfinite(points)):
        raise ValueError("Cannot render nonfinite routing geometry.")
    return points


@eqx.filter_jit
def _sample_routing_paths(robot, q, routing, solution, num_points):
    """Sample resolved spans and join adjacent guides in world coordinates.

    Args:
        robot: Robot whose kinematics define the route.
        q: Configuration of shape ``(robot.num_coordinates,)``.
        routing: Routing strategy and ordered span connectivity.
        solution: Resolved routing for this robot and configuration.
        num_points: Number of equally spaced samples per span, including endpoints.

    Returns:
        One array of shape ``(1, num_spans_in_path * num_points, 3)`` per path,
        in metres. Span order follows ``span_path_index``. Connecting consecutive
        span endpoints forms the straight segment between prescribed joint guides.
    """
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


def _installed_routing_paths(robot, q, routing, num_points):
    """Resolve installed paths once per configuration for trajectory sampling.

    Args:
        robot: Continuum host supplying world-frame path geometry.
        q: Configuration with shape ``(robot.num_coordinates,)``.
        routing: Configuration-dependent routing strategy.
        num_points: Number of uniformly spaced samples per span.

    Returns:
        One world-position array per path, each with shape
        ``(1, num_spans_in_path * num_points, 3)``. Unconverged solutions return
        NaNs so the caller can reject the trajectory outside compiled code.

    Raises:
        ValueError: Fewer than two samples per span are requested.
    """
    if num_points < 2:
        raise ValueError("s_points must have at least two samples.")
    solution = routing.solve_routing(robot, q)
    paths = _sample_routing_paths(robot, q, routing, solution, num_points)
    return tuple(jnp.where(solution.converged, points, jnp.nan) for points in paths)


@eqx.filter_jit
def _solve_visual_routing(robot, q, routing):
    """Solve all paths for a single configuration using the selected strategy.

    Args:
        robot: Robot whose kinematics define the route.
        q: Configuration of shape ``(robot.num_coordinates,)``.
        routing: Configuration-dependent routing strategy and parameters.

    Returns:
        A ``RoutingSolution`` for every span and guide. The caller checks its
        convergence before sampling geometry.
    """
    return routing.solve_routing(robot, q)


def _routing_visual_layers(
    robot: SoftRobot,
    q: Array,
    routing: ThreadlikeRouting | ConfigurationDependentThreadlikeRouting | None,
    *,
    solution: RoutingSolution | None = None,
    s_points: Array,
) -> tuple[ActuatorVisualLayer, ...]:
    """Sample standalone routing into world-space tendon polylines.

    Args:
        robot: Host used to solve the routing.
        q: Configuration, shape ``(robot.num_coordinates,)``.
        routing: Optional fixed, shortest-path, or free-space routing.
            ``None`` returns no layers and requires ``solution=None``.
        solution: Existing solution for exactly this robot, routing, and configuration.
            If supplied, no routing optimization or solve is performed. Callers must
            ensure it matches ``q``; solutions do not store their input configuration.
            Must be ``None`` for fixed routing.
        s_points: Backbone sample coordinates, shape ``(num_points,)`` with at
            least two entries. Fixed routing uses these coordinates; other
            strategies use this sample count uniformly within each span.

    Returns:
        One tendon layer per physical path, named ``routing/path_<index>``. Keeping
        paths separate supports unequal span counts without padding geometry.
        Consecutive span endpoints form the straight cable across their joint guides.
        Positions have shape ``(1, samples_in_path, dim)`` and are in metres.
        Fixed routing uses the host dimension (2 or 3); other strategies use 3.
        Use existing layer styling to display other filament roles.

    Raises:
        TypeError: Routing has an unsupported type.
        ValueError: A solution lacks routing, or the configuration, sample count,
            solution topology, convergence, or sampled geometry is invalid.
    """
    if routing is None:
        if solution is not None:
            raise ValueError("routing is required when routing_solution is supplied.")
        return ()
    q = jnp.asarray(q)
    if q.shape != (robot.num_coordinates,):
        raise ValueError("q must have shape (robot.num_coordinates,).")
    if isinstance(routing, ThreadlikeRouting):
        points = _fixed_routing_points(robot, q, routing, s_points, solution)
        return tuple(
            ActuatorVisualLayer(
                name=f"routing/path_{index}",
                kind="tendon",
                points=points[index : index + 1],
            )
            for index in range(routing.num_paths)
        )
    if not isinstance(routing, ConfigurationDependentThreadlikeRouting):
        raise TypeError(
            "routing must be fixed or configuration-dependent threadlike routing."
        )
    s_points = jnp.asarray(s_points)
    if s_points.ndim != 1 or s_points.shape[0] < 2:
        raise ValueError("s_points must be a vector with at least two samples.")
    num_points = s_points.shape[0]
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

    Args:
        solution: Optional solution PyTree whose array leaves share the leading
            axes of ``path_lengths`` (all axes except its final path axis).
        leading_shape: Target configuration axes after renderer normalization:
            ``(N,)`` for a batch or ``(N, T)`` for a trajectory.

    Returns:
        ``None`` or a solution with the requested leading axes. Only singleton
        axes may be inserted or removed; non-singleton axes retain their order.
        Solutions are never broadcast to additional robots or timesteps.

    Raises:
        ValueError: Non-singleton axes differ or a solution leaf lacks the shared
            leading axes.
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
    """Select a configuration or frame from a normalized solution PyTree.

    Args:
        solution: Optional solution with normalized batch/time axes.
        index: Integer, slice, or tuple selecting the same leading axes from
            every array leaf, for example ``(robot_index, frame_index)``.

    Returns:
        ``None`` or the selected solution, retaining any unselected axes.
    """
    return (
        None if solution is None else jax.tree.map(lambda value: value[index], solution)
    )


__all__ = [
    "actuator_visual_layers",
    "vectorized_actuator_visual_layers_trajectory",
]
