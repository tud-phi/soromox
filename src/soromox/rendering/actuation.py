"""Renderer-side adapters for semantic actuator geometry."""

from __future__ import annotations

import jax
from jax import Array
from jax import numpy as jnp

from soromox.actuation.mckibben import ArticulatedMcKibbenActuator
from soromox.actuation.threadlike import ThreadlikeActuator, ThreadlikeImpedance
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
        q: Generalized coordinates of shape ``(robot.num_dofs,)``.
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


__all__ = ["actuator_visual_layers", "vectorized_actuator_visual_layers_trajectory"]
