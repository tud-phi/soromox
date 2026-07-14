"""Renderer-side adapters for semantic actuator geometry."""

from __future__ import annotations

from jax import Array
from jax import numpy as jnp

from soromox.actuation.mckibben import ArticulatedMcKibbenActuator
from soromox.actuation.threadlike import ThreadlikeActuator, ThreadlikeImpedance

from .actuators import ActuatorVisualLayer


def actuator_visual_layers(
    robot,
    q: Array,
    s_points: Array,
    *,
    actuator_inputs: Array | None = None,
) -> tuple[ActuatorVisualLayer, ...]:
    """Adapt installed actuator mechanics to renderer-facing visual layers.

    Actuation objects provide path geometry and semantic kinds. Visual styling is
    deliberately left unset here so renderer configuration owns colors, radii,
    line widths, and scalar colormaps.
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


__all__ = ["actuator_visual_layers"]
