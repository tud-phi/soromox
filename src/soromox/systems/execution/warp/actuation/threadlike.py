"""Preparation and eligibility for Warp linear-threadlike actuation."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from soromox.actuation.core import DirectEffort
from soromox.actuation.threadlike import (
    LinearThreadlikeRoutingParams,
    ThreadlikeActuator,
    linear_threadlike_routing,
    linear_threadlike_routing_derivative,
)


class LinearThreadlikeActuationData(eqx.Module):
    """Flattened ordered routing data shared by Warp family executors.

    The arrays concatenate actuator groups in the same order as the public
    actuator input vector and :meth:`SoftRobot.actuation_matrix` columns.
    Keeping the path count static lets the execution layer construct output
    shapes without embedding concrete actuator objects in Warp modules.
    """

    num_paths: int = eqx.field(static=True)
    intercepts: Array
    slopes: Array
    start_segments: Array
    end_segments: Array
    coordinate_scales: Array

    @classmethod
    def from_model(cls, model) -> LinearThreadlikeActuationData:
        """Flatten eligible active actuator groups without changing ordering."""

        if not supports_linear_threadlike_matrix(model):
            raise TypeError(
                "Warp threadlike operands require one or more built-in linear "
                "ThreadlikeActuator groups and no other active transmissions."
            )
        actuators = tuple(model.actuators)

        def concatenate_or_view(values: list[Array]) -> Array:
            return values[0] if len(values) == 1 else jnp.concatenate(values, axis=0)

        params = [actuator.transmission.routing.params for actuator in actuators]
        return cls(
            num_paths=sum(item.num_paths for item in params),
            intercepts=concatenate_or_view(
                [jnp.asarray(item.intercept) for item in params]
            ),
            slopes=concatenate_or_view([jnp.asarray(item.slope) for item in params]),
            start_segments=concatenate_or_view(
                [item.start_segment_index_array for item in params]
            ),
            end_segments=concatenate_or_view(
                [item.end_segment_index_array for item in params]
            ),
            coordinate_scales=concatenate_or_view(
                [
                    jnp.asarray(actuator.params.transmission.coordinate_scale)
                    for actuator in actuators
                ]
            ),
        )


def _uses_builtin_linear_routing(actuator) -> bool:
    """Return whether one actuator has the exact built-in linear runtime."""

    if not isinstance(actuator, ThreadlikeActuator):
        return False
    routing = actuator.transmission.routing
    return (
        isinstance(routing.params, LinearThreadlikeRoutingParams)
        and routing.offset_fn is linear_threadlike_routing
        and routing.derivative_fn is linear_threadlike_routing_derivative
    )


def supports_linear_threadlike_matrix(model) -> bool:
    """Return whether all active transmission columns have Warp-linear routing."""

    actuators = tuple(getattr(model, "actuators", ()))
    return bool(actuators) and all(_uses_builtin_linear_routing(a) for a in actuators)


def supports_linear_threadlike_force(model) -> bool:
    """Return whether the matrix may be fused with direct actuator controls."""

    return supports_linear_threadlike_matrix(model) and all(
        type(actuator.effort_model) is DirectEffort for actuator in model.actuators
    )


__all__ = [
    "LinearThreadlikeActuationData",
    "supports_linear_threadlike_force",
    "supports_linear_threadlike_matrix",
]
