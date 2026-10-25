"""JAX batching and differentiation for Warp straight-span geometry."""

from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp

from soromox.actuation.implicit_threadlike import StraightSpanSolution
from soromox.execution.warp.actuation.straight_span import (
    single_segment_spans,
    straight_spans,
)
from soromox.utils.geometry.poses import spatial_transform


@jax.custom_batching.custom_vmap
def _execute_batch(routing, robot, configurations):
    """Resolve spans for a canonical batch using shared host link kinematics.

    Args:
        routing: Validated StraightSpanRouting with at least one span.
        robot: SoftRobot supplying relative link kinematics.
        configurations: Configuration batch, ``(E, D)``.

    Returns:
        Tuple of span solutions with an environment dimension on every leaf.
    """
    if configurations.dtype != jnp.float64:
        raise ValueError("Warp straight routing requires JAX 64-bit mode.")
    if configurations.shape[0] == 0:
        return tuple(
            StraightSpanSolution(
                jnp.empty((0, 3)),
                jnp.empty((0, 3)),
                jnp.empty((0,)),
                jnp.empty((0, robot.num_coordinates)),
                jnp.empty((0,), dtype=bool),
            )
            for _ in range(routing.spans.num_spans)
        )
    bounds = tuple(
        zip(routing.spans.start_segment_index, routing.spans.end_segment_index)
    )
    intervals = tuple(dict.fromkeys(bounds))
    links, body_jacobians, joints = jax.vmap(robot.relative_segment_kinematics)(
        configurations
    )
    bases = jax.vmap(
        lambda configuration: spatial_transform(
            robot.base_transform_from_configuration(configuration)
        )
    )(configurations)[:, None]
    batch, segments = links.shape[:2]
    span_count, interval_count = len(bounds), len(intervals)
    coordinates = configurations.shape[1]
    output_dims = {
        "entries": (batch, segments, 4, 4),
        "exits": (batch, segments, 4, 4),
        "geometry": (batch, span_count, 7),
        "jacobians": (batch, span_count, coordinates),
        "valid": (batch, span_count),
    }
    if all(first == last for first, last in bounds):
        evaluator = single_segment_spans
        interval_array = jnp.tile(
            jnp.arange(segments, dtype=jnp.int32)[:, None], (1, 2)
        )
        indices = jnp.asarray(routing.spans.start_segment_index, dtype=jnp.int32)
    else:
        evaluator = straight_spans
        interval_array = jnp.asarray(intervals, dtype=jnp.int32)
        indices = jnp.asarray(
            [intervals.index(bound) for bound in bounds], dtype=jnp.int32
        )
        output_dims.update(
            {
                "transforms": (batch, interval_count, 4, 4),
                "interval_jacobians": (batch, interval_count, 6, coordinates),
            }
        )
    outputs = evaluator(
        links,
        joints,
        body_jacobians,
        bases,
        interval_array,
        indices,
        routing.spans.start_offset.astype(jnp.float64),
        routing.spans.end_offset.astype(jnp.float64),
        output_dims=output_dims,
    )
    geometry, jacobians, valid = outputs[-3:]
    return tuple(
        StraightSpanSolution(
            geometry[:, index, :3],
            geometry[:, index, 3:6],
            geometry[:, index, 6],
            jacobians[:, index],
            valid[:, index],
        )
        for index in range(span_count)
    )


@_execute_batch.def_vmap
def _execute_batch_vmap(axis_size, in_batched, routing, robot, configurations):
    """Flatten nested configuration batches into one set of Warp launches.

    Args:
        axis_size: Number of environments.
        in_batched: Batched flags for routing, model, and configuration leaves.
        routing: Shared routing parameters.
        robot: Shared host model.
        configurations: Configuration batch with an optional added leading axis.

    Returns:
        Batched span solutions and matching output batching flags.

    Raises:
        ValueError: If model or guide parameters are batched.
    """
    routing_batched, robot_batched, configuration_batched = in_batched
    if any(jax.tree.leaves((routing_batched, robot_batched))):
        raise ValueError(
            "Warp straight routing batches configurations with shared parameters."
        )
    if not configuration_batched:
        configurations = jnp.broadcast_to(
            configurations, (axis_size, *configurations.shape)
        )
    leading = configurations.shape[:-1]
    result = _execute_batch(
        routing,
        robot,
        configurations.reshape((prod(leading), configurations.shape[-1])),
    )
    result = jax.tree.map(
        lambda value: value.reshape((*leading, *value.shape[1:])), result
    )
    return result, jax.tree.map(lambda _: True, result)


@eqx.filter_custom_jvp
def resolve_spans(routing, robot, configuration):
    """Resolve Warp spans while preserving JAX derivatives through the routing.

    Args:
        routing: Validated straight routing with at least one span.
        robot: Host model.
        configuration: Configuration vector, ``(D,)``.

    Returns:
        Tuple of straight span solutions. Explicit length Jacobians are evaluated
        in Warp; transformed calls use the JAX reference, as other Warp executors do.
    """
    return jax.tree.map(
        lambda value: value[0], _execute_batch(routing, robot, configuration[None])
    )


@resolve_spans.def_jvp
def _resolve_spans_jvp(primals, tangents):
    """Evaluate routing sensitivities with the JAX reference implementation.

    Args:
        primals: Routing, host, and configuration.
        tangents: Corresponding perturbations.

    Returns:
        Span solutions and their directional derivatives.
    """
    return eqx.filter_jvp(
        lambda routing, robot, configuration: (
            routing.solve_routing(robot, configuration, backend="jax").spans
        ),
        primals,
        tangents,
    )
