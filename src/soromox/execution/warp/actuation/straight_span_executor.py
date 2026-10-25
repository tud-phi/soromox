"""JAX batching and differentiation around native Warp free-space actuation."""

from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp

from soromox.actuation.implicit_threadlike import (
    RoutingGuideSolution,
    RoutingSolution,
    StraightSpanSolution,
)
from soromox.execution.warp.actuation import straight_span_launch as native
from soromox.execution.warp.actuation.straight_span_actuation import (
    interval_workspace_count,
    routing_topology,
    workspace_shapes,
)
from soromox.execution.warp.actuation.straight_span_kinematics import (
    StraightKinematicsOperands,
)
from soromox.utils.geometry.poses import spatial_transform


def _batch_evaluator(operation):
    """Specialize batching for solutions, matrices, or direct forces."""

    @jax.custom_batching.custom_vmap
    def execute(routing, robot, configurations, tensions):
        """Pass prepared model data and runtime state to a single native FFI call."""
        model = StraightKinematicsOperands.from_model(robot)
        topology = routing_topology(routing, model)
        batch, coordinates = configurations.shape
        spans, paths = routing.spans.num_spans, routing.spans.num_paths
        internal = robot.num_internal_dofs
        shapes = workspace_shapes(
            batch,
            robot.num_segments,
            internal,
            interval_workspace_count(routing),
            model.distances.shape[1],
        )
        arguments = [
            configurations,
            model.basis1.astype(jnp.float64),
            model.basis2.astype(jnp.float64),
            model.reference1.astype(jnp.float64),
            model.reference2.astype(jnp.float64),
            model.distances.astype(jnp.float64),
            *topology,
            routing.spans.start_offset.astype(jnp.float64),
            routing.spans.end_offset.astype(jnp.float64),
        ]
        if operation == "force":
            arguments.append(tensions)
            shapes["force"] = (batch, robot.num_velocities)
        elif operation == "matrix":
            shapes["matrix"] = (batch, robot.num_velocities, paths)
        else:
            fixed_base = jnp.eye(4)
            if not robot.floating_base:
                fixed_base = spatial_transform(
                    robot.base_transform_from_configuration(
                        jnp.zeros(robot.num_coordinates)
                    )
                )
            arguments.append(fixed_base.astype(jnp.float64))
            shapes.update(
                {
                    "entries": (batch, robot.num_segments, 4, 4),
                    "exits": (batch, robot.num_segments, 4, 4),
                    "geometry": (batch, spans, 7),
                    "span_jacobians": (batch, spans, internal),
                    "valid": (batch, spans),
                    "guide_geometry": (batch, spans, 4),
                    "guide_valid": (batch, spans),
                    "lengths": (batch, paths),
                    "matrix": (batch, coordinates, paths),
                }
            )
        if batch == 0:
            outputs = [
                jnp.empty(
                    shape,
                    dtype=bool if name in ("valid", "guide_valid") else jnp.float64,
                )
                for name, shape in shapes.items()
            ]
        else:
            outputs = getattr(native, operation)(*arguments, output_dims=shapes)
        if operation != "routing":
            return outputs[-1]
        geometry, jacobians, valid, guides, guide_valid, lengths, matrix = outputs[-7:]
        jacobians = jnp.pad(
            jacobians, ((0, 0), (0, 0), (robot.num_base_coordinates, 0))
        )
        # Invalid spans invalidate every configuration derivative, including base columns.
        jacobians = jnp.where(valid[..., None], jacobians, jnp.nan)
        solved = tuple(
            StraightSpanSolution(
                geometry[:, i, :3],
                geometry[:, i, 3:6],
                geometry[:, i, 6],
                jacobians[:, i],
                valid[:, i],
            )
            for i in range(spans)
        )
        previous, connections = set(), []
        for index, path in enumerate(routing.spans.span_path_index):
            if path in previous:
                connections.append(
                    RoutingGuideSolution(
                        index,
                        guides[:, index, 3],
                        guides[:, index, :3],
                        guide_valid[:, index],
                    )
                )
            previous.add(path)
        return RoutingSolution(
            solved, lengths, tuple(connections), -jnp.swapaxes(matrix, -1, -2)
        )

    @execute.def_vmap
    def batch(axis_size, in_batched, routing, robot, configurations, tensions):
        """Flatten nested batches while retaining shared numerical parameters."""
        routing_batched, robot_batched, q_batched, tension_batched = in_batched
        if any(jax.tree.leaves((routing_batched, robot_batched))):
            raise ValueError(
                "Warp straight routing batches configurations with shared parameters."
            )
        if not q_batched:
            configurations = jnp.broadcast_to(
                configurations, (axis_size, *configurations.shape)
            )
        if not tension_batched:
            tensions = jnp.broadcast_to(tensions, (axis_size, *tensions.shape))
        leading = configurations.shape[:-1]
        result = execute(
            routing,
            robot,
            configurations.reshape((prod(leading), configurations.shape[-1])),
            tensions.reshape((prod(leading), tensions.shape[-1])),
        )
        result = jax.tree.map(
            lambda value: value.reshape((*leading, *value.shape[1:])), result
        )
        return result, jax.tree.map(lambda _: True, result)

    return execute


_EVALUATORS = {
    operation: _batch_evaluator(operation)
    for operation in ("routing", "matrix", "force")
}


@eqx.filter_custom_jvp
def _resolve(routing, robot, configuration, tensions, operation):
    """Use native forward execution and the analytical JAX reference for AD."""
    result = _EVALUATORS[operation](routing, robot, configuration[None], tensions[None])
    return jax.tree.map(lambda value: value[0], result)


@_resolve.def_jvp
def _resolve_jvp(primals, tangents):
    """Differentiate configuration, guide, model, and tension parameters in JAX."""

    def reference(routing, robot, configuration, tensions, operation):
        if operation == "routing":
            return routing.solve_routing(robot, configuration, backend="jax")
        if operation == "matrix":
            return routing.actuation_matrix(robot, configuration, backend="jax")
        return routing.generalized_force(robot, configuration, tensions, backend="jax")

    return eqx.filter_jvp(reference, primals, tangents)


def evaluate(routing, robot, configuration, tensions=None, operation="routing"):
    """Resolve geometry or evaluate actuation without JAX runtime kinematics."""
    if tensions is None:
        tensions = jnp.empty((0,), dtype=configuration.dtype)
    return _resolve(routing, robot, configuration, tensions, operation)
