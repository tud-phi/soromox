"""JAX-facing batched Warp executor for GVS threadlike actuation."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from soromox.systems.execution.warp.gvs.actuation.operands import (
    GVSThreadlikeOperands,
    GVSThreadlikeShapes,
)
from soromox.systems.execution.warp.gvs.actuation.threadlike import (
    gvs_threadlike_actuation_force,
    gvs_threadlike_actuation_matrix,
)


def _common(operands: GVSThreadlikeOperands, q: Array) -> tuple:
    """Return ordered inputs shared by GVS matrix and force callables."""
    routing = operands.routing
    return (
        q,
        operands.link_local_to_global,
        operands.link_basis,
        operands.reference_strain,
        operands.integration_points,
        operands.integration_weights,
        operands.segment_lengths,
        operands.segment_starts,
        routing.intercepts,
        routing.slopes,
        routing.start_segments,
        routing.end_segments,
        routing.coordinate_scales,
        jnp.asarray(
            [int(operands.scale_rotational_basis_by_length)], dtype=jnp.int32
        ),
        jnp.asarray([operands.global_eps], dtype=jnp.float64),
    )


def execute_actuation_matrix(operands: GVSThreadlikeOperands, q: Array) -> Array:
    """Return a canonical batched GVS actuation matrix with shape ``(E,D,A)``."""
    shapes = GVSThreadlikeShapes.from_operands(operands, batch_size=q.shape[0])
    return gvs_threadlike_actuation_matrix(
        *_common(operands, q), output_dims=shapes.matrix_output()
    )[0]


def execute_actuation_force(
    operands: GVSThreadlikeOperands, q: Array, controls: Array
) -> Array:
    """Return a canonical batched GVS fused force with shape ``(E,D)``."""
    shapes = GVSThreadlikeShapes.from_operands(operands, batch_size=q.shape[0])
    common = _common(operands, q)
    return gvs_threadlike_actuation_force(
        common[0], controls, *common[1:], output_dims=shapes.force_output()
    )[0]


__all__ = ["execute_actuation_force", "execute_actuation_matrix"]
