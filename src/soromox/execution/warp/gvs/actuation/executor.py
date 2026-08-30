"""JAX-facing batched Warp executor for GVS threadlike actuation."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from soromox.execution.warp.gvs.actuation.operands import (
    GVSThreadlikeOperands,
    GVSThreadlikeShapes,
)
from soromox.execution.warp.gvs.actuation.threadlike import (
    gvs_threadlike_actuation_force,
    gvs_threadlike_actuation_matrix,
)


def _common(operands: GVSThreadlikeOperands, q: Array) -> tuple:
    """Return ordered inputs shared by GVS matrix and force callables."""
    routing = operands.routing
    return (
        q,
        operands.link_local_to_global,
        operands.link_global_to_local,
        operands.link_basis,
        operands.reference_strain,
        operands.physical_points,
        operands.integration_weights,
        routing.intercepts,
        routing.slopes,
        routing.start_segments,
        routing.end_segments,
        routing.coordinate_scales,
        jnp.asarray([operands.global_eps], dtype=jnp.float64),
    )


def execute_actuation_matrix(operands: GVSThreadlikeOperands, q: Array) -> Array:
    """Return a canonical batched GVS actuation matrix with shape ``(E,D,A)``."""
    shapes = GVSThreadlikeShapes.from_operands(operands, batch_size=q.shape[0])
    return gvs_threadlike_actuation_matrix(
        *_common(operands, q), output_dims=shapes.matrix_outputs()
    )[-1]


def execute_actuation_force(
    operands: GVSThreadlikeOperands, q: Array, controls: Array
) -> Array:
    """Return a canonical batched GVS fused force with shape ``(E,D)``."""
    shapes = GVSThreadlikeShapes.from_operands(operands, batch_size=q.shape[0])
    common = _common(operands, q)
    return gvs_threadlike_actuation_force(
        common[0], controls, *common[1:], output_dims=shapes.force_outputs()
    )[-1]


__all__ = ["execute_actuation_force", "execute_actuation_matrix"]
