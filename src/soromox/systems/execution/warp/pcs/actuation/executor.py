"""JAX-facing batched Warp executor for PCS threadlike actuation."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from soromox.systems.execution.warp.pcs.actuation.operands import (
    PCSThreadlikeOperands,
    PCSThreadlikeShapes,
)
from soromox.systems.execution.warp.pcs.actuation.threadlike import (
    planar_threadlike_actuation_force,
    planar_threadlike_actuation_matrix,
    spatial_threadlike_actuation_force,
    spatial_threadlike_actuation_matrix,
)


def _common(operands: PCSThreadlikeOperands, q: Array) -> tuple:
    """Return ordered inputs shared by PCS matrix and force callables."""
    spatial_dim = 3 if operands.is_planar else 6
    routing = operands.routing
    return (
        q,
        operands.active_strain_indices,
        operands.active_strain_scales,
        operands.reference_strain.reshape(operands.num_segments, spatial_dim),
        operands.physical_points,
        operands.physical_weights,
        routing.intercepts,
        routing.slopes,
        routing.start_segments,
        routing.end_segments,
        routing.coordinate_scales,
        jnp.asarray([operands.global_eps], dtype=jnp.float64),
    )


def execute_actuation_matrix(operands: PCSThreadlikeOperands, q: Array) -> Array:
    """Return a canonical batched PCS actuation matrix with shape ``(E,D,A)``."""
    shapes = PCSThreadlikeShapes.from_operands(operands, batch_size=q.shape[0])
    evaluator = (
        planar_threadlike_actuation_matrix
        if operands.is_planar
        else spatial_threadlike_actuation_matrix
    )
    return evaluator(*_common(operands, q), output_dims=shapes.matrix_output())[0]


def execute_actuation_force(
    operands: PCSThreadlikeOperands, q: Array, controls: Array
) -> Array:
    """Return a canonical batched PCS fused force with shape ``(E,D)``."""
    shapes = PCSThreadlikeShapes.from_operands(operands, batch_size=q.shape[0])
    evaluator = (
        planar_threadlike_actuation_force
        if operands.is_planar
        else spatial_threadlike_actuation_force
    )
    common = _common(operands, q)
    return evaluator(
        common[0], controls, *common[1:], output_dims=shapes.force_output()
    )[0]


__all__ = ["execute_actuation_force", "execute_actuation_matrix"]
