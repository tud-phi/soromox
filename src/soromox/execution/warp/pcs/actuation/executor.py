"""JAX-facing batched Warp executor for PCS threadlike actuation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from soromox.execution.warp.pcs.actuation.cooperative import (
    PCS_THREADLIKE_FORCE_BLOCK_DIM,
    PCS_THREADLIKE_PATH_POINT_MAX_PATHS,
    PCS_THREADLIKE_SERIAL_MAX_PATHS,
    planar_threadlike_force_path,
    planar_threadlike_force_path_point,
    spatial_threadlike_force_path,
    spatial_threadlike_force_path_point,
)
from soromox.execution.warp.pcs.actuation.operands import (
    PCSThreadlikeOperands,
    PCSThreadlikeShapes,
)
from soromox.execution.warp.pcs.actuation.threadlike import (
    planar_threadlike_force,
    planar_threadlike_matrix,
    spatial_threadlike_force,
    spatial_threadlike_matrix,
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
        planar_threadlike_matrix if operands.is_planar else spatial_threadlike_matrix
    )
    return evaluator(
        *_common(operands, q),
        launch_dims=(q.shape[0] * operands.num_segments * operands.routing.num_paths,),
        output_dims=shapes.matrix_output(),
    )[0]


def execute_actuation_force(
    operands: PCSThreadlikeOperands, q: Array, controls: Array
) -> Array:
    """Return a canonical batched PCS fused force with shape ``(E,D)``."""
    shapes = PCSThreadlikeShapes.from_operands(operands, batch_size=q.shape[0])
    cooperative = not (
        jax.default_backend() != "gpu"
        or operands.routing.num_paths <= PCS_THREADLIKE_SERIAL_MAX_PATHS
    )
    if not cooperative:
        evaluator = (
            planar_threadlike_force if operands.is_planar else spatial_threadlike_force
        )
    elif operands.routing.num_paths <= PCS_THREADLIKE_PATH_POINT_MAX_PATHS:
        evaluator = (
            planar_threadlike_force_path_point
            if operands.is_planar
            else spatial_threadlike_force_path_point
        )
    else:
        evaluator = (
            planar_threadlike_force_path
            if operands.is_planar
            else spatial_threadlike_force_path
        )
    common = _common(operands, q)
    kwargs = {"output_dims": shapes.force_output()}
    if cooperative:
        kwargs["launch_dims"] = (
            q.shape[0] * operands.num_segments,
            PCS_THREADLIKE_FORCE_BLOCK_DIM,
        )
    else:
        kwargs["launch_dims"] = (q.shape[0] * operands.num_segments,)
    return evaluator(common[0], controls, *common[1:], **kwargs)[0]


__all__ = ["execute_actuation_force", "execute_actuation_matrix"]
