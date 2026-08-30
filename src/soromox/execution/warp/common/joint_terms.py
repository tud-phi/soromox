"""JAX-facing orchestration for reusable spatial-joint Warp kernels."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from soromox.execution.warp.common.joints import scalable_joint_terms
from soromox.execution.warp.common.joints_cooperative import (
    cooperative_joint_terms,
)

SPATIAL_DIM = 6


def _evaluate_joint_terms(
    q: Array,
    qd: Array,
    basis: Array,
    reference: Array,
    local_to_global: Array,
    global_to_local: Array,
    *,
    cooperative: bool,
) -> tuple[Array, Array, Array, Array, Array]:
    """Evaluate a batch of general spatial joints with runtime-shaped kernels.

    The caller supplies the basis and coordinate maps rather than a concrete
    GVS model or operand type. Consequently the same execution path can be
    reused by future PCS joint support and by other system families that encode
    their joints as exponential coordinates.

    Args:
        q: Batched active generalized coordinates with shape
            ``(batch_size, num_dofs)``.
        qd: Batched generalized velocities with the same shape as ``q``.
        basis: Padded local joint bases with shape
            ``(num_segments, 6, max_dof)``.
        reference: Reference joint strains with shape ``(num_segments, 6)``.
        local_to_global: Padded local-to-active coordinate map with shape
            ``(num_segments, max_dof)``. Negative entries denote padding.
        global_to_local: Active-to-local coordinate map with shape
            ``(num_segments, num_dofs)``. Negative entries denote inactive
            coordinates for the segment.
        cooperative: Whether to evaluate one joint with a cooperative CUDA
            block or the one-thread CPU-compatible kernel.

    Returns:
        Joint adjoints, adjoint derivatives, active-coordinate tangents,
        tangent-derivative actions, and joint velocities, each with leading
        ``(batch_size, num_segments, 6)`` dimensions.
    """

    batch_size, num_dofs = q.shape
    num_segments, _, max_dof = basis.shape
    matrix_rows = batch_size * num_segments * SPATIAL_DIM
    output_dims = {
        "adjoint": (matrix_rows, SPATIAL_DIM),
        "adjoint_dot": (matrix_rows, SPATIAL_DIM),
        "tangent_local": (matrix_rows, max_dof),
        "tangent_dot_qd": (matrix_rows, 1),
        "joint_velocity": (matrix_rows, 1),
    }
    executor = cooperative_joint_terms if cooperative else scalable_joint_terms
    outputs = executor(
        q,
        qd,
        basis.reshape(num_segments * SPATIAL_DIM, max_dof),
        reference,
        local_to_global,
        output_dims=output_dims,
    )
    leading = (batch_size, num_segments, SPATIAL_DIM)
    tangent_local = outputs[2].reshape(*leading, max_dof)
    local_columns = jnp.maximum(global_to_local, 0)
    local_columns = jnp.broadcast_to(
        local_columns[None, :, None, :],
        (batch_size, num_segments, SPATIAL_DIM, num_dofs),
    )
    tangent_active = jnp.take_along_axis(tangent_local, local_columns, axis=-1)
    tangent_active *= (global_to_local >= 0)[None, :, None, :]
    return (
        outputs[0].reshape(*leading, SPATIAL_DIM),
        outputs[1].reshape(*leading, SPATIAL_DIM),
        tangent_active,
        outputs[3].reshape(*leading),
        outputs[4].reshape(*leading),
    )


__all__: list[str] = []
