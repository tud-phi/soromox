"""Prepared immutable GVS operands for repeated batched Warp term evaluation."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..common.joint_terms import _evaluate_joint_terms
from .executor import execute_dynamics_terms
from .operands import GVSFloatingOperands, GVSOperands


def prepare_dynamics_terms(
    operands: GVSOperands | GVSFloatingOperands, *, cache_fixed_joints: bool = True
) -> Callable:
    """Prepare a pure batched term function for immutable model operands.

    Args:
        operands: Fixed or floating model operands. Construct outside ``jit``.
        cache_fixed_joints: Cache invariant joint terms when every joint is
            fixed. Models with moving joints retain normal joint evaluation.

    Returns:
        A callable ``evaluate(q, qd) -> (M, Cqd, G)`` accepting one leading batch
        dimension, suitable for ``jax.jit``. Rebuild this callable when any
        model operand changes. This low-level forward executor has no custom
        derivative rule; use the model's public API for differentiation.
    """
    cached = None
    if cache_fixed_joints and not np.any(
        np.asarray(operands.joint_local_to_global) >= 0
    ):
        n = operands.joint_global_to_local.shape[1]
        zeros = jnp.zeros((1, n), dtype=jnp.float64)
        cached = _evaluate_joint_terms(
            zeros,
            zeros,
            operands.joint_basis,
            operands.joint_reference,
            operands.joint_local_to_global,
            operands.joint_global_to_local,
            cooperative=operands.block_dim > 1,
        )
        jax.block_until_ready(cached)

    def evaluate(q: jax.Array, qd: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Evaluate batched dynamics using the prepared immutable operands.

        Args:
            q: Configurations with shape ``(E, num_coordinates)``.
            qd: Generalized velocities with shape ``(E, num_velocities)``.

        Returns:
            Batched inertia matrices, convective forces and gravity forces.
        """
        terms = (
            None
            if cached is None
            else tuple(jnp.broadcast_to(x, (q.shape[0], *x.shape[1:])) for x in cached)
        )
        return execute_dynamics_terms(operands, q, qd, fixed_joint_terms=terms)

    return evaluate
