"""Shared assertions for real JAX/Warp system-equivalence tests."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose


def assert_backend_equivalence(
    jax_model: Any,
    warp_model: Any,
    q: jnp.ndarray,
    qd: jnp.ndarray,
    y: jnp.ndarray,
) -> None:
    """Compare public dynamics APIs for scalar and batched execution.

    Args:
        jax_model: Model configured to use JAX/XLA dynamics assembly.
        warp_model: Parameter-identical model configured to use Warp assembly.
        q: Two-dimensional batch of generalized coordinates.
        qd: Velocity batch matching ``q``.
        y: Batched first-order states formed as ``[q, qd]``.
    """

    expected_batch = jax_model.dynamics_terms(q, qd)
    actual_batch = warp_model.dynamics_terms(q, qd)
    actual_mapped = jax.vmap(warp_model.dynamics_terms)(q, qd)
    actual_scalar = warp_model.dynamics_terms(q[0], qd[0])

    for result in (actual_batch, actual_mapped):
        for actual, expected in zip(result, expected_batch, strict=True):
            assert_allclose(actual, expected, rtol=2e-8, atol=2e-10)
    for actual, expected in zip(actual_scalar, expected_batch, strict=True):
        assert_allclose(actual, expected[0], rtol=2e-8, atol=2e-10)

    def terms_objective(model: Any, q_value: jnp.ndarray) -> jnp.ndarray:
        """Reduce public dynamics terms to a scalar differentiation target."""

        inertia, coriolis_qd, gravity = model.dynamics_terms(q_value, qd[0])
        return jnp.sum(inertia) + 0.25 * jnp.sum(coriolis_qd) - 0.5 * jnp.sum(gravity)

    expected_gradient = jax.grad(lambda q_: terms_objective(jax_model, q_))(q[0])
    actual_gradient = jax.grad(lambda q_: terms_objective(warp_model, q_))(q[0])
    assert_allclose(actual_gradient, expected_gradient, rtol=3e-8, atol=3e-10)

    t = jnp.asarray(0.125, dtype=jnp.float64)
    u = jnp.zeros((jax_model.num_actuators,), dtype=jnp.float64)
    tau_ext = jnp.linspace(-0.01, 0.015, jax_model.num_internal_dofs, dtype=jnp.float64)
    actuation_args = (u, tau_ext)
    expected_forward = jax.vmap(
        jax_model.forward_dynamics,
        in_axes=(None, 0, None),
    )(t, y, actuation_args)
    actual_forward = jax.vmap(
        warp_model.forward_dynamics,
        in_axes=(None, 0, None),
    )(t, y, actuation_args)
    actual_forward_scalar = warp_model.forward_dynamics(t, y[0], actuation_args)

    assert_allclose(actual_forward, expected_forward, rtol=3e-8, atol=3e-10)
    assert_allclose(
        actual_forward_scalar,
        expected_forward[0],
        rtol=3e-8,
        atol=3e-10,
    )

    expected_forward_gradient = jax.grad(
        lambda y_: jnp.sum(jax_model.forward_dynamics(t, y_, actuation_args))
    )(y[0])
    actual_forward_gradient = jax.grad(
        lambda y_: jnp.sum(warp_model.forward_dynamics(t, y_, actuation_args))
    )(y[0])
    assert_allclose(
        actual_forward_gradient,
        expected_forward_gradient,
        rtol=3e-8,
        atol=3e-10,
    )


__all__ = ["assert_backend_equivalence"]
