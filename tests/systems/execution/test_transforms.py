"""Tests for shared batching and differentiation transformation rules."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from numpy.testing import assert_allclose

from soromox.systems.execution import (
    DynamicsTerms,
    ExecutionBackend,
    evaluate_forward_dynamics,
    make_dynamics_evaluator,
)

jax.config.update("jax_enable_x64", True)


class _TransformProbe(eqx.Module):
    """Differentiable reference model for execution transform tests."""

    scale: Array
    backend: ExecutionBackend = eqx.field(static=True, default="warp")
    num_dofs: int = eqx.field(static=True, default=3)
    num_actuators: int = eqx.field(static=True, default=0)

    def _assemble_dynamics_terms(self, q: Array, qd: Array) -> DynamicsTerms:
        """Return nonlinear terms so both tangent paths are observable."""

        inertia = self.scale * jnp.eye(self.num_dofs, dtype=q.dtype) + jnp.outer(q, q)
        coriolis_qd = self.scale * q * qd
        gravity = jnp.sin(q) + self.scale
        return inertia, coriolis_qd, gravity

    def dynamics_terms(
        self,
        q: Array,
        qd: Array,
        *,
        backend: ExecutionBackend | None = None,
    ) -> DynamicsTerms:
        """Expose which backend the outer differentiation boundary selected."""

        del qd
        factor = self.scale if backend == "jax" else 10.0 * self.scale
        return jnp.eye(self.num_dofs), factor * jnp.sin(q), jnp.zeros_like(q)

    def elastic_force(self, q: Array) -> Array:
        """Return zero elastic force for the execution probe."""

        return jnp.zeros_like(q)

    def damping_matrix(self, q: Array) -> Array:
        """Return zero damping for the execution probe."""

        return jnp.zeros((q.size, q.size), dtype=q.dtype)

    def actuation_force(
        self,
        q: Array,
        u: Array,
        *,
        qd: Array,
        backend: ExecutionBackend | None = None,
    ) -> Array:
        """Expose automatic actuation routing in the primal calculation."""

        del u, qd
        factor = 2.0 if backend == "auto" else 0.0
        return factor * jnp.ones_like(q)

    def _solve_inertia(self, inertia: Array, rhs: Array) -> Array:
        """Solve the probe's generalized inertia system."""

        return jnp.linalg.solve(inertia, rhs)


def _assert_terms_close(actual: DynamicsTerms, expected: DynamicsTerms) -> None:
    """Compare the three dynamics outputs with strict FP64 tolerances."""

    for actual_term, expected_term in zip(actual, expected, strict=True):
        assert_allclose(actual_term, expected_term, rtol=1e-12, atol=1e-12)


def test_dynamics_evaluator_uses_jax_for_forward_and_reverse_derivatives() -> None:
    """Treat custom JVP as routing, not as a new analytical derivative."""

    model = _TransformProbe(scale=jnp.asarray(2.0, dtype=jnp.float64))
    q = jnp.asarray([0.1, -0.2, 0.3], dtype=jnp.float64)
    qd = jnp.asarray([-0.4, 0.5, 0.6], dtype=jnp.float64)
    tangent = jnp.asarray([0.7, -0.8, 0.9], dtype=jnp.float64)

    def execute_batch(model: _TransformProbe, q: Array, qd: Array) -> DynamicsTerms:
        del qd
        batch_size = q.shape[0]
        return (
            jnp.ones((batch_size, model.num_dofs, model.num_dofs)),
            2.0 * jnp.ones((batch_size, model.num_dofs)),
            3.0 * jnp.ones((batch_size, model.num_dofs)),
        )

    evaluate = make_dynamics_evaluator(execute_batch, family_name="test")
    expected_jvp = jax.jvp(
        lambda q_: model._assemble_dynamics_terms(q_, qd),
        (q,),
        (tangent,),
    )
    actual_jvp = jax.jvp(lambda q_: evaluate(model, q_, qd), (q,), (tangent,))
    _assert_terms_close(actual_jvp[0], expected_jvp[0])
    _assert_terms_close(actual_jvp[1], expected_jvp[1])

    expected_gradient = jax.grad(
        lambda q_: sum(
            jnp.sum(value) for value in model._assemble_dynamics_terms(q_, qd)
        )
    )(q)
    actual_gradient = jax.grad(
        lambda q_: sum(jnp.sum(value) for value in evaluate(model, q_, qd))
    )(q)
    assert_allclose(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


def test_forward_dynamics_boundary_routes_derivatives_to_jax() -> None:
    """Differentiate the complete calculation without staging its Warp primal."""

    model = _TransformProbe(scale=jnp.asarray(2.0, dtype=jnp.float64))
    time = jnp.asarray(0.0, dtype=jnp.float64)
    y = jnp.linspace(-0.3, 0.4, 2 * model.num_dofs, dtype=jnp.float64)
    tangent = jnp.linspace(0.5, -0.2, y.size, dtype=jnp.float64)

    primal = evaluate_forward_dynamics(model, time, y, None, None)
    expected_primal = jnp.concatenate(
        (
            y[model.num_dofs :],
            2.0 - 10.0 * model.scale * jnp.sin(y[: model.num_dofs]),
        )
    )
    assert_allclose(primal, expected_primal, rtol=0.0, atol=0.0)

    expected_jvp = jax.jvp(
        lambda y_: evaluate_forward_dynamics(model, time, y_, None, "jax"),
        (y,),
        (tangent,),
    )
    actual_jvp = jax.jvp(
        lambda y_: evaluate_forward_dynamics(model, time, y_, None, None),
        (y,),
        (tangent,),
    )
    for actual, expected in zip(actual_jvp, expected_jvp, strict=True):
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    expected_gradient = jax.grad(
        lambda y_: jnp.sum(evaluate_forward_dynamics(model, time, y_, None, "jax"))
    )(y)
    actual_gradient = jax.grad(
        lambda y_: jnp.sum(evaluate_forward_dynamics(model, time, y_, None, None))
    )(y)
    assert_allclose(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)
