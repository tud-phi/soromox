"""Tests for spatial constant-strain operators."""

import jax
import pytest
from jax import Array
from jax import numpy as jnp
from numpy.testing import assert_allclose

from soromox.utils.lie_algebra import se3
from soromox.utils.lie_algebra.constant_strain import se3 as constant_strain_se3
from soromox.utils.tolerance import Tolerance

jax.config.update("jax_enable_x64", True)

RTOL = Tolerance.rtol()
ATOL = Tolerance.atol()
EPS = float(jnp.finfo(jnp.float64).eps)


def test_adjoint_inverse_matches_identity():
    xi = jnp.array([0.2, -0.1, 0.4, 0.3, -0.5, 0.1])
    s = jnp.array(0.45)

    adjoint = constant_strain_se3.adjoint(xi, s, eps=EPS)
    adjoint_inverse = constant_strain_se3.adjoint_inverse(xi, s, eps=EPS)

    assert_allclose(adjoint @ adjoint_inverse, jnp.eye(6), rtol=RTOL, atol=ATOL)
    assert_allclose(adjoint_inverse @ adjoint, jnp.eye(6), rtol=RTOL, atol=ATOL)


def test_tangent_zero_theta_matches_reduced_series():
    xi = jnp.array([0.0, 0.0, 0.0, 0.4, -0.3, 0.2])
    s = jnp.array(0.35)
    expected = s * jnp.eye(6) + 0.5 * s**2 * se3.small_adjoint(xi)

    result = constant_strain_se3.tangent(xi, s, eps=EPS)

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_tangent_derivative_is_zero_without_motion():
    xi = jnp.array([0.3, -0.2, 0.4, 0.5, -0.1, 0.2])
    xid = jnp.zeros((6,))
    s = jnp.array(0.6)

    result = constant_strain_se3.tangent_derivative(xi, xid, s, eps=EPS)

    assert_allclose(result, jnp.zeros((6, 6)), rtol=RTOL, atol=ATOL)


def test_tangent_derivative_matches_autodiff_random(N: int = 10):
    key = jax.random.PRNGKey(2)
    samples = 0
    while samples < N:
        key, subkey1, subkey2, subkey3 = jax.random.split(key, num=4)
        xi = jax.random.uniform(subkey1, shape=(6,), minval=-1.0, maxval=1.0)
        xid = jax.random.uniform(subkey2, shape=(6,), minval=-1.0, maxval=1.0)
        s = jax.random.uniform(subkey3, shape=(), minval=0.1, maxval=1.0)
        if jnp.linalg.norm(xi[:3]) < 1e-3:
            continue

        def tangent_map(xi_, s_current=s):
            return constant_strain_se3.tangent(xi_, s_current, eps=EPS)

        _, autodiff = jax.jvp(tangent_map, (xi,), (xid,))
        analytic = constant_strain_se3.tangent_derivative(xi, xid, s, eps=EPS)

        assert_allclose(autodiff, analytic, rtol=RTOL, atol=ATOL)
        samples += 1


def test_constant_strain_helpers_are_autodiff_finite_at_zero():
    xi_zero = jnp.zeros((6,))
    xid_zero = jnp.zeros((6,))
    s = jnp.array(0.35)
    functions_and_arguments = (
        (lambda xi: constant_strain_se3.tangent(xi, s, eps=EPS), xi_zero),
        (
            lambda xi: constant_strain_se3.tangent_derivative(xi, xid_zero, s, eps=EPS),
            xi_zero,
        ),
        (
            lambda xid: constant_strain_se3.tangent_derivative(
                xi_zero, xid, s, eps=EPS
            ),
            xid_zero,
        ),
    )
    for function, argument in functions_and_arguments:
        assert jnp.isfinite(jax.jacrev(function)(argument)).all()
        assert jnp.isfinite(jax.jacfwd(function)(argument)).all()


BRANCH_EPS = 1e1 * float(jnp.finfo(jnp.float64).eps)
TANGENT_BRANCH_EPS = float(jnp.sqrt(jnp.asarray(BRANCH_EPS)))
XI_STRAIGHT = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
ARC_LENGTH = jnp.array(0.1)


def _batched_grad(function, argument: Array, batch: int = 3) -> Array:
    stacked = jnp.broadcast_to(argument, (batch,) + argument.shape)
    return jax.grad(lambda values: jnp.sum(jax.vmap(function)(values)))(stacked)


BRANCH_SAFETY_CASES = {
    "tangent": lambda xi: constant_strain_se3.tangent(
        xi, ARC_LENGTH, TANGENT_BRANCH_EPS
    ),
    "adjoint": lambda xi: constant_strain_se3.adjoint(xi, ARC_LENGTH, BRANCH_EPS),
    "adjoint_inverse": lambda xi: constant_strain_se3.adjoint_inverse(
        xi, ARC_LENGTH, BRANCH_EPS
    ),
    "tangent_derivative": lambda xi: constant_strain_se3.tangent_derivative(
        xi, jnp.zeros(6), ARC_LENGTH, TANGENT_BRANCH_EPS
    ),
}


@pytest.mark.parametrize("name", sorted(BRANCH_SAFETY_CASES))
def test_grad_of_vmap_is_finite_at_zero_rotation(name: str):
    function = BRANCH_SAFETY_CASES[name]
    gradient = _batched_grad(lambda xi: jnp.sum(function(xi)), XI_STRAIGHT)
    assert jnp.isfinite(gradient).all()
