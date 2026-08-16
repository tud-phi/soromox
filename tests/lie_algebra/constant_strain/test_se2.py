"""Tests for planar constant-strain operators."""

import jax
from jax import numpy as jnp
from numpy.testing import assert_allclose

from soromox.utils.lie_algebra import se2
from soromox.utils.lie_algebra.constant_strain import se2 as constant_strain_se2
from soromox.utils.tolerance import Tolerance

jax.config.update("jax_enable_x64", True)

RTOL = Tolerance.rtol()
ATOL = Tolerance.atol()
EPS = float(jnp.finfo(jnp.float64).eps)


def test_adjoint_zero_theta_matches_first_order_series():
    xi = jnp.array([0.0, 1.0, -2.0])
    s = jnp.array(0.3)
    expected = jnp.eye(3) + s * se2.small_adjoint(xi)

    result = constant_strain_se2.adjoint(xi, s, eps=EPS)

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_adjoint_inverse_matches_identity():
    xi = jnp.array([0.7, 0.25, -0.1])
    s = jnp.array(0.5)

    adjoint = constant_strain_se2.adjoint(xi, s, eps=EPS)
    adjoint_inverse = constant_strain_se2.adjoint_inverse(xi, s, eps=EPS)

    assert_allclose(adjoint @ adjoint_inverse, jnp.eye(3), rtol=RTOL, atol=ATOL)
    assert_allclose(adjoint_inverse @ adjoint, jnp.eye(3), rtol=RTOL, atol=ATOL)


def test_tangent_zero_theta_matches_reduced_series():
    xi = jnp.array([0.0, 1.0, -0.5])
    s = jnp.array(0.4)
    expected = s * jnp.eye(3) + 0.5 * s**2 * se2.small_adjoint(xi)

    result = constant_strain_se2.tangent(xi, s, eps=EPS)

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_tangent_derivative_is_zero_without_motion():
    xi = jnp.array([0.7, 0.2, -0.3])
    xid = jnp.zeros((3,))
    s = jnp.array(0.6)

    result = constant_strain_se2.tangent_derivative(xi, xid, s, eps=EPS)

    assert_allclose(result, jnp.zeros((3, 3)), rtol=RTOL, atol=ATOL)


def test_tangent_derivative_zero_theta_matches_exact_reduced_series():
    xi = jnp.array([0.0, 0.5, -0.1])
    xid = jnp.array([0.1, -0.4, 0.2])
    s = jnp.array(0.2)
    ad_xi = se2.small_adjoint(xi)
    ad_xid = se2.small_adjoint(xid)
    expected = (
        0.5 * s**2 * ad_xid
        + (1.0 / 6.0) * s**3 * (ad_xid @ ad_xi + ad_xi @ ad_xid)
        + (1.0 / 24.0) * s**4 * ad_xi @ ad_xid @ ad_xi
    )

    result = constant_strain_se2.tangent_derivative(xi, xid, s, eps=EPS)

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_tangent_derivative_matches_autodiff(N: int = 10):
    key = jax.random.PRNGKey(0)
    for _ in range(N):
        key, subkey1, subkey2, subkey3 = jax.random.split(key, num=4)
        xi = jax.random.uniform(subkey1, shape=(3,), minval=-1.0, maxval=1.0)
        xid = jax.random.uniform(subkey2, shape=(3,), minval=-1.0, maxval=1.0)
        s = jax.random.uniform(subkey3, shape=(), minval=0.1, maxval=1.0)

        def tangent_map(xi_, s_current=s):
            return constant_strain_se2.tangent(xi_, s_current, eps=EPS)

        _, autodiff = jax.jvp(tangent_map, (xi,), (xid,))
        analytic = constant_strain_se2.tangent_derivative(xi, xid, s, eps=EPS)

        assert_allclose(autodiff, analytic, rtol=RTOL, atol=ATOL)


def test_constant_strain_helpers_are_autodiff_finite_at_zero():
    xi_zero = jnp.zeros((3,))
    xid_zero = jnp.zeros((3,))
    s = jnp.array(0.4)

    functions_and_arguments = (
        (lambda xi: constant_strain_se2.tangent(xi, s, eps=EPS), xi_zero),
        (
            lambda xi: constant_strain_se2.tangent_derivative(xi, xid_zero, s, eps=EPS),
            xi_zero,
        ),
        (
            lambda xid: constant_strain_se2.tangent_derivative(
                xi_zero, xid, s, eps=EPS
            ),
            xid_zero,
        ),
    )
    for function, argument in functions_and_arguments:
        assert jnp.isfinite(jax.jacrev(function)(argument)).all()
        assert jnp.isfinite(jax.jacfwd(function)(argument)).all()
