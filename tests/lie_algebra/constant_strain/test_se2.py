"""Tests for planar constant-strain operators."""

import jax
import pytest
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


def _matrix_exp_series(matrix, terms=20):
    result = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    power = result
    factorial = 1.0
    for order in range(1, terms):
        power = power @ matrix
        factorial *= order
        result = result + power / factorial
    return result


def _integrated_exp_series(matrix, s, terms=20):
    result = jnp.zeros_like(matrix)
    power = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    factorial = 1.0
    for order in range(terms):
        factorial *= order + 1
        result = result + s ** (order + 1) * power / factorial
        power = power @ matrix
    return result


def _integrated_exp_series_directional(matrix, matrix_direction, s, terms=20):
    result_direction = jnp.zeros_like(matrix)
    power = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    power_direction = jnp.zeros_like(matrix)
    factorial = 1.0
    for order in range(terms):
        factorial *= order + 1
        result_direction = (
            result_direction + s ** (order + 1) * power_direction / factorial
        )
        power_direction = power_direction @ matrix + power @ matrix_direction
        power = power @ matrix
    return result_direction


@pytest.mark.parametrize("scale", [-1.25, 0.0, 1.0, 1.25])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(jnp.float64, 2e-9, 2e-10), (jnp.float32, 1e-3, 1e-4)],
)
def test_operators_match_independent_series(scale, dtype, rtol, atol):
    s = jnp.asarray(0.73, dtype=dtype)
    threshold = se2._left_jacobian_cutoff(0.0, dtype)
    theta = scale * threshold / s
    xi = jnp.array([theta, 0.7, -0.4], dtype=dtype)
    xid = jnp.array([0.1, -0.2, 0.3], dtype=dtype)
    eps = jnp.zeros((), dtype=dtype)

    ad_xi = se2.small_adjoint(xi)
    actual_adjoint = constant_strain_se2.adjoint(xi, s, eps)
    actual_tangent = constant_strain_se2.tangent(xi, s, eps)
    expected_adjoint = _matrix_exp_series(s * ad_xi)
    expected_tangent = _integrated_exp_series(ad_xi, s)

    assert_allclose(actual_adjoint, expected_adjoint, rtol=rtol, atol=atol)
    assert_allclose(actual_tangent, expected_tangent, rtol=rtol, atol=atol)
    assert_allclose(
        jax.jacrev(lambda value: constant_strain_se2.tangent(value, s, eps))(xi),
        jax.jacrev(lambda value: _integrated_exp_series(se2.small_adjoint(value), s))(
            xi
        ),
        rtol=rtol,
        atol=atol,
    )
    weights = (
        jnp.arange(actual_tangent.size, dtype=dtype).reshape(actual_tangent.shape) + 1.0
    )
    assert_allclose(
        jax.jacrev(
            jax.jacrev(
                lambda value: jnp.sum(
                    weights * constant_strain_se2.tangent(value, s, eps)
                )
            )
        )(xi),
        jax.jacrev(
            jax.jacrev(
                lambda value: jnp.sum(
                    weights * _integrated_exp_series(se2.small_adjoint(value), s)
                )
            )
        )(xi),
        rtol=rtol,
        atol=atol,
    )
    expected_direction = _integrated_exp_series_directional(
        ad_xi, se2.small_adjoint(xid), s
    )
    assert_allclose(
        constant_strain_se2.tangent_derivative(xi, xid, s, eps),
        expected_direction,
        rtol=rtol,
        atol=atol,
    )


def test_zero_arc_length_limits():
    xi = jnp.array([0.3, 0.7, -0.4])
    xid = jnp.array([0.1, -0.2, 0.3])

    assert_allclose(constant_strain_se2.adjoint(xi, 0.0, 0.0), jnp.eye(3))
    assert_allclose(constant_strain_se2.tangent(xi, 0.0, 0.0), jnp.zeros((3, 3)))
    assert_allclose(
        constant_strain_se2.tangent_derivative(xi, xid, 0.0, 0.0),
        jnp.zeros((3, 3)),
    )


@pytest.mark.parametrize("scale", [0.0, 1.0, 1.25])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(jnp.float64, 2e-9, 2e-10), (jnp.float32, 1e-3, 1e-4)],
)
def test_fused_operators_match_public_operators(scale, dtype, rtol, atol):
    s = jnp.asarray(0.73, dtype=dtype)
    threshold = se2._left_jacobian_cutoff(0.0, dtype)
    theta = scale * threshold / s
    xi = jnp.array([theta, 0.7, -0.4], dtype=dtype)
    xid = jnp.array([0.1, -0.2, 0.3], dtype=dtype)
    adjoint_eps = jnp.zeros((), dtype=dtype)
    tangent_eps = 2.0 * threshold / jnp.abs(s)

    pair = constant_strain_se2._operators(xi, s, adjoint_eps, tangent_eps)
    triple = constant_strain_se2._operators(xi, s, adjoint_eps, tangent_eps, xid)
    public = constant_strain_se2.operators(xi, s, tangent_eps, xid)
    expected = (
        constant_strain_se2.adjoint_inverse(xi, s, adjoint_eps),
        constant_strain_se2.tangent(xi, s, tangent_eps),
        constant_strain_se2.tangent_derivative(xi, xid, s, tangent_eps),
    )

    for actual, reference in zip(pair, expected[:2], strict=True):
        assert_allclose(actual, reference, rtol=rtol, atol=atol)
    for actual, reference in zip(triple, expected, strict=True):
        assert_allclose(actual, reference, rtol=rtol, atol=atol)
    public_expected = (
        constant_strain_se2.adjoint(xi, s, tangent_eps),
        expected[0],
        expected[1],
        expected[2],
    )
    for actual, reference in zip(public, public_expected, strict=True):
        assert_allclose(actual, reference, rtol=rtol, atol=atol)
    assert constant_strain_se2.operators(xi, s, tangent_eps).tangent_derivative is None
