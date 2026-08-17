"""Tests for spatial constant-strain operators."""

import jax
import pytest
from jax import Array
from jax import numpy as jnp
from numpy.testing import assert_allclose

from soromox.autodiff import strict_singularities_mode
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
    threshold = se3._left_jacobian_series_threshold(dtype)
    theta = scale * threshold / s
    axis = jnp.array([1.0, -2.0, 0.5], dtype=dtype)
    axis = axis / jnp.sqrt(jnp.dot(axis, axis))
    xi = jnp.concatenate([theta * axis, jnp.array([0.7, -0.4, 0.2], dtype=dtype)])
    xid = jnp.array([0.1, -0.2, 0.05, -0.3, 0.4, 0.2], dtype=dtype)
    eps = jnp.zeros((), dtype=dtype)

    ad_xi = se3.small_adjoint(xi)
    actual_adjoint = constant_strain_se3.adjoint(xi, s, eps)
    actual_tangent = constant_strain_se3.tangent(xi, s, eps)
    expected_adjoint = _matrix_exp_series(s * ad_xi)
    expected_tangent = _integrated_exp_series(ad_xi, s)

    assert_allclose(actual_adjoint, expected_adjoint, rtol=rtol, atol=atol)
    assert_allclose(actual_tangent, expected_tangent, rtol=rtol, atol=atol)
    assert_allclose(
        jax.jacrev(lambda value: constant_strain_se3.tangent(value, s, eps))(xi),
        jax.jacrev(lambda value: _integrated_exp_series(se3.small_adjoint(value), s))(
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
                    weights * constant_strain_se3.tangent(value, s, eps)
                )
            )
        )(xi),
        jax.jacrev(
            jax.jacrev(
                lambda value: jnp.sum(
                    weights * _integrated_exp_series(se3.small_adjoint(value), s)
                )
            )
        )(xi),
        rtol=rtol,
        atol=atol,
    )
    expected_direction = _integrated_exp_series_directional(
        ad_xi, se3.small_adjoint(xid), s
    )
    assert_allclose(
        constant_strain_se3.tangent_derivative(xi, xid, s, eps),
        expected_direction,
        rtol=rtol,
        atol=atol,
    )


def test_zero_arc_length_limits():
    xi = jnp.array([0.2, -0.1, 0.3, 0.7, -0.4, 0.2])
    xid = jnp.array([0.1, -0.2, 0.05, -0.3, 0.4, 0.2])

    assert_allclose(constant_strain_se3.adjoint(xi, 0.0, 0.0), jnp.eye(6))
    assert_allclose(constant_strain_se3.tangent(xi, 0.0, 0.0), jnp.zeros((6, 6)))
    assert_allclose(
        constant_strain_se3.tangent_derivative(xi, xid, 0.0, 0.0),
        jnp.zeros((6, 6)),
    )


@pytest.mark.parametrize("scale", [0.0, 1.0, 1.25])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(jnp.float64, 2e-9, 2e-10), (jnp.float32, 1e-3, 1e-4)],
)
def test_fused_operators_match_public_operators(scale, dtype, rtol, atol):
    s = jnp.asarray(0.73, dtype=dtype)
    threshold = se3._left_jacobian_series_threshold(dtype)
    theta = scale * threshold / s
    axis = jnp.array([1.0, -2.0, 0.5], dtype=dtype)
    axis = axis / jnp.linalg.norm(axis)
    xi = jnp.concatenate([theta * axis, jnp.array([0.7, -0.4, 0.2], dtype=dtype)])
    xid = jnp.array([0.1, -0.2, 0.05, -0.3, 0.4, 0.2], dtype=dtype)
    adjoint_eps = jnp.zeros((), dtype=dtype)
    tangent_eps = 2.0 * threshold / jnp.abs(s)

    pair = constant_strain_se3._operators(xi, s, adjoint_eps, tangent_eps)
    triple = constant_strain_se3._operators(xi, s, adjoint_eps, tangent_eps, xid)
    quadruple = constant_strain_se3._operators(
        xi, s, adjoint_eps, tangent_eps, xid, return_adjoint=True
    )
    accumulated_eps = jnp.abs(s) * tangent_eps
    jacobian_pair = se3.left_jacobian_and_directional_derivative(
        s * xi, s * xid, accumulated_eps
    )
    tangent_pair = tuple(s * value for value in jacobian_pair)
    public = constant_strain_se3.operators(xi, s, tangent_eps, xid)
    expected = (
        constant_strain_se3.adjoint_inverse(xi, s, adjoint_eps),
        constant_strain_se3.tangent(xi, s, tangent_eps),
        constant_strain_se3.tangent_derivative(xi, xid, s, tangent_eps),
    )

    for actual, reference in zip(pair, expected[:2], strict=True):
        assert_allclose(actual, reference, rtol=rtol, atol=atol)
    for actual, reference in zip(triple, expected, strict=True):
        assert_allclose(actual, reference, rtol=rtol, atol=atol)
    for actual, reference in zip(tangent_pair, expected[1:], strict=True):
        assert_allclose(actual, reference, rtol=rtol, atol=atol)
    for actual, reference in zip(quadruple[:3], expected, strict=True):
        assert_allclose(actual, reference, rtol=rtol, atol=atol)
    assert_allclose(
        quadruple[3],
        constant_strain_se3.adjoint(xi, s, adjoint_eps),
        rtol=rtol,
        atol=atol,
    )
    public_expected = (
        constant_strain_se3.adjoint(xi, s, tangent_eps),
        expected[0],
        expected[1],
        expected[2],
    )
    for actual, reference in zip(public, public_expected, strict=True):
        assert_allclose(actual, reference, rtol=rtol, atol=atol)
    assert constant_strain_se3.operators(xi, s, tangent_eps).tangent_derivative is None


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(jnp.float64, 2e-9, 2e-10), (jnp.float32, 1e-3, 1e-4)],
)
def test_prepared_powers_match_pointwise_operator_evaluation(dtype, rtol, atol):
    xi = jnp.array([0.2, -0.1, 0.3, 0.7, -0.4, 0.2], dtype=dtype)
    xid = jnp.array([0.1, -0.2, 0.05, -0.3, 0.4, 0.2], dtype=dtype)
    arclengths = jnp.array([-0.3, 0.0, 0.2, 0.73], dtype=dtype)
    eps = jnp.asarray(1e-8, dtype=dtype)
    prepared = constant_strain_se3._prepare_adjoint_powers(xi, xid)

    actual = jax.vmap(
        lambda s: constant_strain_se3._operators_from_prepared(prepared, s, eps, eps)
    )(arclengths)
    expected = jax.vmap(lambda s: constant_strain_se3._operators(xi, s, eps, eps, xid))(
        arclengths
    )

    for actual_operator, expected_operator in zip(actual, expected, strict=True):
        assert_allclose(actual_operator, expected_operator, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(jnp.float64, 2e-9, 2e-10), (jnp.float32, 1e-3, 1e-4)],
)
def test_prepared_kinematic_operators_match_explicit_products(dtype, rtol, atol):
    xi = jnp.array([0.2, -0.1, 0.3, 0.7, -0.4, 0.2], dtype=dtype)
    xid = jnp.array([0.1, -0.2, 0.05, -0.3, 0.4, 0.2], dtype=dtype)
    s = jnp.asarray(0.73, dtype=dtype)
    eps = jnp.asarray(1e-8, dtype=dtype)
    prepared = constant_strain_se3._prepare_adjoint_powers(xi, xid)

    adjoint_inverse, transported_tangent, tangent, tangent_derivative = (
        constant_strain_se3._kinematic_operators_from_prepared(
            prepared, s, eps, eps, convective_only=False
        )
    )
    convective = constant_strain_se3._kinematic_operators_from_prepared(
        prepared, s, eps, eps, convective_only=True
    )

    assert_allclose(
        transported_tangent,
        adjoint_inverse @ tangent,
        rtol=rtol,
        atol=atol,
    )
    for actual, expected in zip(
        convective,
        (adjoint_inverse, transported_tangent, tangent_derivative),
        strict=True,
    ):
        assert_allclose(actual, expected, rtol=rtol, atol=atol)


def test_strict_mode_exposes_tangent_quotients_at_zero_rotation():
    with strict_singularities_mode():
        result = constant_strain_se3.tangent(
            jnp.array([0.0, 0.0, 0.0, 0.7, -0.4, 0.2]), 0.73, 0.0
        )

    assert not jnp.isfinite(result).all()
