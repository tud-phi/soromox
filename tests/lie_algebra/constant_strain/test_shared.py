"""Tests for shared constant-strain identities and cross-algebra behavior."""

import jax
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose

from soromox.autodiff import strict_singularities_mode
from soromox.utils.lie_algebra import se2, se3, so3
from soromox.utils.lie_algebra.constant_strain import se2 as constant_strain_se2
from soromox.utils.lie_algebra.constant_strain import se3 as constant_strain_se3
from soromox.utils.lie_algebra.constant_strain._shared import (
    _adjoint_coefficients,
    _constant_strain_series_threshold,
    _tangent_coefficients_and_x_derivatives,
)
from soromox.utils.lie_algebra.jacobian_coefficients import (
    angle_minus_sine_over_angle_cubed,
    forward_left_jacobian_series_threshold,
    one_minus_cosine_over_angle_squared,
    sine_over_angle,
)

jax.config.update("jax_enable_x64", True)


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


def _integrated_exp_series_directional(matrix, matrix_dot, s, terms=20):
    result_dot = jnp.zeros_like(matrix)
    power = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    power_dot = jnp.zeros_like(matrix)
    factorial = 1.0
    for order in range(terms):
        factorial *= order + 1
        result_dot = result_dot + s ** (order + 1) * power_dot / factorial
        power_dot = power_dot @ matrix + power @ matrix_dot
        power = power @ matrix
    return result_dot


def _constant_coefficient_references(x):
    a1 = 1.0 + x**2 * (
        -1.0 / 120.0 + x * (1.0 / 2520.0 + x * (-1.0 / 120960.0 + x / 9979200.0))
    )
    a2 = 0.5 + x**2 * (
        -1.0 / 720.0 + x * (1.0 / 20160.0 + x * (-1.0 / 1209600.0 + x / 119750400.0))
    )
    a3 = 1.0 / 6.0 + x * (
        -1.0 / 60.0
        + x
        * (
            1.0 / 1680.0
            + x * (-1.0 / 90720.0 + x * (1.0 / 7983360.0 - x / 1037836800.0))
        )
    )
    a4 = 1.0 / 24.0 + x * (
        -1.0 / 360.0
        + x
        * (
            1.0 / 13440.0
            + x * (-1.0 / 907200.0 + x * (1.0 / 95800320.0 - x / 14529715200.0))
        )
    )
    c2 = 1.0 / 6.0 + x**2 * (
        -1.0 / 5040.0
        + x * (1.0 / 181440.0 + x * (-1.0 / 13305600.0 + x / 1556755200.0))
    )
    c4 = 1.0 / 120.0 + x * (
        -1.0 / 2520.0
        + x
        * (
            1.0 / 120960.0
            + x * (-1.0 / 9979200.0 + x * (1.0 / 1245404160.0 - x / 217945728000.0))
        )
    )
    return jnp.stack([a1, a2, a3, a4, c2, c4])


@pytest.mark.parametrize("scale", [0.0, 1.0, 1.25])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol", "derivative_atol"),
    [
        (jnp.float64, 3e-9, 3e-11, 5e-10),
        (jnp.float32, 8e-4, 8e-5, 8e-5),
    ],
)
def test_constant_strain_scalar_coefficients_and_derivatives(
    scale, dtype, rtol, atol, derivative_atol
):
    z = scale * _constant_strain_series_threshold(dtype)
    x = z**2
    adjoint_actual = jnp.stack(_adjoint_coefficients(x, jnp.ones((), dtype), 0.0))
    tangent_actual, derivative_actual = _tangent_coefficients_and_x_derivatives(
        x, jnp.ones((), dtype), 0.0
    )
    reference = _constant_coefficient_references(x)
    tangent_reference = reference[jnp.array([1, 4, 3, 5])]
    derivative_reference = jax.jacrev(
        lambda value: _constant_coefficient_references(value)[jnp.array([1, 4, 3, 5])]
    )(x)

    assert_allclose(adjoint_actual, reference[:4], rtol=rtol, atol=atol)
    assert_allclose(jnp.stack(tangent_actual), tangent_reference, rtol=rtol, atol=atol)
    assert_allclose(
        jnp.stack(derivative_actual),
        derivative_reference,
        rtol=rtol,
        atol=derivative_atol,
    )


@pytest.mark.parametrize("scale", [-1.25, 1.0, 1.25])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(jnp.float64, 3e-9, 3e-10), (jnp.float32, 8e-4, 8e-5)],
)
def test_exponential_maps_match_independent_matrix_series(scale, dtype, rtol, atol):
    theta = scale * forward_left_jacobian_series_threshold(dtype)
    axis = jnp.array([1.0, -2.0, 0.5], dtype=dtype)
    axis = axis / jnp.sqrt(jnp.dot(axis, axis))
    eps = jnp.zeros((), dtype=dtype)

    omega = theta * axis
    xi2 = jnp.array([theta, 0.7, -0.4], dtype=dtype)
    xi3 = jnp.concatenate([omega, jnp.array([0.7, -0.4, 0.2], dtype=dtype)])
    cases = (
        (
            lambda value: so3.exp(value, eps),
            lambda value: _matrix_exp_series(so3.skew(value)),
            omega,
        ),
        (
            lambda value: se2.exp(value, eps),
            lambda value: _matrix_exp_series(se2.hat(value)),
            xi2,
        ),
        (
            lambda value: se3.exp(value, eps),
            lambda value: _matrix_exp_series(se3.hat(value)),
            xi3,
        ),
    )

    for actual_fn, reference_fn, argument in cases:
        assert_allclose(
            actual_fn(argument), reference_fn(argument), rtol=rtol, atol=atol
        )
        assert_allclose(
            jax.jacrev(actual_fn)(argument),
            jax.jacrev(reference_fn)(argument),
            rtol=rtol,
            atol=atol,
        )
        weights = (
            jnp.arange(actual_fn(argument).size, dtype=dtype).reshape(
                actual_fn(argument).shape
            )
            + 1.0
        )

        def actual_scalar(value, function=actual_fn, output_weights=weights):
            return jnp.sum(output_weights * function(value))

        def reference_scalar(value, function=reference_fn, output_weights=weights):
            return jnp.sum(output_weights * function(value))

        assert_allclose(
            jax.jacrev(jax.jacrev(actual_scalar))(argument),
            jax.jacrev(jax.jacrev(reference_scalar))(argument),
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize("spatial", [False, True])
@pytest.mark.parametrize("scale", [-1.25, 0.0, 1.0, 1.25])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(jnp.float64, 2e-9, 2e-10), (jnp.float32, 1e-3, 1e-4)],
)
def test_constant_strain_operators_match_independent_series(
    spatial, scale, dtype, rtol, atol
):
    s = jnp.asarray(0.73, dtype=dtype)
    theta = scale * _constant_strain_series_threshold(dtype) / s
    eps = jnp.zeros((), dtype=dtype)
    if spatial:
        axis = jnp.array([1.0, -2.0, 0.5], dtype=dtype)
        axis = axis / jnp.sqrt(jnp.dot(axis, axis))
        xi = jnp.concatenate([theta * axis, jnp.array([0.7, -0.4, 0.2], dtype=dtype)])
        xid = jnp.array([0.1, -0.2, 0.05, -0.3, 0.4, 0.2], dtype=dtype)
        small_adjoint = se3.small_adjoint
        adjoint = constant_strain_se3.adjoint
        tangent = constant_strain_se3.tangent
        tangent_dot = constant_strain_se3.tangent_derivative
    else:
        xi = jnp.array([theta, 0.7, -0.4], dtype=dtype)
        xid = jnp.array([0.1, -0.2, 0.3], dtype=dtype)
        small_adjoint = se2.small_adjoint
        adjoint = constant_strain_se2.adjoint
        tangent = constant_strain_se2.tangent
        tangent_dot = constant_strain_se2.tangent_derivative

    def adjoint_fn(value):
        return adjoint(value, s, eps)

    def tangent_fn(value):
        return tangent(value, s, eps)

    def adjoint_reference(value):
        return _matrix_exp_series(s * small_adjoint(value))

    def tangent_reference(value):
        return _integrated_exp_series(small_adjoint(value), s)

    assert_allclose(adjoint_fn(xi), adjoint_reference(xi), rtol=rtol, atol=atol)
    assert_allclose(tangent_fn(xi), tangent_reference(xi), rtol=rtol, atol=atol)
    assert_allclose(
        jax.jacrev(tangent_fn)(xi),
        jax.jacrev(tangent_reference)(xi),
        rtol=rtol,
        atol=atol,
    )
    weights = (
        jnp.arange(tangent_fn(xi).size, dtype=dtype).reshape(tangent_fn(xi).shape) + 1.0
    )
    assert_allclose(
        jax.jacrev(jax.jacrev(lambda value: jnp.sum(weights * tangent_fn(value))))(xi),
        jax.jacrev(
            jax.jacrev(lambda value: jnp.sum(weights * tangent_reference(value)))
        )(xi),
        rtol=rtol,
        atol=atol,
    )
    derivative_reference = _integrated_exp_series_directional(
        small_adjoint(xi), small_adjoint(xid), s
    )
    assert_allclose(
        tangent_dot(xi, xid, s, eps), derivative_reference, rtol=rtol, atol=atol
    )


def test_batched_small_angle_gradients_are_finite():
    xi2 = jnp.array([[0.0, 0.7, -0.4], [1e-8, 0.7, -0.4]])
    xi3 = jnp.array(
        [[0.0, 0.0, 0.0, 0.7, -0.4, 0.2], [1e-8, -2e-8, 0.5e-8, 0.7, -0.4, 0.2]]
    )
    functions = (
        (
            lambda values: jnp.sum(jax.vmap(lambda value: se2.exp(value, 0.0))(values)),
            xi2,
        ),
        (
            lambda values: jnp.sum(jax.vmap(lambda value: se3.exp(value, 0.0))(values)),
            xi3,
        ),
        (
            lambda values: jnp.sum(
                jax.vmap(lambda value: constant_strain_se3.tangent(value, 0.73, 0.0))(
                    values
                )
            ),
            xi3,
        ),
    )
    for function, argument in functions:
        assert jnp.isfinite(jax.grad(function)(argument)).all()


def test_constant_strain_zero_arc_length_limits():
    xi2 = jnp.array([0.3, 0.7, -0.4])
    xid2 = jnp.array([0.1, -0.2, 0.3])
    xi3 = jnp.array([0.2, -0.1, 0.3, 0.7, -0.4, 0.2])
    xid3 = jnp.array([0.1, -0.2, 0.05, -0.3, 0.4, 0.2])

    assert_allclose(constant_strain_se2.adjoint(xi2, 0.0, 0.0), jnp.eye(3))
    assert_allclose(constant_strain_se2.tangent(xi2, 0.0, 0.0), jnp.zeros((3, 3)))
    assert_allclose(
        constant_strain_se2.tangent_derivative(xi2, xid2, 0.0, 0.0),
        jnp.zeros((3, 3)),
    )
    assert_allclose(constant_strain_se3.adjoint(xi3, 0.0, 0.0), jnp.eye(6))
    assert_allclose(constant_strain_se3.tangent(xi3, 0.0, 0.0), jnp.zeros((6, 6)))
    assert_allclose(
        constant_strain_se3.tangent_derivative(xi3, xid3, 0.0, 0.0),
        jnp.zeros((6, 6)),
    )


@pytest.mark.parametrize("scale", [0.0, 1.0, 1.25])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(jnp.float64, 2e-9, 2e-10), (jnp.float32, 1e-3, 1e-4)],
)
def test_fused_se2_operators_match_the_public_operators(scale, dtype, rtol, atol):
    s = jnp.asarray(0.73, dtype=dtype)
    threshold = _constant_strain_series_threshold(dtype)
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

    assert_allclose(pair[0], expected[0], rtol=rtol, atol=atol)
    assert_allclose(pair[1], expected[1], rtol=rtol, atol=atol)
    for actual, reference in zip(triple, expected, strict=True):
        assert_allclose(actual, reference, rtol=rtol, atol=atol)
    public_expected = (
        constant_strain_se2.adjoint(xi, s, tangent_eps),
        constant_strain_se2.adjoint_inverse(xi, s, tangent_eps),
        expected[1],
        expected[2],
    )
    for actual, reference in zip(public, public_expected, strict=True):
        assert_allclose(actual, reference, rtol=rtol, atol=atol)
    assert constant_strain_se2.operators(xi, s, tangent_eps).tangent_derivative is None


@pytest.mark.parametrize("scale", [0.0, 1.0, 1.25])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(jnp.float64, 2e-9, 2e-10), (jnp.float32, 1e-3, 1e-4)],
)
def test_fused_se3_operators_match_the_public_operators(scale, dtype, rtol, atol):
    s = jnp.asarray(0.73, dtype=dtype)
    threshold = _constant_strain_series_threshold(dtype)
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
    tangent_pair = constant_strain_se3._tangent_and_derivative(xi, xid, s, tangent_eps)
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
        constant_strain_se3.adjoint_inverse(xi, s, tangent_eps),
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
def test_prepared_se3_powers_match_pointwise_operator_evaluation(dtype, rtol, atol):
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
def test_prepared_se3_kinematic_operators_match_explicit_products(dtype, rtol, atol):
    xi = jnp.array([0.2, -0.1, 0.3, 0.7, -0.4, 0.2], dtype=dtype)
    xid = jnp.array([0.1, -0.2, 0.05, -0.3, 0.4, 0.2], dtype=dtype)
    s = jnp.asarray(0.73, dtype=dtype)
    eps = jnp.asarray(1e-8, dtype=dtype)
    prepared = constant_strain_se3._prepare_adjoint_powers(xi, xid)

    Ad_inv, Ad_inv_T, T, Td = constant_strain_se3._kinematic_operators_from_prepared(
        prepared, s, eps, eps, convective_only=False
    )
    convective = constant_strain_se3._kinematic_operators_from_prepared(
        prepared, s, eps, eps, convective_only=True
    )

    assert_allclose(Ad_inv_T, Ad_inv @ T, rtol=rtol, atol=atol)
    for actual, expected in zip(convective, (Ad_inv, Ad_inv_T, Td), strict=True):
        assert_allclose(actual, expected, rtol=rtol, atol=atol)


def test_strict_mode_exposes_forward_and_constant_strain_quotients():
    with strict_singularities_mode():
        results = (
            sine_over_angle(jnp.array(0.0)),
            one_minus_cosine_over_angle_squared(jnp.array(0.0)),
            angle_minus_sine_over_angle_cubed(jnp.array(0.0)),
            so3.exp(jnp.zeros(3), 0.0),
            se2.exp(jnp.array([0.0, 0.7, -0.4]), 0.0),
            se3.exp(jnp.array([0.0, 0.0, 0.0, 0.7, -0.4, 0.2]), 0.0),
            constant_strain_se3.tangent(
                jnp.array([0.0, 0.0, 0.0, 0.7, -0.4, 0.2]), 0.73, 0.0
            ),
        )
    assert all(not jnp.isfinite(result).all() for result in results)
