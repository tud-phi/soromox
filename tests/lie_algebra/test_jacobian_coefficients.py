"""Tests for the public scalar Lie-Jacobian coefficient module."""

import jax
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose

from soromox.autodiff import strict_singularities_mode
from soromox.utils.lie_algebra import jacobian_coefficients

jax.config.update("jax_enable_x64", True)


def _forward_reference_from_x(kind, theta_sq):
    if kind == "sinc":
        return 1.0 + theta_sq * (
            -1.0 / 6.0
            + theta_sq
            * (
                1.0 / 120.0
                + theta_sq
                * (-1.0 / 5040.0 + theta_sq * (1.0 / 362880.0 - theta_sq / 39916800.0))
            )
        )
    if kind == "cosc":
        return 0.5 + theta_sq * (
            -1.0 / 24.0
            + theta_sq
            * (
                1.0 / 720.0
                + theta_sq
                * (
                    -1.0 / 40320.0
                    + theta_sq * (1.0 / 3628800.0 - theta_sq / 479001600.0)
                )
            )
        )
    return 1.0 / 6.0 + theta_sq * (
        -1.0 / 120.0
        + theta_sq
        * (
            1.0 / 5040.0
            + theta_sq
            * (
                -1.0 / 362880.0
                + theta_sq * (1.0 / 39916800.0 - theta_sq / 6227020800.0)
            )
        )
    )


def _forward_reference(kind, theta):
    return _forward_reference_from_x(kind, theta**2)


@pytest.mark.parametrize(
    ("kind", "function"),
    [
        ("sinc", jacobian_coefficients.sine_over_angle),
        ("cosc", jacobian_coefficients.one_minus_cosine_over_angle_squared),
        ("tanc", jacobian_coefficients.angle_minus_sine_over_angle_cubed),
    ],
)
@pytest.mark.parametrize("scale", [0.0, 1.0, 1.25])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(jnp.float64, 2e-9, 2e-11), (jnp.float32, 5e-4, 5e-5)],
)
def test_forward_coefficients_match_analytic_value_gradient_and_hessian(
    kind, function, scale, dtype, rtol, atol
):
    theta = scale * jacobian_coefficients.forward_left_jacobian_series_threshold(dtype)

    def reference(value):
        return _forward_reference(kind, value)

    assert_allclose(function(theta), reference(theta), rtol=rtol, atol=atol)
    assert_allclose(
        jax.jacrev(function)(theta), jax.jacrev(reference)(theta), rtol=rtol, atol=atol
    )
    assert_allclose(
        jax.jacrev(jax.jacrev(function))(theta),
        jax.jacrev(jax.jacrev(reference))(theta),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("scale", [-1.25, 0.0, 1.0, 1.25])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(jnp.float64, 2e-9, 2e-11), (jnp.float32, 5e-4, 5e-5)],
)
def test_fused_forward_coefficients_and_explicit_x_derivatives(
    scale, dtype, rtol, atol
):
    theta = scale * jacobian_coefficients.forward_left_jacobian_series_threshold(dtype)
    x = theta**2
    coefficients, derivatives = (
        jacobian_coefficients._forward_coefficients_and_x_derivatives(theta)
    )

    def reference(value):
        return jnp.stack(
            [
                _forward_reference_from_x("sinc", value),
                _forward_reference_from_x("cosc", value),
                _forward_reference_from_x("tanc", value),
            ]
        )

    assert_allclose(jnp.stack(coefficients), reference(x), rtol=rtol, atol=atol)
    assert_allclose(
        jnp.stack(derivatives), jax.jacrev(reference)(x), rtol=rtol, atol=atol
    )


def test_strict_mode_exposes_forward_coefficient_quotients():
    with strict_singularities_mode():
        results = (
            jacobian_coefficients.sine_over_angle(jnp.array(0.0)),
            jacobian_coefficients.one_minus_cosine_over_angle_squared(jnp.array(0.0)),
            jacobian_coefficients.angle_minus_sine_over_angle_cubed(jnp.array(0.0)),
        )

    assert all(not jnp.isfinite(result).all() for result in results)
