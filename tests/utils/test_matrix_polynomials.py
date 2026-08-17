"""Tests for private matrix-polynomial evaluation helpers."""

import jax.numpy as jnp
from numpy.testing import assert_allclose

import soromox.utils._matrix_polynomials as _matrix_polynomials


def test_powers_builds_requested_orders():
    matrix = jnp.array([[1.0, 2.0], [-0.5, 3.0]])

    actual = _matrix_polynomials.powers(matrix, 3)

    assert len(actual) == 4
    assert_allclose(actual[0], jnp.eye(2))
    assert_allclose(actual[1], matrix)
    assert_allclose(actual[2], matrix @ matrix)
    assert_allclose(actual[3], matrix @ matrix @ matrix)


def test_power_directions_match_centered_difference():
    matrix = jnp.array([[1.0, 2.0], [-0.5, 3.0]])
    direction = jnp.array([[0.2, -0.1], [0.4, 0.3]])
    step = 1e-3

    powers, directions = _matrix_polynomials.powers_and_directional_derivatives(
        matrix, direction, 4
    )
    forward = _matrix_polynomials.powers(matrix + step * direction, 4)
    backward = _matrix_polynomials.powers(matrix - step * direction, 4)

    for power, power_direction, forward_power, backward_power in zip(
        powers, directions, forward, backward, strict=True
    ):
        assert power.shape == matrix.shape
        assert_allclose(
            power_direction,
            (forward_power - backward_power) / (2.0 * step),
            rtol=2e-3,
            atol=2e-4,
        )


def test_evaluate_and_directional_derivative_match_centered_difference():
    matrix = jnp.array([[1.0, 2.0], [-0.5, 3.0]])
    direction = jnp.array([[0.2, -0.1], [0.4, 0.3]])
    coefficients = jnp.array([0.4, -0.2, 0.1])
    coefficient_directions = jnp.array([-0.3, 0.2, 0.05])
    step = 1e-3
    powers, power_directions = _matrix_polynomials.powers_and_directional_derivatives(
        matrix, direction, 3
    )

    actual = _matrix_polynomials.evaluate(powers, coefficients)
    actual_direction = _matrix_polynomials.evaluate_directional_derivative(
        powers,
        power_directions,
        coefficients,
        coefficient_directions,
    )

    forward = _matrix_polynomials.evaluate(
        _matrix_polynomials.powers(matrix + step * direction, 3),
        coefficients + step * coefficient_directions,
    )
    backward = _matrix_polynomials.evaluate(
        _matrix_polynomials.powers(matrix - step * direction, 3),
        coefficients - step * coefficient_directions,
    )
    expected = (
        jnp.eye(2)
        + coefficients[0] * matrix
        + coefficients[1] * matrix @ matrix
        + coefficients[2] * matrix @ matrix @ matrix
    )

    assert_allclose(actual, expected)
    assert_allclose(
        actual_direction,
        (forward - backward) / (2.0 * step),
        rtol=2e-3,
        atol=2e-4,
    )
