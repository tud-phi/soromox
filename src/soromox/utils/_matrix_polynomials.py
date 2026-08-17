"""Private helpers for evaluating matrix polynomials and their derivatives."""

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array


def powers(matrix: Array, degree: int) -> list[Array]:
    """Return matrix powers from order zero through ``degree``."""
    result = [jnp.eye(matrix.shape[0], dtype=matrix.dtype)]
    for _ in range(degree):
        result.append(result[-1] @ matrix)
    return result


def powers_and_directional_derivatives(
    matrix: Array, matrix_direction: Array, degree: int
) -> tuple[list[Array], list[Array]]:
    """Return powers and their derivatives along ``matrix_direction``."""
    current = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    current_direction = jnp.zeros_like(matrix)
    result = [current]
    result_directions = [current_direction]
    for _ in range(degree):
        next_direction = current_direction @ matrix + current @ matrix_direction
        current = current @ matrix
        result.append(current)
        result_directions.append(next_direction)
        current_direction = next_direction
    return result, result_directions


def evaluate(matrix_powers: Sequence[Array], coefficients: Sequence[Array]) -> Array:
    """Evaluate a polynomial with an implicit unit constant coefficient."""
    result = matrix_powers[0]
    for coefficient, matrix_power in zip(coefficients, matrix_powers[1:], strict=True):
        result = result + coefficient * matrix_power
    return result


def evaluate_directional_derivative(
    matrix_powers: Sequence[Array],
    matrix_power_directions: Sequence[Array],
    coefficients: Sequence[Array],
    coefficient_directions: Sequence[Array],
) -> Array:
    """Evaluate a matrix polynomial's directional derivative."""
    result = jnp.zeros_like(matrix_powers[0])
    for coefficient, coefficient_direction, matrix_power, matrix_power_direction in zip(
        coefficients,
        coefficient_directions,
        matrix_powers[1:],
        matrix_power_directions[1:],
        strict=True,
    ):
        result = (
            result
            + coefficient_direction * matrix_power
            + coefficient * matrix_power_direction
        )
    return result
