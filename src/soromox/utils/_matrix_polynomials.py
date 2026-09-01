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


def powers_and_state_directional_derivatives(
    matrix: Array,
    velocity_direction: Array,
    configuration_direction: Array,
    rate_direction: Array,
    degree: int,
) -> tuple[list[Array], list[Array], list[Array], list[Array], list[Array]]:
    """
    Return matrix powers and the state derivatives used by PCS kinematics.

    Args:
        matrix: Matrix whose powers are evaluated.
        velocity_direction: First direction associated with the strain rate.
        configuration_direction: Second direction associated with the
            configuration tangent.
        rate_direction: Independent direction associated with the velocity
            tangent.
        degree: Highest matrix-power order to return.

    Returns:
        Tuple containing powers from order zero through ``degree``, their
        derivatives along the three supplied directions, and the mixed
        velocity/configuration derivative.
    """
    current = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    current_velocity = jnp.zeros_like(matrix)
    current_configuration = jnp.zeros_like(matrix)
    current_rate = jnp.zeros_like(matrix)
    current_mixed = jnp.zeros_like(matrix)
    powers = [current]
    velocity_derivatives = [current_velocity]
    configuration_derivatives = [current_configuration]
    rate_derivatives = [current_rate]
    mixed_derivatives = [current_mixed]
    for _ in range(degree):
        next_mixed = (
            current_mixed @ matrix
            + current_velocity @ configuration_direction
            + current_configuration @ velocity_direction
        )
        next_velocity = current_velocity @ matrix + current @ velocity_direction
        next_configuration = (
            current_configuration @ matrix + current @ configuration_direction
        )
        next_rate = current_rate @ matrix + current @ rate_direction
        current = current @ matrix
        powers.append(current)
        velocity_derivatives.append(next_velocity)
        configuration_derivatives.append(next_configuration)
        rate_derivatives.append(next_rate)
        mixed_derivatives.append(next_mixed)
        current_velocity = next_velocity
        current_configuration = next_configuration
        current_rate = next_rate
        current_mixed = next_mixed
    return (
        powers,
        velocity_derivatives,
        configuration_derivatives,
        rate_derivatives,
        mixed_derivatives,
    )


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


def evaluate_mixed_directional_derivative(
    matrix_powers: Sequence[Array],
    velocity_power_derivatives: Sequence[Array],
    configuration_power_derivatives: Sequence[Array],
    mixed_power_derivatives: Sequence[Array],
    coefficients: Sequence[Array],
    velocity_coefficient_derivatives: Sequence[Array],
    configuration_coefficient_derivatives: Sequence[Array],
    mixed_coefficient_derivatives: Sequence[Array],
) -> Array:
    """
    Evaluate a matrix polynomial's mixed state derivative.

    Args:
        matrix_powers: Powers from order zero through the polynomial degree.
        velocity_power_derivatives: Power derivatives along the velocity
            direction.
        configuration_power_derivatives: Power derivatives along the
            configuration direction.
        mixed_power_derivatives: Mixed velocity/configuration derivatives of
            the powers.
        coefficients: Polynomial coefficients for powers one and above.
        velocity_coefficient_derivatives: Coefficient derivatives along the
            velocity direction.
        configuration_coefficient_derivatives: Coefficient derivatives along
            the configuration direction.
        mixed_coefficient_derivatives: Mixed velocity/configuration
            derivatives of the coefficients.

    Returns:
        Matrix containing the mixed directional derivative of the polynomial.
    """
    result = jnp.zeros_like(matrix_powers[0])
    for (
        coefficient,
        coefficient_velocity,
        coefficient_configuration,
        coefficient_mixed,
        power,
        power_velocity,
        power_configuration,
        power_mixed,
    ) in zip(
        coefficients,
        velocity_coefficient_derivatives,
        configuration_coefficient_derivatives,
        mixed_coefficient_derivatives,
        matrix_powers[1:],
        velocity_power_derivatives[1:],
        configuration_power_derivatives[1:],
        mixed_power_derivatives[1:],
        strict=True,
    ):
        result = (
            result
            + coefficient_mixed * power
            + coefficient_velocity * power_configuration
            + coefficient_configuration * power_velocity
            + coefficient * power_mixed
        )
    return result
