r"""Private reduced-polynomial machinery for the :math:`SE(3)` left Jacobian.

The functions in this module operate on an accumulated Lie-algebra coordinate;
they make no assumption about how that coordinate was produced. Constant-strain
rod operators reuse them after scaling strain by arclength.
"""

import jax.numpy as jnp
from jax import Array

from soromox.autodiff import strict_singularities_enabled


def _series_threshold(dtype: jnp.dtype) -> Array:
    """Return the dtype-aware dimensionless SE(3) Jacobian cutoff."""
    return jnp.asarray(jnp.finfo(dtype).eps ** (1.0 / 14.0), dtype=dtype)


def _series_arguments(
    angle_sq: Array, eps: float | Array
) -> tuple[Array, Array, Array]:
    """Prepare stable scalar arguments for SE(3) Jacobian coefficients."""
    x = angle_sq
    requested = jnp.abs(jnp.asarray(eps, dtype=x.dtype))
    threshold = jnp.maximum(requested, _series_threshold(x.dtype))
    use_series = x <= threshold**2
    x_safe = jnp.where(use_series, jnp.ones_like(x), x)
    return x, use_series, jnp.sqrt(x_safe)


def _coefficients_and_x_derivatives(
    angle_sq: Array, eps: float | Array
) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
    r"""Return exact reduced SE(3) left-Jacobian coefficients.

    For ``A = ad_xi`` and ``x = dot(omega, omega)``, the Jacobian is

    .. math::

        J_l(\xi) = I + c_1(x)A + c_2(x)A^2 + c_3(x)A^3 + c_4(x)A^4.

    The second tuple contains analytic derivatives ``dc_k / dx`` used by the
    directional-derivative implementation.
    """
    x, use_series, theta_safe = _series_arguments(angle_sq, eps)
    series = (
        0.5 + x**2 * (-1.0 / 720.0 + x * (1.0 / 20160.0 - x / 1209600.0)),
        1.0 / 6.0 + x**2 * (-1.0 / 5040.0 + x * (1.0 / 181440.0 - x / 13305600.0)),
        1.0 / 24.0
        + x
        * (-1.0 / 360.0 + x * (1.0 / 13440.0 + x * (-1.0 / 907200.0 + x / 95800320.0))),
        1.0 / 120.0
        + x
        * (
            -1.0 / 2520.0
            + x * (1.0 / 120960.0 + x * (-1.0 / 9979200.0 + x / 1245404160.0))
        ),
    )
    derivative_series = (
        x * (-1.0 / 360.0 + x * (1.0 / 6720.0 - x / 302400.0)),
        x * (-1.0 / 2520.0 + x * (1.0 / 60480.0 - x / 3326400.0)),
        -1.0 / 360.0 + x * (1.0 / 6720.0 + x * (-1.0 / 302400.0 + x / 23950080.0)),
        -1.0 / 2520.0 + x * (1.0 / 60480.0 + x * (-1.0 / 3326400.0 + x / 311351040.0)),
    )
    if strict_singularities_enabled():
        theta_safe = jnp.sqrt(x)
        use_series = jnp.zeros_like(x, dtype=jnp.bool_)
    sin_theta = jnp.sin(theta_safe)
    cos_theta = jnp.cos(theta_safe)
    common = -8.0 + (8.0 - theta_safe**2) * cos_theta + 5.0 * theta_safe * sin_theta
    alternate = (
        -8.0 * theta_safe
        + (15.0 - theta_safe**2) * sin_theta
        - 7.0 * theta_safe * cos_theta
    )
    closed = (
        (4.0 - 4.0 * cos_theta - theta_safe * sin_theta) / (2.0 * theta_safe**2),
        (4.0 * theta_safe - 5.0 * sin_theta + theta_safe * cos_theta)
        / (2.0 * theta_safe**3),
        (2.0 - 2.0 * cos_theta - theta_safe * sin_theta) / (2.0 * theta_safe**4),
        (2.0 * theta_safe - 3.0 * sin_theta + theta_safe * cos_theta)
        / (2.0 * theta_safe**5),
    )
    derivative_closed = (
        common / (4.0 * theta_safe**4),
        alternate / (4.0 * theta_safe**5),
        common / (4.0 * theta_safe**6),
        alternate / (4.0 * theta_safe**7),
    )
    coefficients = tuple(
        jnp.where(use_series, a, b) for a, b in zip(series, closed, strict=True)
    )
    derivatives = tuple(
        jnp.where(use_series, a, b)
        for a, b in zip(derivative_series, derivative_closed, strict=True)
    )
    return coefficients, derivatives


def _matrix_powers(matrix: Array) -> list[Array]:
    """Build matrix powers from order zero through four."""
    powers = [jnp.eye(matrix.shape[0], dtype=matrix.dtype)]
    for _ in range(4):
        powers.append(powers[-1] @ matrix)
    return powers


def _matrix_powers_with_derivatives(
    matrix: Array, matrix_dot: Array
) -> tuple[list[Array], list[Array]]:
    """Build fourth-order powers and their directional derivatives."""
    current = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    dot_current = jnp.zeros_like(matrix)
    powers = [current]
    dot_powers = [dot_current]
    for _ in range(4):
        dot_next = dot_current @ matrix + current @ matrix_dot
        current = current @ matrix
        powers.append(current)
        dot_powers.append(dot_next)
        dot_current = dot_next
    return powers, dot_powers


def _from_powers(powers: list[Array], coefficients: tuple[Array, ...]) -> Array:
    """Assemble an SE(3) left Jacobian from powers of ``ad_xi``."""
    result = powers[0]
    for coefficient, power in zip(coefficients, powers[1:], strict=True):
        result = result + coefficient * power
    return result


def _directional_derivative_from_powers(
    powers: list[Array],
    dot_powers: list[Array],
    coefficients: tuple[Array, ...],
    derivatives: tuple[Array, ...],
    angle_sq_dot: Array,
) -> Array:
    """Assemble ``D J_l(xi)[xid]`` from shared polynomial intermediates."""
    result = jnp.zeros_like(powers[0])
    for coefficient, derivative, power, dot_power in zip(
        coefficients, derivatives, powers[1:], dot_powers[1:], strict=True
    ):
        result = result + derivative * angle_sq_dot * power + coefficient * dot_power
    return result


def _left_jacobian(ad_xi: Array, angle_sq: Array, eps: float | Array) -> Array:
    """Evaluate the exact SE(3) left Jacobian from algebra-adjoint data."""
    powers = _matrix_powers(ad_xi)
    coefficients, _ = _coefficients_and_x_derivatives(angle_sq, eps)
    return _from_powers(powers, coefficients)


def _left_jacobian_and_directional_derivative(
    ad_xi: Array,
    ad_xid: Array,
    angle_sq: Array,
    angle_sq_dot: Array,
    eps: float | Array,
) -> tuple[Array, Array]:
    """Evaluate ``J_l(xi)`` and ``D J_l(xi)[xid]`` with shared work."""
    powers, dot_powers = _matrix_powers_with_derivatives(ad_xi, ad_xid)
    coefficients, derivatives = _coefficients_and_x_derivatives(angle_sq, eps)
    jacobian = _from_powers(powers, coefficients)
    jacobian_derivative = _directional_derivative_from_powers(
        powers, dot_powers, coefficients, derivatives, angle_sq_dot
    )
    return jacobian, jacobian_derivative


def _left_jacobian_directional_derivative(
    ad_xi: Array,
    ad_xid: Array,
    angle_sq: Array,
    angle_sq_dot: Array,
    eps: float | Array,
) -> Array:
    """Evaluate only ``D J_l(xi)[xid]`` without assembling ``J_l``."""
    powers, dot_powers = _matrix_powers_with_derivatives(ad_xi, ad_xid)
    coefficients, derivatives = _coefficients_and_x_derivatives(angle_sq, eps)
    return _directional_derivative_from_powers(
        powers, dot_powers, coefficients, derivatives, angle_sq_dot
    )
