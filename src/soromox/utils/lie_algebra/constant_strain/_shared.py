r"""Reduced-polynomial machinery for constant-strain operators.

This private module owns the dimension-independent coefficient selection and
matrix-power recurrences used by the algebra-specific :mod:`.se2` and
:mod:`.se3` implementations. It also defines the result type re-exported by
the public package API.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from soromox.autodiff import strict_singularities_enabled

from .. import _se3_left_jacobian


class ConstantStrainOperators(NamedTuple):
    """Related constant-strain Lie operators evaluated as one shared bundle.

    Attributes:
        adjoint: Forward adjoint ``exp(s * ad_xi)``.
        adjoint_inverse: Inverse of ``adjoint`` assembled from Lie-group
            blocks rather than a dense matrix inverse.
        tangent: Integrated adjoint
            ``integral_0^s exp(sigma * ad_xi) d sigma``.
        tangent_derivative: Analytic directional derivative
            ``D_xi tangent[xid]``. This is ``None`` when the bundle was
            requested without ``xid``.

    Notes:
        The named result gives the public bundled evaluators a fixed interface;
        unlike the private selective helpers, its field layout never depends
        on which values an internal system recurrence happens to consume.
    """

    adjoint: Array
    adjoint_inverse: Array
    tangent: Array
    tangent_derivative: Array | None


# Shared reduced-polynomial coefficients and matrix-power recurrences.


def _constant_strain_series_threshold(dtype: jnp.dtype) -> Array:
    """Return the dtype-aware dimensionless constant-strain cutoff.

    The worst closed-form Hessian roundoff grows as ``O(u / z**6)`` for
    ``z = s * theta``. The retained scalar series have Hessian truncation
    ``O(z**8)``. Balancing the errors yields ``z**14 ~= u``.

    Args:
        dtype: Real floating-point dtype for which to construct the cutoff.

    Returns:
        Scalar array of the requested dtype containing ``u**(1/14)``, where
        ``u`` is the dtype's machine epsilon.

    Notes:
        This dimensionless cutoff applies to the accumulated rotation ``z``,
        not directly to the rotational strain ``theta``. It therefore remains
        meaningful for negative ``s`` and becomes independent of the units
        used to parameterize arclength.
    """
    return _se3_left_jacobian._series_threshold(dtype)


def _series_arguments(
    angle_sq: Array, s: Array, eps: float | Array
) -> tuple[Array, Array, Array]:
    """Prepare stable scalar arguments for constant-strain coefficients.

    The coefficient functions are even in ``z = s * theta`` and are therefore
    represented using ``x = z**2 = s**2 * angle_sq``. The effective
    dimensionless cutoff is ``max(abs(s) * abs(eps), u**(1/14))``. In the
    series region ``x`` is replaced by one before the regular closed forms take
    a square root or divide by powers of ``z``. This prevents a singular dead
    branch from contaminating derivatives after ``vmap`` lowers selection.

    Args:
        angle_sq: Squared rotational strain. For ``se(2)`` this is
            ``xi[0]**2``; for ``se(3)`` it is ``dot(omega, omega)``.
        s: Scalar arclength, which may be positive, negative, or zero.
        eps: User-requested minimum threshold in rotational-strain units.

    Returns:
        Tuple ``(x, use_series, z_safe)`` containing the squared accumulated
        rotation, the scalar series predicate, and a sanitized nonnegative
        magnitude ``sqrt(x_safe)`` for closed-form evaluation.
    """
    accumulated_angle_sq = s**2 * angle_sq
    accumulated_eps = jnp.abs(s) * jnp.abs(
        jnp.asarray(eps, dtype=accumulated_angle_sq.dtype)
    )
    return _se3_left_jacobian._series_arguments(accumulated_angle_sq, accumulated_eps)


def _adjoint_coefficients(
    angle_sq: Array, s: Array, eps: float | Array
) -> tuple[Array, Array, Array, Array]:
    r"""Return the four coefficients of the reduced adjoint exponential.

    With ``B = s * ad_xi`` and ``z = s * theta``, the exact reduced polynomial
    is

    .. math::

        \exp(B) = I + a_1(z)B + a_2(z)B^2
                    + a_3(z)B^3 + a_4(z)B^4.

    The regular coefficients are

    .. math::

        a_1 &= \frac{3\sin z-z\cos z}{2z}, &
        a_2 &= \frac{4-4\cos z-z\sin z}{2z^2}, \\
        a_3 &= \frac{\sin z-z\cos z}{2z^3}, &
        a_4 &= \frac{2-2\cos z-z\sin z}{2z^4}.

    Each removable quotient is replaced near zero by its even Taylor series
    through ``z**8``. Although ``z_safe`` is nonnegative, the result is valid
    for signed ``s`` because all four coefficients are even and the sign is
    carried by the powers of ``B``.

    Args:
        angle_sq: Squared rotational-strain magnitude.
        s: Scalar arclength used in ``B`` and ``z``.
        eps: Minimum rotational-strain series threshold.

    Returns:
        Four scalar arrays ``(a1, a2, a3, a4)`` in ascending matrix-power
        order.

    Notes:
        Strict-singularity mode disables the series and sanitization so the raw
        quotients remain observable at ``z == 0``.
    """
    x, use_series, z_safe = _series_arguments(angle_sq, s, eps)
    series = (
        1.0 + x**2 * (-1.0 / 120.0 + x * (1.0 / 2520.0 - x / 120960.0)),
        0.5 + x**2 * (-1.0 / 720.0 + x * (1.0 / 20160.0 - x / 1209600.0)),
        1.0 / 6.0
        + x * (-1.0 / 60.0 + x * (1.0 / 1680.0 + x * (-1.0 / 90720.0 + x / 7983360.0))),
        1.0 / 24.0
        + x
        * (-1.0 / 360.0 + x * (1.0 / 13440.0 + x * (-1.0 / 907200.0 + x / 95800320.0))),
    )
    if strict_singularities_enabled():
        z_safe = jnp.sqrt(x)
        use_series = jnp.zeros_like(x, dtype=jnp.bool_)
    sin_z = jnp.sin(z_safe)
    cos_z = jnp.cos(z_safe)
    closed = (
        (3.0 * sin_z - z_safe * cos_z) / (2.0 * z_safe),
        (4.0 - 4.0 * cos_z - z_safe * sin_z) / (2.0 * z_safe**2),
        (sin_z - z_safe * cos_z) / (2.0 * z_safe**3),
        (2.0 - 2.0 * cos_z - z_safe * sin_z) / (2.0 * z_safe**4),
    )
    return tuple(
        jnp.where(use_series, a, b) for a, b in zip(series, closed, strict=True)
    )


def _tangent_coefficients_and_x_derivatives(
    angle_sq: Array, s: Array, eps: float | Array
) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
    r"""Return reduced tangent coefficients and their derivatives in ``x``.

    With ``B = s * ad_xi`` and ``x = (s * theta)**2``, the tangent operator is

    .. math::

        T(s,\xi) = s\left(I + c_1(x)B + c_2(x)B^2
                            + c_3(x)B^3 + c_4(x)B^4\right).

    This helper returns both ``c_k`` and the explicit analytic derivatives
    ``dc_k / dx``. The latter are consumed by the production implementation of
    ``D_xi T[xid]``; they are not generated with runtime autodiff. Both sets
    use Taylor polynomials through ``z**8`` in the small-angle branch and exact
    trigonometric quotients otherwise.

    Args:
        angle_sq: Squared rotational-strain magnitude.
        s: Scalar arclength held fixed by the tangent derivative.
        eps: Minimum rotational-strain series threshold.

    Returns:
        Pair ``(coefficients, derivatives)``. Each member is a four-tuple in
        ascending matrix-power order; ``derivatives[k]`` is
        ``d coefficients[k] / d x``.

    Notes:
        Differentiating with respect to ``x`` avoids the undefined expression
        ``dot(omega, omega_dot) / norm(omega)`` at zero spatial rotation. The
        caller supplies ``x_dot = s**2 * d(angle_sq)/dt`` instead.
    """
    accumulated_angle_sq = s**2 * angle_sq
    accumulated_eps = jnp.abs(s) * jnp.abs(
        jnp.asarray(eps, dtype=accumulated_angle_sq.dtype)
    )
    return _se3_left_jacobian._coefficients_and_x_derivatives(
        accumulated_angle_sq, accumulated_eps
    )


def _constant_strain_adjoint(
    ad_xi: Array, angle_sq: Array, s: Array, eps: float | Array
) -> Array:
    """Evaluate the dimension-independent constant-strain adjoint polynomial.

    Args:
        ad_xi: ``se(2)`` or ``se(3)`` algebra-adjoint matrix with shape
            ``(dim, dim)``.
        angle_sq: Squared magnitude of the rotational part of ``xi``.
        s: Scalar arclength measured from the segment base.
        eps: Minimum rotational-strain series threshold.

    Returns:
        Matrix ``exp(s * ad_xi)`` with shape ``(dim, dim)``.

    Notes:
        The fourth-order reduced polynomial is algebraically exact for the
        supported Lie algebras. Only its scalar coefficients switch between
        Taylor and trigonometric representations.
    """
    powers = _se3_left_jacobian._matrix_powers(s * ad_xi)
    return _reduced_adjoint_from_powers(powers, angle_sq, s, eps)


def _reduced_adjoint_from_powers(
    powers: list[Array], angle_sq: Array, s: Array, eps: float | Array
) -> Array:
    r"""Assemble the exact reduced adjoint from precomputed powers of ``B``.

    ``powers`` must contain ``[I, B, ..., B**4]`` for
    ``B = s * ad_xi``. Supplying the powers lets a bundled evaluator reuse
    them for the tangent and its derivative. Only the scalar coefficients are
    selected here; this helper performs no additional matrix multiplication.
    """
    result = powers[0]
    for coefficient, power in zip(
        _adjoint_coefficients(angle_sq, s, eps), powers[1:], strict=True
    ):
        result = result + coefficient * power
    return result


def _reduced_tangent_from_powers(
    powers: list[Array],
    coefficients: tuple[Array, ...],
    s: Array,
) -> Array:
    r"""Assemble the exact reduced tangent from shared powers and coefficients.

    The result is ``s * (I + sum_k coefficients[k-1] * B**k)``. The caller is
    responsible for providing coefficients computed from the same ``xi`` and
    ``s`` as the powers, which allows ``T`` and ``Td`` to share both objects.
    """
    return s * _se3_left_jacobian._from_powers(powers, coefficients)


def _reduced_transported_tangent_from_powers(
    powers: list[Array],
    coefficients: tuple[Array, ...],
    s: Array,
) -> Array:
    r"""Assemble ``exp(-B) @ T`` directly from shared powers.

    Since the tangent coefficients are even in ``z = s * theta``, changing
    ``B`` to ``-B`` only alternates the matrix-power signs. The identity

    .. math::

        \exp(-B)T(s,\xi) = -T(-s,\xi)
        = s\left(I-c_1B+c_2B^2-c_3B^3+c_4B^4\right)

    avoids a dense matrix product in PCS Jacobian recurrences while remaining
    exactly equivalent to the separately assembled operators.
    """
    result = powers[0]
    sign = -1.0
    for coefficient, power in zip(coefficients, powers[1:], strict=True):
        result = result + sign * coefficient * power
        sign = -sign
    return s * result


def _reduced_tangent_derivative_from_powers(
    powers: list[Array],
    dot_powers: list[Array],
    coefficients: tuple[Array, ...],
    derivatives: tuple[Array, ...],
    angle_sq_dot: Array,
    s: Array,
) -> Array:
    r"""Assemble the analytic tangent derivative from shared intermediates.

    Each term applies the product rule to its scalar coefficient and matrix
    power. ``derivatives`` are with respect to ``x = (s theta)**2`` and
    ``dot_powers[k]`` is ``D(B**k)[B_dot]``. Consequently no rotational norm
    or runtime autodiff transform is required at zero rotation.
    """
    accumulated_angle_sq_dot = s**2 * angle_sq_dot
    return s * _se3_left_jacobian._directional_derivative_from_powers(
        powers,
        dot_powers,
        coefficients,
        derivatives,
        accumulated_angle_sq_dot,
    )
