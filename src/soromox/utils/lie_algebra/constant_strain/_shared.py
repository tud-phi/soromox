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
    return jnp.asarray(jnp.finfo(dtype).eps ** (1.0 / 14.0), dtype=dtype)


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
    x = s**2 * angle_sq
    requested = jnp.abs(s) * jnp.abs(jnp.asarray(eps, dtype=x.dtype))
    threshold = jnp.maximum(requested, _constant_strain_series_threshold(x.dtype))
    use_series = x <= threshold**2
    x_safe = jnp.where(use_series, jnp.ones_like(x), x)
    return x, use_series, jnp.sqrt(x_safe)


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
    x, use_series, z_safe = _series_arguments(angle_sq, s, eps)
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
        z_safe = jnp.sqrt(x)
        use_series = jnp.zeros_like(x, dtype=jnp.bool_)
    sin_z = jnp.sin(z_safe)
    cos_z = jnp.cos(z_safe)
    common = -8.0 + (8.0 - z_safe**2) * cos_z + 5.0 * z_safe * sin_z
    alternate = -8.0 * z_safe + (15.0 - z_safe**2) * sin_z - 7.0 * z_safe * cos_z
    closed = (
        (4.0 - 4.0 * cos_z - z_safe * sin_z) / (2.0 * z_safe**2),
        (4.0 * z_safe - 5.0 * sin_z + z_safe * cos_z) / (2.0 * z_safe**3),
        (2.0 - 2.0 * cos_z - z_safe * sin_z) / (2.0 * z_safe**4),
        (2.0 * z_safe - 3.0 * sin_z + z_safe * cos_z) / (2.0 * z_safe**5),
    )
    derivative_closed = (
        common / (4.0 * z_safe**4),
        alternate / (4.0 * z_safe**5),
        common / (4.0 * z_safe**6),
        alternate / (4.0 * z_safe**7),
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
    """Build matrix powers through fourth order with shared products.

    Args:
        matrix: Square matrix ``B`` with shape ``(dim, dim)``.

    Returns:
        List ``[I, B, B**2, B**3, B**4]``. Every entry has the same shape and
        dtype as ``matrix``.

    Notes:
        Fourth order is exact for the reduced ``se(2)`` and ``se(3)`` adjoint
        polynomials; this is not a fourth-order approximation to a generic
        matrix exponential.
    """
    powers = [jnp.eye(matrix.shape[0], dtype=matrix.dtype)]
    for _ in range(4):
        powers.append(powers[-1] @ matrix)
    return powers


def _matrix_powers_with_derivatives(
    matrix: Array, matrix_dot: Array
) -> tuple[list[Array], list[Array]]:
    r"""Build fourth-order matrix powers and their directional derivatives.

    Starting with ``P_0 = I`` and ``Pdot_0 = 0``, the recurrence

    .. math::

        P_{k+1} &= P_k B, \\
        \dot P_{k+1} &= \dot P_k B + P_k \dot B

    applies the product rule without independently expanding every derivative.

    Args:
        matrix: Square matrix ``B`` with shape ``(dim, dim)``.
        matrix_dot: Directional derivative ``B_dot`` with the same shape.

    Returns:
        Pair ``(powers, dot_powers)``. The lists contain entries for orders zero
        through four, with ``dot_powers[k] = D(B**k)[B_dot]``.
    """
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
    powers = _matrix_powers(s * ad_xi)
    return _reduced_adjoint_from_powers(powers, angle_sq, s, eps)


def _constant_strain_tangent(
    ad_xi: Array, angle_sq: Array, s: Array, eps: float | Array
) -> Array:
    r"""Evaluate the dimension-independent constant-strain tangent operator.

    This computes

    .. math::

        T(s,\xi) = \int_0^s \exp(\sigma\,\operatorname{ad}_\xi)\,d\sigma

    using the exact reduced fourth-order polynomial.

    Args:
        ad_xi: ``se(2)`` or ``se(3)`` algebra-adjoint matrix.
        angle_sq: Squared magnitude of the rotational strain.
        s: Scalar integration limit and segment arclength.
        eps: Minimum rotational-strain series threshold.

    Returns:
        Tangent matrix with the same shape and dtype as ``ad_xi``. At ``s == 0``
        the result is exactly the zero matrix in default mode.
    """
    powers = _matrix_powers(s * ad_xi)
    coefficients, _ = _tangent_coefficients_and_x_derivatives(angle_sq, s, eps)
    return _reduced_tangent_from_powers(powers, coefficients, s)


def _constant_strain_tangent_derivative(
    ad_xi: Array,
    ad_xid: Array,
    angle_sq: Array,
    angle_sq_dot: Array,
    s: Array,
    eps: float | Array,
) -> Array:
    r"""Evaluate the tangent's explicit directional derivative in ``xi``.

    The result is ``D_xi T(s, xi)[xid]`` with ``s`` held fixed. It combines the
    coefficient contribution
    ``(dc_k/dx) * x_dot * B**k`` and the power contribution
    ``c_k * D(B**k)[B_dot]``, where ``B_dot = s * ad_xid`` and
    ``x_dot = s**2 * angle_sq_dot``.

    Args:
        ad_xi: Algebra-adjoint matrix of the current strain.
        ad_xid: Algebra-adjoint matrix of the strain direction/rate.
        angle_sq: Squared rotational-strain magnitude.
        angle_sq_dot: Directional derivative of ``angle_sq``. It is
            ``2 * xi[0] * xid[0]`` for ``se(2)`` and
            ``2 * dot(omega, omega_dot)`` for ``se(3)``.
        s: Scalar arclength held fixed during differentiation.
        eps: Minimum rotational-strain series threshold.

    Returns:
        Directional derivative matrix with the same shape and dtype as
        ``ad_xi``.

    Notes:
        This is an analytic production path. JAX autodiff is used only by the
        test suite as an independent oracle.
    """
    powers, dot_powers = _matrix_powers_with_derivatives(s * ad_xi, s * ad_xid)
    coefficients, derivatives = _tangent_coefficients_and_x_derivatives(
        angle_sq, s, eps
    )
    return _reduced_tangent_derivative_from_powers(
        powers, dot_powers, coefficients, derivatives, angle_sq_dot, s
    )


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
    result = powers[0]
    for coefficient, power in zip(coefficients, powers[1:], strict=True):
        result = result + coefficient * power
    return s * result


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
    x_dot = s**2 * angle_sq_dot
    result = jnp.zeros_like(powers[0])
    for coefficient, derivative, power, dot_power in zip(
        coefficients, derivatives, powers[1:], dot_powers[1:], strict=True
    ):
        result = result + derivative * x_dot * power + coefficient * dot_power
    return s * result
