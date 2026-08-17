r"""Closed-form constant-strain operators for :math:`SE(2)`.

The planar implementation uses exact rotation/translation blocks and stable
forward-Jacobian coefficients. Public names are unsuffixed because the module
itself supplies the algebra context.
"""

__all__ = [
    "adjoint",
    "adjoint_inverse",
    "operators",
    "tangent",
    "tangent_derivative",
]

import jax.numpy as jnp
from jax import Array

from .. import se2 as lie_se2
from ..jacobian_coefficients import (
    one_minus_cosine_over_angle_squared,
    sine_over_angle,
)
from ._types import ConstantStrainOperators


def _accumulated_eps(s: Array, eps: float | Array, dtype: jnp.dtype) -> Array:
    """Convert a rotational-strain threshold to accumulated-angle units."""
    return jnp.abs(s) * jnp.abs(jnp.asarray(eps, dtype=dtype))


def _forward_coefficients(
    theta: Array, s: Array, eps: float | Array
) -> tuple[Array, tuple[Array, Array, Array]]:
    r"""Return ``z`` and the stable forward coefficient triple.

    The coefficients are ``sinc(z)``, ``cosc(z)``, and ``tanc(z)`` with
    ``z = s * theta``. They are evaluated together so their shared branch
    predicate and trigonometric work can be reused by the direct tangent-block
    assembler.
    """
    z = s * theta
    coefficients, _ = lie_se2._left_jacobian_coefficients_and_x_derivatives(
        z, _accumulated_eps(s, eps, z.dtype)
    )
    return z, coefficients


def _adjoint_coefficients(
    theta: Array, s: Array, eps: float | Array
) -> tuple[Array, Array, Array]:
    r"""Return ``z``, ``sinc(z)``, and ``cosc(z)`` for an SE(2) adjoint.

    The adjoint does not depend on ``tanc``. Avoiding its evaluation keeps the
    single-operator public adjoint functions smaller than the full bundled
    tangent path while retaining the same cutoff policy.
    """
    z = s * theta
    cutoff = lie_se2._left_jacobian_cutoff(_accumulated_eps(s, eps, z.dtype), z.dtype)
    return (
        z,
        sine_over_angle(z, cutoff),
        one_minus_cosine_over_angle_squared(z, cutoff),
    )


def _forward_coefficients_and_derivatives(
    theta: Array, s: Array, eps: float | Array
) -> tuple[
    Array,
    tuple[Array, Array, Array],
    tuple[Array, Array, Array],
]:
    r"""Return forward coefficients and their analytic ``z**2`` derivatives.

    The derivative triple is ``d(sinc, cosc, tanc) / d(z**2)``. Representing
    these even functions in ``z**2`` avoids dividing by ``theta`` at zero and
    lets :func:`_tangent_derivative_from_coefficients` apply the chain rule
    using ``d(z**2)/dt = 2 z s xid[0]``.
    """
    z = s * theta
    coefficients, derivatives = lie_se2._left_jacobian_coefficients_and_x_derivatives(
        z, _accumulated_eps(s, eps, z.dtype)
    )
    return z, coefficients, derivatives


def _adjoint_components(
    xi: Array, s: Array, eps: float | Array
) -> tuple[Array, Array, Array]:
    r"""Evaluate the shared translation and rotation components of an adjoint.

    For ``xi = [theta, v]``, ``z = s * theta``, and
    ``a = -J v``, the lower blocks of the forward adjoint are

    .. math::

        q = s\left(\operatorname{sinc}(z)a
            + z\operatorname{cosc}(z)v\right),
        \qquad R = \exp(zJ).

    Returns:
        Tuple ``(q, sin_z, cos_z)``. These values are sufficient to assemble
        both the forward block ``[q, R]`` and inverse block ``[-R.T @ q, R.T]``
        without reevaluating a coefficient.

    Notes:
        This exact form replaces four generic ``3 x 3`` matrix products. The
        helper deliberately returns components rather than either matrix so a
        public operator bundle can construct both adjoints with shared work.
    """
    xi = jnp.asarray(xi).reshape(-1)
    s = jnp.asarray(s, dtype=xi.dtype)
    theta = xi[0]
    v = xi[1:]
    z, sinc, cosc = _adjoint_coefficients(theta, s, eps)
    z_cosc = z * cosc
    minus_j_v = jnp.stack([v[1], -v[0]])
    q = s * (sinc * minus_j_v + z_cosc * v)

    return q, jnp.sin(z), jnp.cos(z)


def _adjoint_from_components(
    q: Array, sin_z: Array, cos_z: Array, *, inverse: bool
) -> Array:
    r"""Assemble an SE(2) adjoint from precomputed scalar/block components.

    Args:
        q: Forward-adjoint translation column with shape ``(2,)``.
        sin_z: Sine of the signed accumulated rotation.
        cos_z: Cosine of the signed accumulated rotation.
        inverse: Select ``[-R.T @ q, R.T]`` instead of ``[q, R]``.

    Returns:
        Forward or inverse planar adjoint with shape ``(3, 3)``.
    """
    zero = jnp.zeros((), dtype=q.dtype)
    one = jnp.ones((), dtype=q.dtype)
    if inverse:
        q = -jnp.stack([cos_z * q[0] + sin_z * q[1], -sin_z * q[0] + cos_z * q[1]])
        return jnp.stack(
            [
                jnp.stack([one, zero, zero]),
                jnp.stack([q[0], cos_z, sin_z]),
                jnp.stack([q[1], -sin_z, cos_z]),
            ]
        )
    return jnp.stack(
        [
            jnp.stack([one, zero, zero]),
            jnp.stack([q[0], cos_z, -sin_z]),
            jnp.stack([q[1], sin_z, cos_z]),
        ]
    )


def _adjoint_block(xi: Array, s: Array, eps: float | Array, *, inverse: bool) -> Array:
    """Evaluate one planar adjoint while sharing its component assembly."""
    return _adjoint_from_components(*_adjoint_components(xi, s, eps), inverse=inverse)


def _tangent_from_coefficients(
    xi: Array,
    s: Array,
    z: Array,
    coefficients: tuple[Array, Array, Array],
) -> Array:
    r"""Assemble the exact planar tangent from scalar coefficient blocks.

    For ``xi = [theta, v]`` and ``z = s * theta``, the lower-right block is
    ``s * (sinc(z) I + z cosc(z) J)`` and the lower-left column is
    ``s**2 * (cosc(z) (-J v) + z tanc(z) v)``. This block formula is exactly
    equal to the reduced matrix polynomial, including pure translation.
    """
    accumulated = s * xi
    return s * lie_se2._left_jacobian_from_coefficients(accumulated, z, coefficients)


def _tangent_derivative_from_coefficients(
    xi: Array,
    xid: Array,
    s: Array,
    z: Array,
    coefficients: tuple[Array, Array, Array],
    derivatives: tuple[Array, Array, Array],
) -> Array:
    r"""Assemble ``D_xi T[xid]`` from analytic scalar derivatives.

    The chain rule is applied to the stable coefficient representation in
    ``x = z**2``. Both curvature-rate and translation-rate contributions are
    retained explicitly, so the result remains linear in ``xid`` and finite at
    zero curvature without invoking a JAX differentiation transform.
    """
    accumulated = s * xi
    accumulated_dot = s * xid
    return s * lie_se2._left_jacobian_derivative_from_coefficients(
        accumulated,
        accumulated_dot,
        z,
        coefficients,
        derivatives,
    )


def _operators(
    xi: Array,
    s: Array,
    adjoint_eps: float | Array,
    tangent_eps: float | Array,
    xid: Array | None = None,
) -> tuple[Array, Array] | tuple[Array, Array, Array]:
    r"""Evaluate an internal SE(2) operator bundle with shared scalar work.

    The returned operators are ``(Ad_inv, T)`` when ``xid`` is omitted and
    ``(Ad_inv, T, Td)`` otherwise. The latter path evaluates the tangent
    coefficients only once and reuses them for the tangent and its explicit
    directional derivative. Separate adjoint and tangent epsilon arguments
    preserve the threshold policies of internal system recurrences.

    This helper is private because it is an execution optimization for callers
    that need several public operators together; it is not a new Lie-algebra
    API.

    Args:
        xi: Planar strain in angular-first order.
        s: Scalar arclength.
        adjoint_eps: Threshold for the inverse-adjoint coefficients.
        tangent_eps: Threshold for tangent coefficients and their derivatives.
        xid: Optional strain direction/rate. Supplying it requests ``Td``.

    Returns:
        ``(Ad_inv, T)`` or ``(Ad_inv, T, Td)``. Users who want a stable public
        named result should call :func:`operators` instead.
    """
    xi = jnp.asarray(xi).reshape(-1)
    s = jnp.asarray(s, dtype=xi.dtype)
    adjoint_inverse = _adjoint_block(xi, s, adjoint_eps, inverse=True)
    if xid is None:
        z, coefficients = _forward_coefficients(xi[0], s, tangent_eps)
        return adjoint_inverse, _tangent_from_coefficients(xi, s, z, coefficients)

    xid = jnp.asarray(xid).reshape(-1)
    z, coefficients, derivatives = _forward_coefficients_and_derivatives(
        xi[0], s, tangent_eps
    )
    tangent = _tangent_from_coefficients(xi, s, z, coefficients)
    tangent_derivative = _tangent_derivative_from_coefficients(
        xi, xid, s, z, coefficients, derivatives
    )
    return adjoint_inverse, tangent, tangent_derivative


def adjoint(xi: Array, s: Array, eps: float | Array) -> Array:
    r"""Return the adjoint accumulated along a constant planar strain segment.

    This is a rod-kinematics helper, not a generic transform adjoint. For a
    segment with constant planar strain ``xi`` evaluated at arclength ``s``,
    the returned matrix represents the closed-form adjoint associated with the
    segment transform ``exp(s * xi)`` under the angular-first ``se(2)``
    convention. Equivalently, the returned matrix is
    ``exp(s * se2.small_adjoint(xi))``. It transports planar twists between the
    material frames at the segment base and at arclength ``s``.

    Args:
        xi: Constant planar strain with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, v_x, v_y]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Minimum rotational-strain threshold. The actual dimensionless
            cutoff also accounts for ``s`` and the input dtype.

    Returns:
        Array with shape ``(3, 3)`` mapping planar twists through the segment
        adjoint at arclength ``s``.

    Notes:
        The implementation uses the exact SE(2) rotation/translation block
        form, not a truncated matrix-exponential series. At zero curvature it
        retains the complete translation/shear dependence of
        ``exp(s * ad_xi)``. The Taylor branch affects only removable scalar
        coefficient quotients.
    """
    xi = jnp.asarray(xi).reshape(-1)
    return _adjoint_block(xi, s, eps, inverse=False)


def adjoint_inverse(xi: Array, s: Array, eps: float | Array) -> Array:
    """Return the inverse adjoint for a constant planar strain segment.

    This computes the inverse of :func:`adjoint` without explicitly
    inverting the full homogeneous segment transform. It is used when
    propagating planar body-frame Jacobians backward through a constant-strain
    segment. For ``Ad = adjoint(xi, s, eps)``, this function returns
    ``Ad**-1 = adjoint(-xi, s, eps)`` while exploiting the block structure
    of an ``SE(2)`` adjoint instead of applying a generic matrix inverse.

    Args:
        xi: Constant planar strain with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, v_x, v_y]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Minimum rotational-strain threshold forwarded to
            :func:`adjoint`.

    Returns:
        Array with shape ``(3, 3)`` representing the inverse segment adjoint.

    Notes:
        This path shares the stabilized forward-adjoint block assembler, so
        forward and inverse transport have one near-zero policy. The inverse
        translation is formed explicitly as ``-R.T @ q`` without a generic
        matrix inverse. It is intended for body-frame PCS/GVS recurrences.
    """
    return _adjoint_block(xi, s, eps, inverse=True)


def tangent(xi: Array, s: Array, eps: float | Array) -> Array:
    r"""Return the tangent operator for a constant planar strain segment.

    The tangent operator integrates the local strain perturbation over
    arclength for a segment whose strain is constant. In PCS/GVS recurrences it
    maps strain-basis contributions into body-frame Jacobian contributions at
    arclength ``s``. Mathematically,

    .. math::

        T(s,\xi) = \int_0^s
        \exp(\sigma\,\operatorname{ad}_\xi)\,d\sigma.

    Args:
        xi: Constant planar strain with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, v_x, v_y]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Minimum rotational-strain threshold. The actual dimensionless
            cutoff also accounts for ``s`` and the input dtype.

    Returns:
        Array with shape ``(3, 3)`` containing the planar constant-strain
        tangent operator at arclength ``s``.

    Notes:
        The exact limits are ``T(0, xi) = 0`` and ``T(s, 0) = s I``. Pure
        translation is handled by the same reduced polynomial and does not
        require a separate curvature-zero formula.
    """
    xi = jnp.asarray(xi).reshape(-1)
    s = jnp.asarray(s, dtype=xi.dtype)
    accumulated_eps = jnp.abs(s) * jnp.abs(jnp.asarray(eps, dtype=xi.dtype))
    return s * lie_se2.left_jacobian(s * xi, accumulated_eps)


def tangent_derivative(xi: Array, xid: Array, s: Array, eps: float | Array) -> Array:
    r"""Return the directional derivative of the planar tangent operator.

    This differentiates :func:`tangent` with respect to time through the
    strain ``xi`` and strain rate ``xid`` while holding arclength ``s`` fixed.
    The function uses the same angular-first planar convention as ``se2`` and
    computes ``D_xi T(s, xi)[xid]`` analytically.

    Args:
        xi: Constant planar strain with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, v_x, v_y]`` order.
        xid: Time derivative of ``xi`` with the same shape and coordinate
            order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Minimum rotational-strain threshold. The actual dimensionless
            cutoff also accounts for ``s`` and the input dtype.

    Returns:
        Array with shape ``(3, 3)`` containing ``d/dt tangent(xi, s)``.

    Notes:
        The SE(2) rotation/translation blocks and scalar coefficients are
        differentiated explicitly. No ``jvp``, ``grad``, or Jacobian transform
        is invoked in this production path. The result is linear in ``xid``
        and is exactly zero when either ``s == 0`` or ``xid == 0``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    xid = jnp.asarray(xid).reshape(-1)
    s = jnp.asarray(s, dtype=xi.dtype)
    accumulated_eps = jnp.abs(s) * jnp.abs(jnp.asarray(eps, dtype=xi.dtype))
    return s * lie_se2.left_jacobian_directional_derivative(
        s * xi, s * xid, accumulated_eps
    )


def operators(
    xi: Array,
    s: Array,
    eps: float | Array,
    xid: Array | None = None,
) -> ConstantStrainOperators:
    r"""Return the related SE(2) constant-strain operators as one bundle.

    Use this function when a caller needs more than one of
    :func:`adjoint`, :func:`adjoint_inverse`,
    :func:`tangent`, and :func:`tangent_derivative` at the same
    ``(xi, s)``. The bundled evaluation shares stabilized scalar coefficients
    and adjoint components instead of repeating them in separate public calls.

    Args:
        xi: Constant planar strain with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, v_x, v_y]`` order.
        s: Scalar arclength position measured from the segment base.
        eps: Minimum rotational-strain threshold used consistently by all
            returned operators.
        xid: Optional strain direction/rate with the same shape as ``xi``.
            When supplied, ``tangent_derivative`` contains
            ``D_xi T(s, xi)[xid]``; otherwise that field is ``None``.

    Returns:
        :class:`ConstantStrainOperators` containing the forward and inverse
        adjoints, tangent, and optional analytic tangent derivative.

    Notes:
        The result has a fixed named-field interface and is compatible with
        JAX transformations. No autodiff operation is executed internally.
        Call a single-operator function when only one expression is needed;
        this avoids assembling unused matrices.
    """
    xi = jnp.asarray(xi).reshape(-1)
    s = jnp.asarray(s, dtype=xi.dtype)
    adjoint_components = _adjoint_components(xi, s, eps)
    adjoint = _adjoint_from_components(*adjoint_components, inverse=False)
    adjoint_inverse = _adjoint_from_components(*adjoint_components, inverse=True)
    if xid is None:
        z, coefficients = _forward_coefficients(xi[0], s, eps)
        tangent = _tangent_from_coefficients(xi, s, z, coefficients)
        tangent_derivative = None
    else:
        xid = jnp.asarray(xid).reshape(-1)
        z, coefficients, derivatives = _forward_coefficients_and_derivatives(
            xi[0], s, eps
        )
        tangent = _tangent_from_coefficients(xi, s, z, coefficients)
        tangent_derivative = _tangent_derivative_from_coefficients(
            xi, xid, s, z, coefficients, derivatives
        )
    return ConstantStrainOperators(
        adjoint, adjoint_inverse, tangent, tangent_derivative
    )
