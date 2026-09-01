r"""Closed-form constant-strain operators for :math:`SE(3)`.

The spatial implementation evaluates exact reduced fourth-order polynomials
and supports prepared powers shared across arclength evaluations. Public names
are unsuffixed because the module itself supplies the algebra context.
"""

__all__ = [
    "adjoint",
    "adjoint_inverse",
    "operators",
    "tangent",
    "tangent_derivative",
]

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from ... import _matrix_polynomials
from .. import se3 as lie_se3
from ._types import ConstantStrainOperators

_REDUCED_POLYNOMIAL_DEGREE = 4


def _accumulated_eps(s: Array, eps: float | Array, dtype: jnp.dtype) -> Array:
    """Convert a rotational-strain threshold to accumulated-angle units."""
    return jnp.abs(s) * jnp.abs(jnp.asarray(eps, dtype=dtype))


def _adjoint_coefficients(
    angle_sq: Array, s: Array, eps: float | Array
) -> tuple[Array, ...]:
    """Return adjoint coefficients for the accumulated coordinate ``s * xi``."""
    accumulated_angle_sq = s**2 * angle_sq
    return lie_se3._adjoint_exponential_coefficients(
        accumulated_angle_sq,
        _accumulated_eps(s, eps, accumulated_angle_sq.dtype),
    )


def _tangent_coefficients_and_x_derivatives(
    angle_sq: Array, s: Array, eps: float | Array
) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
    """Return Jacobian coefficients for the accumulated coordinate ``s * xi``."""
    accumulated_angle_sq = s**2 * angle_sq
    return lie_se3._left_jacobian_coefficients_and_x_derivatives(
        accumulated_angle_sq,
        _accumulated_eps(s, eps, accumulated_angle_sq.dtype),
    )


class _PreparedAdjointPowers(NamedTuple):
    """Unscaled powers of an SE(3) algebra-adjoint matrix shared across arclength evaluations.

    ``powers[k]`` contains ``ad_xi**k`` and ``dot_powers[k]`` its analytic
    directional derivative under ``ad_xid``. The scalar rotational invariants
    are stored alongside the matrices so an evaluator only needs to apply
    ``s**k`` and evaluate the stable coefficients at a requested arclength.
    """

    angle_sq: Array
    angle_sq_dot: Array
    powers: Array
    dot_powers: Array


class _PreparedStateDirectionalPowers(NamedTuple):
    """Unscaled adjoint powers and contracted state derivatives.

    Every matrix field stacks powers from order zero through four. The
    rotational invariants are stored before arclength scaling so one segment
    preparation can serve its tip and all quadrature points.
    """

    angle_sq: Array
    angle_sq_velocity: Array
    angle_sq_configuration: Array
    angle_sq_rate: Array
    angle_sq_mixed: Array
    powers: Array
    velocity_power_derivatives: Array
    configuration_power_derivatives: Array
    rate_power_derivatives: Array
    mixed_power_derivatives: Array


def _prepare_state_directional_powers(
    xi: Array,
    xid: Array,
    configuration_direction: Array,
    rate_direction: Array,
) -> _PreparedStateDirectionalPowers:
    """Precompute arclength-independent powers for one PCS state direction.

    Args:
        xi: Constant spatial strain in angular-first order.
        xid: Spatial strain rate with the same shape as ``xi``.
        configuration_direction: Contracted configuration direction in strain
            coordinates.
        rate_direction: Contracted velocity direction in strain-rate
            coordinates.

    Returns:
        Prepared powers, first directional derivatives, the mixed
        strain-rate/configuration derivative, and their rotational
        invariants.
    """
    xi = jnp.asarray(xi).reshape(-1)
    xid = jnp.asarray(xid).reshape(-1)
    configuration_direction = jnp.asarray(configuration_direction).reshape(-1)
    rate_direction = jnp.asarray(rate_direction).reshape(-1)
    (
        powers,
        velocity_power_derivatives,
        configuration_power_derivatives,
        rate_power_derivatives,
        mixed_power_derivatives,
    ) = _matrix_polynomials.powers_and_state_directional_derivatives(
        lie_se3.small_adjoint(xi),
        lie_se3.small_adjoint(xid),
        lie_se3.small_adjoint(configuration_direction),
        lie_se3.small_adjoint(rate_direction),
        _REDUCED_POLYNOMIAL_DEGREE,
    )
    return _PreparedStateDirectionalPowers(
        angle_sq=jnp.dot(xi[:3], xi[:3]),
        angle_sq_velocity=2.0 * jnp.dot(xi[:3], xid[:3]),
        angle_sq_configuration=2.0 * jnp.dot(xi[:3], configuration_direction[:3]),
        angle_sq_rate=2.0 * jnp.dot(xi[:3], rate_direction[:3]),
        angle_sq_mixed=2.0 * jnp.dot(xid[:3], configuration_direction[:3]),
        powers=jnp.stack(powers),
        velocity_power_derivatives=jnp.stack(velocity_power_derivatives),
        configuration_power_derivatives=jnp.stack(configuration_power_derivatives),
        rate_power_derivatives=jnp.stack(rate_power_derivatives),
        mixed_power_derivatives=jnp.stack(mixed_power_derivatives),
    )


def _kinematic_operators_from_state_directional_powers(
    prepared: _PreparedStateDirectionalPowers,
    s: Array,
    adjoint_eps: float | Array,
    tangent_eps: float | Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Evaluate transported operators and one contracted state direction.

    The tangent operators are returned after transport by the inverse
    adjoint. This directly supplies the body-frame PCS recurrence and avoids
    multiplying each tangent derivative by the inverse adjoint afterward.

    Args:
        prepared: Arclength-independent segment powers and state derivatives.
        s: Local arclength at which to evaluate the operators.
        adjoint_eps: Rotational-strain threshold for the adjoint exponential.
        tangent_eps: Rotational-strain threshold for tangent coefficients.

    Returns:
        Tuple ``(Ad_inv, P, Pv, Ph, Pvh, Pk)``. Here ``P = Ad_inv @ T``;
        ``Pv`` and ``Ph`` are its strain-rate and configuration directional
        derivatives; ``Pvh`` is their mixed derivative; and ``Pk`` is the
        derivative along the contracted velocity direction.
    """
    s = jnp.asarray(s, dtype=prepared.powers.dtype)
    scale = jnp.ones((), dtype=prepared.powers.dtype)
    powers: list[Array] = []
    velocity_power_derivatives: list[Array] = []
    configuration_power_derivatives: list[Array] = []
    rate_power_derivatives: list[Array] = []
    mixed_power_derivatives: list[Array] = []
    for order in range(_REDUCED_POLYNOMIAL_DEGREE + 1):
        powers.append(scale * prepared.powers[order])
        velocity_power_derivatives.append(
            scale * prepared.velocity_power_derivatives[order]
        )
        configuration_power_derivatives.append(
            scale * prepared.configuration_power_derivatives[order]
        )
        rate_power_derivatives.append(scale * prepared.rate_power_derivatives[order])
        mixed_power_derivatives.append(scale * prepared.mixed_power_derivatives[order])
        scale = scale * s

    s_sq = s**2
    angle_sq = s_sq * prepared.angle_sq
    coefficients, derivatives, second_derivatives = (
        lie_se3._left_jacobian_coefficients_with_x_derivatives(
            angle_sq,
            _accumulated_eps(s, tangent_eps, angle_sq.dtype),
        )
    )
    angle_sq_velocity = s_sq * prepared.angle_sq_velocity
    angle_sq_configuration = s_sq * prepared.angle_sq_configuration
    velocity_coefficient_derivatives = tuple(
        derivative * angle_sq_velocity for derivative in derivatives
    )
    configuration_coefficient_derivatives = tuple(
        derivative * angle_sq_configuration for derivative in derivatives
    )
    rate_coefficient_derivatives = tuple(
        derivative * s_sq * prepared.angle_sq_rate for derivative in derivatives
    )
    mixed_coefficient_derivatives = tuple(
        second_derivative * angle_sq_velocity * angle_sq_configuration
        + derivative * s_sq * prepared.angle_sq_mixed
        for derivative, second_derivative in zip(
            derivatives,
            second_derivatives,
            strict=True,
        )
    )

    signs = tuple(
        -1.0 if order % 2 else 1.0 for order in range(1, _REDUCED_POLYNOMIAL_DEGREE + 1)
    )
    transported_coefficients = tuple(
        sign * coefficient
        for sign, coefficient in zip(signs, coefficients, strict=True)
    )
    transported_velocity_coefficient_derivatives = tuple(
        sign * derivative
        for sign, derivative in zip(
            signs,
            velocity_coefficient_derivatives,
            strict=True,
        )
    )
    transported_configuration_coefficient_derivatives = tuple(
        sign * derivative
        for sign, derivative in zip(
            signs,
            configuration_coefficient_derivatives,
            strict=True,
        )
    )
    transported_rate_coefficient_derivatives = tuple(
        sign * derivative
        for sign, derivative in zip(
            signs,
            rate_coefficient_derivatives,
            strict=True,
        )
    )
    transported_mixed_coefficient_derivatives = tuple(
        sign * derivative
        for sign, derivative in zip(
            signs,
            mixed_coefficient_derivatives,
            strict=True,
        )
    )

    transported_tangent = s * _matrix_polynomials.evaluate(
        powers,
        transported_coefficients,
    )
    transported_tangent_velocity_direction = s * (
        _matrix_polynomials.evaluate_directional_derivative(
            powers,
            velocity_power_derivatives,
            transported_coefficients,
            transported_velocity_coefficient_derivatives,
        )
    )
    transported_tangent_configuration_direction = s * (
        _matrix_polynomials.evaluate_directional_derivative(
            powers,
            configuration_power_derivatives,
            transported_coefficients,
            transported_configuration_coefficient_derivatives,
        )
    )
    transported_tangent_rate_direction = s * (
        _matrix_polynomials.evaluate_directional_derivative(
            powers,
            rate_power_derivatives,
            transported_coefficients,
            transported_rate_coefficient_derivatives,
        )
    )
    transported_tangent_mixed_direction = s * (
        _matrix_polynomials.evaluate_mixed_directional_derivative(
            powers,
            velocity_power_derivatives,
            configuration_power_derivatives,
            mixed_power_derivatives,
            transported_coefficients,
            transported_velocity_coefficient_derivatives,
            transported_configuration_coefficient_derivatives,
            transported_mixed_coefficient_derivatives,
        )
    )
    adjoint = lie_se3._adjoint_from_powers(
        powers,
        _adjoint_coefficients(prepared.angle_sq, s, adjoint_eps),
    )
    return (
        lie_se3._adjoint_inverse_from_adjoint(adjoint),
        transported_tangent,
        transported_tangent_velocity_direction,
        transported_tangent_configuration_direction,
        transported_tangent_mixed_direction,
        transported_tangent_rate_direction,
    )


def _prepare_adjoint_powers(xi: Array, xid: Array) -> _PreparedAdjointPowers:
    r"""Precompute SE(3) powers that do not depend on arclength.

    PCS evaluates a constant segment strain at its tip and at several
    quadrature points. For ``A = ad_xi`` and fixed ``xid``, the identities

    .. math::

        (sA)^k = s^k A^k, \qquad
        D[(sA)^k][s\dot A] = s^k D[A^k][\dot A]

    let all matrix products be evaluated once per segment. This helper stores
    those unscaled powers; :func:`_operators_from_prepared` applies the
    inexpensive scalar powers for each requested ``s``.

    Args:
        xi: Spatial strain in angular-first order.
        xid: Spatial strain direction/rate with the same shape.

    Returns:
        Prepared rotational invariants and stacked powers with shapes
        ``(5, 6, 6)``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    xid = jnp.asarray(xid).reshape(-1)
    powers, dot_powers = _matrix_polynomials.powers_and_directional_derivatives(
        lie_se3.small_adjoint(xi),
        lie_se3.small_adjoint(xid),
        _REDUCED_POLYNOMIAL_DEGREE,
    )
    return _PreparedAdjointPowers(
        angle_sq=jnp.dot(xi[:3], xi[:3]),
        angle_sq_dot=2.0 * jnp.dot(xi[:3], xid[:3]),
        powers=jnp.stack(powers),
        dot_powers=jnp.stack(dot_powers),
    )


def _scaled_powers_from_prepared(
    prepared: _PreparedAdjointPowers, s: Array
) -> tuple[list[Array], list[Array]]:
    """Scale prepared ``A**k`` and directional powers by ``s**k``."""
    scale = jnp.ones((), dtype=prepared.powers.dtype)
    powers: list[Array] = []
    dot_powers: list[Array] = []
    for order in range(_REDUCED_POLYNOMIAL_DEGREE + 1):
        powers.append(scale * prepared.powers[order])
        dot_powers.append(scale * prepared.dot_powers[order])
        scale = scale * s
    return powers, dot_powers


def _operators_from_powers(
    powers: list[Array],
    dot_powers: list[Array] | None,
    angle_sq: Array,
    angle_sq_dot: Array | None,
    s: Array,
    adjoint_eps: float | Array,
    tangent_eps: float | Array,
    *,
    return_adjoint: bool = False,
) -> (
    tuple[Array, Array] | tuple[Array, Array, Array] | tuple[Array, Array, Array, Array]
):
    """Assemble selected SE(3) operators from already scaled powers."""
    adjoint = lie_se3._adjoint_from_powers(
        powers, _adjoint_coefficients(angle_sq, s, adjoint_eps)
    )
    adjoint_inverse = lie_se3._adjoint_inverse_from_adjoint(adjoint)
    coefficients, derivatives = _tangent_coefficients_and_x_derivatives(
        angle_sq, s, tangent_eps
    )
    tangent = s * lie_se3._left_jacobian_from_powers(powers, coefficients)

    outputs = (adjoint_inverse, tangent)
    if dot_powers is not None:
        assert angle_sq_dot is not None
        tangent_derivative = (
            s
            * lie_se3._left_jacobian_directional_derivative_from_powers(
                powers,
                dot_powers,
                coefficients,
                derivatives,
                s**2 * angle_sq_dot,
            )
        )
        outputs += (tangent_derivative,)
    if return_adjoint:
        outputs += (adjoint,)
    return outputs


def _operators_from_prepared(
    prepared: _PreparedAdjointPowers,
    s: Array,
    adjoint_eps: float | Array,
    tangent_eps: float | Array,
) -> tuple[Array, Array, Array]:
    """Evaluate ``(Ad_inv, T, Td)`` from per-segment prepared powers."""
    s = jnp.asarray(s, dtype=prepared.powers.dtype)
    powers, dot_powers = _scaled_powers_from_prepared(prepared, s)
    operators = _operators_from_powers(
        powers,
        dot_powers,
        prepared.angle_sq,
        prepared.angle_sq_dot,
        s,
        adjoint_eps,
        tangent_eps,
    )
    assert len(operators) == 3
    return operators


def _kinematic_operators_from_prepared(
    prepared: _PreparedAdjointPowers,
    s: Array,
    adjoint_eps: float | Array,
    tangent_eps: float | Array,
    *,
    convective_only: bool,
) -> tuple[Array, Array, Array] | tuple[Array, Array, Array, Array]:
    r"""Evaluate the operator subset consumed by PCS recurrences.

    Both paths return ``Ad_inv`` and the directly assembled transported
    tangent ``Ad_inv @ T``. The convective path returns ``Td`` as its third
    value and deliberately omits ``T``; forward dynamics only needs
    ``Ad_inv @ (Td @ xid)``. The full-derivative path returns
    ``(Ad_inv, Ad_inv_T, T, Td)``.

    This selective helper keeps the public operator bundle fixed while
    avoiding matrices that an internal recurrence would immediately discard.
    """
    s = jnp.asarray(s, dtype=prepared.powers.dtype)
    powers, dot_powers = _scaled_powers_from_prepared(prepared, s)
    adjoint = lie_se3._adjoint_from_powers(
        powers, _adjoint_coefficients(prepared.angle_sq, s, adjoint_eps)
    )
    adjoint_inverse = lie_se3._adjoint_inverse_from_adjoint(adjoint)
    coefficients, derivatives = _tangent_coefficients_and_x_derivatives(
        prepared.angle_sq, s, tangent_eps
    )
    transported_tangent = s * lie_se3._transported_left_jacobian_from_powers(
        powers, coefficients
    )
    tangent_derivative = s * lie_se3._left_jacobian_directional_derivative_from_powers(
        powers,
        dot_powers,
        coefficients,
        derivatives,
        s**2 * prepared.angle_sq_dot,
    )
    if convective_only:
        return adjoint_inverse, transported_tangent, tangent_derivative
    tangent = s * lie_se3._left_jacobian_from_powers(powers, coefficients)
    return adjoint_inverse, transported_tangent, tangent, tangent_derivative


def _operators(
    xi: Array,
    s: Array,
    adjoint_eps: float | Array,
    tangent_eps: float | Array,
    xid: Array | None = None,
    *,
    return_adjoint: bool = False,
) -> (
    tuple[Array, Array] | tuple[Array, Array, Array] | tuple[Array, Array, Array, Array]
):
    r"""Evaluate related SE(3) constant-strain operators with shared work.

    The standard result is ``(Ad_inv, T)``. Supplying ``xid`` adds ``Td``;
    requesting ``return_adjoint`` adds the forward adjoint as the final
    element. Matrix powers through ``B**4`` are built once, while ``T`` and
    ``Td`` also share their scalar coefficients. Separate adjoint and tangent
    thresholds preserve the caller's existing numerical policy.

    This private execution helper mirrors :func:`.se2._operators` while
    keeping the spatial fourth-order polynomial.

    Args:
        xi: Spatial strain in angular-first order.
        s: Scalar arclength.
        adjoint_eps: Threshold for forward/inverse adjoint coefficients.
        tangent_eps: Threshold for tangent coefficients and derivatives.
        xid: Optional strain direction/rate. Supplying it requests ``Td``.
        return_adjoint: Append the already-computed forward adjoint. This is
            used by arc-length derivatives and the public named bundle.

    Returns:
        Selective tuple beginning with ``(Ad_inv, T)`` and followed by ``Td``
        and/or ``Ad`` according to the two optional arguments. The public
        :func:`operators` function provides a fixed named return type.
    """
    xi = jnp.asarray(xi).reshape(-1)
    s = jnp.asarray(s, dtype=xi.dtype)
    ad_xi = lie_se3.small_adjoint(xi)
    angle_sq = jnp.dot(xi[:3], xi[:3])

    if xid is None:
        powers = _matrix_polynomials.powers(s * ad_xi, _REDUCED_POLYNOMIAL_DEGREE)
        dot_powers = None
    else:
        xid = jnp.asarray(xid).reshape(-1)
        powers, dot_powers = _matrix_polynomials.powers_and_directional_derivatives(
            s * ad_xi,
            s * lie_se3.small_adjoint(xid),
            _REDUCED_POLYNOMIAL_DEGREE,
        )

    angle_sq_dot = None if xid is None else 2.0 * jnp.dot(xi[:3], xid[:3])
    return _operators_from_powers(
        powers,
        dot_powers,
        angle_sq,
        angle_sq_dot,
        s,
        adjoint_eps,
        tangent_eps,
        return_adjoint=return_adjoint,
    )


def adjoint(xi: Array, s: Array, eps: float | Array) -> Array:
    r"""Return the adjoint accumulated along a constant spatial strain segment.

    This is a rod-kinematics helper for spatial PCS/GVS segments. For constant
    strain ``xi`` evaluated at arclength ``s``, the returned matrix represents
    the closed-form adjoint associated with the segment transform
    ``exp(s * xi)`` under the angular-first ``se(3)`` convention.
    Equivalently, it is ``exp(s * lie_se3.small_adjoint(xi))``.

    Args:
        xi: Constant spatial strain with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Minimum rotational-strain threshold. The actual dimensionless
            cutoff also accounts for ``s`` and the input dtype.

    Returns:
        Array with shape ``(6, 6)`` mapping spatial twists through the segment
        adjoint at arclength ``s``.

    Notes:
        Rotation enters the scalar coefficients only through
        ``dot(omega, omega)``. This avoids differentiating a Euclidean norm at
        zero while retaining arbitrary-axis behavior. The fourth-order matrix
        polynomial is exact for an ``se(3)`` adjoint.
    """
    xi = jnp.asarray(xi).reshape(-1)
    s = jnp.asarray(s, dtype=xi.dtype)
    angle_sq = jnp.dot(xi[:3], xi[:3])
    powers = _matrix_polynomials.powers(
        s * lie_se3.small_adjoint(xi), _REDUCED_POLYNOMIAL_DEGREE
    )
    return lie_se3._adjoint_from_powers(powers, _adjoint_coefficients(angle_sq, s, eps))


def adjoint_inverse(xi: Array, s: Array, eps: float | Array) -> Array:
    """Return the inverse adjoint for a constant spatial strain segment.

    This computes the inverse of :func:`adjoint` without explicitly
    constructing and inverting the homogeneous segment transform. It is used
    when propagating body-frame Jacobians backward through spatial
    constant-strain segments. It uses the rotation and translation-skew blocks
    of the stabilized forward adjoint rather than a generic ``6 x 6`` inverse.

    Args:
        xi: Constant spatial strain with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Minimum rotational-strain threshold forwarded to
            :func:`adjoint`.

    Returns:
        Array with shape ``(6, 6)`` representing the inverse segment adjoint.

    Notes:
        Building the inverse from the forward blocks keeps strict-singularity
        and stable-default behavior aligned with :func:`adjoint` and avoids
        the cost and conditioning of a dense solve.
    """
    forward_adjoint = adjoint(xi, s, eps=eps)
    return lie_se3._adjoint_inverse_from_adjoint(forward_adjoint)


def tangent(xi: Array, s: Array, eps: float | Array) -> Array:
    r"""Return the tangent operator for a constant spatial strain segment.

    The tangent operator integrates local spatial strain perturbations over
    arclength for a segment whose strain is constant. In PCS/GVS recurrences it
    maps strain-basis contributions into body-frame Jacobian contributions at
    arclength ``s``. It evaluates

    .. math::

        T(s,\xi) = \int_0^s
        \exp(\sigma\,\operatorname{ad}_\xi)\,d\sigma

    with the exact reduced ``se(3)`` polynomial.

    Args:
        xi: Constant spatial strain with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Minimum rotational-strain threshold. The actual dimensionless
            cutoff also accounts for ``s`` and the input dtype.

    Returns:
        Array with shape ``(6, 6)`` containing the spatial constant-strain
        tangent operator at arclength ``s``.

    Notes:
        The implementation covers pure translation and arbitrary-axis rotation
        with one formula. At zero arclength the result is exactly zero; at zero
        strain it is ``s * I``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    s = jnp.asarray(s, dtype=xi.dtype)
    accumulated_eps = jnp.abs(s) * jnp.abs(jnp.asarray(eps, dtype=xi.dtype))
    return s * lie_se3.left_jacobian(s * xi, accumulated_eps)


def tangent_derivative(xi: Array, xid: Array, s: Array, eps: float | Array) -> Array:
    r"""Return the directional derivative of the spatial tangent operator.

    This differentiates :func:`tangent` with respect to time through the
    strain ``xi`` and strain rate ``xid`` while holding arclength ``s`` fixed.
    The coefficient derivative is expressed through the squared rotational
    magnitude, avoiding a division by ``norm(omega)`` at zero rotation. The
    returned matrix is ``D_xi T(s, xi)[xid]``.

    Args:
        xi: Constant spatial strain with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
        xid: Time derivative of ``xi`` with the same shape and coordinate
            order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Minimum rotational-strain threshold. The actual dimensionless
            cutoff also accounts for ``s`` and the input dtype.

    Returns:
        Array with shape ``(6, 6)`` containing ``d/dt tangent(xi, s)``.

    Notes:
        The derivative uses ``d dot(omega, omega) / dt =
        2 * dot(omega, omega_dot)`` and the generic product-rule recurrence for
        powers of ``s * ad_xi``. It is fully analytic and linear in ``xid``;
        autodiff appears only in tests that validate this implementation.
    """
    xi = jnp.asarray(xi).reshape(-1)
    xid = jnp.asarray(xid).reshape(-1)
    s = jnp.asarray(s, dtype=xi.dtype)
    accumulated_eps = jnp.abs(s) * jnp.abs(jnp.asarray(eps, dtype=xi.dtype))
    return s * lie_se3.left_jacobian_directional_derivative(
        s * xi, s * xid, accumulated_eps
    )


def operators(
    xi: Array,
    s: Array,
    eps: float | Array,
    xid: Array | None = None,
) -> ConstantStrainOperators:
    r"""Return the related SE(3) constant-strain operators as one bundle.

    This is the spatial counterpart of :func:`.se2.operators`. It is intended
    for users who require multiple expressions at the same constant strain and
    arclength. Matrix powers through ``B**4`` are constructed once; the tangent
    and its optional derivative also share their stable coefficients.

    Args:
        xi: Constant spatial strain with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
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
        The fixed named result is JAX-transformable. The implementation uses
        explicit matrix-power and coefficient derivatives and performs no
        runtime autodiff. For one expression, prefer the corresponding
        single-operator function to avoid unused assembly.
    """
    if xid is None:
        adjoint_inverse, tangent, adjoint = _operators(
            xi, s, eps, eps, return_adjoint=True
        )
        tangent_derivative = None
    else:
        adjoint_inverse, tangent, tangent_derivative, adjoint = _operators(
            xi, s, eps, eps, xid, return_adjoint=True
        )
    return ConstantStrainOperators(
        adjoint, adjoint_inverse, tangent, tangent_derivative
    )
