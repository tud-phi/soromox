__all__ = [
    "adjoint_se2",
    "adjoint_inverse_se2",
    "tangent_se2",
    "tangent_derivative_se2",
    "adjoint_se3",
    "adjoint_inverse_se3",
    "tangent_se3",
    "tangent_derivative_se3",
]

import jax.numpy as jnp
from jax import Array, lax

from . import se2, se3


def _rotational_strain_magnitude_se3(xi: Array, eps: float | Array) -> Array:
    """Return a regularized norm of a spatial strain's rotational component.

    Constant-strain spatial vectors use the same angular-first convention as
    ``se3``: ``xi = [omega_x, omega_y, omega_z, v_x, v_y, v_z]``. This helper
    returns ``norm(omega)`` except inside the ``eps`` ball, where it returns an
    exact scalar zero to keep the closed-form branch selection and autodiff
    behavior well-defined at zero rotation.

    Args:
        xi: Spatial strain with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
        eps: Small positive scalar threshold. If ``dot(omega, omega) <= eps**2``,
            the helper returns zero.

    Returns:
        Scalar array containing the regularized rotational strain magnitude.
    """
    k = xi[:3]
    theta_sq = jnp.dot(k, k)
    return lax.cond(
        theta_sq <= eps**2,
        lambda _: jnp.zeros((), dtype=xi.dtype),
        lambda _: jnp.sqrt(theta_sq),
        operand=None,
    )


def _adjoint_powers_with_derivatives(ad_xi: Array, ad_xid: Array, dim: int):
    """Compute powers of ``ad_xi`` and their directional time derivatives.

    The constant-strain tangent derivative formulas are written as polynomial
    combinations of ``ad_xi`` through fourth order. This helper builds
    ``[I, A, A^2, A^3, A^4]`` with ``A = ad_xi`` and the corresponding
    derivatives under ``A_dot = ad_xid`` using the product rule.

    Args:
        ad_xi: Algebra adjoint matrix with shape ``(dim, dim)``.
        ad_xid: Time derivative of ``ad_xi`` with shape ``(dim, dim)``.
        dim: Matrix dimension, either ``3`` for planar strains or ``6`` for
            spatial strains.

    Returns:
        Tuple ``(powers, dot_powers)``. Each item is a Python list of five
        arrays with shape ``(dim, dim)``. Entry ``k`` contains ``A^k`` and
        ``d/dt A^k`` respectively.
    """
    current = jnp.eye(dim, dtype=ad_xi.dtype)
    dot_current = jnp.zeros_like(ad_xi)
    powers = [current]
    dot_powers = [dot_current]

    for _ in range(4):
        dot_next = dot_current @ ad_xi + current @ ad_xid
        current = current @ ad_xi
        powers.append(current)
        dot_powers.append(dot_next)
        dot_current = dot_next

    return powers, dot_powers


def _constant_strain_adjoint(
    ad_xi: Array, theta: Array, s: Array, eps: float | Array
) -> Array:
    """Evaluate the shared closed form for constant-strain adjoint transport.

    This dimension-independent helper is used by both planar and spatial
    public functions. The caller supplies ``ad_xi`` and the appropriate
    rotational magnitude ``theta``: ``abs(xi[0])`` for ``se(2)`` and
    ``norm(xi[:3])`` for ``se(3)``. The helper switches to a first-order
    series in the small-angle branch.

    Args:
        ad_xi: Algebra adjoint matrix of the constant strain, with shape
            ``(dim, dim)``.
        theta: Scalar rotational strain magnitude used in the closed-form
            trigonometric coefficients.
        s: Scalar arclength position measured from the segment base.
        eps: Small positive scalar threshold for the small-angle branch.

    Returns:
        Array with shape ``(dim, dim)`` representing the constant-strain
        adjoint transport at arclength ``s``.
    """
    dim = ad_xi.shape[0]

    def _series_branch() -> Array:
        return jnp.eye(dim, dtype=ad_xi.dtype) + s * ad_xi

    def _general_branch(theta_val: Array) -> Array:
        cos_theta = jnp.cos(s * theta_val)
        sin_theta = jnp.sin(s * theta_val)

        ad_xi_square = ad_xi @ ad_xi
        ad_xi_cube = ad_xi_square @ ad_xi
        ad_xi_quad = ad_xi_cube @ ad_xi

        term1 = (3 * sin_theta - s * theta_val * cos_theta) / (2 * theta_val) * ad_xi
        term2 = (
            (4 - 4 * cos_theta - s * theta_val * sin_theta)
            / (2 * theta_val**2)
            * ad_xi_square
        )
        term3 = (
            (sin_theta - s * theta_val * cos_theta) / (2 * theta_val**3) * ad_xi_cube
        )
        term4 = (
            (2 - 2 * cos_theta - s * theta_val * sin_theta)
            / (2 * theta_val**4)
            * ad_xi_quad
        )

        return jnp.eye(dim, dtype=ad_xi.dtype) + term1 + term2 + term3 + term4

    return lax.cond(
        theta <= eps,
        lambda _: _series_branch(),
        lambda _: _general_branch(theta),
        operand=None,
    )


def _constant_strain_tangent(
    ad_xi: Array, theta: Array, s: Array, eps: float | Array
) -> Array:
    """Evaluate the shared closed form for the constant-strain tangent map.

    This helper implements the dimension-independent polynomial/trigonometric
    expression used by ``tangent_se2`` and ``tangent_se3``. It assumes the
    caller has already converted the strain into its Lie algebra adjoint matrix
    and supplied the matching rotational magnitude.

    Args:
        ad_xi: Algebra adjoint matrix of the constant strain, with shape
            ``(dim, dim)``.
        theta: Scalar rotational strain magnitude, ``abs(theta)`` for planar
            strain or ``norm(omega)`` for spatial strain.
        s: Scalar arclength position measured from the segment base.
        eps: Small positive scalar threshold for the small-angle series branch.

    Returns:
        Array with shape ``(dim, dim)`` containing the tangent operator at
        arclength ``s``.
    """
    dim = ad_xi.shape[0]

    def _series_branch() -> Array:
        return s * jnp.eye(dim, dtype=ad_xi.dtype) + 0.5 * s**2 * ad_xi

    def _general_branch(theta_val: Array) -> Array:
        cos_theta = jnp.cos(s * theta_val)
        sin_theta = jnp.sin(s * theta_val)

        ad_xi_square = ad_xi @ ad_xi
        ad_xi_cube = ad_xi_square @ ad_xi
        ad_xi_quad = ad_xi_cube @ ad_xi

        term1 = (
            (4 - 4 * cos_theta - s * theta_val * sin_theta) / (2 * theta_val**2) * ad_xi
        )
        term2 = (
            (4 * s * theta_val - 5 * sin_theta + s * theta_val * cos_theta)
            / (2 * theta_val**3)
            * ad_xi_square
        )
        term3 = (
            (2 - 2 * cos_theta - s * theta_val * sin_theta)
            / (2 * theta_val**4)
            * ad_xi_cube
        )
        term4 = (
            (2 * s * theta_val - 3 * sin_theta + s * theta_val * cos_theta)
            / (2 * theta_val**5)
            * ad_xi_quad
        )

        return s * jnp.eye(dim, dtype=ad_xi.dtype) + term1 + term2 + term3 + term4

    return lax.cond(
        theta <= eps,
        lambda _: _series_branch(),
        lambda _: _general_branch(theta),
        operand=None,
    )


def _constant_strain_tangent_derivative(
    ad_xi: Array,
    ad_xid: Array,
    theta: Array,
    theta_dot: Array,
    s: Array,
    eps: float | Array,
) -> Array:
    """Evaluate the shared time derivative of the constant-strain tangent map.

    The derivative is taken with respect to the strain trajectory while holding
    arclength ``s`` fixed. The caller supplies both ``ad_xi`` and ``ad_xid`` so
    the helper remains independent of the planar/spatial coordinate dimension.
    The scalar ``theta_dot`` must be the derivative of the rotational magnitude
    used in the coefficients.

    Args:
        ad_xi: Algebra adjoint matrix of the strain, with shape ``(dim, dim)``.
        ad_xid: Algebra adjoint matrix of the strain rate, with shape
            ``(dim, dim)``.
        theta: Scalar rotational strain magnitude used by the tangent
            coefficients.
        theta_dot: Scalar time derivative of ``theta``. For planar strain this
            is ``xid[0]``; for spatial strain it is ``dot(omega_dot, omega) /
            norm(omega)`` in the nonzero branch.
        s: Scalar arclength position measured from the segment base.
        eps: Small positive scalar threshold for the small-angle series branch.

    Returns:
        Array with shape ``(dim, dim)`` containing the time derivative of the
        tangent operator at arclength ``s``.
    """
    dim = ad_xi.shape[0]

    def _series_branch() -> Array:
        return 0.5 * s**2 * ad_xid

    def _general_branch(theta_val: Array) -> Array:
        cos_theta = jnp.cos(s * theta_val)
        sin_theta = jnp.sin(s * theta_val)

        powers, dot_powers = _adjoint_powers_with_derivatives(ad_xi, ad_xid, dim)

        coeff_theta_common = (
            -8 + (8 - s**2 * theta_val**2) * cos_theta + 5 * s * theta_val * sin_theta
        )
        coeff_theta_alt = (
            -8 * s * theta_val
            + (15 - s**2 * theta_val**2) * sin_theta
            - 7 * s * theta_val * cos_theta
        )

        coeff1_theta = theta_dot / (2 * theta_val**3) * coeff_theta_common
        coeff1 = (4 - 4 * cos_theta - s * theta_val * sin_theta) / (2 * theta_val**2)

        coeff2_theta = theta_dot / (2 * theta_val**4) * coeff_theta_alt
        coeff2 = (4 * s * theta_val - 5 * sin_theta + s * theta_val * cos_theta) / (
            2 * theta_val**3
        )

        coeff3_theta = theta_dot / (2 * theta_val**5) * coeff_theta_common
        coeff3 = (2 - 2 * cos_theta - s * theta_val * sin_theta) / (2 * theta_val**4)

        coeff4_theta = theta_dot / (2 * theta_val**6) * coeff_theta_alt
        coeff4 = (2 * s * theta_val - 3 * sin_theta + s * theta_val * cos_theta) / (
            2 * theta_val**5
        )

        return (
            coeff1_theta * powers[1]
            + coeff1 * dot_powers[1]
            + coeff2_theta * powers[2]
            + coeff2 * dot_powers[2]
            + coeff3_theta * powers[3]
            + coeff3 * dot_powers[3]
            + coeff4_theta * powers[4]
            + coeff4 * dot_powers[4]
        )

    return lax.cond(
        theta <= eps,
        lambda _: _series_branch(),
        lambda _: _general_branch(theta),
        operand=None,
    )


def adjoint_se2(xi: Array, s: Array, eps: float | Array) -> Array:
    """Return the adjoint accumulated along a constant planar strain segment.

    This is a rod-kinematics helper, not a generic transform adjoint. For a
    segment with constant planar strain ``xi`` evaluated at arclength ``s``,
    the returned matrix represents the closed-form adjoint associated with the
    segment transform ``exp(s * xi)`` under the angular-first ``se(2)``
    convention.

    Args:
        xi: Constant planar strain with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, v_x, v_y]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Small positive scalar threshold used to select the small-angle
            series branch when ``abs(theta) <= eps``.

    Returns:
        Array with shape ``(3, 3)`` mapping planar twists through the segment
        adjoint at arclength ``s``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    theta = jnp.abs(xi[0])
    return _constant_strain_adjoint(se2.small_adjoint(xi), theta, s, eps)


def adjoint_inverse_se2(xi: Array, s: Array, eps: float | Array) -> Array:
    """Return the inverse adjoint for a constant planar strain segment.

    This computes the inverse of :func:`adjoint_se2` without explicitly
    inverting the full homogeneous segment transform. It is used when
    propagating planar body-frame Jacobians backward through a constant-strain
    segment.

    Args:
        xi: Constant planar strain with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, v_x, v_y]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Small positive scalar threshold for the same small-angle branch
            used by :func:`adjoint_se2`.

    Returns:
        Array with shape ``(3, 3)`` representing the inverse segment adjoint.
    """
    Ad = adjoint_se2(xi, s, eps=eps)

    R = Ad[1:, 1:]
    mJt = Ad[1:, 0].reshape(-1, 1)
    R_inv = jnp.transpose(R)

    return jnp.concatenate(
        [
            jnp.concatenate(
                [jnp.ones((1, 1), dtype=Ad.dtype), jnp.zeros((1, 2), dtype=Ad.dtype)],
                axis=1,
            ),
            jnp.concatenate([-R_inv @ mJt, R_inv], axis=1),
        ]
    )


def tangent_se2(xi: Array, s: Array, eps: float | Array) -> Array:
    """Return the tangent operator for a constant planar strain segment.

    The tangent operator integrates the local strain perturbation over
    arclength for a segment whose strain is constant. In PCS/GVS recurrences it
    maps strain-basis contributions into body-frame Jacobian contributions at
    arclength ``s``.

    Args:
        xi: Constant planar strain with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, v_x, v_y]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Small positive scalar threshold used for the small-angle Taylor
            series when ``abs(theta) <= eps``.

    Returns:
        Array with shape ``(3, 3)`` containing the planar constant-strain
        tangent operator at arclength ``s``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    theta = jnp.abs(xi[0])
    return _constant_strain_tangent(se2.small_adjoint(xi), theta, s, eps)


def tangent_derivative_se2(
    xi: Array, xid: Array, s: Array, eps: float | Array
) -> Array:
    """Return the time derivative of the planar tangent operator.

    This differentiates :func:`tangent_se2` with respect to time through the
    strain ``xi`` and strain rate ``xid`` while holding arclength ``s`` fixed.
    The function uses the same angular-first planar convention as ``se2``.

    Args:
        xi: Constant planar strain with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, v_x, v_y]`` order.
        xid: Time derivative of ``xi`` with the same shape and coordinate
            order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Small positive scalar threshold used for the small-angle series
            branch.

    Returns:
        Array with shape ``(3, 3)`` containing ``d/dt tangent_se2(xi, s)``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    xid = jnp.asarray(xid).reshape(-1)
    theta = jnp.abs(xi[0])
    theta_dot = xid[0]
    return _constant_strain_tangent_derivative(
        se2.small_adjoint(xi),
        se2.small_adjoint(xid),
        theta,
        theta_dot,
        s,
        eps,
    )


def adjoint_se3(xi: Array, s: Array, eps: float | Array) -> Array:
    """Return the adjoint accumulated along a constant spatial strain segment.

    This is a rod-kinematics helper for spatial PCS/GVS segments. For constant
    strain ``xi`` evaluated at arclength ``s``, the returned matrix represents
    the closed-form adjoint associated with the segment transform
    ``exp(s * xi)`` under the angular-first ``se(3)`` convention.

    Args:
        xi: Constant spatial strain with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Small positive scalar threshold used to select the small-angle
            branch when ``norm(omega) <= eps``.

    Returns:
        Array with shape ``(6, 6)`` mapping spatial twists through the segment
        adjoint at arclength ``s``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    theta = _rotational_strain_magnitude_se3(xi, eps)
    return _constant_strain_adjoint(se3.small_adjoint(xi), theta, s, eps)


def adjoint_inverse_se3(xi: Array, s: Array, eps: float | Array) -> Array:
    """Return the inverse adjoint for a constant spatial strain segment.

    This computes the inverse of :func:`adjoint_se3` without explicitly
    constructing and inverting the homogeneous segment transform. It is used
    when propagating body-frame Jacobians backward through spatial
    constant-strain segments.

    Args:
        xi: Constant spatial strain with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Small positive scalar threshold for the same small-angle branch
            used by :func:`adjoint_se3`.

    Returns:
        Array with shape ``(6, 6)`` representing the inverse segment adjoint.
    """
    Ad = adjoint_se3(xi, s, eps=eps)

    R = Ad[:3, :3]
    t_hat_R = Ad[3:, :3]
    R_inv = jnp.transpose(R)
    t_hat = t_hat_R @ R_inv

    return jnp.block(
        [[R_inv, jnp.zeros((3, 3), dtype=Ad.dtype)], [-R_inv @ t_hat, R_inv]]
    )


def tangent_se3(xi: Array, s: Array, eps: float | Array) -> Array:
    """Return the tangent operator for a constant spatial strain segment.

    The tangent operator integrates local spatial strain perturbations over
    arclength for a segment whose strain is constant. In PCS/GVS recurrences it
    maps strain-basis contributions into body-frame Jacobian contributions at
    arclength ``s``.

    Args:
        xi: Constant spatial strain with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Small positive scalar threshold used for the small-angle Taylor
            series when ``norm(omega) <= eps``.

    Returns:
        Array with shape ``(6, 6)`` containing the spatial constant-strain
        tangent operator at arclength ``s``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    theta = _rotational_strain_magnitude_se3(xi, eps)
    return _constant_strain_tangent(se3.small_adjoint(xi), theta, s, eps)


def tangent_derivative_se3(
    xi: Array, xid: Array, s: Array, eps: float | Array
) -> Array:
    """Return the time derivative of the spatial tangent operator.

    This differentiates :func:`tangent_se3` with respect to time through the
    strain ``xi`` and strain rate ``xid`` while holding arclength ``s`` fixed.
    The rotational magnitude derivative is regularized in the zero-rotation
    branch to keep autodiff finite.

    Args:
        xi: Constant spatial strain with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
        xid: Time derivative of ``xi`` with the same shape and coordinate
            order.
        s: Scalar arclength position measured from the base of the segment.
        eps: Small positive scalar threshold used for the small-angle series
            branch.

    Returns:
        Array with shape ``(6, 6)`` containing ``d/dt tangent_se3(xi, s)``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    xid = jnp.asarray(xid).reshape(-1)

    theta = _rotational_strain_magnitude_se3(xi, eps)
    theta_dot = lax.cond(
        theta <= eps,
        lambda _: jnp.zeros((), dtype=xi.dtype),
        lambda _: jnp.dot(xid[:3], xi[:3]) / theta,
        operand=None,
    )

    return _constant_strain_tangent_derivative(
        se3.small_adjoint(xi),
        se3.small_adjoint(xid),
        theta,
        theta_dot,
        s,
        eps,
    )
