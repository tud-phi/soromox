__all__ = [
    "hat",
    "exp",
    "log",
    "small_adjoint",
    "coadjoint",
    "adjoint",
    "adjoint_inverse",
    "left_jacobian",
    "left_jacobian_and_directional_derivative",
    "left_jacobian_directional_derivative",
]

import jax.numpy as jnp
from jax import Array, lax

from . import _se3_left_jacobian, so3
from .jacobian_coefficients import (
    angle_minus_sine_over_angle_cubed,
    inverse_left_jacobian_quadratic_coefficient,
    one_minus_cosine_over_angle_squared,
    sine_over_angle,
)


def _rotational_strain_magnitude(xi: Array, eps: float | Array) -> Array:
    """Return a differentiability-aware norm of the rotational twist part.

    The spatial twist convention is angular-first, so this helper computes
    ``norm(xi[:3])`` for ``xi = [omega, v]``. Near zero rotation it returns an
    exact scalar zero instead of evaluating ``sqrt(dot(omega, omega))``. This
    avoids reverse-mode autodiff singularities at ``omega == 0`` and matches
    the small-angle branch used by :func:`exp`.

    Args:
        xi: Spatial twist with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
        eps: Small positive scalar threshold. If ``dot(omega, omega) <= eps**2``,
            the helper returns zero.

    Returns:
        Scalar array containing the regularized rotational magnitude.
    """
    k = xi[:3]
    theta_sq = jnp.dot(k, k)
    return lax.cond(
        theta_sq <= eps**2,
        lambda _: jnp.zeros((), dtype=xi.dtype),
        lambda _: jnp.sqrt(theta_sq),
        operand=None,
    )


def hat(xi: Array) -> Array:
    """Return the homogeneous matrix representation of an ``se(3)`` twist.

    Spatial twists use angular-first coordinates
    ``xi = [omega_x, omega_y, omega_z, v_x, v_y, v_z]``. The returned matrix is
    ``[[so3.skew(omega), v], [0, 0, 0, 0]]`` and is suitable for use in the
    matrix exponential.

    Args:
        xi: Spatial twist with shape ``(6,)`` or ``(6, 1)``. The first three
            entries are the angular component ``omega`` and the final three
            entries are the translational component ``v``.

    Returns:
        Array with shape ``(4, 4)`` representing the same algebra element in
        homogeneous matrix form.
    """
    xi = jnp.asarray(xi).reshape(-1)

    omega = xi[:3].reshape((3, 1))
    v = xi[3:].reshape((3, 1))

    return jnp.block(
        [
            [so3.skew(omega), v],
            [jnp.zeros((1, 3), dtype=xi.dtype), jnp.zeros((1, 1), dtype=xi.dtype)],
        ]
    )


def log(g: Array, eps: float | Array) -> Array:
    """Compute the Lie-group logarithm from ``SE(3)`` to ``se(3)``.

    The returned vector is a spatial twist in angular-first coordinates. The
    translational component is not the raw transform translation; it is the
    result of applying the inverse ``SE(3)`` left Jacobian to the translation.
    This makes the function the inverse of :func:`exp` for regular transforms.

    Args:
        g: Homogeneous ``SE(3)`` transform with shape ``(4, 4)``. The rotation
            is read from ``g[:3, :3]`` and the translation from ``g[:3, 3]``.
        eps: Small positive scalar threshold passed to ``so3.log`` for
            rotation extraction. The inverse Jacobian uses a dtype-aware
            Taylor series near zero rotation.

    Returns:
        Array with shape ``(6,)`` in
        ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
    """
    R = g[:3, :3]
    p = g[:3, 3].reshape((3, 1))

    omega = so3.log(R, eps=eps).reshape((3, 1))
    omega_hat = so3.skew(omega)
    theta_sq = jnp.dot(omega.reshape(-1), omega.reshape(-1))
    theta = lax.cond(
        theta_sq <= eps**2,
        lambda _: jnp.zeros((), dtype=R.dtype),
        lambda _: jnp.sqrt(theta_sq),
        operand=None,
    )
    B = inverse_left_jacobian_quadratic_coefficient(theta)
    omega_sq = omega_hat @ omega_hat
    V_inv = jnp.eye(3, dtype=R.dtype) - 0.5 * omega_hat + B * omega_sq

    v = V_inv @ p

    return jnp.vstack([omega, v]).reshape(-1)


def exp(xi: Array, eps: float | Array) -> Array:
    """Compute the Lie-group exponential from ``se(3)`` to ``SE(3)``.

    This is the matrix exponential of :func:`hat`. The translational component
    of ``xi`` is integrated through the ``SE(3)`` left Jacobian, so it is a
    twist coordinate rather than a direct pose translation. Use
    ``poses.quaternion_pose_to_transform`` for direct quaternion-plus-position
    pose coordinates.

    Args:
        xi: Spatial twist with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
        eps: Minimum small-angle series threshold. A dtype-aware threshold may
            select the series over a larger interval to preserve Hessian
            accuracy near zero rotation.

    Returns:
        Homogeneous ``SE(3)`` transform with shape ``(4, 4)``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    theta = _rotational_strain_magnitude(xi, eps)
    omega_hat = so3.skew(xi[:3])
    omega_hat_sq = omega_hat @ omega_hat
    cosc = one_minus_cosine_over_angle_squared(theta, eps)
    tanc = angle_minus_sine_over_angle_cubed(theta, eps)
    V = jnp.eye(3, dtype=xi.dtype) + cosc * omega_hat + tanc * omega_hat_sq
    p = V @ xi[3:]
    R = so3._exp_from_skew(omega_hat, theta, eps)

    return jnp.block(
        [
            [R, p.reshape((3, 1))],
            [jnp.zeros((1, 3), dtype=xi.dtype), jnp.ones((1, 1), dtype=xi.dtype)],
        ]
    )


def small_adjoint(xi: Array) -> Array:
    """Return the Lie algebra adjoint matrix ``ad_xi``.

    The returned matrix implements the spatial Lie bracket in angular-first
    coordinates: ``small_adjoint(xi) @ eta == [xi, eta]`` for twists
    ``xi`` and ``eta`` written as ``[omega, v]``.

    Args:
        xi: Spatial twist with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.

    Returns:
        Array with shape ``(6, 6)`` representing ``ad_xi``.
    """
    xi = jnp.asarray(xi).reshape(-1)

    omega = xi[:3].reshape((3, 1))
    v = xi[3:].reshape((3, 1))

    omega_hat = so3.skew(omega)
    v_hat = so3.skew(v)

    return jnp.block(
        [[omega_hat, jnp.zeros((3, 3), dtype=xi.dtype)], [v_hat, omega_hat]]
    )


def left_jacobian(xi: Array, eps: float | Array) -> Array:
    r"""Return the left Jacobian of the :math:`SE(3)` exponential.

    For an accumulated spatial twist ``xi``, this evaluates

    .. math::

        J_l(\xi) = \int_0^1 \exp(\sigma\,\operatorname{ad}_\xi)\,d\sigma.

    The operation is intrinsic to ``SE(3)`` and does not assume that ``xi``
    came from a constant-strain rod segment.

    Args:
        xi: Accumulated spatial twist with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.
        eps: Minimum small-angle threshold for the stable coefficient series.

    Returns:
        Left Jacobian with shape ``(6, 6)``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    angle_sq = jnp.dot(xi[:3], xi[:3])
    return _se3_left_jacobian._left_jacobian(small_adjoint(xi), angle_sq, eps)


def left_jacobian_directional_derivative(
    xi: Array, xid: Array, eps: float | Array
) -> Array:
    r"""Return ``D J_l(xi)[xid]`` for the :math:`SE(3)` left Jacobian.

    ``xid`` is an arbitrary direction in the Lie algebra; it need not be a
    physical time derivative or a constant-strain rate.

    Args:
        xi: Accumulated spatial twist in angular-first order.
        xid: Direction with the same shape and coordinate order as ``xi``.
        eps: Minimum small-angle threshold for the stable coefficient series.

    Returns:
        Directional derivative with shape ``(6, 6)``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    xid = jnp.asarray(xid).reshape(-1)
    angle_sq = jnp.dot(xi[:3], xi[:3])
    angle_sq_dot = 2.0 * jnp.dot(xi[:3], xid[:3])
    return _se3_left_jacobian._left_jacobian_directional_derivative(
        small_adjoint(xi),
        small_adjoint(xid),
        angle_sq,
        angle_sq_dot,
        eps,
    )


def left_jacobian_and_directional_derivative(
    xi: Array, xid: Array, eps: float | Array
) -> tuple[Array, Array]:
    r"""Evaluate ``J_l(xi)`` and ``D J_l(xi)[xid]`` with shared work.

    Args:
        xi: Accumulated spatial twist in angular-first order.
        xid: Direction with the same shape and coordinate order as ``xi``.
        eps: Minimum small-angle threshold for the stable coefficient series.

    Returns:
        Tuple ``(J, Jd)`` whose arrays both have shape ``(6, 6)``.
    """
    xi = jnp.asarray(xi).reshape(-1)
    xid = jnp.asarray(xid).reshape(-1)
    angle_sq = jnp.dot(xi[:3], xi[:3])
    angle_sq_dot = 2.0 * jnp.dot(xi[:3], xid[:3])
    return _se3_left_jacobian._left_jacobian_and_directional_derivative(
        small_adjoint(xi),
        small_adjoint(xid),
        angle_sq,
        angle_sq_dot,
        eps,
    )


def _exp_left_jacobian_and_directional_derivative(
    xi: Array,
    xid: Array,
    exp_eps: float | Array,
    jacobian_eps: float | Array,
) -> tuple[Array, Array, Array]:
    r"""Evaluate ``exp(xi)``, ``J_l(xi)``, and ``D J_l(xi)[xid]`` together.

    This private execution kernel exploits the block structure of
    ``ad_xi`` and the ``so(3)`` minimal polynomial. It is kept private because
    bundling the group exponential with two differential operators is a GVS
    performance concern rather than a separate mathematical API.

    Separate thresholds preserve the caller's numerical policies for the
    group exponential and left Jacobian.
    """
    xi = jnp.asarray(xi).reshape(-1)
    xid = jnp.asarray(xid).reshape(-1)

    omega_hat = so3.skew(xi[:3])
    linear_hat = so3.skew(xi[3:])
    omega_hat_dot = so3.skew(xid[:3])
    linear_hat_dot = so3.skew(xid[3:])

    identity = jnp.eye(3, dtype=xi.dtype)
    zero = jnp.zeros((3, 3), dtype=xi.dtype)
    angle_sq = jnp.dot(xi[:3], xi[:3])
    angle_sq_dot = 2.0 * jnp.dot(xi[:3], xid[:3])
    coefficients, derivatives = _se3_left_jacobian._coefficients_and_x_derivatives(
        angle_sq, jacobian_eps
    )
    c1, c2, c3, c4 = coefficients
    d1, d2, d3, d4 = derivatives

    omega_hat_sq = omega_hat @ omega_hat
    omega_hat_sq_dot = omega_hat_dot @ omega_hat + omega_hat @ omega_hat_dot
    linear_omega = linear_hat @ omega_hat
    linear_omega_dot = linear_hat_dot @ omega_hat + linear_hat @ omega_hat_dot
    omega_linear = omega_hat @ linear_hat
    omega_linear_dot = omega_hat_dot @ linear_hat + omega_hat @ linear_hat_dot

    second_lower = linear_omega + omega_linear
    second_lower_dot = linear_omega_dot + omega_linear_dot

    linear_omega_sq = linear_hat @ omega_hat_sq
    linear_omega_sq_dot = linear_hat_dot @ omega_hat_sq + linear_hat @ omega_hat_sq_dot
    omega_sq_linear = omega_hat_sq @ linear_hat
    omega_sq_linear_dot = omega_hat_sq_dot @ linear_hat + omega_hat_sq @ linear_hat_dot
    omega_linear_omega = omega_linear @ omega_hat
    omega_linear_omega_dot = omega_linear_dot @ omega_hat + omega_linear @ omega_hat_dot
    third_lower = linear_omega_sq + omega_linear_omega + omega_sq_linear
    third_lower_dot = linear_omega_sq_dot + omega_linear_omega_dot + omega_sq_linear_dot

    omega_linear_omega_sq = omega_linear @ omega_hat_sq
    omega_linear_omega_sq_dot = (
        omega_linear_dot @ omega_hat_sq + omega_linear @ omega_hat_sq_dot
    )
    omega_sq_linear_omega = omega_sq_linear @ omega_hat
    omega_sq_linear_omega_dot = (
        omega_sq_linear_dot @ omega_hat + omega_sq_linear @ omega_hat_dot
    )
    fourth_lower_remainder = omega_linear_omega_sq + omega_sq_linear_omega
    fourth_lower_remainder_dot = omega_linear_omega_sq_dot + omega_sq_linear_omega_dot

    diagonal_linear = c1 - angle_sq * c3
    diagonal_quadratic = c2 - angle_sq * c4
    diagonal_linear_dot = (d1 - c3 - angle_sq * d3) * angle_sq_dot
    diagonal_quadratic_dot = (d2 - c4 - angle_sq * d4) * angle_sq_dot
    jacobian_diagonal = (
        identity + diagonal_linear * omega_hat + diagonal_quadratic * omega_hat_sq
    )
    jacobian_diagonal_dot = (
        diagonal_linear_dot * omega_hat
        + diagonal_linear * omega_hat_dot
        + diagonal_quadratic_dot * omega_hat_sq
        + diagonal_quadratic * omega_hat_sq_dot
    )

    c1_dot = d1 * angle_sq_dot
    c3_dot = d3 * angle_sq_dot
    c4_dot = d4 * angle_sq_dot
    jacobian_lower = (
        c1 * linear_hat
        + diagonal_quadratic * second_lower
        + c3 * third_lower
        + c4 * fourth_lower_remainder
    )
    jacobian_lower_dot = (
        c1_dot * linear_hat
        + c1 * linear_hat_dot
        + diagonal_quadratic_dot * second_lower
        + diagonal_quadratic * second_lower_dot
        + c3_dot * third_lower
        + c3 * third_lower_dot
        + c4_dot * fourth_lower_remainder
        + c4 * fourth_lower_remainder_dot
    )

    jacobian = jnp.block(
        [[jacobian_diagonal, zero], [jacobian_lower, jacobian_diagonal]]
    )
    jacobian_dot = jnp.block(
        [
            [jacobian_diagonal_dot, zero],
            [jacobian_lower_dot, jacobian_diagonal_dot],
        ]
    )

    theta = _rotational_strain_magnitude(xi, exp_eps)
    sinc = sine_over_angle(theta, exp_eps)
    cosc = one_minus_cosine_over_angle_squared(theta, exp_eps)
    tanc = angle_minus_sine_over_angle_cubed(theta, exp_eps)
    rotation = identity + sinc * omega_hat + cosc * omega_hat_sq
    translation = (identity + cosc * omega_hat + tanc * omega_hat_sq) @ xi[3:]
    transform = jnp.block(
        [
            [rotation, translation.reshape((3, 1))],
            [
                jnp.zeros((1, 3), dtype=xi.dtype),
                jnp.ones((1, 1), dtype=xi.dtype),
            ],
        ]
    )
    return transform, jacobian, jacobian_dot


def coadjoint(xi: Array) -> Array:
    """Return the spatial coadjoint matrix ``-ad_xi.T``.

    Dual vectors are ordered consistently with angular-first twists, i.e.
    ``[moment_x, moment_y, moment_z, force_x, force_y, force_z]``. The returned
    matrix acts on those dual vectors and is used in the dynamics terms where
    spatial inertia and wrench quantities are expressed in the same convention.

    Args:
        xi: Spatial twist with shape ``(6,)`` or ``(6, 1)`` in
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` order.

    Returns:
        Array with shape ``(6, 6)`` representing the coadjoint action on
        angular-first dual vectors.
    """
    return -small_adjoint(xi).T


def adjoint(g: Array) -> Array:
    """Return the group adjoint matrix ``Ad_g`` for an ``SE(3)`` transform.

    The adjoint maps spatial twists between frames according to the homogeneous
    transform ``g`` and the angular-first twist convention. It is constructed
    so that ``hat(adjoint(g) @ xi) == g @ hat(xi) @ inverse(g)``.

    Args:
        g: Homogeneous ``SE(3)`` transform with shape ``(4, 4)``. The rotation
            block is ``g[:3, :3]`` and the translation is ``g[:3, 3]``.

    Returns:
        Array with shape ``(6, 6)`` representing ``Ad_g`` in
        ``[omega, v]`` coordinates.
    """
    R = g[:3, :3]
    t = g[:3, 3].reshape((3, 1))
    t_hat = so3.skew(t)

    return jnp.block([[R, jnp.zeros((3, 3), dtype=g.dtype)], [t_hat @ R, R]])


def adjoint_inverse(g: Array) -> Array:
    """Return the inverse group adjoint matrix ``Ad_g^{-1}``.

    This is equivalent to ``adjoint(inverse(g))`` but avoids explicitly
    constructing the inverse homogeneous transform. The result maps spatial
    twists in the opposite direction from :func:`adjoint`.

    Args:
        g: Homogeneous ``SE(3)`` transform with shape ``(4, 4)``.

    Returns:
        Array with shape ``(6, 6)`` representing the inverse adjoint in
        angular-first coordinates.
    """
    R = g[:3, :3]
    t = g[:3, 3].reshape((3, 1))
    t_hat = so3.skew(t)
    R_inv = jnp.transpose(R)

    return jnp.block(
        [[R_inv, jnp.zeros((3, 3), dtype=g.dtype)], [-R_inv @ t_hat, R_inv]]
    )
