__all__ = [
    "hat",
    "exp",
    "log",
    "small_adjoint",
    "coadjoint",
    "adjoint",
    "adjoint_inverse",
]

import jax.numpy as jnp
from jax import Array, lax

from . import so3
from .jacobian_coefficients import (
    angle_minus_sine_over_angle_cubed,
    inverse_left_jacobian_quadratic_coefficient,
    one_minus_cosine_over_angle_squared,
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
