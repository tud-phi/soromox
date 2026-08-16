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
from jax import Array

from . import so2
from .jacobian_coefficients import (
    half_angle_cotangent,
    one_minus_cosine_over_angle_squared,
    sine_over_angle,
)


def hat(xi: Array) -> Array:
    """Return the homogeneous matrix representation of an ``se(2)`` twist.

    Twists use angular-first coordinates ``xi = [theta, v_x, v_y]``. The first
    entry is the scalar angular component and the last two entries are the
    translational twist coordinates. The resulting matrix is
    ``[[so2.skew(theta), v], [0, 0, 0]]`` and is suitable for use in the
    matrix exponential.

    Args:
        xi: Planar twist with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, v_x, v_y]`` order.

    Returns:
        Array with shape ``(3, 3)`` representing the same algebra element in
        homogeneous matrix form.
    """
    xi = jnp.asarray(xi).reshape(-1)

    omega = xi[0]
    v = xi[1:].reshape((2, 1))

    return jnp.block(
        [
            [so2.skew(omega), v],
            [jnp.zeros((1, 2), dtype=xi.dtype), jnp.zeros((1, 1), dtype=xi.dtype)],
        ]
    )


def exp(xi: Array, eps: float | Array) -> Array:
    """Compute the Lie-group exponential from ``se(2)`` to ``SE(2)``.

    This is the matrix exponential of :func:`hat`. The translational part is
    integrated through the ``SE(2)`` left Jacobian, so the input translation is
    a twist coordinate, not a direct pose translation. Use
    ``poses.planar_pose_to_transform`` when the input is a direct
    ``[theta, x, y]`` pose.

    Args:
        xi: Planar twist with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, v_x, v_y]`` order.
        eps: Minimum small-angle series threshold. A dtype-aware threshold may
            select the series over a larger interval to preserve Hessian
            accuracy near zero rotation.

    Returns:
        Homogeneous ``SE(2)`` transform with shape ``(3, 3)``.
    """
    R, p = _exp_rotation_and_translation(xi, eps)

    return jnp.block(
        [
            [R, p.reshape((2, 1))],
            [jnp.zeros((1, 2), dtype=R.dtype), jnp.ones((1, 1), dtype=R.dtype)],
        ]
    )


def _exp_rotation_and_translation(xi: Array, eps: float | Array) -> tuple[Array, Array]:
    """Return the stable rotation and translation blocks of ``exp(xi)``."""
    xi = jnp.asarray(xi).reshape(-1)
    return so2.exp(xi[0]), _exp_translation(xi, eps)


def _exp_translation(xi: Array, eps: float | Array) -> Array:
    """Return the stable translation block of ``exp(xi)``."""
    xi = jnp.asarray(xi).reshape(-1)
    theta = xi[0]
    v = xi[1:]
    sinc = sine_over_angle(theta, eps)
    theta_cosc = theta * one_minus_cosine_over_angle_squared(theta, eps)
    return jnp.stack([sinc * v[0] - theta_cosc * v[1], theta_cosc * v[0] + sinc * v[1]])


def log(g: Array, eps: float | Array) -> Array:
    """Compute the Lie-group logarithm from ``SE(2)`` to ``se(2)``.

    This function is the inverse of :func:`exp` for regular planar transforms:
    it returns twist coordinates, not direct pose coordinates. In particular,
    the returned translational coordinates are obtained by applying the inverse
    ``SE(2)`` left Jacobian to the transform translation. Use
    ``poses.planar_pose_from_transform`` when the desired output is direct
    pose coordinates ``[theta, x, y]``.

    Args:
        g: Homogeneous ``SE(2)`` transform with shape ``(3, 3)``. The rotation
            is read from ``g[:2, :2]`` and the translation from ``g[:2, 2]``.
        eps: Small positive scalar threshold forwarded to :func:`so2.log`.
            Recovered angles with magnitude below this threshold are rounded
            to zero.

    Returns:
        Array with shape ``(3,)`` containing twist coordinates
        ``[theta, v_x, v_y]``.
    """
    R = g[:2, :2]
    p = g[:2, 2].reshape((2, 1))

    theta = so2.log(R, eps=eps)
    J = so2.skew(jnp.ones((), dtype=R.dtype))
    half_theta = 0.5 * theta
    half_theta_cotangent = half_angle_cotangent(theta)
    V_inv = half_theta_cotangent * jnp.eye(2, dtype=R.dtype) - half_theta * J
    v = V_inv @ p

    return jnp.concatenate([jnp.array([theta], dtype=R.dtype), v.reshape(-1)])


def small_adjoint(xi: Array) -> Array:
    """Return the Lie algebra adjoint matrix ``ad_xi``.

    The returned matrix implements the planar Lie bracket in angular-first
    coordinates: ``small_adjoint(xi) @ eta == [xi, eta]`` for twists
    ``xi`` and ``eta`` written as ``[theta, v_x, v_y]``.

    Args:
        xi: Planar twist with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, v_x, v_y]`` order.

    Returns:
        Array with shape ``(3, 3)`` representing ``ad_xi``.
    """
    xi = jnp.asarray(xi).reshape(-1)

    omega = xi[0]
    v = xi[1:].reshape((2, 1))
    J = so2.skew(jnp.ones((), dtype=xi.dtype))

    return jnp.concatenate(
        [
            jnp.zeros((1, 3), dtype=xi.dtype),
            jnp.concatenate([-J @ v, omega * J], axis=1),
        ]
    )


def coadjoint(xi: Array) -> Array:
    """Return the planar coadjoint matrix used for dual wrench dynamics.

    Vectors are angular-first, so dual quantities are ordered as
    ``[moment_z, force_x, force_y]``. With ``xi = [theta, v_x, v_y]`` this
    function returns the package's planar coadjoint convention:
    ``[[0, v_y, -v_x], [0, 0, -theta], [0, theta, 0]]``. This matches the
    existing dynamics convention for planar mass and force terms.

    Args:
        xi: Planar twist with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, v_x, v_y]`` order.

    Returns:
        Array with shape ``(3, 3)`` acting on angular-first dual vectors.
    """
    xi = jnp.asarray(xi).reshape(-1)

    omega = xi[0]
    v = xi[1:].reshape((2, 1))
    J = so2.skew(jnp.ones((), dtype=xi.dtype))

    return jnp.concatenate(
        [
            jnp.zeros((3, 1), dtype=xi.dtype),
            jnp.concatenate([v.T @ J, omega * J], axis=0),
        ],
        axis=1,
    )


def adjoint(g: Array) -> Array:
    """Return the group adjoint matrix ``Ad_g`` for an ``SE(2)`` transform.

    The adjoint maps planar twists between frames according to the homogeneous
    transform ``g`` and the angular-first twist convention. It is constructed
    so that ``hat(adjoint(g) @ xi) == g @ hat(xi) @ inverse(g)``.

    Args:
        g: Homogeneous ``SE(2)`` transform with shape ``(3, 3)``. The rotation
            block is ``g[:2, :2]`` and the translation is ``g[:2, 2]``.

    Returns:
        Array with shape ``(3, 3)`` representing ``Ad_g`` in
        ``[theta, v_x, v_y]`` coordinates.
    """
    R = g[:2, :2]
    t = g[:2, 2].reshape((2, 1))
    J = so2.skew(jnp.ones((), dtype=g.dtype))

    return jnp.concatenate(
        [
            jnp.concatenate(
                [jnp.ones((1, 1), dtype=g.dtype), jnp.zeros((1, 2), dtype=g.dtype)],
                axis=1,
            ),
            jnp.concatenate([-J @ t, R], axis=1),
        ]
    )


def adjoint_inverse(g: Array) -> Array:
    """Return the inverse group adjoint matrix ``Ad_g^{-1}``.

    This is equivalent to ``adjoint(inverse(g))`` but avoids explicitly
    constructing the inverse homogeneous transform. The result maps twists in
    the opposite direction from :func:`adjoint`.

    Args:
        g: Homogeneous ``SE(2)`` transform with shape ``(3, 3)``.

    Returns:
        Array with shape ``(3, 3)`` representing the inverse adjoint in
        angular-first coordinates.
    """
    Ad = adjoint(g)

    R = Ad[1:, 1:]
    mJt = Ad[1:, 0].reshape(-1, 1)
    R_inv = jnp.transpose(R)

    return jnp.concatenate(
        [
            jnp.concatenate(
                [jnp.ones((1, 1), dtype=g.dtype), jnp.zeros((1, 2), dtype=g.dtype)],
                axis=1,
            ),
            jnp.concatenate([-R_inv @ mJt, R_inv], axis=1),
        ]
    )
