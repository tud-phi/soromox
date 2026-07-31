"""Lie group and Lie algebra operators for spatial rotations.

This module contains pure ``SO(3)`` / ``so(3)`` operations. Rotation vectors
use axis-angle coordinates: the vector direction is the rotation axis and its
Euclidean norm is the rotation angle in radians.
"""

__all__ = ["skew", "vee", "exp", "log"]

import jax.numpy as jnp
from jax import Array, lax

from soromox.utils._numerics import eps_for_dtype


def _rotation_magnitude(omega: Array, eps: float | Array) -> Array:
    """Return a differentiability-aware norm of a rotation vector.

    The helper returns ``norm(omega)`` outside the ``eps`` ball and exact zero
    inside it. This avoids evaluating ``sqrt(dot(omega, omega))`` at the origin
    in branches that are meant to use small-angle series expansions.

    Args:
        omega: Rotation vector with shape ``(3,)``.
        eps: Small nonnegative scalar threshold.

    Returns:
        Scalar array containing the regularized rotation angle.
    """
    theta_sq = jnp.dot(omega, omega)
    return lax.cond(
        theta_sq <= eps**2,
        lambda _: jnp.zeros((), dtype=omega.dtype),
        lambda _: jnp.sqrt(theta_sq),
        operand=None,
    )


def vee(matrix: Array) -> Array:
    """Extract the vector coordinate of the skew part of a matrix.

    This is the inverse of :func:`skew` for matrices in ``so(3)``. For a
    general matrix, it first projects onto ``so(3)`` using
    ``0.5 * (matrix - matrix.T)``. In particular,
    ``vee(skew(omega)) == omega``.

    Args:
        matrix: Matrix with shape ``(3, 3)``.

    Returns:
        Vector with shape ``(3,)`` in ``[x, y, z]`` order.
    """
    matrix = jnp.asarray(matrix)
    return 0.5 * jnp.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=matrix.dtype,
    )


def _normalize_vector(vec: Array, eps: float | Array) -> Array:
    """Normalize a vector while keeping zero vectors finite.

    Args:
        vec: Vector with shape ``(3,)``.
        eps: Small nonnegative scalar threshold. Vectors with norm less than or
            equal to ``eps`` are returned unchanged.

    Returns:
        Vector with shape ``(3,)``. Nonzero vectors are unit length; near-zero
        vectors remain finite.
    """
    norm = jnp.linalg.norm(vec)
    safe_norm = jnp.where(norm > eps, norm, jnp.ones((), dtype=vec.dtype))
    return jnp.where(norm > eps, vec / safe_norm, vec)


def _axis_near_pi(R: Array, eps: float | Array) -> Array:
    """Extract a rotation axis from a matrix whose angle is close to ``pi``.

    The usual skew-part formula divides by ``sin(theta)`` and is singular at
    ``theta = pi``. This helper instead reconstructs the principal axis from
    the largest diagonal entry of the rotation matrix. The sign is determined
    by the corresponding symmetric off-diagonal entries, which is the standard
    near-180-degree branch for the ``SO(3)`` logarithm.

    Args:
        R: Rotation matrix with shape ``(3, 3)`` whose principal angle is close
            to ``pi``.
        eps: Small nonnegative scalar used to keep denominators finite.

    Returns:
        Unit rotation axis with shape ``(3,)``.
    """

    def _x_axis(_: None) -> Array:
        x = 0.5 * jnp.sqrt(jnp.maximum(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
        denom = jnp.maximum(4.0 * x, eps)
        return jnp.array(
            [x, (R[0, 1] + R[1, 0]) / denom, (R[0, 2] + R[2, 0]) / denom],
            dtype=R.dtype,
        )

    def _y_axis(_: None) -> Array:
        y = 0.5 * jnp.sqrt(jnp.maximum(0.0, 1.0 + R[1, 1] - R[0, 0] - R[2, 2]))
        denom = jnp.maximum(4.0 * y, eps)
        return jnp.array(
            [(R[0, 1] + R[1, 0]) / denom, y, (R[1, 2] + R[2, 1]) / denom],
            dtype=R.dtype,
        )

    def _z_axis(_: None) -> Array:
        z = 0.5 * jnp.sqrt(jnp.maximum(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1]))
        denom = jnp.maximum(4.0 * z, eps)
        return jnp.array(
            [(R[0, 2] + R[2, 0]) / denom, (R[1, 2] + R[2, 1]) / denom, z],
            dtype=R.dtype,
        )

    axis = lax.cond(
        (R[0, 0] >= R[1, 1]) & (R[0, 0] >= R[2, 2]),
        _x_axis,
        lambda _: lax.cond(R[1, 1] >= R[2, 2], _y_axis, _z_axis, operand=None),
        operand=None,
    )
    return _normalize_vector(axis, eps)


def skew(vec: Array) -> Array:
    """Return the ``so(3)`` skew-symmetric cross-product matrix.

    For a vector ``a = [x, y, z]``, the returned matrix ``a_hat`` satisfies
    ``a_hat @ b == cross(a, b)`` for any Cartesian 3-vector ``b``. This is the
    matrix representation used by ``SO(3)`` and the rotational blocks of
    spatial ``SE(3)`` operators.

    Args:
        vec: Vector with shape ``(3,)`` or ``(3, 1)`` in Cartesian
            ``[x, y, z]`` order.

    Returns:
        Array with shape ``(3, 3)`` containing the skew-symmetric
        cross-product matrix.
    """
    vec = jnp.asarray(vec).reshape(-1)
    x, y, z = vec
    return jnp.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=vec.dtype,
    )


def exp(omega: Array, eps: float | Array = 1e-10) -> Array:
    """Compute the Lie-group exponential from ``so(3)`` to ``SO(3)``.

    The input is an axis-angle rotation vector. The function evaluates
    Rodrigues' formula and switches to a Taylor series near zero rotation to
    avoid singular divisions.

    Args:
        omega: Rotation vector with shape ``(3,)`` or ``(3, 1)``. Its direction
            is the rotation axis and its Euclidean norm is the angle in
            radians.
        eps: Small nonnegative scalar threshold. If ``norm(omega) <= eps``,
            the small-angle series branch is used.

    Returns:
        Rotation matrix with shape ``(3, 3)``.
    """
    omega = jnp.asarray(omega).reshape(-1)
    theta = _rotation_magnitude(omega, eps)
    omega_hat = skew(omega)
    omega_hat_sq = omega_hat @ omega_hat

    def _series(_: None) -> Array:
        return jnp.eye(3, dtype=omega.dtype) + omega_hat + 0.5 * omega_hat_sq

    def _general(theta_val: Array) -> Array:
        return (
            jnp.eye(3, dtype=omega.dtype)
            + (jnp.sin(theta_val) / theta_val) * omega_hat
            + ((1.0 - jnp.cos(theta_val)) / theta_val**2) * omega_hat_sq
        )

    return lax.cond(theta <= eps, _series, _general, theta)


def log(R: Array, eps: float | Array = 1e-10) -> Array:
    """Compute the principal Lie-group logarithm from ``SO(3)`` to ``so(3)``.

    The returned vector is the principal axis-angle coordinate of ``R``. Its
    norm is in ``[0, pi]`` and its direction is the corresponding rotation
    axis. The implementation uses the skew part of ``R`` in the regular case,
    a first-order series near identity, and a diagonal-axis reconstruction near
    ``pi`` where the skew part becomes singular.

    Args:
        R: Rotation matrix with shape ``(3, 3)``.
        eps: Small nonnegative scalar threshold used for normalization and
            small-angle handling.

    Returns:
        Rotation vector with shape ``(3,)`` whose Euclidean norm is the
        principal rotation angle in radians.
    """
    R = jnp.asarray(R)
    trace_R = jnp.trace(R)
    skew_part = R - R.T
    skew_vec = vee(skew_part)
    eps_arr = eps_for_dtype(eps, R.dtype)

    cos_theta = jnp.clip((trace_R - 1.0) * 0.5, -1.0, 1.0)
    skew_norm_sq = jnp.dot(skew_vec, skew_vec)
    sin_theta = 0.5 * lax.cond(
        skew_norm_sq <= eps_arr**2,
        lambda _: jnp.zeros((), dtype=R.dtype),
        lambda _: jnp.sqrt(skew_norm_sq),
        operand=None,
    )
    theta = jnp.arctan2(sin_theta, cos_theta)
    pi = jnp.asarray(jnp.pi, dtype=R.dtype)

    small_threshold = jnp.maximum(jnp.asarray(1e-8, dtype=R.dtype), eps_arr)
    near_pi_threshold = jnp.asarray(1e-6, dtype=R.dtype)

    def _small(_: None) -> Array:
        return 0.5 * skew_vec

    def _regular(_: None) -> Array:
        scale = theta / (2.0 * jnp.sin(theta))
        return scale * skew_vec

    def _large(_: None) -> Array:
        axis = _axis_near_pi(R, eps_arr)
        return theta * axis

    return lax.cond(
        theta <= small_threshold,
        _small,
        lambda _: lax.cond(
            jnp.abs(pi - theta) <= near_pi_threshold,
            _large,
            _regular,
            operand=None,
        ),
        operand=None,
    )
