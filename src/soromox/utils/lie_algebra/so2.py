"""Lie group and Lie algebra operators for planar rotations.

This module contains pure ``SO(2)`` / ``so(2)`` operations. It does not handle
translations or pose-coordinate encodings; use ``se2`` for planar rigid-body
transforms and ``soromox.utils.geometry.poses`` for direct pose coordinates.
"""

__all__ = ["skew", "exp", "log"]

import jax.numpy as jnp
from jax import Array, lax


def _skew_basis(dtype) -> Array:
    """Return the canonical ``so(2)`` basis matrix with the requested dtype.

    The returned matrix is ``J = [[0, -1], [1, 0]]``. It represents a unit
    positive right-handed rotation about the out-of-plane z-axis and is the
    generator used by :func:`skew` and the planar ``SE(2)`` operators.

    Args:
        dtype: JAX dtype to use for the floating-point entries.

    Returns:
        Array with shape ``(2, 2)`` and entries matching the canonical
        right-handed planar rotation convention.
    """
    return jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=dtype)


def skew(angle: Array) -> Array:
    """Return the ``so(2)`` skew-symmetric matrix for a scalar angle.

    The planar Lie algebra is one-dimensional. This function maps the scalar
    coordinate ``theta`` to ``theta * J`` where
    ``J = [[0, -1], [1, 0]]``. Positive ``theta`` follows the standard
    right-handed convention for counter-clockwise rotation in the xy-plane.

    Args:
        angle: Scalar or array-like value broadcastable to a scalar. The first
            flattened entry is interpreted as the angular coordinate ``theta``
            in radians.

    Returns:
        Array with shape ``(2, 2)`` containing the skew-symmetric matrix
        ``theta * J``.
    """
    theta = jnp.asarray(angle).reshape(-1)[0]
    return theta * _skew_basis(theta.dtype)


def exp(angle: Array) -> Array:
    """Compute the Lie-group exponential from ``so(2)`` to ``SO(2)``.

    For ``SO(2)``, the exponential map is the standard planar rotation matrix
    ``R(theta)``. The input is an angular Lie-algebra coordinate, not a direct
    planar pose; translations are handled by ``se2.exp`` or by geometry pose
    helpers depending on the coordinate convention.

    Args:
        angle: Scalar or array-like value broadcastable to a scalar. The first
            flattened entry is interpreted as the rotation angle ``theta`` in
            radians.

    Returns:
        Rotation matrix with shape ``(2, 2)``.
    """
    theta = jnp.asarray(angle).reshape(-1)[0]
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    return jnp.array(
        [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
        dtype=theta.dtype,
    )


def log(R: Array, eps: float | Array = 1e-10) -> Array:
    """Compute the principal Lie-group logarithm from ``SO(2)`` to ``so(2)``.

    The returned scalar is the principal angle obtained from ``atan2`` and
    therefore lies in the half-open wrapping convention induced by
    ``atan2``. Very small angles are regularized to exact zero to keep
    downstream comparisons and control errors stable near identity.

    Args:
        R: Rotation matrix with shape ``(2, 2)``. The angle is read from
            ``atan2(R[1, 0], R[0, 0])``.
        eps: Small nonnegative scalar threshold. If the recovered angle has
            magnitude less than ``eps``, the function returns exact zero.

    Returns:
        Scalar array containing the principal rotation angle in radians.
    """
    R = jnp.asarray(R)
    theta = jnp.arctan2(R[1, 0], R[0, 0])
    return lax.cond(
        jnp.abs(theta) < eps,
        lambda _: jnp.zeros((), dtype=R.dtype),
        lambda _: theta,
        operand=None,
    )
