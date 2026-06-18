"""Geodesic error utilities for planar and spatial rotations.

These functions compute shortest-path orientation errors on the appropriate
rotation group. They are intended for control and trajectory tracking where
subtracting representation coordinates directly would produce discontinuities
or long-way-around errors.
"""

__all__ = [
    "wrap_angle",
    "angle_geodesic_error",
    "rotation_matrix_geodesic_error",
    "quaternion_geodesic_error",
]

import jax.numpy as jnp
from jax import Array

from .rotations import (
    DEFAULT_ROTATION_EPS,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_rotation_vector,
    rotation_matrix_to_rotation_vector,
)


def wrap_angle(angle: Array) -> Array:
    """Wrap an angle to the principal interval ``[-pi, pi)``.

    This helper applies elementwise and is useful whenever planar angles need
    to be compared modulo ``2*pi``. The interval is half-open: both ``pi`` and
    ``-pi`` represent the same orientation, and this implementation maps that
    boundary to ``-pi``.

    Args:
        angle: Scalar or array of angles in radians.

    Returns:
        Array with the same shape as ``angle`` containing wrapped angles in
        radians.
    """
    return jnp.mod(angle + jnp.pi, 2.0 * jnp.pi) - jnp.pi


def angle_geodesic_error(theta_current: Array, theta_desired: Array) -> Array:
    """Compute the shortest-path error between two planar orientations.

    The result is the signed angular displacement that rotates the current
    orientation into the desired orientation along the shortest path on
    ``SO(2)``. Positive values follow the right-handed counter-clockwise
    convention.

    Args:
        theta_current: Current planar angle in radians. May be a scalar or an
            array of angles.
        theta_desired: Desired planar angle in radians. Must be broadcastable
            with ``theta_current``.

    Returns:
        Scalar or array with the broadcasted shape of the inputs. Values are in
        the principal interval ``[-pi, pi)``.
    """
    return wrap_angle(theta_desired - theta_current)


def rotation_matrix_geodesic_error(
    R_current: Array,
    R_desired: Array,
    eps: float = DEFAULT_ROTATION_EPS,
) -> Array:
    """Compute the geodesic orientation error between two rotation matrices.

    The error is the ``SO(3)`` logarithm of the relative rotation
    ``R_desired @ R_current.T``. It is a rotation vector representing the
    shortest-path correction from the current orientation to the desired
    orientation. This is the appropriate orientation error for operational
    space control; naive subtraction of rotation-vector or quaternion
    coordinates is representation-dependent and can choose the long path.

    Args:
        R_current: Current rotation matrix with shape ``(3, 3)``.
        R_desired: Desired rotation matrix with shape ``(3, 3)``.
        eps: Small nonnegative scalar threshold passed to the ``SO(3)``
            logarithm.

    Returns:
        Rotation vector with shape ``(3,)``. Its norm is the shortest
        correction angle in radians and its direction is the correction axis.
    """
    R_error = R_desired @ R_current.T
    return rotation_matrix_to_rotation_vector(R_error, eps=eps)


def quaternion_geodesic_error(
    q_current: Array,
    q_desired: Array,
    eps: float = DEFAULT_ROTATION_EPS,
) -> Array:
    """Compute the geodesic orientation error between two quaternions.

    Quaternions use the scalar-first Hamilton convention
    ``[qw, qx, qy, qz]``. The relative quaternion is
    ``q_desired * conjugate(q_current)`` and is flipped when needed so the
    scalar component is nonnegative. That resolves the antipodal ambiguity and
    returns the shortest-path rotation vector.

    Args:
        q_current: Current quaternion with shape ``(4,)`` in scalar-first
            Hamilton order.
        q_desired: Desired quaternion with shape ``(4,)`` in scalar-first
            Hamilton order.
        eps: Small nonnegative scalar threshold passed to quaternion-to-vector
            conversion.

    Returns:
        Rotation vector with shape ``(3,)`` representing the shortest-path
        correction from ``q_current`` to ``q_desired``.
    """
    q_current_inv = quaternion_conjugate(q_current)
    q_error = quaternion_multiply(q_desired, q_current_inv)
    q_error = jnp.where(q_error[0] < 0.0, -q_error, q_error)
    return quaternion_to_rotation_vector(q_error, eps=eps)
