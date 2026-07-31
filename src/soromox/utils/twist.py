"""
Twist (velocity) utilities for converting pose callables to twist callables.

This module provides utilities for deriving velocity (twist) functions from
pose functions, properly handling different rotation representations.
"""

__all__ = ["twist_callable_from_pose_callable_factory"]

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

from soromox.utils.geometry.rotations import (
    RotationRepresentation,
    quaternion_to_rotation_matrix,
    rotation_6d_to_rotation_matrix,
    rotation_vector_to_rotation_matrix,
)
from soromox.utils.lie_algebra import so3


def twist_callable_from_pose_callable_factory(
    pose_fn: Callable[[Array], Array],
    n_points: int,
    n_pose_dim: int,
    n_orientation_dim: int,
    rotation_representation: RotationRepresentation | None = None,
    is_planar: bool = False,
) -> Callable[[Array], Array]:
    """
    Create a twist (velocity) callable from a pose callable, properly handling rotation.

    For 3D rotation representations, angular velocity is NOT generally the
    simple time derivative of the orientation coordinates. This function:
    - Computes linear velocity as the time derivative of position
    - Computes inertial-frame angular velocity as ``vee(R_dot @ R.T)``

    The returned velocity is in the FULL velocity space (not task-selected),
    with dimension (n_points * n_velocity_dim,).

    Parameters
    ----------
    pose_fn : Callable[[Array], Array]
        A function that takes scalar time t and returns the full pose of shape
        (n_points * n_pose_dim,).
    n_points : int
        Number of points in the pose vector.
    n_pose_dim : int
        Dimension of pose per point (e.g., 6 for ROTATION_VECTOR, 7 for QUATERNION,
        9 for ROTATION_MATRIX_6D, 3 for planar).
    n_orientation_dim : int
        Dimension of orientation component (e.g., 3 for ROTATION_VECTOR, 4 for
        QUATERNION, 6 for ROTATION_MATRIX_6D, 1 for planar).
    rotation_representation : RotationRepresentation | None
        The rotation representation used in the pose. If None, assumes planar
        (scalar angle). For 3D robots, this determines how to convert orientation
        derivative to angular velocity.
    is_planar : bool
        If True, the pose is for a planar robot with pose = [theta, x, y].
        In this case, velocity = d(pose)/dt directly.

    Returns
    -------
    Callable[[Array], Array]
        A function that takes scalar time t and returns the full twist (velocity)
        of shape (n_points * n_velocity_dim,).

    The inertial-frame convention matches the operational-space Jacobians used
    throughout SoRoMoX. Differentiating the rotation matrix also keeps the
    conversion exact for rotation vectors, quaternions, and the continuous 6D
    representation, including trajectories whose rotation axis changes.

    Examples
    --------
    >>> def x_des_fn(t):
    ...     # Returns full pose [rot, pos] as function of time for 1 point
    ...     return jnp.array([0., 0., t * 0.1, 0., 0., 0.2 + 0.01 * t])
    >>> twist_fn = twist_callable_from_pose_callable_factory(
    ...     x_des_fn, n_points=1, n_pose_dim=6, n_orientation_dim=3,
    ...     rotation_representation=RotationRepresentation.ROTATION_VECTOR
    ... )
    >>> twist = twist_fn(1.0)  # Full velocity at t=1.0
    """

    def twist_fn(t: Array) -> Array:
        # Compute pose derivative using JAX autodiff
        pose_dot = jax.jacfwd(pose_fn)(t)

        if is_planar or rotation_representation is None:
            # For planar robots, velocity = d(pose)/dt directly
            # pose = [theta, x, y], velocity = [omega_z, v_x, v_y]
            return pose_dot
        else:
            # For 3D robots, we need to convert orientation derivative
            # to angular velocity

            # Reshape to per-point
            pose_dot_reshaped = pose_dot.reshape(n_points, n_pose_dim)
            pose = pose_fn(t).reshape(n_points, n_pose_dim)

            # Process each point
            velocities = []
            for i in range(n_points):
                orient = pose[i, :n_orientation_dim]
                orient_dot = pose_dot_reshaped[i, :n_orientation_dim]
                pos_dot = pose_dot_reshaped[i, n_orientation_dim:]

                # Convert the representation and its tangent to a rotation matrix
                # and R_dot. The operational-space Jacobian uses inertial-frame
                # angular velocity, omega_hat = R_dot @ R.T.
                if rotation_representation == RotationRepresentation.ROTATION_VECTOR:
                    orientation_to_rotation_matrix = rotation_vector_to_rotation_matrix
                elif rotation_representation == RotationRepresentation.QUATERNION:
                    orientation_to_rotation_matrix = quaternion_to_rotation_matrix
                elif (
                    rotation_representation == RotationRepresentation.ROTATION_MATRIX_6D
                ):
                    orientation_to_rotation_matrix = rotation_6d_to_rotation_matrix
                else:
                    raise ValueError(
                        f"Unknown rotation representation: {rotation_representation}"
                    )

                R, R_dot = jax.jvp(
                    orientation_to_rotation_matrix,
                    (orient,),
                    (orient_dot,),
                )
                omega_hat = R_dot @ R.T
                angular_vel = so3.vee(omega_hat)

                # Combine angular velocity and linear velocity
                point_vel = jnp.concatenate([angular_vel, pos_dot])
                velocities.append(point_vel)

            return jnp.stack(velocities).flatten()

    return twist_fn
