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

from soromox.utils.rotations import (
    RotationRepresentation,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_rotation_vector,
    rotation_6d_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)


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

    For rotation representations like QUATERNION and ROTATION_MATRIX_6D, the
    velocity (angular velocity) is NOT the simple time derivative of the
    orientation. This function creates a velocity function that:
    - Computes linear velocity as the time derivative of position
    - Computes angular velocity properly from the orientation derivative

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

    Notes
    -----
    For ROTATION_VECTOR representation, this uses the small-angle approximation
    where angular_velocity ≈ d(rotation_vector)/dt. This is exact for infinitesimal
    changes and accurate for smooth trajectories.

    For QUATERNION and ROTATION_MATRIX_6D, this properly converts the orientation
    derivative to angular velocity using the geometric relationship.

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

                # Convert orientation derivative to angular velocity
                if rotation_representation == RotationRepresentation.ROTATION_VECTOR:
                    # For rotation vector: ω ≈ d(rotation_vector)/dt
                    # This is the small-angle approximation, exact for
                    # infinitesimal changes
                    angular_vel = orient_dot

                elif rotation_representation == RotationRepresentation.QUATERNION:
                    # For scalar-first quaternion q = [qw, qx, qy, qz]:
                    # ω = 2 * Im(q^{-1} ⊗ dq/dt)
                    # where q^{-1} = q* (conjugate) for unit quaternions
                    q_conj = quaternion_conjugate(orient)
                    # Note: orient_dot is dq/dt, not a unit quaternion
                    omega_quat = 2.0 * quaternion_multiply(q_conj, orient_dot)
                    # Angular velocity is the imaginary (vector) part
                    angular_vel = omega_quat[1:4]

                elif (
                    rotation_representation == RotationRepresentation.ROTATION_MATRIX_6D
                ):
                    # For rotation_6d, we need to:
                    # 1. Convert to rotation matrix R
                    # 2. Compute dR/dt from d(rotation_6d)/dt
                    # 3. Compute ω from R^T @ dR/dt (skew-symmetric)
                    #
                    # We use autodiff through the full conversion:

                    def orient_to_rotvec(orient_6d):
                        R = rotation_6d_to_rotation_matrix(orient_6d)
                        quat = rotation_matrix_to_quaternion(R)
                        return quaternion_to_rotation_vector(quat)

                    # Compute Jacobian of rotation_vector w.r.t. rotation_6d
                    J_orient = jax.jacfwd(orient_to_rotvec)(orient)
                    angular_vel = J_orient @ orient_dot

                else:
                    raise ValueError(
                        f"Unknown rotation representation: {rotation_representation}"
                    )

                # Combine angular velocity and linear velocity
                point_vel = jnp.concatenate([angular_vel, pos_dot])
                velocities.append(point_vel)

            return jnp.stack(velocities).flatten()

    return twist_fn
