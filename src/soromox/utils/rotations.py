"""
Rotation utilities for converting between different rotation representations.

This module provides numerically stable conversions between rotation matrices,
quaternions, and rotation vectors (axis-angle representation). The implementations
are designed to handle edge cases such as small rotations (θ ≈ 0) and near-180-degree
rotations (θ ≈ π) correctly.

All functions are JAX-compatible and can be used with jit, vmap, and grad.

Quaternion Convention:
    This module uses the [x, y, z, w] convention for quaternions, where w is
    the scalar part and [x, y, z] is the vector part. The quaternion represents
    a rotation by angle θ around unit axis n as:
        q = [sin(θ/2) * n_x, sin(θ/2) * n_y, sin(θ/2) * n_z, cos(θ/2)]

Rotation Representations:
    This module supports multiple rotation representations via the RotationRepresentation
    enum. Each has different properties suitable for different use cases:

    - ROTATION_VECTOR (3D): Compact, axis-angle representation. The magnitude is the
      rotation angle and the direction is the rotation axis.
    - QUATERNION (4D): Unit quaternion representation. Good for interpolation and
      avoiding gimbal lock, but has antipodal ambiguity (q and -q represent same rotation).
    - ROTATION_MATRIX_6D (6D): Continuous representation using first two columns of
      the rotation matrix (Zhou et al. 2019). Excellent for neural networks as it has
      no discontinuities, but not suitable for direct error computation.

Geometric Orientation Errors:
    For control applications (especially PID), the orientation error should be
    computed as the geometric (geodesic) error, not as a naive difference of
    representations. The geometric error is:
        e_orientation = log(R_des @ R_current^T)
    This gives the rotation vector representing the shortest-path rotation from
    the current orientation to the desired orientation.
"""

__all__ = [
    # Enum
    "RotationRepresentation",
    # Conversion functions
    "rotation_matrix_to_quaternion",
    "quaternion_to_rotation_matrix",
    "quaternion_to_rotation_vector",
    "rotation_vector_to_quaternion",
    "rotation_matrix_to_rotation_vector",
    "rotation_vector_to_rotation_matrix",
    # 6D continuous representation
    "rotation_matrix_to_6d",
    "rotation_6d_to_rotation_matrix",
    # Quaternion operations
    "quaternion_multiply",
    "quaternion_conjugate",
    # Geometric error computation
    "rotation_matrix_error",
    "rotation_quat_error",
    "angle_error",
    "wrap_angle",
]

from enum import Enum

import jax.numpy as jnp
from jax import Array


class RotationRepresentation(Enum):
    """
    Enumeration of supported rotation representations.

    Each representation has different properties and is suitable for different use cases:

    - ROTATION_VECTOR: 3D representation where magnitude = angle, direction = axis.
      Compact but has discontinuity at angle = ±π.
    - QUATERNION: 4D unit quaternion representation. Good for interpolation but
      has antipodal ambiguity.
    - ROTATION_MATRIX_6D: 6D continuous representation (first two columns of rotation
      matrix). Continuous everywhere, ideal for neural networks.

    Note:
        For computing orientation errors in control applications, the geometric error
        should always be used regardless of the representation choice. See
        `rotation_matrix_error` for details.
    """

    ROTATION_VECTOR = "rotation_vector"
    QUATERNION = "quaternion"
    ROTATION_MATRIX_6D = "rotation_matrix_6d"


# Default epsilon for numerical stability in rotation conversions
DEFAULT_ROTATION_EPS = 1e-10


def rotation_matrix_to_quaternion(R: Array, eps: float = DEFAULT_ROTATION_EPS) -> Array:
    """
    Convert a rotation matrix to a unit quaternion.

    Uses Shepperd's method, which is numerically stable for all rotation
    angles including near-180-degree rotations (θ ≈ π).

    Args:
        R: Rotation matrix of shape (3, 3).
        eps: Small value to avoid division by zero. Default is 1e-10.

    Returns:
        q: Unit quaternion [x, y, z, w] of shape (4,).

    Example:
        >>> import jax.numpy as jnp
        >>> from soromox.utils.rotations import rotation_matrix_to_quaternion
        >>> R = jnp.eye(3)  # Identity rotation
        >>> q = rotation_matrix_to_quaternion(R)
        >>> # q ≈ [0, 0, 0, 1] (identity quaternion)

    References:
        Shepperd, S. W. (1978). Quaternion from rotation matrix.
        Journal of Guidance and Control, 1(3), 223-224.
    """
    trace = jnp.trace(R)

    # Compute all possible quaternion reconstructions
    # Case 1: trace > 0 (best for small rotation angles)
    s1 = jnp.sqrt(jnp.maximum(trace + 1.0, 0.0)) * 2  # s = 4 * q_w
    q_w1 = 0.25 * s1
    q_x1 = (R[2, 1] - R[1, 2]) / jnp.maximum(s1, eps)
    q_y1 = (R[0, 2] - R[2, 0]) / jnp.maximum(s1, eps)
    q_z1 = (R[1, 0] - R[0, 1]) / jnp.maximum(s1, eps)
    quat1 = jnp.array([q_x1, q_y1, q_z1, q_w1])

    # Case 2: R[0,0] is largest diagonal (best for rotations around x-axis)
    s2 = jnp.sqrt(jnp.maximum(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 0.0)) * 2
    q_w2 = (R[2, 1] - R[1, 2]) / jnp.maximum(s2, eps)
    q_x2 = 0.25 * s2
    q_y2 = (R[0, 1] + R[1, 0]) / jnp.maximum(s2, eps)
    q_z2 = (R[0, 2] + R[2, 0]) / jnp.maximum(s2, eps)
    quat2 = jnp.array([q_x2, q_y2, q_z2, q_w2])

    # Case 3: R[1,1] is largest diagonal (best for rotations around y-axis)
    s3 = jnp.sqrt(jnp.maximum(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 0.0)) * 2
    q_w3 = (R[0, 2] - R[2, 0]) / jnp.maximum(s3, eps)
    q_x3 = (R[0, 1] + R[1, 0]) / jnp.maximum(s3, eps)
    q_y3 = 0.25 * s3
    q_z3 = (R[1, 2] + R[2, 1]) / jnp.maximum(s3, eps)
    quat3 = jnp.array([q_x3, q_y3, q_z3, q_w3])

    # Case 4: R[2,2] is largest diagonal (best for rotations around z-axis)
    s4 = jnp.sqrt(jnp.maximum(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 0.0)) * 2
    q_w4 = (R[1, 0] - R[0, 1]) / jnp.maximum(s4, eps)
    q_x4 = (R[0, 2] + R[2, 0]) / jnp.maximum(s4, eps)
    q_y4 = (R[1, 2] + R[2, 1]) / jnp.maximum(s4, eps)
    q_z4 = 0.25 * s4
    quat4 = jnp.array([q_x4, q_y4, q_z4, q_w4])

    # Select the most numerically stable case based on largest component
    quat = jnp.where(
        trace > 0,
        quat1,
        jnp.where(
            (R[0, 0] > R[1, 1]) & (R[0, 0] > R[2, 2]),
            quat2,
            jnp.where(R[1, 1] > R[2, 2], quat3, quat4),
        ),
    )

    # Ensure canonical form with positive scalar part (w >= 0)
    quat = jnp.where(quat[3] < 0, -quat, quat)

    # Normalize to unit quaternion
    quat = quat / jnp.linalg.norm(quat)

    return quat


def quaternion_to_rotation_matrix(quat: Array) -> Array:
    """
    Convert a unit quaternion to a rotation matrix.

    Args:
        quat: Unit quaternion [x, y, z, w] of shape (4,).

    Returns:
        R: Rotation matrix of shape (3, 3).

    Example:
        >>> import jax.numpy as jnp
        >>> from soromox.utils.rotations import quaternion_to_rotation_matrix
        >>> q = jnp.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
        >>> R = quaternion_to_rotation_matrix(q)
        >>> # R ≈ eye(3) (identity rotation matrix)

    Note:
        The quaternion should be normalized. If not, the output rotation
        matrix may not be orthonormal.
    """
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]

    # Pre-compute squared terms
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w

    # Pre-compute products
    xy, xz, xw = x * y, x * z, x * w
    yz, yw = y * z, y * w
    zw = z * w

    # Construct rotation matrix
    R = jnp.array(
        [
            [w2 + x2 - y2 - z2, 2 * (xy - zw), 2 * (xz + yw)],
            [2 * (xy + zw), w2 - x2 + y2 - z2, 2 * (yz - xw)],
            [2 * (xz - yw), 2 * (yz + xw), w2 - x2 - y2 + z2],
        ]
    )

    return R


def quaternion_to_rotation_vector(
    quat: Array, eps: float = DEFAULT_ROTATION_EPS
) -> Array:
    """
    Convert a unit quaternion to a rotation vector (axis-angle representation).

    This correctly handles both small rotations (θ ≈ 0) and near-180-degree
    rotations (θ ≈ π), avoiding the singularity that affects direct
    rotation matrix to rotation vector conversion.

    The rotation vector ω has magnitude equal to the rotation angle θ and
    direction equal to the rotation axis. For a rotation by angle θ around
    unit axis n, the rotation vector is ω = θ * n.

    Args:
        quat: Unit quaternion [x, y, z, w] of shape (4,).
        eps: Small value to avoid division by zero. Default is 1e-10.

    Returns:
        omega: Rotation vector of shape (3,).

    Example:
        >>> import jax.numpy as jnp
        >>> from soromox.utils.rotations import quaternion_to_rotation_vector
        >>> # 90-degree rotation around z-axis
        >>> q = jnp.array([0, 0, jnp.sin(jnp.pi/4), jnp.cos(jnp.pi/4)])
        >>> omega = quaternion_to_rotation_vector(q)
        >>> # omega ≈ [0, 0, π/2]
    """
    # Extract quaternion components: q = [x, y, z, w]
    q_vec = quat[:3]  # [x, y, z]
    q_w = quat[3]  # w

    # Compute the rotation angle using atan2 for numerical stability
    q_vec_norm = jnp.linalg.norm(q_vec)
    half_angle = jnp.arctan2(q_vec_norm, q_w)
    angle = 2.0 * half_angle

    # Compute the rotation vector: omega = angle * axis
    # For small rotations: angle ≈ 2 * q_vec_norm, so scale ≈ 2
    # For general rotations: omega = (angle / q_vec_norm) * q_vec
    scale = jnp.where(
        q_vec_norm > eps,
        angle / q_vec_norm,
        2.0,  # Small angle approximation: angle ≈ 2 * ||q_vec||
    )

    omega = q_vec * scale
    return omega


def rotation_vector_to_quaternion(
    omega: Array, eps: float = DEFAULT_ROTATION_EPS
) -> Array:
    """
    Convert a rotation vector (axis-angle representation) to a unit quaternion.

    The rotation vector ω has magnitude equal to the rotation angle θ and
    direction equal to the rotation axis. For a rotation by angle θ around
    unit axis n, the rotation vector is ω = θ * n.

    Args:
        omega: Rotation vector of shape (3,).
        eps: Small value for numerical stability at small angles. Default is 1e-10.

    Returns:
        q: Unit quaternion [x, y, z, w] of shape (4,).

    Example:
        >>> import jax.numpy as jnp
        >>> from soromox.utils.rotations import rotation_vector_to_quaternion
        >>> # 90-degree rotation around z-axis
        >>> omega = jnp.array([0.0, 0.0, jnp.pi / 2])
        >>> q = rotation_vector_to_quaternion(omega)
        >>> # q ≈ [0, 0, sin(π/4), cos(π/4)]
    """
    angle = jnp.linalg.norm(omega)
    half_angle = angle / 2.0

    # For small angles, use Taylor expansion: sin(θ/2) ≈ θ/2, cos(θ/2) ≈ 1
    # For general angles: q_vec = sin(θ/2) * (ω / ||ω||) = sin(θ/2) / ||ω|| * ω
    scale = jnp.where(
        angle > eps,
        jnp.sin(half_angle) / angle,
        0.5,  # Small angle approximation: sin(θ/2) / θ ≈ 0.5
    )

    q_vec = omega * scale
    q_w = jnp.where(
        angle > eps,
        jnp.cos(half_angle),
        1.0,  # Small angle approximation: cos(θ/2) ≈ 1
    )

    quat = jnp.array([q_vec[0], q_vec[1], q_vec[2], q_w])

    # Normalize to ensure unit quaternion (handles numerical errors)
    quat = quat / jnp.linalg.norm(quat)

    return quat


def rotation_matrix_to_rotation_vector(
    R: Array, eps: float = DEFAULT_ROTATION_EPS
) -> Array:
    """
    Convert a rotation matrix to a rotation vector (axis-angle representation).

    This is a convenience function that combines rotation_matrix_to_quaternion
    and quaternion_to_rotation_vector. It correctly handles both small rotations
    (θ ≈ 0) and near-180-degree rotations (θ ≈ π).

    The rotation vector ω has magnitude equal to the rotation angle θ and
    direction equal to the rotation axis. For a rotation by angle θ around
    unit axis n, the rotation vector is ω = θ * n.

    Args:
        R: Rotation matrix of shape (3, 3).
        eps: Small value to avoid division by zero. Default is 1e-10.

    Returns:
        omega: Rotation vector of shape (3,).

    Example:
        >>> import jax.numpy as jnp
        >>> from soromox.utils.rotations import rotation_matrix_to_rotation_vector
        >>> # 180-degree rotation around z-axis
        >>> R = jnp.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1.0]])
        >>> omega = rotation_matrix_to_rotation_vector(R)
        >>> # omega ≈ [0, 0, π]

    Note:
        This function uses quaternions as an intermediate representation to
        avoid the singularity at θ = π that affects direct rotation matrix
        to rotation vector conversion methods.
    """
    quat = rotation_matrix_to_quaternion(R, eps=eps)
    return quaternion_to_rotation_vector(quat, eps=eps)


def rotation_vector_to_rotation_matrix(
    omega: Array, eps: float = DEFAULT_ROTATION_EPS
) -> Array:
    """
    Convert a rotation vector (axis-angle representation) to a rotation matrix.

    This is a convenience function that combines rotation_vector_to_quaternion
    and quaternion_to_rotation_matrix. It correctly handles both small rotations
    (θ ≈ 0) and all other rotation angles.

    The rotation vector ω has magnitude equal to the rotation angle θ and
    direction equal to the rotation axis. For a rotation by angle θ around
    unit axis n, the rotation vector is ω = θ * n.

    Args:
        omega: Rotation vector of shape (3,).
        eps: Small value for numerical stability at small angles. Default is 1e-10.

    Returns:
        R: Rotation matrix of shape (3, 3).

    Example:
        >>> import jax.numpy as jnp
        >>> from soromox.utils.rotations import rotation_vector_to_rotation_matrix
        >>> # 90-degree rotation around z-axis
        >>> omega = jnp.array([0.0, 0.0, jnp.pi / 2])
        >>> R = rotation_vector_to_rotation_matrix(omega)
        >>> # R is the rotation matrix for 90° around z

    Note:
        This function uses quaternions as an intermediate representation.
        Alternatively, you can use Rodrigues' formula directly, which may
        be slightly more efficient for this particular conversion.
    """
    quat = rotation_vector_to_quaternion(omega, eps=eps)
    return quaternion_to_rotation_matrix(quat)


# =============================================================================
# 6D Continuous Representation (Zhou et al. 2019)
# =============================================================================


def rotation_matrix_to_6d(R: Array) -> Array:
    """
    Convert a rotation matrix to a 6D continuous representation.

    The 6D representation uses the first two columns of the rotation matrix.
    This representation is continuous everywhere in SO(3), making it ideal
    for neural network outputs.

    Args:
        R: Rotation matrix of shape (3, 3).

    Returns:
        r6d: 6D representation of shape (6,), containing [R[:, 0], R[:, 1]].

    Example:
        >>> import jax.numpy as jnp
        >>> from soromox.utils.rotations import rotation_matrix_to_6d
        >>> R = jnp.eye(3)  # Identity rotation
        >>> r6d = rotation_matrix_to_6d(R)
        >>> # r6d = [1, 0, 0, 0, 1, 0]

    References:
        Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2019).
        On the continuity of rotation representations in neural networks.
        CVPR 2019.
    """
    # Extract first two columns and flatten
    return jnp.concatenate([R[:, 0], R[:, 1]])


def rotation_6d_to_rotation_matrix(r6d: Array) -> Array:
    """
    Convert a 6D continuous representation to a rotation matrix.

    Uses Gram-Schmidt orthonormalization to ensure the output is a valid
    rotation matrix. The third column is computed as the cross product
    of the first two orthonormalized columns.

    Args:
        r6d: 6D representation of shape (6,).

    Returns:
        R: Rotation matrix of shape (3, 3).

    Example:
        >>> import jax.numpy as jnp
        >>> from soromox.utils.rotations import rotation_6d_to_rotation_matrix
        >>> r6d = jnp.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        >>> R = rotation_6d_to_rotation_matrix(r6d)
        >>> # R ≈ eye(3)

    References:
        Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2019).
        On the continuity of rotation representations in neural networks.
        CVPR 2019.
    """
    # Extract the two column vectors
    a1 = r6d[:3]
    a2 = r6d[3:6]

    # Gram-Schmidt orthonormalization
    b1 = a1 / jnp.linalg.norm(a1)
    b2 = a2 - jnp.dot(b1, a2) * b1
    b2 = b2 / jnp.linalg.norm(b2)

    # Third column via cross product
    b3 = jnp.cross(b1, b2)

    # Construct rotation matrix
    R = jnp.column_stack([b1, b2, b3])
    return R


# =============================================================================
# Quaternion Operations
# =============================================================================


def quaternion_multiply(q1: Array, q2: Array) -> Array:
    """
    Multiply two quaternions (Hamilton product).

    The result represents the composition of rotations: first q2, then q1.
    That is, q1 * q2 represents rotating by q2 first, then by q1.

    Args:
        q1: First quaternion [x, y, z, w] of shape (4,).
        q2: Second quaternion [x, y, z, w] of shape (4,).

    Returns:
        q: Product quaternion [x, y, z, w] of shape (4,).

    Example:
        >>> import jax.numpy as jnp
        >>> from soromox.utils.rotations import quaternion_multiply
        >>> q1 = jnp.array([0, 0, 0, 1])  # Identity
        >>> q2 = jnp.array([0, 0, jnp.sin(jnp.pi/4), jnp.cos(jnp.pi/4)])  # 90° around z
        >>> q = quaternion_multiply(q1, q2)
        >>> # q ≈ q2
    """
    x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
    x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]

    return jnp.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def quaternion_conjugate(q: Array) -> Array:
    """
    Compute the conjugate of a quaternion.

    For a unit quaternion, the conjugate is also the inverse and represents
    the opposite rotation.

    Args:
        q: Quaternion [x, y, z, w] of shape (4,).

    Returns:
        q_conj: Conjugate quaternion [-x, -y, -z, w] of shape (4,).

    Example:
        >>> import jax.numpy as jnp
        >>> from soromox.utils.rotations import quaternion_conjugate
        >>> q = jnp.array([0, 0, jnp.sin(jnp.pi/4), jnp.cos(jnp.pi/4)])  # 90° around z
        >>> q_conj = quaternion_conjugate(q)
        >>> # q_conj represents -90° around z
    """
    return jnp.array([-q[0], -q[1], -q[2], q[3]])


# =============================================================================
# Geometric Error Computation
# =============================================================================


def wrap_angle(angle: Array) -> Array:
    """
    Wrap an angle to the range [-π, π].

    This is useful for computing the shortest angular difference between
    two angles, particularly for planar systems.

    Args:
        angle: Angle in radians (scalar or array).

    Returns:
        wrapped: Angle wrapped to [-π, π].

    Example:
        >>> import jax.numpy as jnp
        >>> from soromox.utils.rotations import wrap_angle
        >>> wrap_angle(jnp.array(3 * jnp.pi))  # Returns approximately π
        >>> wrap_angle(jnp.array(-3 * jnp.pi))  # Returns approximately -π
    """
    return jnp.mod(angle + jnp.pi, 2 * jnp.pi) - jnp.pi


def angle_error(theta_current: Array, theta_desired: Array) -> Array:
    """
    Compute the geometric angular error for planar rotations.

    This computes the shortest angular path from the current angle to the
    desired angle, properly handling angle wrapping. The result is always
    in the range [-π, π].

    Args:
        theta_current: Current angle in radians (scalar or array).
        theta_desired: Desired angle in radians (scalar or array).

    Returns:
        error: Angular error in radians, in range [-π, π].
            Positive error means rotate counter-clockwise to reach desired.

    Example:
        >>> import jax.numpy as jnp
        >>> from soromox.utils.rotations import angle_error
        >>> # Current: 10°, Desired: 350° → Error should be -20° (not +340°)
        >>> e = angle_error(jnp.deg2rad(10), jnp.deg2rad(350))
        >>> # e ≈ -0.349 rad (≈ -20°)

    Note:
        This is the proper error computation for planar PID control,
        ensuring the controller takes the shortest path.
    """
    return wrap_angle(theta_desired - theta_current)


def rotation_matrix_error(
    R_current: Array,
    R_desired: Array,
    eps: float = DEFAULT_ROTATION_EPS,
) -> Array:
    """
    Compute the geometric orientation error between two rotation matrices.

    This computes the rotation vector representing the shortest-path rotation
    from the current orientation to the desired orientation:
        e = log(R_desired @ R_current^T)

    The result is suitable for use in PID controllers as it:
    - Represents the axis-angle of the required correction
    - Always takes the shortest path (magnitude ≤ π)
    - Is continuous for small perturbations
    - Lives in the tangent space (same as angular velocity)

    Args:
        R_current: Current rotation matrix of shape (3, 3).
        R_desired: Desired rotation matrix of shape (3, 3).
        eps: Small value for numerical stability. Default is 1e-10.

    Returns:
        error: Rotation vector of shape (3,) representing the geometric error.
            The magnitude is the angle to rotate, the direction is the axis.

    Example:
        >>> import jax.numpy as jnp
        >>> from soromox.utils.rotations import rotation_matrix_error, rotation_vector_to_rotation_matrix
        >>> # Current: 10° around z, Desired: 350° around z
        >>> R_current = rotation_vector_to_rotation_matrix(jnp.array([0, 0, jnp.deg2rad(10)]))
        >>> R_desired = rotation_vector_to_rotation_matrix(jnp.array([0, 0, jnp.deg2rad(350)]))
        >>> e = rotation_matrix_error(R_current, R_desired)
        >>> # e ≈ [0, 0, -0.349] (≈ -20° around z, the shortest path)

    Note:
        This should be used instead of naive subtraction of rotation vectors
        (ω_des - ω) in control applications, as the naive subtraction does
        not represent the geometric error.
    """
    # Compute relative rotation: R_error = R_desired @ R_current^T
    R_error = R_desired @ R_current.T

    # Convert to rotation vector (log map)
    return rotation_matrix_to_rotation_vector(R_error, eps=eps)


def rotation_quat_error(
    q_current: Array,
    q_desired: Array,
    eps: float = DEFAULT_ROTATION_EPS,
) -> Array:
    """
    Compute the geometric orientation error between two quaternions.

    This computes the rotation vector representing the shortest-path rotation
    from the current orientation to the desired orientation using quaternion
    arithmetic:
        q_error = q_desired ⊗ q_current^{-1}
        e = log(q_error)

    The function automatically handles the antipodal ambiguity of quaternions
    to ensure the shortest path is taken.

    Args:
        q_current: Current quaternion [x, y, z, w] of shape (4,).
        q_desired: Desired quaternion [x, y, z, w] of shape (4,).
        eps: Small value for numerical stability. Default is 1e-10.

    Returns:
        error: Rotation vector of shape (3,) representing the geometric error.

    Example:
        >>> import jax.numpy as jnp
        >>> from soromox.utils.rotations import rotation_quat_error, rotation_vector_to_quaternion
        >>> q_current = rotation_vector_to_quaternion(jnp.array([0, 0, 0.1]))
        >>> q_desired = rotation_vector_to_quaternion(jnp.array([0, 0, 0.2]))
        >>> e = rotation_quat_error(q_current, q_desired)
        >>> # e ≈ [0, 0, 0.1]
    """
    # Compute the error quaternion: q_error = q_desired * q_current^{-1}
    q_current_inv = quaternion_conjugate(q_current)
    q_error = quaternion_multiply(q_desired, q_current_inv)

    # Ensure shortest path by flipping if scalar part is negative
    q_error = jnp.where(q_error[3] < 0, -q_error, q_error)

    # Convert to rotation vector
    return quaternion_to_rotation_vector(q_error, eps=eps)
