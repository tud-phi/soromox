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
"""

__all__ = [
    "rotation_matrix_to_quaternion",
    "quaternion_to_rotation_matrix",
    "quaternion_to_rotation_vector",
    "rotation_vector_to_quaternion",
    "rotation_matrix_to_rotation_vector",
    "rotation_vector_to_rotation_matrix",
]

from jax import Array
import jax.numpy as jnp


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
    R = jnp.array([
        [w2 + x2 - y2 - z2, 2 * (xy - zw), 2 * (xz + yw)],
        [2 * (xy + zw), w2 - x2 + y2 - z2, 2 * (yz - xw)],
        [2 * (xz - yw), 2 * (yz + xw), w2 - x2 - y2 + z2],
    ])

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


def rotation_vector_to_quaternion(omega: Array, eps: float = DEFAULT_ROTATION_EPS) -> Array:
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


def rotation_matrix_to_rotation_vector(R: Array, eps: float = DEFAULT_ROTATION_EPS) -> Array:
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
