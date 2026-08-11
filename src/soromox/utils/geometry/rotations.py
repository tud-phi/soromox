"""Rotation utilities for converting between representation encodings.

This module provides numerically stable conversions between rotation matrices,
quaternions, and rotation vectors (axis-angle representation). The implementations
are designed to handle edge cases such as small rotations (θ ≈ 0) and near-180-degree
rotations (θ ≈ π) correctly.

All functions are JAX-compatible and can be used with jit, vmap, and grad.

Quaternion Convention:
    This module uses scalar-first Hamilton quaternions in ``[qw, qx, qy, qz]``
    order, where ``qw`` is the scalar part and ``[qx, qy, qz]`` is the vector
    part. A rotation by angle θ around unit axis n is represented as:
        q = [cos(θ/2), sin(θ/2) * n_x, sin(θ/2) * n_y, sin(θ/2) * n_z]

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
"""

__all__ = [
    # Enum
    "RotationRepresentation",
    # Conversion functions
    "normalize_quaternion",
    "rotation_matrix_to_quaternion",
    "quaternion_to_rotation_matrix",
    "quaternion_to_rotation_vector",
    "rotation_vector_to_quaternion",
    "rotation_matrix_to_rotation_vector",
    "rotation_vector_to_rotation_matrix",
    "principal_axis_rotation_matrix",
    "principal_axis_rotation_matrix_derivative",
    # 6D continuous representation
    "rotation_matrix_to_6d",
    "rotation_6d_to_rotation_matrix",
    # Quaternion operations
    "quaternion_multiply",
    "quaternion_conjugate",
]

from enum import Enum
from typing import Literal

import jax.numpy as jnp
from jax import Array, lax

from soromox.utils._numerics import eps_for_dtype
from soromox.utils.lie_algebra import so3


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
        For computing orientation errors in control applications, use
        ``soromox.utils.geometry.errors`` regardless of the representation
        chosen for pose coordinates.
    """

    ROTATION_VECTOR = "rotation_vector"
    QUATERNION = "quaternion"
    ROTATION_MATRIX_6D = "rotation_matrix_6d"


# Default epsilon for numerical stability in rotation conversions
DEFAULT_ROTATION_EPS = 1e-10


def principal_axis_rotation_matrix(angle: Array, axis: Literal["x", "y", "z"]) -> Array:
    """Return the active rotation matrix about a Cartesian principal axis.

    Args:
        angle: Right-handed rotation angle in radians.
        axis: Static Cartesian axis identifier.

    Returns:
        Active rotation matrix with shape ``(3, 3)``.
    """
    c, s = jnp.cos(angle), jnp.sin(angle)
    z, o = jnp.zeros_like(angle), jnp.ones_like(angle)
    if axis == "x":
        return jnp.stack(
            [jnp.stack([o, z, z]), jnp.stack([z, c, -s]), jnp.stack([z, s, c])]
        )
    if axis == "y":
        return jnp.stack(
            [jnp.stack([c, z, s]), jnp.stack([z, o, z]), jnp.stack([-s, z, c])]
        )
    if axis == "z":
        return jnp.stack(
            [jnp.stack([c, -s, z]), jnp.stack([s, c, z]), jnp.stack([z, z, o])]
        )
    raise ValueError("axis must be one of 'x', 'y', or 'z'.")


def principal_axis_rotation_matrix_derivative(
    angle: Array, axis: Literal["x", "y", "z"]
) -> Array:
    """Differentiate a principal-axis rotation matrix with respect to angle."""
    c, s = jnp.cos(angle), jnp.sin(angle)
    z = jnp.zeros_like(angle)
    if axis == "x":
        return jnp.stack(
            [jnp.stack([z, z, z]), jnp.stack([z, -s, -c]), jnp.stack([z, c, -s])]
        )
    if axis == "y":
        return jnp.stack(
            [jnp.stack([-s, z, c]), jnp.stack([z, z, z]), jnp.stack([-c, z, -s])]
        )
    if axis == "z":
        return jnp.stack(
            [jnp.stack([-s, -c, z]), jnp.stack([c, -s, z]), jnp.stack([z, z, z])]
        )
    raise ValueError("axis must be one of 'x', 'y', or 'z'.")


def normalize_quaternion(
    quaternion: Array,
    eps: float | Array | None = None,
) -> Array:
    """Normalize a scalar-first Hamilton quaternion.

    Args:
        quaternion: Quaternion with shape ``(4,)`` in ``[qw, qx, qy, qz]``
            order. ``qw`` is the scalar component and ``[qx, qy, qz]`` is the
            vector component.
        eps: Optional nonnegative norm threshold. If the quaternion norm is
            less than or equal to ``eps``, the identity quaternion is returned.
            Defaults to machine epsilon for the quaternion dtype.

    Returns:
        Array: Unit quaternion with shape ``(4,)`` in ``[qw, qx, qy, qz]``
        order. If the input norm is numerically smaller than ``eps``, the
        identity quaternion ``[1, 0, 0, 0]`` is returned to avoid
        division-by-zero NaNs in traced computations. Parameter validators
        reject zero-norm configured poses.
    """
    dtype = jnp.result_type(quaternion, 1.0)
    q = jnp.asarray(quaternion, dtype=dtype).reshape(-1)
    eps_arr = eps_for_dtype(eps, q.dtype)
    identity = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=q.dtype)
    norm_sq = jnp.dot(q, q)

    def _normalize(_: None) -> Array:
        return q / jnp.sqrt(norm_sq)

    return lax.cond(norm_sq > eps_arr**2, _normalize, lambda _: identity, operand=None)


def rotation_matrix_to_quaternion(R: Array, eps: float = DEFAULT_ROTATION_EPS) -> Array:
    """
    Convert a rotation matrix to a unit quaternion.

    Uses Shepperd's method, which is numerically stable for all rotation
    angles including near-180-degree rotations (θ ≈ π).

    Args:
        R: Rotation matrix of shape (3, 3).
        eps: Small value to avoid division by zero. Default is 1e-10.

    Returns:
        q: Unit quaternion with shape ``(4,)`` in scalar-first
            ``[qw, qx, qy, qz]`` order.

    Example:
        ```python
        import jax.numpy as jnp
        from soromox.utils.geometry.rotations import rotation_matrix_to_quaternion

        R = jnp.eye(3)
        q = rotation_matrix_to_quaternion(R)  # approximately [1, 0, 0, 0]
        ```

    References:
        Shepperd, S. W. (1978). Quaternion from rotation matrix.
        Journal of Guidance and Control, 1(3), 223-224.
    """
    R = jnp.asarray(R, dtype=jnp.result_type(R, 1.0))
    trace = jnp.trace(R)
    eps_arr = eps_for_dtype(eps, R.dtype)

    def _trace_branch(_: None) -> Array:
        s = jnp.sqrt(jnp.maximum(trace + 1.0, 0.0)) * 2.0
        denom = jnp.maximum(s, eps_arr)
        return jnp.array(
            [
                0.25 * s,
                (R[2, 1] - R[1, 2]) / denom,
                (R[0, 2] - R[2, 0]) / denom,
                (R[1, 0] - R[0, 1]) / denom,
            ],
            dtype=R.dtype,
        )

    def _x_branch(_: None) -> Array:
        s = jnp.sqrt(jnp.maximum(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 0.0)) * 2.0
        denom = jnp.maximum(s, eps_arr)
        return jnp.array(
            [
                (R[2, 1] - R[1, 2]) / denom,
                0.25 * s,
                (R[0, 1] + R[1, 0]) / denom,
                (R[0, 2] + R[2, 0]) / denom,
            ],
            dtype=R.dtype,
        )

    def _y_branch(_: None) -> Array:
        s = jnp.sqrt(jnp.maximum(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 0.0)) * 2.0
        denom = jnp.maximum(s, eps_arr)
        return jnp.array(
            [
                (R[0, 2] - R[2, 0]) / denom,
                (R[0, 1] + R[1, 0]) / denom,
                0.25 * s,
                (R[1, 2] + R[2, 1]) / denom,
            ],
            dtype=R.dtype,
        )

    def _z_branch(_: None) -> Array:
        s = jnp.sqrt(jnp.maximum(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 0.0)) * 2.0
        denom = jnp.maximum(s, eps_arr)
        return jnp.array(
            [
                (R[1, 0] - R[0, 1]) / denom,
                (R[0, 2] + R[2, 0]) / denom,
                (R[1, 2] + R[2, 1]) / denom,
                0.25 * s,
            ],
            dtype=R.dtype,
        )

    def _diagonal_branch(_: None) -> Array:
        return lax.cond(
            (R[0, 0] > R[1, 1]) & (R[0, 0] > R[2, 2]),
            _x_branch,
            lambda _: lax.cond(R[1, 1] > R[2, 2], _y_branch, _z_branch, operand=None),
            operand=None,
        )

    quat = lax.cond(trace > 0.0, _trace_branch, _diagonal_branch, operand=None)

    # Ensure canonical form with positive scalar part (qw >= 0)
    quat = jnp.where(quat[0] < 0, -quat, quat)

    quat = normalize_quaternion(quat, eps=eps)

    return quat


def quaternion_to_rotation_matrix(quat: Array) -> Array:
    """
    Convert a unit quaternion to a rotation matrix.

    Args:
        quat: Quaternion with shape ``(4,)`` in scalar-first Hamilton
            ``[qw, qx, qy, qz]`` order. The quaternion is normalized before
            constructing the matrix.

    Returns:
        R: Rotation matrix of shape (3, 3).

    Example:
        ```python
        import jax.numpy as jnp
        from soromox.utils.geometry.rotations import quaternion_to_rotation_matrix

        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_rotation_matrix(q)  # approximately eye(3)
        ```

    Note:
        If the quaternion norm is numerically zero, identity rotation is used
        to keep traced computations finite. Configured robot base poses are
        validated separately and reject zero-norm quaternions.
    """
    quat = normalize_quaternion(quat)
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

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
        quat: Unit quaternion with shape ``(4,)`` in scalar-first Hamilton
            ``[qw, qx, qy, qz]`` order.
        eps: Small value to avoid division by zero. Default is 1e-10.

    Returns:
        omega: Rotation vector of shape (3,).

    Example:
        ```python
        import jax.numpy as jnp
        from soromox.utils.geometry.rotations import quaternion_to_rotation_vector

        # 90-degree rotation around the z-axis
        q = jnp.array([jnp.cos(jnp.pi / 4), 0, 0, jnp.sin(jnp.pi / 4)])
        omega = quaternion_to_rotation_vector(q)  # approximately [0, 0, pi/2]
        ```
    """
    quat = normalize_quaternion(quat, eps=eps)
    q_w = quat[0]
    q_vec = quat[1:4]
    eps_arr = eps_for_dtype(eps, quat.dtype)
    q_vec_norm_sq = jnp.dot(q_vec, q_vec)

    def _small(_: None) -> Array:
        return 2.0 * q_vec

    def _general(_: None) -> Array:
        q_vec_norm = jnp.sqrt(q_vec_norm_sq)
        angle = 2.0 * jnp.arctan2(q_vec_norm, q_w)
        return q_vec * (angle / q_vec_norm)

    return lax.cond(q_vec_norm_sq > eps_arr**2, _general, _small, operand=None)


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
        q: Unit quaternion with shape ``(4,)`` in scalar-first Hamilton
            ``[qw, qx, qy, qz]`` order.

    Example:
        ```python
        import jax.numpy as jnp
        from soromox.utils.geometry.rotations import rotation_vector_to_quaternion

        # 90-degree rotation around the z-axis
        omega = jnp.array([0.0, 0.0, jnp.pi / 2])
        q = rotation_vector_to_quaternion(omega)
        # q is approximately [cos(pi/4), 0, 0, sin(pi/4)]
        ```
    """
    omega = jnp.asarray(omega).reshape(-1)
    eps_arr = eps_for_dtype(eps, omega.dtype)
    angle_sq = jnp.dot(omega, omega)

    def _small(_: None) -> Array:
        return jnp.array(
            [1.0, 0.5 * omega[0], 0.5 * omega[1], 0.5 * omega[2]],
            dtype=omega.dtype,
        )

    def _general(_: None) -> Array:
        angle = jnp.sqrt(angle_sq)
        half_angle = angle / 2.0
        q_vec = omega * (jnp.sin(half_angle) / angle)
        return jnp.array([jnp.cos(half_angle), q_vec[0], q_vec[1], q_vec[2]])

    quat = lax.cond(angle_sq > eps_arr**2, _general, _small, operand=None)

    return normalize_quaternion(quat, eps=eps)


def rotation_matrix_to_rotation_vector(
    R: Array, eps: float = DEFAULT_ROTATION_EPS
) -> Array:
    """
    Convert a rotation matrix to a rotation vector (axis-angle representation).

    This is a convenience wrapper around ``soromox.utils.lie_algebra.so3.log``.
    It correctly handles both small rotations (θ ≈ 0) and near-180-degree
    rotations (θ ≈ π).

    The rotation vector ω has magnitude equal to the rotation angle θ and
    direction equal to the rotation axis. For a rotation by angle θ around
    unit axis n, the rotation vector is ω = θ * n.

    Args:
        R: Rotation matrix of shape (3, 3).
        eps: Small value to avoid division by zero. Default is 1e-10.

    Returns:
        omega: Rotation vector of shape (3,).

    Example:
        ```python
        import jax.numpy as jnp
        from soromox.utils.geometry.rotations import rotation_matrix_to_rotation_vector

        # 180-degree rotation around the z-axis
        R = jnp.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1.0]])
        omega = rotation_matrix_to_rotation_vector(R)  # approximately [0, 0, pi]
        ```

    Note:
        This delegates to ``soromox.utils.lie_algebra.so3.log`` so that the
        matrix logarithm has one canonical implementation.
    """
    return so3.log(R, eps=eps)


def rotation_vector_to_rotation_matrix(
    omega: Array, eps: float = DEFAULT_ROTATION_EPS
) -> Array:
    """
    Convert a rotation vector (axis-angle representation) to a rotation matrix.

    This is a convenience wrapper around ``soromox.utils.lie_algebra.so3.exp``.
    It correctly handles both small rotations (θ ≈ 0) and all other rotation
    angles.

    The rotation vector ω has magnitude equal to the rotation angle θ and
    direction equal to the rotation axis. For a rotation by angle θ around
    unit axis n, the rotation vector is ω = θ * n.

    Args:
        omega: Rotation vector of shape (3,).
        eps: Small value for numerical stability at small angles. Default is 1e-10.

    Returns:
        R: Rotation matrix of shape (3, 3).

    Example:
        ```python
        import jax.numpy as jnp
        from soromox.utils.geometry.rotations import rotation_vector_to_rotation_matrix

        # 90-degree rotation around the z-axis
        omega = jnp.array([0.0, 0.0, jnp.pi / 2])
        R = rotation_vector_to_rotation_matrix(omega)
        ```

    Note:
        This delegates to ``soromox.utils.lie_algebra.so3.exp`` so that
        Rodrigues' formula has one canonical implementation.
    """
    return so3.exp(omega, eps=eps)


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
        ```python
        import jax.numpy as jnp
        from soromox.utils.geometry.rotations import rotation_matrix_to_6d

        R = jnp.eye(3)
        r6d = rotation_matrix_to_6d(R)  # [1, 0, 0, 0, 1, 0]
        ```

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
        ```python
        import jax.numpy as jnp
        from soromox.utils.geometry.rotations import rotation_6d_to_rotation_matrix

        r6d = jnp.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        R = rotation_6d_to_rotation_matrix(r6d)  # approximately eye(3)
        ```

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
        q1: First quaternion with shape ``(4,)`` in scalar-first Hamilton
            ``[qw, qx, qy, qz]`` order.
        q2: Second quaternion with shape ``(4,)`` in scalar-first Hamilton
            ``[qw, qx, qy, qz]`` order.

    Returns:
        q: Product quaternion with shape ``(4,)`` in ``[qw, qx, qy, qz]``
            order.

    Example:
        ```python
        import jax.numpy as jnp
        from soromox.utils.geometry.rotations import quaternion_multiply

        q1 = jnp.array([1, 0, 0, 0])
        q2 = jnp.array([jnp.cos(jnp.pi / 4), 0, 0, jnp.sin(jnp.pi / 4)])
        q = quaternion_multiply(q1, q2)  # approximately q2
        ```
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    return jnp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quaternion_conjugate(q: Array) -> Array:
    """
    Compute the conjugate of a quaternion.

    For a unit quaternion, the conjugate is also the inverse and represents
    the opposite rotation.

    Args:
        q: Quaternion with shape ``(4,)`` in scalar-first Hamilton
            ``[qw, qx, qy, qz]`` order.

    Returns:
        q_conj: Conjugate quaternion with shape ``(4,)`` and coordinates
            ``[qw, -qx, -qy, -qz]``.

    Example:
        ```python
        import jax.numpy as jnp
        from soromox.utils.geometry.rotations import quaternion_conjugate

        q = jnp.array([jnp.cos(jnp.pi / 4), 0, 0, jnp.sin(jnp.pi / 4)])
        q_conj = quaternion_conjugate(q)  # represents -90 degrees around z
        ```
    """
    return jnp.array([q[0], -q[1], -q[2], -q[3]])
