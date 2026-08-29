__all__ = [
    "mounting_pose",
    "planar_mounting_pose",
    "spatial_mounting_pose",
    "planar_pose_to_transform",
    "planar_pose_from_transform",
    "quaternion_pose_to_transform",
]

import jax.numpy as jnp
from jax import Array, lax

from soromox.utils.lie_algebra import so2

from .rotations import quaternion_to_rotation_matrix


def planar_mounting_pose(
    mounting: str = "upright", position: Array | None = None
) -> Array:
    """Return a named planar mounting pose in ``[theta, x, y]`` order.

    Args:
        mounting: ``"horizontal"``, ``"upright"``, or ``"hanging"``.
        position: Optional finite translation with shape ``(2,)``.

    Returns:
        Planar pose with shape ``(3,)``.

    Raises:
        ValueError: If the mounting name or position is invalid.
    """
    angles = {"horizontal": 0.0, "upright": jnp.pi / 2, "hanging": -jnp.pi / 2}
    if mounting not in angles:
        raise ValueError("mounting must be 'horizontal', 'upright', or 'hanging'.")
    p = jnp.zeros((2,)) if position is None else jnp.asarray(position)
    if p.shape != (2,) or not bool(jnp.isfinite(p).all()):
        raise ValueError("position must be a finite array with shape (2,).")
    return jnp.concatenate([jnp.asarray([angles[mounting]], dtype=p.dtype), p])


def spatial_mounting_pose(
    mounting: str = "upright", position: Array | None = None
) -> Array:
    """Return a named scalar-first quaternion and position mounting pose.

    Args:
        mounting: ``"horizontal"``, ``"upright"``, or ``"hanging"``.
        position: Optional finite translation with shape ``(3,)``.

    Returns:
        Spatial pose ``[qw, qx, qy, qz, x, y, z]`` with shape ``(7,)``.

    Raises:
        ValueError: If the mounting name or position is invalid.
    """
    sqrt_half = jnp.sqrt(jnp.asarray(0.5))
    quaternions = {
        "horizontal": jnp.asarray([1.0, 0.0, 0.0, 0.0]),
        "upright": jnp.asarray([sqrt_half, 0.0, -sqrt_half, 0.0]),
        "hanging": jnp.asarray([sqrt_half, 0.0, sqrt_half, 0.0]),
    }
    if mounting not in quaternions:
        raise ValueError("mounting must be 'horizontal', 'upright', or 'hanging'.")
    p = jnp.zeros((3,)) if position is None else jnp.asarray(position)
    if p.shape != (3,) or not bool(jnp.isfinite(p).all()):
        raise ValueError("position must be a finite array with shape (3,).")
    return jnp.concatenate([quaternions[mounting].astype(p.dtype), p])


def mounting_pose(
    mounting: str = "upright", *, planar: bool, position: Array | None = None
) -> Array:
    """Return a named planar or spatial mounting pose.

    Args:
        mounting: ``"horizontal"``, ``"upright"``, or ``"hanging"``.
        planar: Whether to return planar rather than spatial coordinates.
        position: Optional translation with dimension selected by ``planar``.

    Returns:
        Planar pose with shape ``(3,)`` or spatial pose with shape ``(7,)``.
    """
    if planar:
        return planar_mounting_pose(mounting, position)
    return spatial_mounting_pose(mounting, position)


def planar_pose_to_transform(pose: Array) -> Array:
    """Construct an ``SE(2)`` transform from direct planar pose coordinates.

    This helper treats the translation entries as pose coordinates, not as
    Lie-algebra twist coordinates. A pose ``[theta, x, y]`` maps directly to
    ``[[R(theta), [x, y]], [0, 0, 1]]``. Use ``se2.exp`` when the input is an
    ``se(2)`` twist whose translational part must be integrated through the
    ``SE(2)`` left Jacobian.

    Args:
        pose: Planar pose with shape ``(3,)`` or ``(3, 1)`` in
            ``[theta, x, y]`` order. ``theta`` is a right-handed rotation angle
            in radians about the out-of-plane z-axis. ``x`` and ``y`` are
            direct translation coordinates in the transform parent frame.

    Returns:
        Homogeneous ``SE(2)`` transform with shape ``(3, 3)``.
    """
    pose = jnp.asarray(pose).reshape(-1)

    theta = pose[0]
    p = pose[1:3].reshape((2, 1))

    R = so2.exp(theta)

    return jnp.block(
        [
            [R, p],
            [jnp.zeros((1, 2), dtype=pose.dtype), jnp.ones((1, 1), dtype=pose.dtype)],
        ]
    )


def planar_pose_from_transform(g: Array, eps: float | Array) -> Array:
    """Extract direct planar pose coordinates from an ``SE(2)`` transform.

    This is the inverse of :func:`planar_pose_to_transform` for the principal
    angle returned by ``arctan2``. It returns the raw transform translation, not
    the translational twist coordinates returned by ``se2.log``.

    Args:
        g: Homogeneous ``SE(2)`` transform with shape ``(3, 3)``. The rotation
            is read from ``g[:2, :2]`` and the direct translation from
            ``g[:2, 2]``.
        eps: Small positive scalar threshold. Angles with magnitude below this
            threshold are regularized to exactly zero.

    Returns:
        Array with shape ``(3,)`` in direct pose-coordinate order
        ``[theta, x, y]``.
    """
    R = g[:2, :2]
    p = g[:2, 2]

    theta = jnp.arctan2(R[1, 0], R[0, 0])
    theta = lax.cond(
        jnp.abs(theta) < eps,
        lambda _: jnp.zeros((), dtype=theta.dtype),
        lambda _: theta,
        operand=None,
    )

    return jnp.concatenate([jnp.array([theta], dtype=g.dtype), p])


def quaternion_pose_to_transform(pose: Array) -> Array:
    """Construct an ``SE(3)`` transform from quaternion pose coordinates.

    This helper treats the final three entries as direct translation
    coordinates, not as Lie-algebra twist coordinates. The quaternion is
    scalar-first and is normalized by ``quaternion_to_rotation_matrix`` before
    constructing the rotation block. Use ``se3.exp`` when the input is an
    ``se(3)`` twist whose translational part must be integrated through the
    ``SE(3)`` left Jacobian.

    Args:
        pose: Spatial pose with shape ``(7,)`` or ``(7, 1)`` in
            ``[qw, qx, qy, qz, x, y, z]`` order. The quaternion is scalar-first
            and the translation ``[x, y, z]`` is inserted directly into the
            homogeneous transform.

    Returns:
        Homogeneous ``SE(3)`` transform with shape ``(4, 4)``.
    """
    pose = jnp.asarray(pose).reshape(-1)
    R = quaternion_to_rotation_matrix(pose[:4])
    p = pose[4:].reshape((3, 1))

    return jnp.block(
        [
            [R, p],
            [jnp.zeros((1, 3), dtype=pose.dtype), jnp.ones((1, 1), dtype=pose.dtype)],
        ]
    )
