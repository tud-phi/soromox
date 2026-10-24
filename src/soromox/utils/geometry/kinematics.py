"""Frame composition and point transformations for continuum kinematics."""

import jax.numpy as jnp
from jax import lax


def spatial_transform(transform):
    """Embed a planar homogeneous transform in three dimensions.

    Args:
        transform (Array): Homogeneous transform, shape ``(3, 3)`` or ``(4, 4)``.

    Returns:
        result (Array): Spatial homogeneous transform, shape ``(4, 4)``.
    """
    if transform.shape == (3, 3):
        return (
            jnp.eye(4, dtype=transform.dtype)
            .at[:2, :2]
            .set(transform[:2, :2])
            .at[:2, 3]
            .set(transform[:2, 2])
        )
    return transform


def transform_point(pose, offset):
    """Transform a material point into its parent's coordinate frame.

    Args:
        pose (Array): Planar ``[theta, x, y]`` pose, shape ``(3,)``, or spatial
            homogeneous transform, shape ``(4, 4)``.
        offset (Array): Material point, shape ``(3,)``.

    Returns:
        result (Array): Transformed point, shape ``(3,)``.
    """
    if pose.shape == (3,):
        cosine, sine = jnp.cos(pose[0]), jnp.sin(pose[0])
        return jnp.array(
            [
                pose[1] + cosine * offset[0] - sine * offset[1],
                pose[2] + sine * offset[0] + cosine * offset[1],
                offset[2],
            ]
        )
    return pose[:3, :3] @ offset + pose[:3, 3]


def segment_boundary_poses(base, links, joints):
    """Compose segment entry and exit frames in one traversal.

    Args:
        base (Array): Base transform, shape ``(3, 3)`` or ``(4, 4)``.
        links (Array): Local link transforms, shape ``(num_segments, 4, 4)``.
        joints (Array): Preceding joint transforms with the same shape.

    Returns:
        result (tuple[Array, Array]): World entry and exit transforms, each
            shaped ``(num_segments, 4, 4)``. Entry frames include their joint.
    """

    def advance(previous, transforms):
        """Return the next boundary and both sides of its link."""
        link, joint = transforms
        entry = previous @ joint
        exit_pose = entry @ link
        return exit_pose, (entry, exit_pose)

    _, boundaries = lax.scan(advance, spatial_transform(base), (links, joints))
    return boundaries
