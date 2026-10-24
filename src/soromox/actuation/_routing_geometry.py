"""Prescribed guide geometry built from general robot kinematics."""

import jax.numpy as jnp

from soromox.utils.geometry.kinematics import transform_point


def validate_span(robot, start, end):
    """Require guides wherever a span crosses a moving joint.

    Args:
        robot (SoftRobot): Robot providing static joint topology.
        start (int): Inclusive first segment index.
        end (int): Inclusive last segment index.

    Raises:
        ValueError: A moving joint lies inside the span.
    """
    if any(start < index <= end for index in robot.moving_joint_indices):
        raise ValueError(
            "A span crossing a moving joint requires prescribed guides; split it at the joint."
        )


def guide_kinematics(robot, configuration, segment, upstream, downstream):
    """Evaluate the straight connection between prescribed joint guides.

    Args:
        robot (SoftRobot): Robot providing relative joint kinematics.
        configuration (Array): Configuration, shape ``(num_coordinates,)``.
        segment (int): Segment entered after the joint.
        upstream (Array): Guide in the upstream tip frame, shape ``(3,)``.
        downstream (Array): Guide in the downstream entry frame, shape ``(3,)``.

    Returns:
        result (tuple[Array, Array, Array]): Upstream-frame displacement,
            shape ``(3,)``; configuration Jacobian, shape
            ``(3, num_coordinates)``; and a scalar boolean certifying that the
            guides coincide for every joint state. Base columns are zero.
    """
    transform, spatial_jacobian, generators = robot.joint_kinematics(
        configuration, segment
    )
    endpoint = transform_point(transform, downstream)
    derivative = jnp.cross(spatial_jacobian[:3].T, endpoint).T + spatial_jacobian[3:]
    fixed_point = jnp.cross(generators[:3].T, downstream).T + generators[3:]
    coincident = jnp.all(upstream == downstream) & jnp.all(fixed_point == 0)
    return endpoint - upstream, derivative, coincident


def span_entry_position(robot, configuration, segment, offset):
    """Place an entry guide on the downstream side of its joint.

    Args:
        robot (SoftRobot): Continuum robot.
        configuration (Array): Configuration, shape ``(num_coordinates,)``.
        segment (int): Static index of the first span segment.
        offset (Array): Material guide offset, shape ``(3,)``.

    Returns:
        result (Array): World guide position, shape ``(3,)``.
    """
    position = jnp.sum(robot.segment_length[:segment])
    pose = robot.forward_kinematics(configuration, position)
    if segment > 0:
        joint, _, _ = robot.joint_kinematics(configuration, segment)
        offset = transform_point(joint, offset)
    return transform_point(pose, offset)
