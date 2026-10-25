"""Shared relative kinematics for planar and spatial PCS segments."""

import jax.numpy as jnp
from jax import vmap

from soromox.utils.geometry.poses import as_spatial_transform
from soromox.utils.lie_algebra import se2, se3


def constant_strain_segment_kinematics(robot, configuration):
    """Evaluate PCS link transforms and analytical body Jacobians once.

    Args:
        robot (PCS | PlanarPCS): Constant-strain host.
        configuration (Array): Configuration, shape ``(num_coordinates,)``.

    Returns:
        result (tuple[Array, Array, Array]): Arrays following the
            ``relative_segment_kinematics`` contract, with identity joints.
    """
    strains, bases = robot.strain_and_basis(
        configuration,
        jnp.arange(robot.num_segments),
        jnp.cumsum(robot.segment_length),
    )

    def evaluate(strain, basis, length):
        """Return a constant-strain transform and its body Jacobian."""
        if robot.is_planar:
            increment = length * strain[2:5]
            planar = se2.exp(increment, robot.global_eps)
            transform = as_spatial_transform(planar)
            derivative = se2.left_jacobian(-increment, robot.tangent_eps) @ (
                length * basis[2:5]
            )
            derivative = jnp.pad(derivative, ((2, 1), (0, 0)))
        else:
            increment = length * strain
            transform = se3.exp(increment, robot.global_eps)
            derivative = se3.left_jacobian(-increment, robot.tangent_eps) @ (
                length * basis
            )
        return transform, derivative

    transforms, jacobians = vmap(evaluate)(strains, bases, robot.segment_length)
    return transforms, jacobians, jnp.broadcast_to(jnp.eye(4), transforms.shape)
