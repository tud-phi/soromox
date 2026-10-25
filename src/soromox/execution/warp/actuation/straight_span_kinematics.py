"""Relative continuum kinematics for native free-space actuation.

Model data is prepared once. Runtime configurations are consumed entirely by
Warp. Constant strains use one exact exponential; variable GVS strains retain
the model's padded Magnus grid.
The SE(3) operators are shared with the existing PCS and GVS Warp executors.
"""

import equinox as eqx
import jax.numpy as jnp
import warp as wp

from soromox.execution.warp.common.se3 import (
    Vec6d,
    _ad_action,
    _exponential_transform,
    _left_action,
    _left_coefficients,
)


class StraightKinematicsOperands(eqx.Module):
    """Prepared link and joint strains in internal coordinate order.

    Cell zero is the preceding joint, with unit integration distance. Remaining
    cells describe the link. Basis arrays have shape ``(S, C, 6, D)`` and
    references ``(S, C, 6)``; distances have shape ``(S, C)``. Floating base
    coordinates are excluded from the bases and skipped when reading ``q``.
    """

    basis1: jnp.ndarray
    basis2: jnp.ndarray
    reference1: jnp.ndarray
    reference2: jnp.ndarray
    distances: jnp.ndarray
    num_base_coordinates: int = eqx.field(static=True)

    @classmethod
    def from_model(cls, robot):
        """Prepare numerical operands from PCS, PlanarPCS, or GVS model data."""
        if hasattr(robot, "B_Z1"):
            mapping = robot.active_dof_map_blocks
            basis1 = robot.B_Z1
            basis2 = robot.B_Z2
            if robot.scale_rotational_basis_by_length:
                scale = jnp.concatenate(
                    (
                        jnp.broadcast_to(
                            1 / robot.segment_lengths[:, None], (robot.num_segments, 3)
                        ),
                        jnp.ones((robot.num_segments, 3)),
                    ),
                    axis=1,
                )
                basis1 = basis1 * scale[:, None, :, None]
                basis2 = basis2 * scale[:, None, :, None]
            basis1 = jnp.einsum("scik,skd->scid", basis1, mapping[:, 1])
            basis2 = jnp.einsum("scik,skd->scid", basis2, mapping[:, 1])
            joint = jnp.einsum("sik,skd->sid", robot.B_joint, mapping[:, 0])
            reference1, reference2 = robot.xi_ref_Z1, robot.xi_ref_Z2
            joint_reference = robot.xi_ref_joint
            distances = robot.segment_lengths[:, None] * jnp.diff(
                robot.integration_points, axis=1
            )
            if all(
                segment.basis.type in ("monomial", "legendre", "chebyshev")
                and all(order == 0 for order in segment.basis.basis_order)
                for segment in robot.structure.segments
            ):
                # A constant strain has one exact exponential, independent of
                # the quadrature grid. Reference strain is constant per link.
                basis1, basis2 = basis1[:, :1], basis2[:, :1]
                reference1, reference2 = reference1[:, :1], reference2[:, :1]
                distances = robot.segment_lengths[:, None]
        elif hasattr(robot, "B_xi"):
            size = 3 if robot.is_planar else 6
            basis1 = robot.B_xi.reshape(
                robot.num_segments, 1, size, robot.num_internal_dofs
            )
            reference1 = robot.xi_ref.reshape(robot.num_segments, 1, size)
            if robot.is_planar:
                basis1 = jnp.pad(basis1, ((0, 0), (0, 0), (2, 1), (0, 0)))
                reference1 = jnp.pad(reference1, ((0, 0), (0, 0), (2, 1)))
            basis2, reference2 = basis1, reference1
            joint = jnp.zeros((robot.num_segments, 6, robot.num_internal_dofs))
            joint_reference = jnp.zeros((robot.num_segments, 6))
            distances = robot.segment_length[:, None]
        else:
            raise NotImplementedError(
                "Native Warp routing supports PlanarPCS, PCS, and GVS."
            )
        return cls(
            jnp.concatenate((joint[:, None], basis1), axis=1),
            jnp.concatenate((joint[:, None], basis2), axis=1),
            jnp.concatenate((joint_reference[:, None], reference1), axis=1),
            jnp.concatenate((joint_reference[:, None], reference2), axis=1),
            jnp.concatenate((jnp.ones((robot.num_segments, 1)), distances), axis=1),
            robot.num_base_coordinates,
        )


@wp.func
def _transport(transform: wp.mat44d, vector: Vec6d) -> Vec6d:
    """Apply an inverse adjoint to an angular-linear vector."""
    rotation = wp.mat33d()
    translation = wp.vec3d()
    for i in range(3):
        translation[i] = transform[i, 3]
        for j in range(3):
            rotation[i, j] = transform[i, j]
    angular = wp.vec3d(vector[0], vector[1], vector[2])
    linear = wp.transpose(rotation) @ (
        wp.vec3d(vector[3], vector[4], vector[5]) - wp.cross(translation, angular)
    )
    angular = wp.transpose(rotation) @ angular
    return Vec6d(angular[0], angular[1], angular[2], linear[0], linear[1], linear[2])


@wp.kernel(enable_backward=False)
def strain_samples_kernel(
    q: wp.array2d[wp.float64],
    basis1: wp.array4d[wp.float64],
    basis2: wp.array4d[wp.float64],
    reference1: wp.array3d[wp.float64],
    reference2: wp.array3d[wp.float64],
    strains: wp.array4d[wp.float64],
):
    """Evaluate shared strain samples once, in parallel over spatial components."""
    environment, segment, cell, row = wp.tid()
    first = reference1[segment, cell, row]
    second = reference2[segment, cell, row]
    base = q.shape[1] - basis1.shape[3]
    for column in range(basis1.shape[3]):
        value = q[environment, base + column]
        first += basis1[segment, cell, row, column] * value
        second += basis2[segment, cell, row, column] * value
    index = segment * basis1.shape[1] + cell
    strains[environment, index, 0, row] = first
    strains[environment, index, 1, row] = second


@wp.kernel(enable_backward=False)
def relative_kinematics_kernel(
    strains: wp.array4d[wp.float64],
    basis1: wp.array4d[wp.float64],
    basis2: wp.array4d[wp.float64],
    distances: wp.array2d[wp.float64],
    links: wp.array4d[wp.float64],
    link_jacobians: wp.array4d[wp.float64],
    joints: wp.array4d[wp.float64],
    joint_jacobians: wp.array4d[wp.float64],
):
    """Integrate relative link poses and body Jacobians, with one lane per column.

    Output Jacobians contain only internal coordinates. Each segment is
    independent; no world or chain Jacobians are formed. All output entries
    are overwritten, including inactive columns and padded cells.
    """
    environment, segment, lane = wp.tid()
    pose = wp.identity(n=4, dtype=wp.float64)
    derivative = Vec6d()
    for cell in range(distances.shape[1]):
        xi1, xi2 = Vec6d(), Vec6d()
        b1, b2 = Vec6d(), Vec6d()
        for row in range(6):
            index = segment * distances.shape[1] + cell
            xi1[row] = strains[environment, index, 0, row]
            xi2[row] = strains[environment, index, 1, row]
            if lane > 0:
                b1[row] = basis1[segment, cell, row, lane - 1]
                b2[row] = basis2[segment, cell, row, lane - 1]
        distance = distances[segment, cell]
        alpha = distance * wp.float64(0.5)
        beta = wp.sqrt(wp.float64(3.0)) * distance * distance / wp.float64(12.0)
        increment = alpha * (xi1 + xi2) + beta * _ad_action(xi1, xi2)
        step = _exponential_transform(increment)
        if lane == 0:
            pose = pose @ step
        else:
            variation = alpha * (b1 + b2) + beta * (
                _ad_action(xi1, b2) - _ad_action(xi2, b1)
            )
            angle_sq = (
                increment[0] * increment[0]
                + increment[1] * increment[1]
                + increment[2] * increment[2]
            )
            derivative = _transport(
                step,
                derivative
                + _left_action(increment, variation, _left_coefficients(angle_sq)),
            )
        if cell == 0:
            if lane == 0:
                for row in range(4):
                    for column in range(4):
                        joints[environment, segment, row, column] = pose[row, column]
                pose = wp.identity(n=4, dtype=wp.float64)
            else:
                for row in range(6):
                    joint_jacobians[environment, segment, row, lane - 1] = derivative[
                        row
                    ]
                derivative = Vec6d()
    if lane == 0:
        for row in range(4):
            for column in range(4):
                links[environment, segment, row, column] = pose[row, column]
    else:
        for row in range(6):
            link_jacobians[environment, segment, row, lane - 1] = derivative[row]
