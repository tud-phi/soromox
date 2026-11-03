"""Shared relative continuum kinematics and serial-chain frame composition.

Model data is prepared once. Runtime configurations are consumed entirely by
Warp. Constant strains use one exact exponential; variable GVS strains retain
the model's padded Magnus grid.
The SE(3) operators are shared with the existing PCS and GVS Warp executors.
"""

import equinox as eqx
import jax.numpy as jnp
import warp as wp

from soromox.execution.warp.common.rotations import _quaternion_rotation_matrix
from soromox.execution.warp.common.se3 import (
    Vec6d,
    _ad_action,
    _exponential_transform,
    _left_action,
    _left_coefficients,
)


class RelativeKinematicsOperands(eqx.Module):
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
        """Prepare link and joint operands from continuum robot data.

        Args:
            robot: PCS, PlanarPCS, or GVS host exposing strain bases, reference
                strains, integration grids, and internal coordinate maps.

        Returns:
            RelativeKinematicsOperands: Shared numerical arrays in internal
                coordinate order. Constant-strain GVS uses a single exact link
                exponential; variable strain retains the padded Magnus grid.
        """
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
                "Native relative kinematics supports PlanarPCS, PCS, and GVS."
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
    """Evaluate shared strain samples once, in parallel over spatial components.

    E denotes environments, L links, C cells including the joint, D internal
    coordinates, Q configurations, V velocities, S spans, I intervals, and P paths.
    All buffers use FP64 except int32 topology and boolean flags.

    Args:
        q: Configurations ``(E, Q)`` including any base coordinates.
        basis1: First strain bases ``(L, C, 6, D)`` in internal coordinate order.
        basis2: Second strain bases at the other Magnus sample, same shape.
        reference1: First reference strains ``(L, C, 6)``.
        reference2: Second reference strains ``(L, C, 6)``.
        strains: Shared strain output workspace ``(E, L*C, 2, 6)``.

    Returns:
        None. Launch with ``dim=(E, L, C, 6)``; every strain entry is written.
    """
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

    E denotes environments, L links, C cells including the joint, D internal
    coordinates, Q configurations, V velocities, S spans, I intervals, and P paths.
    All buffers use FP64 except int32 topology and boolean flags.

    Args:
        strains: Prepared shared strains ``(E, L*C, 2, 6)``.
        basis1: First strain bases ``(L, C, 6, D)`` in internal coordinate order.
        basis2: Second strain bases at the other Magnus sample, same shape.
        distances: Integration distances ``(L, C)``; cell zero represents the joint.
        links: Relative link transform output workspace ``(E, L, 4, 4)``.
        link_jacobians: Link body Jacobian output workspace ``(E, L, 6, D)``.
        joints: Joint transform output workspace ``(E, L, 4, 4)``.
        joint_jacobians: Joint body Jacobian output workspace ``(E, L, 6, D)``.

    Returns:
        None. Launch with ``dim=(E, L, D + 1)``; lane zero writes poses.
        All Jacobian entries are overwritten, including inactive columns.
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


@wp.func
def read_transform(values: wp.array4d[wp.float64], environment: int, index: int):
    """Read a homogeneous transform from an ``(E, S, 4, 4)`` array.

    Args:
        values: Transform array.
        environment: Environment index.
        index: Transform index within the environment.

    Returns:
        The selected FP64 matrix.
    """
    result = wp.mat44d()
    for row in range(4):
        for column in range(4):
            result[row, column] = values[environment, index, row, column]
    return result


@wp.kernel(enable_backward=False)
def relative_intervals_kernel(
    links: wp.array4d[wp.float64],
    joints: wp.array4d[wp.float64],
    body_jacobians: wp.array4d[wp.float64],
    intervals: wp.array2d[wp.int32],
    transforms: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Compose serial-chain intervals and their body Jacobians.

    Args:
        links: Link transforms, ``(E, S, 4, 4)``.
        joints: Preceding joint transforms, ``(E, S, 4, 4)``. Interior joints
            of each interval must be fixed.
        body_jacobians: Link body Jacobians, angular-linear order, ``(E, S, 6, D)``.
        intervals: Inclusive first and last segment indices, ``(I, 2)``.
        transforms: Output interval transforms, ``(E, I, 4, 4)``.
        jacobians: Output interval body Jacobians, ``(E, I, 6, D)``.

    Returns:
        None. Launch with dimensions ``(E, I, D + 1)``; lane zero writes poses.
    """
    environment, interval, lane = wp.tid()
    first = intervals[interval, 0]
    last = intervals[interval, 1]
    if lane == 0:
        pose = read_transform(links, environment, first)
        for segment in range(first + 1, last + 1):
            pose = pose @ (
                read_transform(joints, environment, segment)
                @ read_transform(links, environment, segment)
            )
        for row in range(4):
            for column in range(4):
                transforms[environment, interval, row, column] = pose[row, column]
    else:
        coordinate = lane - 1
        angular = wp.vec3d()
        linear = wp.vec3d()
        for component in range(3):
            angular[component] = body_jacobians[
                environment, first, component, coordinate
            ]
            linear[component] = body_jacobians[
                environment, first, component + 3, coordinate
            ]
        for segment in range(first + 1, last + 1):
            step = read_transform(joints, environment, segment) @ read_transform(
                links, environment, segment
            )
            rotation = wp.mat33d()
            translation = wp.vec3d()
            for row in range(3):
                translation[row] = step[row, 3]
                for column in range(3):
                    rotation[row, column] = step[row, column]
            linear = wp.transpose(rotation) @ (linear - wp.cross(translation, angular))
            angular = wp.transpose(rotation) @ angular
            for component in range(3):
                angular[component] += body_jacobians[
                    environment, segment, component, coordinate
                ]
                linear[component] += body_jacobians[
                    environment, segment, component + 3, coordinate
                ]
        for component in range(3):
            jacobians[environment, interval, component, coordinate] = angular[component]
            jacobians[environment, interval, component + 3, coordinate] = linear[
                component
            ]


@wp.kernel(enable_backward=False)
def segment_boundaries_kernel(
    q: wp.array2d[wp.float64],
    joints: wp.array4d[wp.float64],
    internal_dofs: int,
    links: wp.array4d[wp.float64],
    fixed_base: wp.array2d[wp.float64],
    entries: wp.array4d[wp.float64],
    exits: wp.array4d[wp.float64],
):
    """Compose world entry and exit transforms for a serial joint-link chain.

    Args:
        q: FP64 configurations ``(E, Q)``, with leading base coordinates.
        joints: Preceding joint transforms ``(E, L, 4, 4)``.
        internal_dofs: Number of coordinates excluding the base. The remaining
            base width is zero, three (planar), or seven (quaternion then position).
        links: Relative link transforms ``(E, L, 4, 4)``.
        fixed_base: Fixed base transform ``(4, 4)``; ignored for floating bases.
        entries: Output world link-entry transforms ``(E, L, 4, 4)``.
        exits: Output world link-exit transforms ``(E, L, 4, 4)``.

    Returns:
        None. Launch with ``dim=E``; each thread traverses one complete chain.
    """
    environment = wp.tid()
    base_count = q.shape[1] - internal_dofs
    pose = wp.identity(n=4, dtype=wp.float64)
    if base_count == 0:
        for row in range(4):
            for column in range(4):
                pose[row, column] = fixed_base[row, column]
    elif base_count == 3:
        angle = q[environment, 0]
        pose[0, 0] = wp.cos(angle)
        pose[0, 1] = -wp.sin(angle)
        pose[1, 0] = wp.sin(angle)
        pose[1, 1] = wp.cos(angle)
        pose[0, 3] = q[environment, 1]
        pose[1, 3] = q[environment, 2]
    else:
        rotation = _quaternion_rotation_matrix(q, environment)
        for row in range(3):
            pose[row, 3] = q[environment, row + 4]
            for column in range(3):
                pose[row, column] = rotation[row, column]
    for segment in range(links.shape[1]):
        entry = pose @ read_transform(joints, environment, segment)
        pose = entry @ read_transform(links, environment, segment)
        for row in range(4):
            for column in range(4):
                entries[environment, segment, row, column] = entry[row, column]
                exits[environment, segment, row, column] = pose[row, column]


def launch_kinematics(
    q, operands, strains, links, link_jacobians, joints, joint_jacobians
):
    """Enqueue shared strain evaluation followed by relative link kinematics.

    Args:
        q: FP64 configurations ``(E, Q)`` on the target Warp device.
        operands: Warp arrays for basis1, basis2, reference1, reference2, and
            distances, following ``RelativeKinematicsOperands``.
        strains: Output sampled strains ``(E, L*C, 2, 6)``.
        links: Output relative link transforms ``(E, L, 4, 4)``.
        link_jacobians: Output body Jacobians ``(E, L, 6, D)``.
        joints: Output preceding-joint transforms ``(E, L, 4, 4)``.
        joint_jacobians: Output joint body Jacobians ``(E, L, 6, D)``.

    Returns:
        None. Two launches fill caller-owned buffers on the current stream.
        L is the number of links, C the padded cell count including the joint,
        and D the number of internal coordinates.
    """
    wp.launch(
        strain_samples_kernel,
        dim=(q.shape[0], links.shape[1], operands[4].shape[1], 6),
        inputs=[q, *operands[:4]],
        outputs=[strains],
        device=q.device,
    )
    wp.launch(
        relative_kinematics_kernel,
        dim=(q.shape[0], links.shape[1], link_jacobians.shape[3] + 1),
        inputs=[strains, operands[0], operands[1], operands[4]],
        outputs=[links, link_jacobians, joints, joint_jacobians],
        device=q.device,
    )
