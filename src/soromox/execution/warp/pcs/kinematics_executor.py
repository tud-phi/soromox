"""JAX-facing fused Warp kinematics executor for PlanarPCS and PCS."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from soromox.execution.types import KinematicsOperation, KinematicsResult
from soromox.execution.warp.common.floating_kinematics import (
    planar_floating_jacobian_composition,
    planar_floating_kinematics_composition,
    planar_floating_pose_composition,
    spatial_floating_jacobian_composition,
    spatial_floating_kinematics_composition,
    spatial_floating_pose_composition,
)
from soromox.execution.warp.pcs.operands import (
    PCSFloatingKinematicsOperands,
    PCSKinematicsOperands,
    PCSKinematicsShapes,
)
from soromox.execution.warp.pcs.planar_kinematics import (
    planar_forward_kinematics,
    planar_inertial_jacobians,
    planar_kinematics,
)
from soromox.execution.warp.pcs.spatial_kinematics import (
    spatial_cooperative_inertial_jacobians,
    spatial_cooperative_kinematics,
    spatial_forward_kinematics,
    spatial_inertial_jacobians,
    spatial_kinematics,
)


def execute_kinematics(
    operands: PCSKinematicsOperands | PCSFloatingKinematicsOperands,
    q: Array,
    s: Array,
    operation: KinematicsOperation,
) -> KinematicsResult:
    """Execute batched PCS poses and inertial-frame Jacobians in Warp.

    Args:
        operands: Runtime PCS or PlanarPCS model data.
        q: Batched active configurations with shape ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        operation: Select poses, inertial Jacobians, or both.

    Returns:
        Poses, Jacobians, or their tuple with canonical ``(E, N, ...)``
        leading dimensions.
    """

    if isinstance(operands, PCSFloatingKinematicsOperands):
        base_pose = q[:, : operands.num_base_coordinates]
        q_internal = q[:, operands.num_base_coordinates :]
        relative_operation: KinematicsOperation = (
            "pose" if operation == "pose" else "both"
        )
        relative = execute_kinematics(
            operands.relative, q_internal, s, relative_operation
        )
        if operation == "pose":
            relative_pose = relative
            pose_shape = (
                (q.shape[0], s.shape[1], 3)
                if operands.is_planar
                else (q.shape[0], s.shape[1], 4, 4)
            )
            callable_ = (
                planar_floating_pose_composition
                if operands.is_planar
                else spatial_floating_pose_composition
            )
            return callable_(
                base_pose,
                relative_pose,
                output_dims={"poses": pose_shape},
            )[-1]
        relative_pose, internal_jacobian = relative
        jacobian_shape = (
            q.shape[0],
            s.shape[1],
            3 if operands.is_planar else 6,
            operands.num_velocities,
        )
        if operation == "jacobian":
            callable_ = (
                planar_floating_jacobian_composition
                if operands.is_planar
                else spatial_floating_jacobian_composition
            )
            return callable_(
                base_pose,
                relative_pose,
                internal_jacobian,
                output_dims={"jacobians": jacobian_shape},
            )[-1]
        pose_shape = (
            (q.shape[0], s.shape[1], 3)
            if operands.is_planar
            else (q.shape[0], s.shape[1], 4, 4)
        )
        callable_ = (
            planar_floating_kinematics_composition
            if operands.is_planar
            else spatial_floating_kinematics_composition
        )
        outputs = callable_(
            base_pose,
            relative_pose,
            internal_jacobian,
            output_dims={"poses": pose_shape, "jacobians": jacobian_shape},
        )
        return outputs[-2], outputs[-1]

    shapes = PCSKinematicsShapes.from_operands(
        operands,
        batch_size=q.shape[0],
        num_samples=s.shape[1],
    )
    common = (
        q,
        s,
        operands.active_strain_indices,
        operands.active_strain_scales,
        operands.reference_strain.reshape(
            operands.num_segments, 3 if operands.is_planar else 6
        ),
        operands.segment_lengths,
        operands.segment_starts,
        operands.base_pose,
    )
    if operands.is_planar:
        epsilons = jnp.asarray(
            [operands.global_eps, operands.tangent_eps], dtype=jnp.float64
        )
        if operation == "pose":
            output_dims = shapes.pose_workspace()
            output_dims["poses"] = shapes.pose_output()
            return planar_forward_kinematics(
                *common, epsilons, output_dims=output_dims
            )[-1]
        if operation == "jacobian":
            output_dims = shapes.jacobian_workspace()
            output_dims["jacobians"] = shapes.jacobian_output()
            return planar_inertial_jacobians(
                *common, epsilons, output_dims=output_dims
            )[-1]
        output_dims = shapes.workspace()
        output_dims.update(
            {
                "poses": shapes.pose_output(),
                "jacobians": shapes.jacobian_output(),
            }
        )
        outputs = planar_kinematics(*common, epsilons, output_dims=output_dims)
    else:
        if operation == "pose":
            output_dims = shapes.pose_workspace()
            output_dims["poses"] = shapes.pose_output()
            return spatial_forward_kinematics(*common, output_dims=output_dims)[-1]
        if operation == "jacobian":
            output_dims = shapes.jacobian_workspace()
            output_dims["jacobians"] = shapes.jacobian_output()
            evaluator = (
                spatial_cooperative_inertial_jacobians
                if jax.default_backend() == "gpu"
                else spatial_inertial_jacobians
            )
            return evaluator(*common, output_dims=output_dims)[-1]
        output_dims = shapes.workspace()
        output_dims.update(
            {
                "poses": shapes.pose_output(),
                "jacobians": shapes.jacobian_output(),
            }
        )
        evaluator = (
            spatial_cooperative_kinematics
            if jax.default_backend() == "gpu"
            else spatial_kinematics
        )
        outputs = evaluator(*common, output_dims=output_dims)
    poses, jacobians = outputs[-2:]
    if operation == "pose":
        return poses
    if operation == "jacobian":
        return jacobians
    return poses, jacobians


__all__ = ["execute_kinematics"]
