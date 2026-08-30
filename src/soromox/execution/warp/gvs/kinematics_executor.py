"""JAX-facing fused Warp kinematics executor for GVS."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from soromox.execution.types import KinematicsOperation, KinematicsResult
from soromox.execution.warp.common.floating_kinematics import (
    spatial_floating_jacobian_composition,
    spatial_floating_kinematics_composition,
    spatial_floating_pose_composition,
)
from soromox.execution.warp.common.joint_terms import _evaluate_joint_terms
from soromox.execution.warp.gvs.executor import (
    _cell_terms,
    _gather_local,
)
from soromox.execution.warp.gvs.kinematics import (
    gvs_forward_kinematics,
    gvs_inertial_jacobians,
    gvs_kinematics,
)
from soromox.execution.warp.gvs.operands import (
    GVSFloatingKinematicsOperands,
    GVSKinematicsOperands,
    GVSKinematicsShapes,
)

SPATIAL_DIM = 6


def _full_local_kinematics(
    operands: GVSKinematicsOperands,
    q: Array,
    q_link: Array,
) -> tuple[Array, Array, Array, Array]:
    """Evaluate pose/Jacobian local terms using the established operators.

    Args:
        operands: Runtime GVS model data and static dimensions.
        q: Batched active configurations with shape ``(E, D)``.
        q_link: Configurations gathered into padded link-local coordinates.

    Returns:
        Joint adjoints, joint tangents, cell adjoints, and cell tangents.
    """

    joint_outputs = _evaluate_joint_terms(
        q,
        jnp.zeros_like(q),
        operands.joint_basis,
        operands.joint_reference,
        operands.joint_local_to_global,
        operands.joint_global_to_local,
        cooperative=operands.block_dim > 1,
    )
    cell_outputs = _cell_terms(operands, q_link, jnp.zeros_like(q_link))
    return joint_outputs[0], joint_outputs[2], cell_outputs[0], cell_outputs[1]


def execute_kinematics(
    operands: GVSKinematicsOperands | GVSFloatingKinematicsOperands,
    q: Array,
    s: Array,
    operation: KinematicsOperation,
) -> KinematicsResult:
    """Execute batched GVS poses and inertial-frame Jacobians in Warp.

    Args:
        operands: Runtime GVS model data and static dimensions.
        q: Batched active configurations with shape ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        operation: Select poses, inertial Jacobians, or both.

    Returns:
        Poses, Jacobians, or their tuple with canonical ``(E, N, ...)``
        leading dimensions.
    """

    if isinstance(operands, GVSFloatingKinematicsOperands):
        base_pose = q[:, : operands.num_base_coordinates]
        q_internal = q[:, operands.num_base_coordinates :]
        relative_operation: KinematicsOperation = (
            "pose" if operation == "pose" else "both"
        )
        relative = execute_kinematics(
            operands.relative, q_internal, s, relative_operation
        )
        pose_shape = (q.shape[0], s.shape[1], 4, 4)
        if operation == "pose":
            return spatial_floating_pose_composition(
                base_pose,
                relative,
                output_dims={"poses": pose_shape},
            )[-1]
        relative_pose, internal_jacobian = relative
        jacobian_shape = (
            q.shape[0],
            s.shape[1],
            SPATIAL_DIM,
            operands.num_velocities,
        )
        if operation == "jacobian":
            return spatial_floating_jacobian_composition(
                base_pose,
                relative_pose,
                internal_jacobian,
                output_dims={"jacobians": jacobian_shape},
            )[-1]
        outputs = spatial_floating_kinematics_composition(
            base_pose,
            relative_pose,
            internal_jacobian,
            output_dims={"poses": pose_shape, "jacobians": jacobian_shape},
        )
        return outputs[-2], outputs[-1]

    q_link = _gather_local(q, operands.link_local_to_global)
    batch_size = q.shape[0]
    shapes = GVSKinematicsShapes.from_operands(
        operands,
        batch_size=batch_size,
        num_samples=s.shape[1],
    )
    output_dims = shapes.workspace()
    output_dims.update(
        {
            "poses": shapes.pose_output(),
            "jacobians": shapes.jacobian_output(),
        }
    )
    joint_rows = batch_size * operands.num_segments * SPATIAL_DIM
    cell_rows = batch_size * operands.num_segments * operands.num_cells * SPATIAL_DIM
    sample_operands = (
        operands.link_basis_rows,
        operands.link_reference_z1.reshape(
            operands.num_segments * operands.num_cells, SPATIAL_DIM
        ),
        operands.link_reference_z2.reshape(
            operands.num_segments * operands.num_cells, SPATIAL_DIM
        ),
        operands.segment_lengths,
        operands.segment_starts,
        operands.integration_points,
        operands.basis_type_index.astype(jnp.int32),
        operands.basis_active_params,
        operands.basis_order_params,
        operands.magnus_points,
        jnp.asarray([operands.scale_rotational_basis_by_length], dtype=jnp.int32),
        jnp.asarray([operands.global_eps, operands.tangent_eps], dtype=jnp.float64),
        jnp.asarray([operands.num_cells], dtype=jnp.int32),
    )
    if operation == "pose":
        joint_adjoint, _, cell_adjoint, _ = _full_local_kinematics(operands, q, q_link)
        joint_adjoint = joint_adjoint.reshape(joint_rows, SPATIAL_DIM)
        cell_adjoint = cell_adjoint.reshape(cell_rows, SPATIAL_DIM)
        output_dims = shapes.pose_workspace()
        output_dims["poses"] = shapes.pose_output()
        return gvs_forward_kinematics(
            q,
            s,
            joint_adjoint,
            cell_adjoint,
            operands.base_transform,
            operands.link_local_to_global,
            *sample_operands,
            output_dims=output_dims,
        )[-1]
    joint_adjoint, joint_tangent, cell_adjoint, cell_tangent = _full_local_kinematics(
        operands, q, q_link
    )
    joint_adjoint = joint_adjoint.reshape(joint_rows, SPATIAL_DIM)
    cell_adjoint = cell_adjoint.reshape(cell_rows, SPATIAL_DIM)
    if operation == "jacobian":
        output_dims = shapes.workspace()
        output_dims["jacobians"] = shapes.jacobian_output()
        return gvs_inertial_jacobians(
            q,
            s,
            joint_adjoint,
            joint_tangent.reshape(joint_rows, operands.num_dofs),
            cell_adjoint,
            cell_tangent.reshape(cell_rows, operands.max_dof),
            operands.base_transform,
            operands.link_local_to_global,
            operands.link_global_to_local,
            *sample_operands,
            operands.block_dim,
            output_dims=output_dims,
        )[-1]
    outputs = gvs_kinematics(
        q,
        s,
        joint_adjoint,
        joint_tangent.reshape(joint_rows, operands.num_dofs),
        cell_adjoint,
        cell_tangent.reshape(cell_rows, operands.max_dof),
        operands.base_transform,
        operands.link_local_to_global,
        operands.link_global_to_local,
        *sample_operands,
        operands.block_dim,
        output_dims=output_dims,
    )
    poses, jacobians = outputs[-2:]
    return poses, jacobians


__all__ = ["execute_kinematics"]
