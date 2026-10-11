"""Family executor for persistent batched GVS dynamics in Warp."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from soromox.execution.warp.common.joint_terms import _evaluate_joint_terms
from soromox.execution.warp.gvs.cell import scalable_cell_terms
from soromox.execution.warp.gvs.chain import scalable_persistent_chain
from soromox.execution.warp.gvs.floating_chain import (
    scalable_floating_persistent_chain,
)
from soromox.execution.warp.gvs.operands import (
    GVSFloatingOperands,
    GVSFloatingPipelineShapes,
    GVSOperands,
    GVSPipelineShapes,
)

SPATIAL_DIM = 6


def _gather_local(values: Array, local_to_global: Array) -> Array:
    """Gather padded segment-local coordinates from batched active states.

    Args:
        values: Active coordinates with shape ``(batch_size, num_dofs)``.
        local_to_global: Per-segment map from padded local columns to active
            global columns. Negative entries denote padding.

    Returns:
        Padded local values with shape
        ``(batch_size, num_segments, max_dof)``. Padding entries are zero.
    """

    safe_indices = jnp.maximum(local_to_global, 0)
    gathered = jnp.take(values, safe_indices, axis=1)
    return gathered * (local_to_global >= 0)[None, :, :]


def _cell_terms(
    operands: GVSOperands | GVSFloatingOperands,
    q_link: Array,
    qd_link: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Evaluate spatially varying link-cell Lie terms in local coordinates.

    Args:
        operands: GVS runtime operand bundle.
        q_link: Batched padded link coordinates with shape
            ``(batch_size, num_segments, max_dof)``.
        qd_link: Link velocities with the same shape as ``q_link``.

    Returns:
        Cell adjoints, local tangents, link velocities, step velocities, and
        tangent-velocity derivatives for every environment, segment, and cell.
    """

    batch_size = q_link.shape[0]
    num_segments = operands.num_segments
    num_cells = operands.num_cells
    max_dof = operands.max_dof
    shape_type = (
        GVSFloatingPipelineShapes
        if isinstance(operands, GVSFloatingOperands)
        else GVSPipelineShapes
    )
    output_dims = shape_type.from_operands(
        operands, batch_size=batch_size
    ).cell_outputs()
    outputs = scalable_cell_terms(
        q_link.reshape(batch_size * num_segments, max_dof),
        qd_link.reshape(batch_size * num_segments, max_dof),
        operands.link_basis_z1_values.reshape(num_segments * num_cells, max_dof),
        operands.link_basis_z2_values.reshape(num_segments * num_cells, max_dof),
        operands.link_basis_rows,
        operands.link_reference_z1.reshape(num_segments * num_cells, SPATIAL_DIM),
        operands.link_reference_z2.reshape(num_segments * num_cells, SPATIAL_DIM),
        operands.segment_lengths,
        operands.cell_widths.reshape(num_segments * num_cells),
        jnp.asarray([num_cells], dtype=jnp.int32),
        jnp.asarray([0], dtype=jnp.int32),
        output_dims=output_dims,
    )
    leading = (batch_size, num_segments, num_cells, SPATIAL_DIM)
    return (
        outputs[0].reshape(*leading, SPATIAL_DIM),
        outputs[1].reshape(*leading, max_dof),
        outputs[2].reshape(*leading),
        outputs[3].reshape(*leading),
        outputs[4].reshape(*leading),
    )


def _local_dynamics_terms(
    operands: GVSOperands | GVSFloatingOperands,
    q_internal: Array,
    qd_internal: Array,
) -> tuple[Array, ...]:
    """Evaluate the shared joint and link-cell operators once.

    Args:
        operands: Fixed or floating GVS runtime operands.
        q_internal: Batched internal generalized coordinates.
        qd_internal: Batched internal generalized velocities.

    Returns:
        Joint operators followed by cell operators in chain-kernel order.
    """
    joint_terms = _evaluate_joint_terms(
        q_internal,
        qd_internal,
        operands.joint_basis,
        operands.joint_reference,
        operands.joint_local_to_global,
        operands.joint_global_to_local,
        cooperative=operands.block_dim > 1,
    )
    q_link = _gather_local(q_internal, operands.link_local_to_global)
    qd_link = _gather_local(qd_internal, operands.link_local_to_global)
    return (*joint_terms, *_cell_terms(operands, q_link, qd_link))


def execute_dynamics_terms(
    operands: GVSOperands | GVSFloatingOperands, q: Array, qd: Array
) -> tuple[Array, Array, Array]:
    """Assemble batched fixed or floating GVS dynamics in Warp.

    General-joint and spatially varying cell operators are evaluated once in
    their internal coordinates. A statically selected persistent entry kernel
    then walks the same serial chain with fixed or augmented root state.

    Args:
        operands: Fixed or floating shape-generic GVS runtime operands.
        q: FP64 generalized coordinates in the model's storage layout.
        qd: FP64 generalized velocities in the model's velocity layout.

    Returns:
        Batched inertia, convective-force, and gravity-force terms.
    """
    floating = isinstance(operands, GVSFloatingOperands)
    q_internal = q[:, 7:] if floating else q
    qd_internal = qd[:, 6:] if floating else qd
    (
        joint_adjoint,
        joint_adjoint_dot,
        joint_tangent,
        joint_tangent_dot_qd,
        joint_velocity,
        cell_adjoint,
        cell_tangent_local,
        cell_link_velocity,
        cell_step_velocity,
        cell_tangent_velocity_dot,
    ) = _local_dynamics_terms(operands, q_internal, qd_internal)

    batch_size = q.shape[0]
    num_segments = operands.num_segments
    num_cells = operands.num_cells
    num_quadrature = operands.num_quadrature
    num_internal_dofs = q_internal.shape[1]
    max_dof = operands.max_dof
    joint_rows = batch_size * num_segments * SPATIAL_DIM
    cell_rows = batch_size * num_segments * num_cells * SPATIAL_DIM
    shape_type = GVSFloatingPipelineShapes if floating else GVSPipelineShapes
    output_dims = shape_type.from_operands(
        operands, batch_size=batch_size
    ).chain_outputs()
    common = (
        joint_adjoint.reshape(joint_rows, SPATIAL_DIM),
        joint_adjoint_dot.reshape(joint_rows, SPATIAL_DIM),
        joint_tangent.reshape(joint_rows, num_internal_dofs),
        joint_tangent_dot_qd.reshape(joint_rows, 1),
        joint_velocity.reshape(joint_rows, 1),
        cell_adjoint.reshape(cell_rows, SPATIAL_DIM),
        cell_tangent_local.reshape(cell_rows, max_dof),
        cell_link_velocity.reshape(cell_rows, 1),
        cell_step_velocity.reshape(cell_rows, 1),
        cell_tangent_velocity_dot.reshape(cell_rows, 1),
        operands.link_global_to_local,
        operands.active_dofs_per_segment,
    )
    trailing = (
        operands.inertia_upper_rows,
        operands.inertia_upper_columns,
        operands.weighted_mass_diagonals.reshape(
            num_segments * num_quadrature, SPATIAL_DIM
        ),
    )
    launch_sizes = (
        jnp.asarray([num_cells], dtype=jnp.int32),
        jnp.asarray([num_quadrature], dtype=jnp.int32),
        jnp.asarray([operands.block_dim], dtype=jnp.int32),
        operands.block_dim,
    )
    if floating:
        assert isinstance(operands, GVSFloatingOperands)
        outputs = scalable_floating_persistent_chain(
            *common,
            q[:, :7],
            qd,
            *trailing,
            operands.gravity_world,
            *launch_sizes,
            output_dims=output_dims,
        )
    else:
        assert isinstance(operands, GVSOperands)
        outputs = scalable_persistent_chain(
            *common,
            qd,
            *trailing,
            operands.gravity_base,
            *launch_sizes,
            output_dims=output_dims,
        )
    return outputs[-3], outputs[-2], outputs[-1]


__all__ = ["execute_dynamics_terms"]
