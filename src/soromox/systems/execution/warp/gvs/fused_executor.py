"""JAX-facing fused GVS dynamics and threadlike-force executor."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from soromox.systems.execution.types import DynamicsActuationTerms
from soromox.systems.execution.warp.common.joint_terms import _evaluate_joint_terms
from soromox.systems.execution.warp.gvs.actuation.operands import (
    GVSThreadlikeOperands,
    GVSThreadlikeShapes,
)
from soromox.systems.execution.warp.gvs.executor import _cell_terms, _gather_local
from soromox.systems.execution.warp.gvs.fused import (
    gvs_dynamics_and_threadlike_actuation_force,
)
from soromox.systems.execution.warp.gvs.operands import GVSOperands, GVSPipelineShapes

SPATIAL_DIM = 6


def execute_dynamics_and_threadlike_actuation_force(
    dynamics: GVSOperands,
    actuation: GVSThreadlikeOperands,
    q: Array,
    qd: Array,
    controls: Array,
) -> DynamicsActuationTerms:
    """Return batched ``(B, Cqd, G, tau_u)`` from one final Warp callable."""

    joint = _evaluate_joint_terms(
        q,
        qd,
        dynamics.joint_basis,
        dynamics.joint_reference,
        dynamics.joint_local_to_global,
        dynamics.joint_global_to_local,
        cooperative=dynamics.block_dim > 1,
    )
    q_link = _gather_local(q, dynamics.link_local_to_global)
    qd_link = _gather_local(qd, dynamics.link_local_to_global)
    cell = _cell_terms(dynamics, q_link, qd_link)

    batch_size = q.shape[0]
    num_segments = dynamics.num_segments
    joint_rows = batch_size * num_segments * SPATIAL_DIM
    cell_rows = batch_size * num_segments * dynamics.num_cells * SPATIAL_DIM
    dynamics_shapes = GVSPipelineShapes.from_operands(dynamics, batch_size=batch_size)
    actuation_shapes = GVSThreadlikeShapes.from_operands(
        actuation, batch_size=batch_size
    )
    output_dims = {
        **dynamics_shapes.chain_outputs(),
        "strain_workspace": actuation_shapes.strain_workspace(),
        "actuation_force": actuation_shapes.force_output(),
    }
    routing = actuation.routing
    outputs = gvs_dynamics_and_threadlike_actuation_force(
        joint[0].reshape(joint_rows, SPATIAL_DIM),
        joint[1].reshape(joint_rows, SPATIAL_DIM),
        joint[2].reshape(joint_rows, dynamics.num_dofs),
        joint[3].reshape(joint_rows, 1),
        joint[4].reshape(joint_rows, 1),
        cell[0].reshape(cell_rows, SPATIAL_DIM),
        cell[1].reshape(cell_rows, dynamics.max_dof),
        cell[2].reshape(cell_rows, 1),
        cell[3].reshape(cell_rows, 1),
        cell[4].reshape(cell_rows, 1),
        dynamics.link_global_to_local,
        dynamics.active_dofs_per_segment,
        qd,
        dynamics.inertia_upper_rows,
        dynamics.inertia_upper_columns,
        dynamics.weighted_mass_diagonals.reshape(
            num_segments * dynamics.num_quadrature, SPATIAL_DIM
        ),
        dynamics.gravity_base,
        jnp.asarray([dynamics.num_cells], dtype=jnp.int32),
        jnp.asarray([dynamics.num_quadrature], dtype=jnp.int32),
        jnp.asarray([dynamics.block_dim], dtype=jnp.int32),
        q,
        controls,
        actuation.link_local_to_global,
        actuation.link_basis,
        actuation.reference_strain,
        actuation.physical_points,
        actuation.integration_weights,
        routing.intercepts,
        routing.slopes,
        routing.start_segments,
        routing.end_segments,
        routing.coordinate_scales,
        jnp.asarray([actuation.global_eps], dtype=jnp.float64),
        output_dims=output_dims,
    )
    return outputs[8], outputs[9], outputs[10], outputs[12]


__all__ = ["execute_dynamics_and_threadlike_actuation_force"]
