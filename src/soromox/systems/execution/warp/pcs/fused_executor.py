"""JAX-facing fused PlanarPCS/PCS dynamics and threadlike-force executor."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from soromox.systems.execution.types import DynamicsActuationTerms
from soromox.systems.execution.warp.pcs.actuation.operands import (
    PCSThreadlikeOperands,
    PCSThreadlikeShapes,
)
from soromox.systems.execution.warp.pcs.floating_fused import (
    planar_floating_dynamics_and_threadlike_actuation_force,
    spatial_floating_dynamics_and_threadlike_actuation_force,
)
from soromox.systems.execution.warp.pcs.fused import (
    planar_dynamics_and_threadlike_actuation_force,
    spatial_dynamics_and_threadlike_actuation_force,
)
from soromox.systems.execution.warp.pcs.operands import (
    PCSFloatingOperands,
    PCSFloatingPipelineShapes,
    PCSOperands,
    PCSPipelineShapes,
)


def execute_dynamics_and_threadlike_actuation_force(
    dynamics: PCSOperands | PCSFloatingOperands,
    actuation: PCSThreadlikeOperands,
    q: Array,
    qd: Array,
    controls: Array,
) -> DynamicsActuationTerms:
    """Return batched ``(B, Cqd, G, tau_u)`` for either PCS dimension."""

    batch_size = q.shape[0]
    num_segments = dynamics.num_segments
    floating = isinstance(dynamics, PCSFloatingOperands)
    dynamics_shapes = (
        PCSFloatingPipelineShapes.from_operands(dynamics, batch_size=batch_size)
        if floating
        else PCSPipelineShapes.from_operands(dynamics, batch_size=batch_size)
    )
    actuation_shapes = PCSThreadlikeShapes.from_operands(
        actuation, batch_size=batch_size
    )
    output_dims = {
        **dynamics_shapes.operator_outputs(),
        **dynamics_shapes.chain_outputs(),
    }
    if floating:
        output_dims.update(
            {
                "actuation_internal": actuation_shapes.force_output(),
                "actuation_force": (batch_size, dynamics.num_velocities),
            }
        )
    else:
        output_dims["actuation_force"] = actuation_shapes.force_output()
    routing = actuation.routing
    q_internal = q[:, 3 if dynamics.is_planar else 7 :] if floating else q
    qd_internal = qd[:, 3 if dynamics.is_planar else 6 :] if floating else qd
    common = (
        q_internal,
        qd_internal,
        controls,
        dynamics.active_strain_indices,
        dynamics.active_strain_scales,
        dynamics.reference_strain.reshape(num_segments, 3 if dynamics.is_planar else 6),
        dynamics.local_points,
        dynamics.active_dof_ends,
        dynamics.inertia_upper_rows,
        dynamics.inertia_upper_columns,
        dynamics.weighted_mass_diagonals.reshape(
            num_segments * dynamics.num_gauss_points,
            3 if dynamics.is_planar else 6,
        ),
    )
    trailing = (
        actuation.physical_points,
        actuation.physical_weights,
        routing.intercepts,
        routing.slopes,
        routing.start_segments,
        routing.end_segments,
        routing.coordinate_scales,
        jnp.asarray(
            [
                actuation.global_eps,
                dynamics.tangent_eps if dynamics.is_planar else 0.0,
            ],
            dtype=jnp.float64,
        ),
    )
    if floating:
        assert isinstance(dynamics, PCSFloatingOperands)
        callable_ = (
            planar_floating_dynamics_and_threadlike_actuation_force
            if dynamics.is_planar
            else spatial_floating_dynamics_and_threadlike_actuation_force
        )
        outputs = callable_(
            common[0],
            common[1],
            q[:, : 3 if dynamics.is_planar else 7],
            qd,
            *common[2:],
            dynamics.gravity_world,
            dynamics.block_dim,
            *trailing,
            output_dims=output_dims,
        )
        final_offset = 4 + (9 if dynamics.is_planar else 11)
        return (
            outputs[final_offset - 3],
            outputs[final_offset - 2],
            outputs[final_offset - 1],
            outputs[final_offset + 1],
        )
    assert isinstance(dynamics, PCSOperands)
    callable_ = (
        planar_dynamics_and_threadlike_actuation_force
        if dynamics.is_planar
        else spatial_dynamics_and_threadlike_actuation_force
    )
    outputs = callable_(
        *common,
        dynamics.gravity_base,
        dynamics.block_dim,
        *trailing,
        output_dims=output_dims,
    )
    chain_offset = 4
    final_offset = chain_offset + (9 if dynamics.is_planar else 11)
    return (
        outputs[final_offset - 3],
        outputs[final_offset - 2],
        outputs[final_offset - 1],
        outputs[final_offset],
    )


__all__ = ["execute_dynamics_and_threadlike_actuation_force"]
