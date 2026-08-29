"""Family executor for PlanarPCS and PCS dynamics in Warp."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from soromox.systems.execution.warp.pcs.operands import (
    PCSFloatingOperands,
    PCSFloatingPipelineShapes,
    PCSOperands,
    PCSPipelineShapes,
)
from soromox.systems.execution.warp.pcs.planar_floating_kernels import (
    planar_floating_persistent_chain,
)
from soromox.systems.execution.warp.pcs.planar_kernels import (
    planar_local_operators,
    planar_persistent_chain,
)
from soromox.systems.execution.warp.pcs.spatial_floating_kernels import (
    spatial_floating_persistent_chain,
)
from soromox.systems.execution.warp.pcs.spatial_kernels import (
    spatial_local_operators,
    spatial_persistent_chain,
)


def _planar_dynamics_terms(
    operands: PCSOperands, q: Array, qd: Array
) -> tuple[Array, Array, Array]:
    """Execute the staged constant-strain SE(2) dynamics pipeline.

    Args:
        operands: PlanarPCS runtime operands.
        q: FP64 active coordinates with shape ``(batch_size, num_dofs)``.
        qd: FP64 active velocities with the same shape as ``q``.

    Returns:
        Batched planar inertia, convective-force, and gravity-force terms.
    """

    batch_size = q.shape[0]
    num_segments = operands.num_segments
    shapes = PCSPipelineShapes.from_operands(operands, batch_size=batch_size)
    operators = planar_local_operators(
        q,
        qd,
        operands.active_strain_indices,
        operands.active_strain_scales,
        operands.reference_strain.reshape(num_segments, 3),
        operands.local_points,
        jnp.asarray([operands.global_eps, operands.tangent_eps], dtype=jnp.float64),
        output_dims=shapes.operator_outputs(),
    )
    outputs = planar_persistent_chain(
        *operators,
        operands.active_strain_indices,
        operands.active_strain_scales,
        operands.active_dof_ends,
        qd,
        operands.inertia_upper_rows,
        operands.inertia_upper_columns,
        operands.weighted_mass_diagonals.reshape(
            num_segments * operands.num_gauss_points, 3
        ),
        operands.gravity_base,
        operands.block_dim,
        output_dims=shapes.chain_outputs(),
    )
    return outputs[-3], outputs[-2], outputs[-1]


def execute_dynamics_terms(
    operands: PCSOperands | PCSFloatingOperands,
    q: Array,
    qd: Array,
) -> tuple[Array, Array, Array]:
    """Execute the planar or spatial runtime-shaped PCS Warp pipeline.

    Both variants evaluate independent local constant-strain operators in a
    first launch. A persistent block per environment then advances the serial
    segment recurrence and cooperatively assembles only ``B``, ``C @ qd``, and
    ``G``. Planar and spatial systems share the operand and orchestration layer
    but retain dimension-specialized Lie operators and chain kernels.

    Args:
        operands: PlanarPCS or PCS runtime operands. ``operands.is_planar``
            selects the dimension-specific pipeline.
        q: FP64 active coordinates with shape ``(batch_size, num_dofs)``.
        qd: FP64 active velocities with the same shape as ``q``.

    Returns:
        Tuple ``(B, Cqd, G)`` with shapes
        ``(batch_size, num_dofs, num_dofs)``,
        ``(batch_size, num_dofs)``, and ``(batch_size, num_dofs)``.
    """

    if isinstance(operands, PCSFloatingOperands):
        return execute_floating_dynamics_terms(operands, q, qd)
    if operands.is_planar:
        return _planar_dynamics_terms(operands, q, qd)
    batch_size = q.shape[0]
    num_segments = operands.num_segments
    shapes = PCSPipelineShapes.from_operands(operands, batch_size=batch_size)
    operators = spatial_local_operators(
        q,
        qd,
        operands.active_strain_indices,
        operands.active_strain_scales,
        operands.reference_strain.reshape(num_segments, 6),
        operands.local_points,
        output_dims=shapes.operator_outputs(),
    )
    outputs = spatial_persistent_chain(
        *operators,
        operands.active_strain_indices,
        operands.active_strain_scales,
        operands.active_dof_ends,
        qd,
        operands.inertia_upper_rows,
        operands.inertia_upper_columns,
        operands.weighted_mass_diagonals.reshape(
            num_segments * operands.num_gauss_points, 6
        ),
        operands.gravity_base,
        operands.block_dim,
        output_dims=shapes.chain_outputs(),
    )
    return outputs[-3], outputs[-2], outputs[-1]


def execute_floating_dynamics_terms(
    operands: PCSFloatingOperands,
    q: Array,
    velocity: Array,
) -> tuple[Array, Array, Array]:
    """Execute a statically augmented floating PCS Warp pipeline.

    Args:
        operands: Floating PlanarPCS or PCS runtime operands.
        q: Batched total configurations with shape
            ``(batch_size, num_coordinates)``.
        velocity: Batched total generalized velocities with shape
            ``(batch_size, num_velocities)``.

    Returns:
        Batched augmented inertia, convective-force, and gravity-force terms.

    """
    if operands.is_planar:
        num_segments = operands.num_segments
        shapes = PCSFloatingPipelineShapes.from_operands(
            operands, batch_size=q.shape[0]
        )
        q_internal = q[:, 3:]
        velocity_internal = velocity[:, 3:]
        operators = planar_local_operators(
            q_internal,
            velocity_internal,
            operands.active_strain_indices,
            operands.active_strain_scales,
            operands.reference_strain.reshape(num_segments, 3),
            operands.local_points,
            jnp.asarray([operands.global_eps, operands.tangent_eps], dtype=jnp.float64),
            output_dims=shapes.operator_outputs(),
        )
        outputs = planar_floating_persistent_chain(
            *operators,
            operands.active_strain_indices,
            operands.active_strain_scales,
            operands.active_dof_ends,
            q[:, :3],
            velocity,
            operands.inertia_upper_rows,
            operands.inertia_upper_columns,
            operands.weighted_mass_diagonals.reshape(
                num_segments * operands.num_gauss_points, 3
            ),
            operands.gravity_world,
            operands.block_dim,
            output_dims=shapes.chain_outputs(),
        )
        return outputs[-3], outputs[-2], outputs[-1]
    batch_size = q.shape[0]
    num_segments = operands.num_segments
    shapes = PCSFloatingPipelineShapes.from_operands(operands, batch_size=batch_size)
    q_internal = q[:, 7:]
    velocity_internal = velocity[:, 6:]
    operators = spatial_local_operators(
        q_internal,
        velocity_internal,
        operands.active_strain_indices,
        operands.active_strain_scales,
        operands.reference_strain.reshape(num_segments, 6),
        operands.local_points,
        output_dims=shapes.operator_outputs(),
    )
    outputs = spatial_floating_persistent_chain(
        *operators,
        operands.active_strain_indices,
        operands.active_strain_scales,
        operands.active_dof_ends,
        q[:, :7],
        velocity,
        operands.inertia_upper_rows,
        operands.inertia_upper_columns,
        operands.weighted_mass_diagonals.reshape(
            num_segments * operands.num_gauss_points, 6
        ),
        operands.gravity_world,
        operands.block_dim,
        output_dims=shapes.chain_outputs(),
    )
    return outputs[-3], outputs[-2], outputs[-1]


__all__ = ["execute_dynamics_terms", "execute_floating_dynamics_terms"]
