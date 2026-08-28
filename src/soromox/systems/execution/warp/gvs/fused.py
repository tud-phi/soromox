# ruff: noqa: I001, UP018
"""Allocation-free fused GVS dynamics and threadlike-force execution.

The public launcher composes the existing persistent dynamics chain and
threadlike direct-effort launch sequence behind one Warp callable. It introduces
no alternative numerical kernels: callers retain ownership of every workspace
and output, and the fixed sequence remains CUDA-graph-capturable.
"""

from __future__ import annotations

import warp as wp

from soromox.systems.execution.warp.gvs.actuation.threadlike import (
    launch_gvs_threadlike_actuation_force,
)
from soromox.systems.execution.warp.gvs.chain import launch_persistent_chain

wp.set_module_options({"enable_backward": False})


def launch_gvs_dynamics_and_threadlike_actuation_force(
    joint_adjoint: wp.array2d[wp.float64],
    joint_adjoint_dot: wp.array2d[wp.float64],
    joint_tangent: wp.array2d[wp.float64],
    joint_tangent_dot_qd: wp.array2d[wp.float64],
    joint_velocity: wp.array2d[wp.float64],
    cell_adjoint: wp.array2d[wp.float64],
    cell_tangent_local: wp.array2d[wp.float64],
    cell_link_velocity: wp.array2d[wp.float64],
    cell_step_velocity: wp.array2d[wp.float64],
    cell_tangent_velocity_dot: wp.array2d[wp.float64],
    link_global_to_local: wp.array2d[wp.int32],
    active_dofs: wp.array[wp.int32],
    qd: wp.array2d[wp.float64],
    inertia_upper_rows: wp.array[wp.int32],
    inertia_upper_columns: wp.array[wp.int32],
    weighted_masses: wp.array2d[wp.float64],
    gravity_base: wp.array[wp.float64],
    num_cells: wp.array[wp.int32],
    num_quadrature: wp.array[wp.int32],
    lanes_per_block: wp.array[wp.int32],
    q: wp.array2d[wp.float64],
    controls: wp.array2d[wp.float64],
    link_local_to_global: wp.array2d[wp.int32],
    link_basis: wp.array4d[wp.float64],
    reference_strain: wp.array3d[wp.float64],
    physical_points: wp.array2d[wp.float64],
    integration_weights: wp.array2d[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    path_start_segments: wp.array[wp.int32],
    path_end_segments: wp.array[wp.int32],
    coordinate_scales: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    jacobian_first: wp.array2d[wp.float64],
    jacobian_dot_qd_first: wp.array2d[wp.float64],
    velocity_first: wp.array2d[wp.float64],
    gravity_first: wp.array2d[wp.float64],
    jacobian_second: wp.array2d[wp.float64],
    jacobian_dot_qd_second: wp.array2d[wp.float64],
    velocity_second: wp.array2d[wp.float64],
    gravity_second: wp.array2d[wp.float64],
    inertia: wp.array3d[wp.float64],
    coriolis_qd: wp.array2d[wp.float64],
    gravity_force: wp.array2d[wp.float64],
    strain_workspace: wp.array4d[wp.float64],
    actuation_force: wp.array2d[wp.float64],
):
    """Enqueue GVS chain assembly followed by threadlike direct effort.

    All arrays follow the public GVS dynamics and threadlike operand/shape
    contracts. The first eleven outputs are the persistent-chain workspaces and
    results, followed by ``strain_workspace`` with shape ``(E, S, Q, 6)`` and
    ``actuation_force`` with shape ``(E, D)``.

    Returns:
        None. The caller-owned buffers are updated without allocation,
        synchronization, JAX calls, or device transfers.
    """

    launch_persistent_chain(
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
        link_global_to_local,
        active_dofs,
        qd,
        inertia_upper_rows,
        inertia_upper_columns,
        weighted_masses,
        gravity_base,
        num_cells,
        num_quadrature,
        lanes_per_block,
        jacobian_first,
        jacobian_dot_qd_first,
        velocity_first,
        gravity_first,
        jacobian_second,
        jacobian_dot_qd_second,
        velocity_second,
        gravity_second,
        inertia,
        coriolis_qd,
        gravity_force,
    )
    launch_gvs_threadlike_actuation_force(
        q,
        controls,
        link_local_to_global,
        link_global_to_local,
        link_basis,
        reference_strain,
        physical_points,
        integration_weights,
        routing_intercepts,
        routing_slopes,
        path_start_segments,
        path_end_segments,
        coordinate_scales,
        epsilons,
        strain_workspace,
        actuation_force,
    )


gvs_dynamics_and_threadlike_actuation_force = wp.jax_callable(
    launch_gvs_dynamics_and_threadlike_actuation_force,
    num_outputs=13,
)


__all__ = [
    "gvs_dynamics_and_threadlike_actuation_force",
    "launch_gvs_dynamics_and_threadlike_actuation_force",
]
