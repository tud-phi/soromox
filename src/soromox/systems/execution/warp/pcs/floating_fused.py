# ruff: noqa: I001, UP018
"""Allocation-free floating PCS dynamics and threadlike-force execution."""

from __future__ import annotations

import warp as wp

from soromox.systems.execution.warp.pcs.actuation.cooperative import (
    launch_planar_threadlike_force_path,
    launch_planar_threadlike_force_path_point,
    launch_spatial_threadlike_force_path,
    launch_spatial_threadlike_force_path_point,
)
from soromox.systems.execution.warp.pcs.actuation.threadlike import (
    launch_planar_threadlike_actuation_force,
    launch_spatial_threadlike_actuation_force,
)
from soromox.systems.execution.warp.common.floating_base import launch_prepend_zeros
from soromox.systems.execution.warp.pcs.fused import _select_force_launcher
from soromox.systems.execution.warp.pcs.planar_floating_kernels import (
    launch_planar_floating_persistent_chain,
)
from soromox.systems.execution.warp.pcs.planar_kernels import (
    launch_planar_local_operators,
)
from soromox.systems.execution.warp.pcs.spatial_floating_kernels import (
    launch_spatial_floating_persistent_chain,
)
from soromox.systems.execution.warp.pcs.spatial_kernels import (
    launch_spatial_local_operators,
)

wp.set_module_options({"enable_backward": False})

def launch_planar_floating_dynamics_and_threadlike_actuation_force(
    q_internal: wp.array2d[wp.float64],
    qd_internal: wp.array2d[wp.float64],
    base_pose: wp.array2d[wp.float64],
    velocity: wp.array2d[wp.float64],
    controls: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    operator_points: wp.array2d[wp.float64],
    active_dof_ends: wp.array[wp.int32],
    inertia_upper_rows: wp.array[wp.int32],
    inertia_upper_columns: wp.array[wp.int32],
    weighted_masses: wp.array2d[wp.float64],
    gravity_world: wp.array[wp.float64],
    block_dim: int,
    physical_points: wp.array2d[wp.float64],
    physical_weights: wp.array2d[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    path_start_segments: wp.array[wp.int32],
    path_end_segments: wp.array[wp.int32],
    coordinate_scales: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    adjoint_inverse: wp.array2d[wp.float64],
    transported_tangent: wp.array2d[wp.float64],
    local_velocity: wp.array2d[wp.float64],
    transported_tangent_dot_velocity: wp.array2d[wp.float64],
    jacobian_first: wp.array2d[wp.float64],
    derivative_first: wp.array2d[wp.float64],
    gravity_first: wp.array2d[wp.float64],
    jacobian_second: wp.array2d[wp.float64],
    derivative_second: wp.array2d[wp.float64],
    gravity_second: wp.array2d[wp.float64],
    inertia: wp.array3d[wp.float64],
    coriolis_qd: wp.array2d[wp.float64],
    gravity_force: wp.array2d[wp.float64],
    actuation_internal: wp.array2d[wp.float64],
    actuation_force: wp.array2d[wp.float64],
):
    """Enqueue augmented PlanarPCS dynamics and internal direct effort."""
    if not q_internal.device.is_cuda:
        raise NotImplementedError(
            "Fused floating PlanarPCS Warp dynamics and actuation require CUDA."
        )
    launch_planar_local_operators(
        q_internal,
        qd_internal,
        active_indices,
        active_scales,
        reference_strain,
        operator_points,
        epsilons,
        adjoint_inverse,
        transported_tangent,
        local_velocity,
        transported_tangent_dot_velocity,
    )
    launch_planar_floating_persistent_chain(
        adjoint_inverse,
        transported_tangent,
        local_velocity,
        transported_tangent_dot_velocity,
        active_indices,
        active_scales,
        active_dof_ends,
        base_pose,
        velocity,
        inertia_upper_rows,
        inertia_upper_columns,
        weighted_masses,
        gravity_world,
        block_dim,
        jacobian_first,
        derivative_first,
        gravity_first,
        jacobian_second,
        derivative_second,
        gravity_second,
        inertia,
        coriolis_qd,
        gravity_force,
    )
    force_launcher = _select_force_launcher(
        q_internal,
        routing_intercepts,
        launch_planar_threadlike_actuation_force,
        launch_planar_threadlike_force_path_point,
        launch_planar_threadlike_force_path,
    )
    force_launcher(
        q_internal,
        controls,
        active_indices,
        active_scales,
        reference_strain,
        physical_points,
        physical_weights,
        routing_intercepts,
        routing_slopes,
        path_start_segments,
        path_end_segments,
        coordinate_scales,
        epsilons,
        actuation_internal,
    )
    launch_prepend_zeros(actuation_internal, 3, actuation_force)

def launch_spatial_floating_dynamics_and_threadlike_actuation_force(
    q_internal: wp.array2d[wp.float64],
    qd_internal: wp.array2d[wp.float64],
    base_pose: wp.array2d[wp.float64],
    velocity: wp.array2d[wp.float64],
    controls: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    operator_points: wp.array2d[wp.float64],
    active_dof_ends: wp.array[wp.int32],
    inertia_upper_rows: wp.array[wp.int32],
    inertia_upper_columns: wp.array[wp.int32],
    weighted_masses: wp.array2d[wp.float64],
    gravity_world: wp.array[wp.float64],
    block_dim: int,
    physical_points: wp.array2d[wp.float64],
    physical_weights: wp.array2d[wp.float64],
    routing_intercepts: wp.array2d[wp.float64],
    routing_slopes: wp.array2d[wp.float64],
    path_start_segments: wp.array[wp.int32],
    path_end_segments: wp.array[wp.int32],
    coordinate_scales: wp.array[wp.float64],
    epsilons: wp.array[wp.float64],
    adjoint_inverse: wp.array2d[wp.float64],
    transported_tangent: wp.array2d[wp.float64],
    local_velocity: wp.array2d[wp.float64],
    transported_tangent_dot_velocity: wp.array2d[wp.float64],
    jacobian_first: wp.array2d[wp.float64],
    derivative_first: wp.array2d[wp.float64],
    velocity_first: wp.array2d[wp.float64],
    gravity_first: wp.array2d[wp.float64],
    jacobian_second: wp.array2d[wp.float64],
    derivative_second: wp.array2d[wp.float64],
    velocity_second: wp.array2d[wp.float64],
    gravity_second: wp.array2d[wp.float64],
    inertia: wp.array3d[wp.float64],
    coriolis_qd: wp.array2d[wp.float64],
    gravity_force: wp.array2d[wp.float64],
    actuation_internal: wp.array2d[wp.float64],
    actuation_force: wp.array2d[wp.float64],
):
    """Enqueue augmented spatial PCS dynamics and internal direct effort."""
    if not q_internal.device.is_cuda:
        raise NotImplementedError(
            "Fused floating PCS Warp dynamics and actuation require CUDA."
        )
    launch_spatial_local_operators(
        q_internal,
        qd_internal,
        active_indices,
        active_scales,
        reference_strain,
        operator_points,
        adjoint_inverse,
        transported_tangent,
        local_velocity,
        transported_tangent_dot_velocity,
    )
    launch_spatial_floating_persistent_chain(
        adjoint_inverse,
        transported_tangent,
        local_velocity,
        transported_tangent_dot_velocity,
        active_indices,
        active_scales,
        active_dof_ends,
        base_pose,
        velocity,
        inertia_upper_rows,
        inertia_upper_columns,
        weighted_masses,
        gravity_world,
        block_dim,
        jacobian_first,
        derivative_first,
        velocity_first,
        gravity_first,
        jacobian_second,
        derivative_second,
        velocity_second,
        gravity_second,
        inertia,
        coriolis_qd,
        gravity_force,
    )
    force_launcher = _select_force_launcher(
        q_internal,
        routing_intercepts,
        launch_spatial_threadlike_actuation_force,
        launch_spatial_threadlike_force_path_point,
        launch_spatial_threadlike_force_path,
    )
    force_launcher(
        q_internal,
        controls,
        active_indices,
        active_scales,
        reference_strain,
        physical_points,
        physical_weights,
        routing_intercepts,
        routing_slopes,
        path_start_segments,
        path_end_segments,
        coordinate_scales,
        epsilons,
        actuation_internal,
    )
    launch_prepend_zeros(actuation_internal, 6, actuation_force)

planar_floating_dynamics_and_threadlike_actuation_force = wp.jax_callable(
    launch_planar_floating_dynamics_and_threadlike_actuation_force,
    num_outputs=15,
)

spatial_floating_dynamics_and_threadlike_actuation_force = wp.jax_callable(
    launch_spatial_floating_dynamics_and_threadlike_actuation_force,
    num_outputs=17,
)

__all__ = [
    "launch_planar_floating_dynamics_and_threadlike_actuation_force",
    "launch_spatial_floating_dynamics_and_threadlike_actuation_force",
]
