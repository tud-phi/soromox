# ruff: noqa: I001, UP018
"""Allocation-free fused PCS dynamics and threadlike-force execution.

The dimension-specialized public launchers compose the existing local
operators, persistent chain, and linear-threadlike direct-effort launches
behind one Warp callable. They introduce no duplicate numerical kernels and
retain caller ownership of every workspace and result.
"""

from __future__ import annotations

import warp as wp

from soromox.systems.execution.warp.pcs.actuation.cooperative import (
    PCS_THREADLIKE_PATH_POINT_MAX_PATHS,
    PCS_THREADLIKE_SERIAL_MAX_PATHS,
    launch_planar_threadlike_force_path,
    launch_planar_threadlike_force_path_point,
    launch_spatial_threadlike_force_path,
    launch_spatial_threadlike_force_path_point,
)
from soromox.systems.execution.warp.pcs.actuation.threadlike import (
    launch_planar_threadlike_actuation_force,
    launch_spatial_threadlike_actuation_force,
)
from soromox.systems.execution.warp.pcs.planar_kernels import (
    launch_planar_local_operators,
    launch_planar_persistent_chain,
)
from soromox.systems.execution.warp.pcs.spatial_kernels import (
    launch_spatial_local_operators,
    launch_spatial_persistent_chain,
)

wp.set_module_options({"enable_backward": False})


def _select_force_launcher(q, routing_intercepts, serial, path_point, path):
    """Select a topology using only the static path layout and device."""
    if (
        not q.device.is_cuda
        or routing_intercepts.shape[0] <= PCS_THREADLIKE_SERIAL_MAX_PATHS
    ):
        return serial
    if routing_intercepts.shape[0] <= PCS_THREADLIKE_PATH_POINT_MAX_PATHS:
        return path_point
    return path


def launch_planar_dynamics_and_threadlike_actuation_force(
    q: wp.array2d[wp.float64],
    qd: wp.array2d[wp.float64],
    controls: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    operator_points: wp.array2d[wp.float64],
    active_dof_ends: wp.array[wp.int32],
    inertia_upper_rows: wp.array[wp.int32],
    inertia_upper_columns: wp.array[wp.int32],
    weighted_masses: wp.array2d[wp.float64],
    gravity_base: wp.array[wp.float64],
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
    actuation_force: wp.array2d[wp.float64],
):
    """Enqueue the complete PlanarPCS dynamics and direct-effort sequence.

    Returns:
        None. Four local-operator outputs, nine chain outputs, and the final
        ``(E, D)`` actuator force are written without allocation or
        synchronization.

    Raises:
        NotImplementedError: If ``q`` is not stored on a CUDA device.
    """

    if not q.device.is_cuda:
        raise NotImplementedError(
            "Fused PlanarPCS Warp dynamics and actuation require CUDA."
        )

    launch_planar_local_operators(
        q,
        qd,
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
    launch_planar_persistent_chain(
        adjoint_inverse,
        transported_tangent,
        local_velocity,
        transported_tangent_dot_velocity,
        active_indices,
        active_scales,
        active_dof_ends,
        qd,
        inertia_upper_rows,
        inertia_upper_columns,
        weighted_masses,
        gravity_base,
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
        q,
        routing_intercepts,
        launch_planar_threadlike_actuation_force,
        launch_planar_threadlike_force_path_point,
        launch_planar_threadlike_force_path,
    )
    force_launcher(
        q,
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
        actuation_force,
    )


def launch_spatial_dynamics_and_threadlike_actuation_force(
    q: wp.array2d[wp.float64],
    qd: wp.array2d[wp.float64],
    controls: wp.array2d[wp.float64],
    active_indices: wp.array2d[wp.int32],
    active_scales: wp.array2d[wp.float64],
    reference_strain: wp.array2d[wp.float64],
    operator_points: wp.array2d[wp.float64],
    active_dof_ends: wp.array[wp.int32],
    inertia_upper_rows: wp.array[wp.int32],
    inertia_upper_columns: wp.array[wp.int32],
    weighted_masses: wp.array2d[wp.float64],
    gravity_base: wp.array[wp.float64],
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
    actuation_force: wp.array2d[wp.float64],
):
    """Enqueue the complete spatial PCS dynamics and direct-effort sequence.

    Returns:
        None. Four local-operator outputs, eleven chain outputs, and the final
        ``(E, D)`` actuator force are written without allocation or
        synchronization.

    Raises:
        NotImplementedError: If ``q`` is not stored on a CUDA device.
    """

    if not q.device.is_cuda:
        raise NotImplementedError("Fused PCS Warp dynamics and actuation require CUDA.")

    launch_spatial_local_operators(
        q,
        qd,
        active_indices,
        active_scales,
        reference_strain,
        operator_points,
        adjoint_inverse,
        transported_tangent,
        local_velocity,
        transported_tangent_dot_velocity,
    )
    launch_spatial_persistent_chain(
        adjoint_inverse,
        transported_tangent,
        local_velocity,
        transported_tangent_dot_velocity,
        active_indices,
        active_scales,
        active_dof_ends,
        qd,
        inertia_upper_rows,
        inertia_upper_columns,
        weighted_masses,
        gravity_base,
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
        q,
        routing_intercepts,
        launch_spatial_threadlike_actuation_force,
        launch_spatial_threadlike_force_path_point,
        launch_spatial_threadlike_force_path,
    )
    force_launcher(
        q,
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
        actuation_force,
    )


planar_dynamics_and_threadlike_actuation_force = wp.jax_callable(
    launch_planar_dynamics_and_threadlike_actuation_force,
    num_outputs=14,
)
spatial_dynamics_and_threadlike_actuation_force = wp.jax_callable(
    launch_spatial_dynamics_and_threadlike_actuation_force,
    num_outputs=16,
)


__all__ = [
    "launch_planar_dynamics_and_threadlike_actuation_force",
    "launch_spatial_dynamics_and_threadlike_actuation_force",
    "planar_dynamics_and_threadlike_actuation_force",
    "spatial_dynamics_and_threadlike_actuation_force",
]
