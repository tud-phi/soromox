"""Native per-world descriptors for mixed fixed and floating GVS chains.

Each descriptor owns array views for one world. Work maps contain
(world, local item, auxiliary item); arrays remain alive in the caller.
"""

import warp as wp

from .chain import persistent_chain_at
from .floating_chain import floating_persistent_chain_at


@wp.struct
class MixedChainArrays:
    joint_adjoint: wp.array2d[wp.float64]
    joint_adjoint_dot: wp.array2d[wp.float64]
    joint_tangent: wp.array2d[wp.float64]
    joint_tangent_dot_qd: wp.array2d[wp.float64]
    joint_velocity: wp.array2d[wp.float64]
    cell_adjoint: wp.array2d[wp.float64]
    cell_tangent_local: wp.array2d[wp.float64]
    cell_link_velocity: wp.array2d[wp.float64]
    cell_step_velocity: wp.array2d[wp.float64]
    cell_tangent_velocity_dot: wp.array2d[wp.float64]
    global_to_local: wp.array2d[wp.int32]
    active_dofs: wp.array[wp.int32]
    qd: wp.array2d[wp.float64]
    inertia_upper_rows: wp.array[wp.int32]
    inertia_upper_columns: wp.array[wp.int32]
    weighted_masses: wp.array2d[wp.float64]
    gravity_base: wp.array[wp.float64]
    num_cells_array: wp.array[wp.int32]
    num_quadrature_array: wp.array[wp.int32]
    lanes_per_block: wp.array[wp.int32]
    jacobian_first: wp.array2d[wp.float64]
    jacobian_dot_qd_first: wp.array2d[wp.float64]
    velocity_first: wp.array2d[wp.float64]
    gravity_first: wp.array2d[wp.float64]
    jacobian_second: wp.array2d[wp.float64]
    jacobian_dot_qd_second: wp.array2d[wp.float64]
    velocity_second: wp.array2d[wp.float64]
    gravity_second: wp.array2d[wp.float64]
    inertia: wp.array3d[wp.float64]
    coriolis_qd: wp.array2d[wp.float64]
    gravity_force: wp.array2d[wp.float64]
    base_pose: wp.array2d[wp.float64]
    velocity: wp.array2d[wp.float64]
    gravity_world: wp.array[wp.float64]


@wp.kernel(enable_backward=False)
def packed_mixed_chain_kernel(
    worlds: wp.array(dtype=MixedChainArrays),
    work: wp.array(dtype=wp.vec3i),
    floating: wp.array(dtype=wp.int32),
):
    item, lane = wp.tid()
    world = work[item][0]
    d = worlds[world]
    if floating[world] == 0:
        persistent_chain_at(
            0,
            lane,
            d.joint_adjoint,
            d.joint_adjoint_dot,
            d.joint_tangent,
            d.joint_tangent_dot_qd,
            d.joint_velocity,
            d.cell_adjoint,
            d.cell_tangent_local,
            d.cell_link_velocity,
            d.cell_step_velocity,
            d.cell_tangent_velocity_dot,
            d.global_to_local,
            d.active_dofs,
            d.qd,
            d.inertia_upper_rows,
            d.inertia_upper_columns,
            d.weighted_masses,
            d.gravity_base,
            d.num_cells_array,
            d.num_quadrature_array,
            d.lanes_per_block,
            d.jacobian_first,
            d.jacobian_dot_qd_first,
            d.velocity_first,
            d.gravity_first,
            d.jacobian_second,
            d.jacobian_dot_qd_second,
            d.velocity_second,
            d.gravity_second,
            d.inertia,
            d.coriolis_qd,
            d.gravity_force,
        )
    else:
        floating_persistent_chain_at(
            0,
            lane,
            d.joint_adjoint,
            d.joint_adjoint_dot,
            d.joint_tangent,
            d.joint_tangent_dot_qd,
            d.joint_velocity,
            d.cell_adjoint,
            d.cell_tangent_local,
            d.cell_link_velocity,
            d.cell_step_velocity,
            d.cell_tangent_velocity_dot,
            d.global_to_local,
            d.active_dofs,
            d.base_pose,
            d.velocity,
            d.inertia_upper_rows,
            d.inertia_upper_columns,
            d.weighted_masses,
            d.gravity_world,
            d.num_cells_array,
            d.num_quadrature_array,
            d.lanes_per_block,
            d.jacobian_first,
            d.jacobian_dot_qd_first,
            d.velocity_first,
            d.gravity_first,
            d.jacobian_second,
            d.jacobian_dot_qd_second,
            d.velocity_second,
            d.gravity_second,
            d.inertia,
            d.coriolis_qd,
            d.gravity_force,
        )
