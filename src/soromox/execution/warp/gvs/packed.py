"""Native per-world descriptors for mixed fixed and floating GVS chains.

Each descriptor references caller-owned array views for one world. Work maps contain
``(world_index, local_index, 0)`` triples; arrays remain alive in the caller.
"""

import warp as wp

from .chain import persistent_chain_at
from .floating_chain import floating_persistent_chain_at


@wp.struct
class GVSMixedChainData:
    """GVS chain buffers for a world with either a fixed or floating base.

    Array fields are non-owning views; the caller must retain their allocations
    for every launch and captured graph. Batched fields use one environment
    (``E = 1``); structural fields use this world's own dimensions.

    Fixed worlds use ``qd`` and ``gravity_base``; floating worlds use
    ``base_pose``, ``velocity`` and ``gravity_world``. Unused fields may be empty.

    Attributes:
        joint_adjoint: Flattened joint inverse adjoints.
        joint_adjoint_dot: Flattened joint-adjoint derivatives.
        joint_tangent: Internal joint tangents.
        joint_tangent_dot_qd: Joint tangent-derivative actions.
        joint_velocity: Internal joint spatial velocities.
        cell_adjoint: Flattened cell inverse adjoints.
        cell_tangent_local: Cell tangents in padded local coordinates.
        cell_link_velocity: Internal link velocities per cell.
        cell_step_velocity: Internal Magnus-step velocities per cell.
        cell_tangent_velocity_dot: Cell tangent-derivative actions.
        global_to_local: Internal active-to-local link-coordinate map.
        active_dofs: Cumulative internal coordinate count by segment.
        qd: Internal generalized velocities for the fixed-base recurrence.
        inertia_upper_rows: Packed upper-inertia row indices for this world's velocity dimension.
        inertia_upper_columns: Packed upper-inertia column indices for this world's velocity dimension.
        weighted_masses: Quadrature-weighted diagonal spatial inertias.
        gravity_base: Base-frame gravity.
        num_cells_array: One-entry array containing cells per segment.
        num_quadrature_array: One-entry array containing quadrature count.
        lanes_per_block: One-entry array containing cooperative lanes.
        jacobian_first: First caller-owned Jacobian workspace.
        jacobian_dot_qd_first: First derivative-action workspace.
        velocity_first: First spatial-velocity workspace.
        gravity_first: First local-gravity workspace.
        jacobian_second: Second Jacobian workspace.
        jacobian_dot_qd_second: Second derivative-action workspace.
        velocity_second: Second spatial-velocity workspace.
        gravity_second: Second local-gravity workspace.
        inertia: Batched inertia output; includes six base velocities for floating worlds.
        coriolis_qd: Batched convective-force output matching the velocity dimension.
        gravity_force: Batched gravity-force output matching the velocity dimension.
        base_pose: Runtime scalar-last quaternion poses with shape ``(E, 7)``.
        velocity: Total angular-first generalized velocities.
        gravity_world: World-frame spatial gravity.
    """

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
    world_data: wp.array(dtype=GVSMixedChainData),
    work_items: wp.array(dtype=wp.vec3i),
    is_floating: wp.array(dtype=wp.int32),
):
    """Evaluate mixed fixed-base and floating-base GVS chain dynamics using packed world data.

    Each selected descriptor contains one world (E = 1). Structure and numerical
    model parameters may differ between worlds. Descriptors may share allocations
    when their values match.

    Launch with ``wp.launch_tiled(..., dim=len(work_items), block_dim=B)``.
    Each block cooperates on one work item; all lanes must participate.
    Use the same block size as the arrays' ``lanes_per_block[0]``.

    Args:
        world_data: Device array of per-world data views, including caller-owned
            outputs and workspaces. Keep every referenced allocation alive.
        work_items: INT32 triples ``(world_index, local_index, 0)``. The local
            index is zero (one work item per world). The third component is reserved.
        is_floating: One INT32 flag per world; zero selects the fixed-base
            recurrence and one selects the floating-base recurrence.

    Returns:
        None. Updates the selected worlds' output arrays and workspaces in place.
    """
    item, lane = wp.tid()
    world = work_items[item][0]
    data = world_data[world]
    if is_floating[world] == 0:
        persistent_chain_at(
            0,
            lane,
            data.joint_adjoint,
            data.joint_adjoint_dot,
            data.joint_tangent,
            data.joint_tangent_dot_qd,
            data.joint_velocity,
            data.cell_adjoint,
            data.cell_tangent_local,
            data.cell_link_velocity,
            data.cell_step_velocity,
            data.cell_tangent_velocity_dot,
            data.global_to_local,
            data.active_dofs,
            data.qd,
            data.inertia_upper_rows,
            data.inertia_upper_columns,
            data.weighted_masses,
            data.gravity_base,
            data.num_cells_array,
            data.num_quadrature_array,
            data.lanes_per_block,
            data.jacobian_first,
            data.jacobian_dot_qd_first,
            data.velocity_first,
            data.gravity_first,
            data.jacobian_second,
            data.jacobian_dot_qd_second,
            data.velocity_second,
            data.gravity_second,
            data.inertia,
            data.coriolis_qd,
            data.gravity_force,
        )
    else:
        floating_persistent_chain_at(
            0,
            lane,
            data.joint_adjoint,
            data.joint_adjoint_dot,
            data.joint_tangent,
            data.joint_tangent_dot_qd,
            data.joint_velocity,
            data.cell_adjoint,
            data.cell_tangent_local,
            data.cell_link_velocity,
            data.cell_step_velocity,
            data.cell_tangent_velocity_dot,
            data.global_to_local,
            data.active_dofs,
            data.base_pose,
            data.velocity,
            data.inertia_upper_rows,
            data.inertia_upper_columns,
            data.weighted_masses,
            data.gravity_world,
            data.num_cells_array,
            data.num_quadrature_array,
            data.lanes_per_block,
            data.jacobian_first,
            data.jacobian_dot_qd_first,
            data.velocity_first,
            data.gravity_first,
            data.jacobian_second,
            data.jacobian_dot_qd_second,
            data.velocity_second,
            data.gravity_second,
            data.inertia,
            data.coriolis_qd,
            data.gravity_force,
        )
