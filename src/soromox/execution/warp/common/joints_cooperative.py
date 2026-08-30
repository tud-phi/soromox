# ruff: noqa: I001, UP018
"""Cooperative shape-generic preparation kernels for spatial joints."""

from __future__ import annotations

import warp as wp

from soromox.execution.warp.common.se3 import (
    Vec6d,
    _ad_action,
    _adjoint_inverse_action,
    _adjoint_inverse_entry,
    _forward_coefficients,
    _left_action,
    _left_coefficient_x_derivatives,
    _left_coefficients,
    _left_total_derivative_action,
    _translation,
)
from soromox.execution.warp.common.joints import _joint_basis_column

wp.set_module_options({"enable_backward": False})


SPATIAL_DIM = 6
JOINT_BLOCK_DIM = 32


@wp.kernel(enable_backward=False)
def cooperative_joint_terms_kernel(
    q: wp.array2d[wp.float64],
    qd: wp.array2d[wp.float64],
    basis: wp.array2d[wp.float64],
    reference: wp.array2d[wp.float64],
    local_to_global: wp.array2d[wp.int32],
    adjoint: wp.array2d[wp.float64],
    adjoint_dot: wp.array2d[wp.float64],
    tangent_local: wp.array2d[wp.float64],
    tangent_dot_qd: wp.array2d[wp.float64],
    joint_velocity: wp.array2d[wp.float64],
):
    """Evaluate one general joint cooperatively per warp-sized block.

    Args:
        q: Batched active generalized coordinates.
        qd: Batched active generalized velocities.
        basis: Flattened padded joint bases.
        reference: Reference joint strains.
        local_to_global: Padded local-to-active coordinate map.
        adjoint: Caller-owned inverse-adjoint output.
        adjoint_dot: Caller-owned inverse-adjoint derivative output.
        tangent_local: Caller-owned local-tangent output.
        tangent_dot_qd: Caller-owned tangent-derivative action output.
        joint_velocity: Caller-owned joint-velocity output.

    Returns:
        None. All results are written to the caller-owned output arrays.
    """

    work_item, lane = wp.tid()
    num_segments = reference.shape[0]
    environment = work_item // num_segments
    segment = work_item - environment * num_segments
    output_base_row = work_item * SPATIAL_DIM

    xi_shared = wp.tile_zeros(shape=(SPATIAL_DIM,), dtype=wp.float64, storage="shared")
    xid_shared = wp.tile_zeros(shape=(SPATIAL_DIM,), dtype=wp.float64, storage="shared")
    coefficients_shared = wp.tile_zeros(shape=(4,), dtype=wp.float64, storage="shared")
    forward_shared = wp.tile_zeros(shape=(3,), dtype=wp.float64, storage="shared")
    translation_shared = wp.tile_zeros(shape=(3,), dtype=wp.float64, storage="shared")
    eta_shared = wp.tile_zeros(shape=(SPATIAL_DIM,), dtype=wp.float64, storage="shared")
    angle_sq_shared = wp.tile_zeros(shape=(1,), dtype=wp.float64, storage="shared")
    barrier = wp.tile_zeros(shape=(1,), dtype=wp.int32, storage="shared")

    component = lane
    xi_value = wp.float64(0.0)
    xid_value = wp.float64(0.0)
    if component < SPATIAL_DIM:
        xi_value = reference[segment, component]
        local_column = int(0)
        while local_column < local_to_global.shape[1]:
            global_column = local_to_global[segment, local_column]
            if global_column >= 0:
                xi_value += (
                    basis[segment * SPATIAL_DIM + component, local_column]
                    * q[environment, global_column]
                )
                xid_value += (
                    basis[segment * SPATIAL_DIM + component, local_column]
                    * qd[environment, global_column]
                )
            local_column += 1
    wp.tile_scatter_masked(xi_shared, component, xi_value, component < SPATIAL_DIM)
    wp.tile_scatter_masked(xid_shared, component, xid_value, component < SPATIAL_DIM)

    angle_sq_value = wp.float64(0.0)
    if lane == 0:
        angle_sq_value = (
            xi_shared[0] * xi_shared[0]
            + xi_shared[1] * xi_shared[1]
            + xi_shared[2] * xi_shared[2]
        )
    wp.tile_scatter_masked(angle_sq_shared, 0, angle_sq_value, lane == 0)

    coefficient_index = lane
    coefficient_value = wp.float64(0.0)
    if coefficient_index < 4:
        coefficient_value = _left_coefficients(angle_sq_shared[0])[coefficient_index]
    wp.tile_scatter_masked(
        coefficients_shared,
        coefficient_index,
        coefficient_value,
        coefficient_index < 4,
    )

    forward_index = lane
    forward_value = wp.float64(0.0)
    if forward_index < 3:
        forward_value = _forward_coefficients(angle_sq_shared[0])[forward_index]
    wp.tile_scatter_masked(
        forward_shared,
        forward_index,
        forward_value,
        forward_index < 3,
    )

    translation_index = lane
    translation_value = wp.float64(0.0)
    if translation_index < 3:
        translation_value = _translation(
            wp.vec3d(xi_shared[0], xi_shared[1], xi_shared[2]),
            wp.vec3d(xi_shared[3], xi_shared[4], xi_shared[5]),
            wp.vec3d(forward_shared[0], forward_shared[1], forward_shared[2]),
        )[translation_index]
    wp.tile_scatter_masked(
        translation_shared,
        translation_index,
        translation_value,
        translation_index < 3,
    )

    xi = Vec6d(
        xi_shared[0],
        xi_shared[1],
        xi_shared[2],
        xi_shared[3],
        xi_shared[4],
        xi_shared[5],
    )
    xid = Vec6d(
        xid_shared[0],
        xid_shared[1],
        xid_shared[2],
        xid_shared[3],
        xid_shared[4],
        xid_shared[5],
    )
    coefficients = wp.vec4d(
        coefficients_shared[0],
        coefficients_shared[1],
        coefficients_shared[2],
        coefficients_shared[3],
    )
    if lane == 0:
        velocity = _left_action(xi, xid, coefficients)
        coefficient_derivatives = _left_coefficient_x_derivatives(angle_sq_shared[0])
        angle_sq_dot = wp.float64(2.0) * (
            xi[0] * xid[0] + xi[1] * xid[1] + xi[2] * xid[2]
        )
        derivative_velocity = _left_total_derivative_action(
            xi,
            xid,
            xid,
            Vec6d(),
            coefficients,
            coefficient_derivatives * angle_sq_dot,
        )
        row = int(0)
        while row < SPATIAL_DIM:
            tangent_dot_qd[output_base_row + row, 0] = derivative_velocity[row]
            joint_velocity[output_base_row + row, 0] = velocity[row]
            row += 1
    wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)

    eta_component = lane
    eta_value = wp.float64(0.0)
    if eta_component < SPATIAL_DIM:
        velocity = Vec6d(
            joint_velocity[output_base_row + 0, 0],
            joint_velocity[output_base_row + 1, 0],
            joint_velocity[output_base_row + 2, 0],
            joint_velocity[output_base_row + 3, 0],
            joint_velocity[output_base_row + 4, 0],
            joint_velocity[output_base_row + 5, 0],
        )
        eta = _adjoint_inverse_action(
            wp.vec3d(xi[0], xi[1], xi[2]),
            wp.vec3d(
                translation_shared[0],
                translation_shared[1],
                translation_shared[2],
            ),
            angle_sq_shared[0],
            wp.vec3d(forward_shared[0], forward_shared[1], forward_shared[2]),
            velocity,
        )
        eta_value = eta[eta_component]
    wp.tile_scatter_masked(
        eta_shared,
        eta_component,
        eta_value,
        eta_component < SPATIAL_DIM,
    )

    entry = lane
    while entry < SPATIAL_DIM * SPATIAL_DIM:
        row = entry // SPATIAL_DIM
        column = entry - row * SPATIAL_DIM
        adjoint[output_base_row + row, column] = _adjoint_inverse_entry(
            wp.vec3d(xi[0], xi[1], xi[2]),
            wp.vec3d(
                translation_shared[0],
                translation_shared[1],
                translation_shared[2],
            ),
            angle_sq_shared[0],
            wp.vec3d(forward_shared[0], forward_shared[1], forward_shared[2]),
            row,
            column,
        )
        entry += JOINT_BLOCK_DIM
    wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)

    column = lane
    if column < SPATIAL_DIM:
        adjoint_column = Vec6d(
            adjoint[output_base_row + 0, column],
            adjoint[output_base_row + 1, column],
            adjoint[output_base_row + 2, column],
            adjoint[output_base_row + 3, column],
            adjoint[output_base_row + 4, column],
            adjoint[output_base_row + 5, column],
        )
        derivative_column = -_ad_action(
            Vec6d(
                eta_shared[0],
                eta_shared[1],
                eta_shared[2],
                eta_shared[3],
                eta_shared[4],
                eta_shared[5],
            ),
            adjoint_column,
        )
        row = int(0)
        while row < SPATIAL_DIM:
            adjoint_dot[output_base_row + row, column] = derivative_column[row]
            row += 1

    local_column = lane
    while local_column < local_to_global.shape[1]:
        tangent = _left_action(
            xi,
            _joint_basis_column(basis, segment, local_column),
            coefficients,
        )
        row = int(0)
        while row < SPATIAL_DIM:
            tangent_local[output_base_row + row, local_column] = tangent[row]
            row += 1
        local_column += JOINT_BLOCK_DIM


def launch_cooperative_joint_terms(
    q: wp.array2d[wp.float64],
    qd: wp.array2d[wp.float64],
    basis: wp.array2d[wp.float64],
    reference: wp.array2d[wp.float64],
    local_to_global: wp.array2d[wp.int32],
    adjoint: wp.array2d[wp.float64],
    adjoint_dot: wp.array2d[wp.float64],
    tangent_local: wp.array2d[wp.float64],
    tangent_dot_qd: wp.array2d[wp.float64],
    joint_velocity: wp.array2d[wp.float64],
):
    """Launch cooperative spatial-joint evaluation on a CUDA device.

    One CUDA block processes each ``(environment, segment)`` pair. Threads
    cooperatively evaluate active local basis columns while sharing the joint
    Lie quantities needed by every column.

    Args:
        q: FP64 active coordinates with shape ``(batch_size, num_dofs)``.
        qd: FP64 active velocities with the same shape as ``q``.
        basis: Flattened joint bases with shape
            ``(num_segments * 6, max_dof)``.
        reference: Reference joint strains with shape ``(num_segments, 6)``.
        local_to_global: Padded local-to-active coordinate map.
        adjoint: Preallocated flattened joint-adjoint output.
        adjoint_dot: Preallocated flattened joint-adjoint derivative output.
        tangent_local: Preallocated flattened local tangent output.
        tangent_dot_qd: Preallocated flattened tangent-derivative action.
        joint_velocity: Preallocated flattened joint-velocity output.

    Returns:
        None. Outputs are written in place.
    """

    wp.launch_tiled(
        cooperative_joint_terms_kernel,
        dim=q.shape[0] * reference.shape[0],
        inputs=[q, qd, basis, reference, local_to_global],
        outputs=[
            adjoint,
            adjoint_dot,
            tangent_local,
            tangent_dot_qd,
            joint_velocity,
        ],
        block_dim=JOINT_BLOCK_DIM,
    )


cooperative_joint_terms = wp.jax_callable(launch_cooperative_joint_terms, num_outputs=5)


__all__ = [
    "cooperative_joint_terms_kernel",
    "launch_cooperative_joint_terms",
]
