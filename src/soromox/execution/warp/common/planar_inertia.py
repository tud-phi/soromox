"""Diagonal and full planar inertia actions for constant-strain kernels."""

import warp as wp

from soromox.execution.warp.common.storage import _matrix_value, _vector_value

SPATIAL_DIM = 3


@wp.func
def _mass_vector_action(
    masses: wp.array2d[wp.float64],
    item: int,
    row: int,
    first: wp.array2d[wp.float64],
    second: wp.array2d[wp.float64],
    use_first: bool,
    base_row: int,
) -> wp.float64:
    """Apply diagonal or row-major full spatial inertia to a stored vector."""
    if masses.shape[1] == SPATIAL_DIM:
        return masses[item, row] * _vector_value(
            first, second, use_first, base_row, row
        )
    value = wp.float64(0.0)
    for column in range(SPATIAL_DIM):
        value += masses[item, row * SPATIAL_DIM + column] * _vector_value(
            first, second, use_first, base_row, column
        )
    return value


@wp.func
def _mass_jacobian_action(
    masses: wp.array2d[wp.float64],
    item: int,
    row: int,
    first: wp.array2d[wp.float64],
    second: wp.array2d[wp.float64],
    use_first: bool,
    base_row: int,
    column: int,
) -> wp.float64:
    """Apply diagonal or full spatial inertia to one Jacobian column."""
    if masses.shape[1] == SPATIAL_DIM:
        return masses[item, row] * _matrix_value(
            first, second, use_first, base_row, row, column
        )
    value = wp.float64(0.0)
    for k in range(SPATIAL_DIM):
        value += masses[item, row * SPATIAL_DIM + k] * _matrix_value(
            first, second, use_first, base_row, k, column
        )
    return value
