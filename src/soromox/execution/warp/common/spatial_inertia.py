"""Diagonal and full spatial inertia actions shared by SE(3) Warp kernels."""

import warp as wp

from soromox.execution.warp.common.storage import _matrix_value, _vector_value

wp.set_module_options({"enable_backward": False})

SPATIAL_DIM = 6


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
    """Apply diagonal or row-major full spatial inertia to a stored vector.

    Args:
        masses: Weighted inertia rows, shaped ``(items, 6)`` for diagonal
            inertia or ``(items, 36)`` for full matrices in row-major order.
        item: Inertia row for the current quadrature or link-tip contribution.
        row: Spatial component of the resulting vector, in angular-first order.
        first: First flattened spatial-vector buffer.
        second: Second flattened spatial-vector buffer.
        use_first: Whether to read from the first buffer.
        base_row: First flattened row of the stored six-component vector.

    Returns:
        The selected component of the weighted inertia-vector product.
    """
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
    """Apply diagonal or full spatial inertia to one Jacobian column.

    Args:
        masses: Weighted inertia rows, shaped ``(items, 6)`` for diagonal
            inertia or ``(items, 36)`` for full matrices in row-major order.
        item: Inertia row for the current quadrature or link-tip contribution.
        row: Spatial row of the product, in angular-first order.
        first: First Jacobian buffer, with spatial blocks stacked by row.
        second: Second Jacobian buffer, with spatial blocks stacked by row.
        use_first: Whether to read from the first buffer.
        base_row: First row of the stored six-row Jacobian block.
        column: Generalized-velocity column of the Jacobian.

    Returns:
        The selected entry of the weighted inertia-Jacobian product.
    """
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
