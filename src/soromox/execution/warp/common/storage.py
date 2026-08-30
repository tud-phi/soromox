# ruff: noqa: I001, UP018
"""Double-buffer access helpers shared by persistent Warp kernels."""

from __future__ import annotations

import warp as wp


wp.set_module_options({"enable_backward": False})


@wp.func
def _matrix_value(
    first: wp.array2d[wp.float64],
    second: wp.array2d[wp.float64],
    use_first: bool,
    base_row: int,
    row: int,
    column: int,
) -> wp.float64:
    """Read one entry from the active matrix buffer.

    Args:
        first: First flattened matrix buffer.
        second: Second flattened matrix buffer.
        use_first: Whether ``first`` is currently active.
        base_row: First flattened row of the logical matrix.
        row: Logical matrix row.
        column: Logical matrix column.

    Returns:
        The selected matrix entry.
    """
    if use_first:
        return first[base_row + row, column]
    return second[base_row + row, column]


@wp.func
def _vector_value(
    first: wp.array2d[wp.float64],
    second: wp.array2d[wp.float64],
    use_first: bool,
    base_row: int,
    row: int,
) -> wp.float64:
    """Read one entry from the active column-vector buffer.

    Args:
        first: First flattened column-vector buffer.
        second: Second flattened column-vector buffer.
        use_first: Whether ``first`` is currently active.
        base_row: First flattened row of the logical vector.
        row: Logical vector row.

    Returns:
        The selected vector entry.
    """
    if use_first:
        return first[base_row + row, 0]
    return second[base_row + row, 0]


@wp.func
def _write_matrix_value(
    first: wp.array2d[wp.float64],
    second: wp.array2d[wp.float64],
    use_first: bool,
    base_row: int,
    row: int,
    column: int,
    value: wp.float64,
):
    """Write one entry to the active matrix buffer.

    Args:
        first: First flattened matrix buffer.
        second: Second flattened matrix buffer.
        use_first: Whether ``first`` is currently active.
        base_row: First flattened row of the logical matrix.
        row: Logical matrix row.
        column: Logical matrix column.
        value: Value to store.

    Returns:
        None. The selected buffer is updated in place.
    """
    if use_first:
        first[base_row + row, column] = value
    else:
        second[base_row + row, column] = value


@wp.func
def _write_vector_value(
    first: wp.array2d[wp.float64],
    second: wp.array2d[wp.float64],
    use_first: bool,
    base_row: int,
    row: int,
    value: wp.float64,
):
    """Write one entry to the active column-vector buffer.

    Args:
        first: First flattened column-vector buffer.
        second: Second flattened column-vector buffer.
        use_first: Whether ``first`` is currently active.
        base_row: First flattened row of the logical vector.
        row: Logical vector row.
        value: Value to store.

    Returns:
        None. The selected buffer is updated in place.
    """
    if use_first:
        first[base_row + row, 0] = value
    else:
        second[base_row + row, 0] = value
