# ruff: noqa: UP018
"""Optional cooperative FP64 SPD solves with explicit per-world indexing.

One 32-thread block factors a system. Each dot product keeps scalar accumulation
order. Status is one when a pivot is floored; zero otherwise. Callers select this
low-level forward solve explicitly. The public JAX solve policy is unchanged.
"""

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.func
def cholesky_solve_at(
    environment: int,
    lane: int,
    matrix: wp.array3d[wp.float64],
    rhs: wp.array2d[wp.float64],
    diagonal_regularization: wp.float64,
    factor: wp.array3d[wp.float64],
    workspace: wp.array2d[wp.float64],
    solution: wp.array2d[wp.float64],
    status: wp.array[wp.int32],
):
    """Solve FP64 symmetric positive-definite systems with cooperative Cholesky.

    Use 32 threads per system. Inputs, outputs and workspaces must not alias.

    Args:
        environment: Batch row in the supplied arrays; zero for a single-world view.
        lane: Thread index in the 32-thread block; all lanes must participate.
        matrix: Symmetric input matrices with shape ``(E, D, D)``; reads the lower
            triangle.
        rhs: Right-hand sides with shape ``(E, D)``.
        diagonal_regularization: Strictly positive floor applied to Cholesky pivots
            before taking their square root.
        factor: Caller-owned Cholesky-factor workspace with shape ``(E, D, D)``.
        workspace: Caller-owned forward-substitution workspace with shape ``(E, D)``.
        solution: Caller-owned solution output with shape ``(E, D)``.
        status: INT32 output of shape ``(E,)``; one if a pivot was floored, otherwise
            zero.

    Returns:
        None. Writes ``factor``, ``workspace``, ``solution`` and ``status`` in place.
    """
    n = rhs.shape[1]
    barrier = wp.tile_zeros(shape=(1,), dtype=wp.int32, storage="shared")
    if lane == 0:
        status[environment] = 0
    row = lane
    while row < n:
        column = int(0)
        while column < n:
            factor[environment, row, column] = wp.float64(0.0)
            column += 1
        row += 32
    wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
    column = int(0)
    while column < n:
        if lane == 0:
            value = matrix[environment, column, column]
            inner = int(0)
            while inner < column:
                value -= (
                    factor[environment, column, inner]
                    * factor[environment, column, inner]
                )
                inner += 1
            if value <= diagonal_regularization:
                value = diagonal_regularization
                status[environment] = 1
            factor[environment, column, column] = wp.sqrt(value)
        wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
        row = column + 1 + lane
        while row < n:
            value = matrix[environment, row, column]
            inner = int(0)
            while inner < column:
                value -= (
                    factor[environment, row, inner] * factor[environment, column, inner]
                )
                inner += 1
            factor[environment, row, column] = (
                value / factor[environment, column, column]
            )
            row += 32
        wp.tile_scatter_masked(barrier, 0, int(0), lane == 0)
        column += 1
    if lane == 0:
        row = int(0)
        while row < n:
            value = rhs[environment, row]
            column = int(0)
            while column < row:
                value -= (
                    factor[environment, row, column] * workspace[environment, column]
                )
                column += 1
            workspace[environment, row] = value / factor[environment, row, row]
            row += 1
        row = n - 1
        while row >= 0:
            value = workspace[environment, row]
            column = row + 1
            while column < n:
                value -= (
                    factor[environment, column, row] * solution[environment, column]
                )
                column += 1
            solution[environment, row] = value / factor[environment, row, row]
            row -= 1


@wp.kernel(enable_backward=False)
def cooperative_cholesky_kernel(
    matrix: wp.array3d[wp.float64],
    rhs: wp.array2d[wp.float64],
    diagonal_regularization: wp.float64,
    factor: wp.array3d[wp.float64],
    workspace: wp.array2d[wp.float64],
    solution: wp.array2d[wp.float64],
    status: wp.array[wp.int32],
):
    """Solve FP64 symmetric positive-definite systems with cooperative Cholesky.

    Use 32 threads per system. Inputs, outputs and workspaces must not alias.
    Launch with ``wp.launch_tiled(..., dim=E, block_dim=32)`` on CUDA.

    Args:
        matrix: Symmetric input matrices with shape ``(E, D, D)``; reads the lower
            triangle.
        rhs: Right-hand sides with shape ``(E, D)``.
        diagonal_regularization: Strictly positive floor applied to Cholesky pivots
            before taking their square root.
        factor: Caller-owned Cholesky-factor workspace with shape ``(E, D, D)``.
        workspace: Caller-owned forward-substitution workspace with shape ``(E, D)``.
        solution: Caller-owned solution output with shape ``(E, D)``.
        status: INT32 output of shape ``(E,)``; one if a pivot was floored, otherwise
            zero.

    Returns:
        None. Writes ``factor``, ``workspace``, ``solution`` and ``status`` in place.
    """
    environment, lane = wp.tid()
    cholesky_solve_at(
        environment,
        lane,
        matrix,
        rhs,
        diagonal_regularization,
        factor,
        workspace,
        solution,
        status,
    )


@wp.struct
class CholeskySolveData:
    """FP64 linear-system inputs, solve workspaces and outputs for one world.

    Array fields are non-owning views; the caller must retain their allocations
    for every launch and captured graph. Batched fields use one environment
    (``E = 1``); structural fields use this world's own dimensions.

    Attributes:
        matrix: Symmetric input matrices with shape ``(E, D, D)``; reads the lower
            triangle.
        rhs: Right-hand sides with shape ``(E, D)``.
        factor: Caller-owned Cholesky-factor workspace with shape ``(E, D, D)``.
        workspace: Caller-owned forward-substitution workspace with shape ``(E, D)``.
        solution: Caller-owned solution output with shape ``(E, D)``.
        status: INT32 output of shape ``(E,)``; one if a pivot was floored, otherwise
            zero.
    """

    matrix: wp.array3d[wp.float64]
    rhs: wp.array2d[wp.float64]
    factor: wp.array3d[wp.float64]
    workspace: wp.array2d[wp.float64]
    solution: wp.array2d[wp.float64]
    status: wp.array[wp.int32]


@wp.kernel(enable_backward=False)
def packed_cholesky_kernel(
    world_data: wp.array(dtype=CholeskySolveData),
    work_items: wp.array(dtype=wp.vec3i),
    diagonal_regularization: wp.float64,
):
    """Solve FP64 symmetric positive-definite systems with cooperative Cholesky.

    Launch with ``wp.launch_tiled(..., dim=len(work_items), block_dim=32)``.
    Each block cooperates on one work item; all lanes must participate.

    Args:
        world_data: Device array of per-world data views, including caller-owned
            outputs and workspaces. Keep every referenced allocation alive.
        work_items: INT32 triples ``(world_index, local_index, 0)``. The local
            index is zero (one work item per world). The third component is reserved.
        diagonal_regularization: Strictly positive floor applied to Cholesky pivots
            before taking their square root.

    Returns:
        None. Writes ``factor``, ``workspace``, ``solution`` and ``status`` in place.
    """
    item, lane = wp.tid()
    data = world_data[work_items[item][0]]
    cholesky_solve_at(
        0,
        lane,
        data.matrix,
        data.rhs,
        diagonal_regularization,
        data.factor,
        data.workspace,
        data.solution,
        data.status,
    )
