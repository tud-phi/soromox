# ruff: noqa: UP018
"""Optional cooperative FP64 SPD solves with explicit per-world indexing.

One 32-thread block factors a system. Each dot product keeps scalar accumulation
order. Status is one when a pivot is floored; zero otherwise. This is an explicit
low-level API, not a replacement for the differentiable JAX solve policy.
"""

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.func
def cholesky_solve_at(
    environment: int,
    lane: int,
    matrix: wp.array3d[wp.float64],
    rhs: wp.array2d[wp.float64],
    factor: wp.array3d[wp.float64],
    workspace: wp.array2d[wp.float64],
    solution: wp.array2d[wp.float64],
    status: wp.array[wp.int32],
    diagonal_regularization: wp.float64,
):
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
    factor: wp.array3d[wp.float64],
    workspace: wp.array2d[wp.float64],
    solution: wp.array2d[wp.float64],
    status: wp.array[wp.int32],
    diagonal_regularization: wp.float64,
):
    environment, lane = wp.tid()
    cholesky_solve_at(
        environment,
        lane,
        matrix,
        rhs,
        factor,
        workspace,
        solution,
        status,
        diagonal_regularization,
    )


@wp.struct
class CholeskyArrays:
    matrix: wp.array3d[wp.float64]
    rhs: wp.array2d[wp.float64]
    factor: wp.array3d[wp.float64]
    workspace: wp.array2d[wp.float64]
    solution: wp.array2d[wp.float64]
    status: wp.array[wp.int32]


@wp.kernel(enable_backward=False)
def packed_cholesky_kernel(
    worlds: wp.array(dtype=CholeskyArrays),
    work: wp.array(dtype=wp.vec3i),
    diagonal_regularization: wp.float64,
):
    item, lane = wp.tid()
    d = worlds[work[item][0]]
    cholesky_solve_at(
        0,
        lane,
        d.matrix,
        d.rhs,
        d.factor,
        d.workspace,
        d.solution,
        d.status,
        diagonal_regularization,
    )
