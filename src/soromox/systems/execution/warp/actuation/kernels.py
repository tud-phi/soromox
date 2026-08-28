# ruff: noqa: I001
"""Shared allocation-free kernels for Warp-native actuation."""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.kernel(enable_backward=False)
def clear_threadlike_matrix_kernel(output: wp.array3d[wp.float64]):
    """Clear one caller-owned batched matrix output."""
    item = wp.tid()
    columns = output.shape[2]
    dofs = output.shape[1]
    environment = item // (dofs * columns)
    remainder = item - environment * dofs * columns
    dof = remainder // columns
    path = remainder - dof * columns
    output[environment, dof, path] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def clear_threadlike_force_kernel(output: wp.array2d[wp.float64]):
    """Clear one caller-owned batched generalized-force output."""
    item = wp.tid()
    dofs = output.shape[1]
    output[item // dofs, item % dofs] = wp.float64(0.0)


__all__ = ["clear_threadlike_force_kernel", "clear_threadlike_matrix_kernel"]
