# ruff: noqa: I001, UP018
"""Shared floating-base root operators for continuum Warp kernels."""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.func
def _planar_rotation_transpose_entry(
    theta: wp.float64, row: int, column: int
) -> wp.float64:
    """Return one entry of the transpose of a planar rotation.

    Args:
        theta: World-frame base orientation.
        row: Requested matrix row.
        column: Requested matrix column.

    Returns:
        Entry ``(row, column)`` of ``R(theta).T``.
    """
    cosine = wp.cos(theta)
    sine = wp.sin(theta)
    value = wp.float64(0.0)
    if row == 0 and column == 0 or row == 1 and column == 1:
        value = cosine
    elif row == 0 and column == 1:
        value = sine
    elif row == 1 and column == 0:
        value = -sine
    return value


@wp.func
def _quaternion_rotation_transpose_entry(
    base_pose: wp.array2d[wp.float64], environment: int, row: int, column: int
) -> wp.float64:
    """Return one normalized quaternion rotation-transpose entry.

    Args:
        base_pose: Batched scalar-first quaternion poses with shape ``(E, 7)``.
        environment: Environment row in ``base_pose``.
        row: Requested matrix row.
        column: Requested matrix column.

    Returns:
        Entry ``(row, column)`` of the normalized quaternion rotation
        transpose. A zero runtime quaternion follows the trace-safe JAX path
        and maps to identity.
    """
    w = base_pose[environment, 0]
    x = base_pose[environment, 1]
    y = base_pose[environment, 2]
    z = base_pose[environment, 3]
    norm_sq = w * w + x * x + y * y + z * z
    if norm_sq <= wp.float64(0.0):
        value = wp.float64(0.0)
        if row == column:
            value = wp.float64(1.0)
        return value
    inverse_norm = wp.float64(1.0) / wp.sqrt(norm_sq)
    w *= inverse_norm
    x *= inverse_norm
    y *= inverse_norm
    z *= inverse_norm

    value = wp.float64(0.0)
    if row == 0 and column == 0:
        value = wp.float64(1.0) - wp.float64(2.0) * (y * y + z * z)
    elif row == 0 and column == 1:
        value = wp.float64(2.0) * (x * y + w * z)
    elif row == 0 and column == 2:
        value = wp.float64(2.0) * (x * z - w * y)
    elif row == 1 and column == 0:
        value = wp.float64(2.0) * (x * y - w * z)
    elif row == 1 and column == 1:
        value = wp.float64(1.0) - wp.float64(2.0) * (x * x + z * z)
    elif row == 1 and column == 2:
        value = wp.float64(2.0) * (y * z + w * x)
    elif row == 2 and column == 0:
        value = wp.float64(2.0) * (x * z + w * y)
    elif row == 2 and column == 1:
        value = wp.float64(2.0) * (y * z - w * x)
    elif row == 2 and column == 2:
        value = wp.float64(1.0) - wp.float64(2.0) * (x * x + y * y)
    return value


@wp.func
def _spatial_root_jacobian_entry(
    base_pose: wp.array2d[wp.float64], environment: int, row: int, column: int
) -> wp.float64:
    """Return one root body-Jacobian entry for world-frame base velocity.

    Args:
        base_pose: Batched scalar-first quaternion poses with shape ``(E, 7)``.
        environment: Environment row in ``base_pose``.
        row: Requested spatial row.
        column: Requested base-velocity column.

    Returns:
        Entry of ``diag(R.T, R.T)`` in angular-first spatial ordering.
    """
    if row < 3 and column < 3:
        return _quaternion_rotation_transpose_entry(base_pose, environment, row, column)
    if row >= 3 and column >= 3:
        return _quaternion_rotation_transpose_entry(
            base_pose, environment, row - 3, column - 3
        )
    return wp.float64(0.0)


@wp.kernel(enable_backward=False)
def _prepend_zeros_kernel(
    internal: wp.array2d[wp.float64],
    prefix: int,
    output: wp.array2d[wp.float64],
):
    """Copy an internal generalized vector behind an exact zero prefix."""
    environment, column = wp.tid()
    value = wp.float64(0.0)
    if column >= prefix:
        value = internal[environment, column - prefix]
    output[environment, column] = value


def launch_prepend_zeros(
    internal: wp.array2d[wp.float64],
    prefix: int,
    output: wp.array2d[wp.float64],
):
    """Launch a specialized zero-prefix copy into caller-owned storage.

    Args:
        internal: Batched internal generalized vectors.
        prefix: Number of leading floating-base entries.
        output: Batched total generalized-vector output.

    Returns:
        None. ``output`` is updated in place.
    """
    wp.launch(
        _prepend_zeros_kernel,
        dim=output.shape,
        inputs=[internal, prefix],
        outputs=[output],
    )


__all__ = [
    "_planar_rotation_transpose_entry",
    "_quaternion_rotation_transpose_entry",
    "_spatial_root_jacobian_entry",
    "launch_prepend_zeros",
]
