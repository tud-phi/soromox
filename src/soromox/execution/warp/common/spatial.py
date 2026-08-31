# ruff: noqa: I001, UP018
"""Spatial inertia and force helpers shared by SE(3) Warp kernels."""

from __future__ import annotations

import warp as wp

from soromox.execution.warp.common.se3 import Vec6d
from soromox.execution.warp.common.storage import _vector_value

wp.set_module_options({"enable_backward": False})

SPATIAL_DIM = 6


@wp.func
def _load_spatial_vector(values: wp.array2d[wp.float64], item: int) -> Vec6d:
    """Load one six-vector from item-major storage.

    Args:
        values: Item-major spatial-vector storage.
        item: Item row to load.

    Returns:
        Six-dimensional spatial vector stored at ``item``.
    """

    return Vec6d(
        values[item, 0],
        values[item, 1],
        values[item, 2],
        values[item, 3],
        values[item, 4],
        values[item, 5],
    )


@wp.func
def _load_spatial_vector3d(
    values: wp.array3d[wp.float64], environment: int, item: int
) -> Vec6d:
    """Load one six-vector from batched item-major storage.

    Args:
        values: Batched item-major spatial-vector storage.
        environment: Batch row to load.
        item: Item row to load.

    Returns:
        Six-dimensional spatial vector stored at ``environment, item``.
    """

    return Vec6d(
        values[environment, item, 0],
        values[environment, item, 1],
        values[environment, item, 2],
        values[environment, item, 3],
        values[environment, item, 4],
        values[environment, item, 5],
    )


@wp.func
def _store_spatial_vector(
    values: wp.array2d[wp.float64], item: int, value: Vec6d
):
    """Store one six-vector in item-major storage.

    Args:
        values: Item-major spatial-vector storage.
        item: Item row to update.
        value: Six-dimensional spatial vector to store.

    Returns:
        None. ``values`` is updated in place.
    """

    for row in range(SPATIAL_DIM):
        values[item, row] = value[row]


@wp.func
def _flattened_matrix_action(
    matrix: wp.array2d[wp.float64], base_row: int, value: Vec6d
) -> Vec6d:
    """Apply one flattened six-by-six matrix to a spatial vector.

    Args:
        matrix: Row-flattened six-by-six matrices.
        base_row: First row of the matrix to apply.
        value: Spatial vector to transform.

    Returns:
        Matrix action on ``value``.
    """

    result = Vec6d()
    for row in range(SPATIAL_DIM):
        for column in range(SPATIAL_DIM):
            result[row] += matrix[base_row + row, column] * value[column]
    return result


@wp.func
def _flattened_matrix_transpose_action(
    matrix: wp.array2d[wp.float64], base_row: int, value: Vec6d
) -> Vec6d:
    """Apply the transpose of one flattened six-by-six matrix.

    Args:
        matrix: Row-flattened six-by-six matrices.
        base_row: First row of the matrix to apply.
        value: Spatial vector to transform.

    Returns:
        Transposed matrix action on ``value``.
    """

    result = Vec6d()
    for column in range(SPATIAL_DIM):
        for row in range(SPATIAL_DIM):
            result[column] += matrix[base_row + row, column] * value[row]
    return result


@wp.func
def _rotate_spatial_to_inertial(pose: wp.mat44d, value: Vec6d) -> Vec6d:
    """Rotate a body-frame spatial vector into the inertial frame.

    Args:
        pose: Homogeneous pose whose rotation maps body to inertial coordinates.
        value: Body-frame spatial vector.

    Returns:
        Inertial-frame spatial vector.
    """

    result = Vec6d()
    for row in range(SPATIAL_DIM):
        local_row = row if row < 3 else row - 3
        offset = 0 if row < 3 else 3
        for column in range(3):
            result[row] += pose[local_row, column] * value[offset + column]
    return result


@wp.func
def _rotate_spatial_from_inertial(pose: wp.mat44d, value: Vec6d) -> Vec6d:
    """Rotate an inertial-frame spatial cotangent into the body frame.

    Args:
        pose: Homogeneous pose whose rotation maps body to inertial coordinates.
        value: Inertial-frame spatial cotangent.

    Returns:
        Body-frame spatial cotangent.
    """

    result = Vec6d()
    for row in range(SPATIAL_DIM):
        local_row = row if row < 3 else row - 3
        offset = 0 if row < 3 else 3
        for column in range(3):
            result[row] += pose[column, local_row] * value[offset + column]
    return result


@wp.func
def _coadjoint_wrench(
    velocity_first: wp.array2d[wp.float64],
    velocity_second: wp.array2d[wp.float64],
    derivative_first: wp.array2d[wp.float64],
    derivative_second: wp.array2d[wp.float64],
    use_first: bool,
    base_row: int,
    masses: wp.array2d[wp.float64],
    quadrature: int,
) -> Vec6d:
    """Combine inertial acceleration and coadjoint momentum terms.

    Args:
        velocity_first: First flattened spatial-velocity buffer.
        velocity_second: Second flattened spatial-velocity buffer.
        derivative_first: First flattened velocity-derivative buffer.
        derivative_second: Second flattened velocity-derivative buffer.
        use_first: Whether the first pair of buffers is active.
        base_row: First flattened row of the current spatial vector.
        masses: Diagonal spatial inertias for all quadrature items.
        quadrature: Row of ``masses`` used by the current item.

    Returns:
        Spatial wrench ``M @ eta_dot + ad(eta).T @ M @ eta``.
    """
    omega = wp.vec3d(
        _vector_value(velocity_first, velocity_second, use_first, base_row, 0),
        _vector_value(velocity_first, velocity_second, use_first, base_row, 1),
        _vector_value(velocity_first, velocity_second, use_first, base_row, 2),
    )
    linear_velocity = wp.vec3d(
        _vector_value(velocity_first, velocity_second, use_first, base_row, 3),
        _vector_value(velocity_first, velocity_second, use_first, base_row, 4),
        _vector_value(velocity_first, velocity_second, use_first, base_row, 5),
    )
    moment = wp.vec3d(
        masses[quadrature, 0] * omega[0],
        masses[quadrature, 1] * omega[1],
        masses[quadrature, 2] * omega[2],
    )
    force = wp.vec3d(
        masses[quadrature, 3] * linear_velocity[0],
        masses[quadrature, 4] * linear_velocity[1],
        masses[quadrature, 5] * linear_velocity[2],
    )
    angular = wp.cross(omega, moment) + wp.cross(linear_velocity, force)
    linear = wp.cross(omega, force)
    return Vec6d(
        masses[quadrature, 0]
        * _vector_value(derivative_first, derivative_second, use_first, base_row, 0)
        + angular[0],
        masses[quadrature, 1]
        * _vector_value(derivative_first, derivative_second, use_first, base_row, 1)
        + angular[1],
        masses[quadrature, 2]
        * _vector_value(derivative_first, derivative_second, use_first, base_row, 2)
        + angular[2],
        masses[quadrature, 3]
        * _vector_value(derivative_first, derivative_second, use_first, base_row, 3)
        + linear[0],
        masses[quadrature, 4]
        * _vector_value(derivative_first, derivative_second, use_first, base_row, 4)
        + linear[1],
        masses[quadrature, 5]
        * _vector_value(derivative_first, derivative_second, use_first, base_row, 5)
        + linear[2],
    )
