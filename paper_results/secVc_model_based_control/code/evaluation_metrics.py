"""Evaluation metrics reported for the Section V.C control experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

STEADY_STATE_FRACTION = 0.1
REGULATION_SETPOINT_INTERVALS = (
    (3.0, 6.0),
    (6.0, 9.0),
    (9.0, 12.0),
    (12.0, 15.0),
)


def componentwise_rmse(errors: np.ndarray) -> np.ndarray:
    """Return root-mean-square error for every signal coordinate."""
    error_array = _sample_matrix(errors, "errors")
    return np.sqrt(np.mean(np.square(error_array), axis=0))


def terminal_window_masks(
    t: np.ndarray,
    intervals: Sequence[tuple[float, float]],
    *,
    fraction: float = STEADY_STATE_FRACTION,
) -> np.ndarray:
    """Select the final fraction of each half-open constant-reference interval."""
    timestamps = np.asarray(t, dtype=float)
    if timestamps.ndim != 1:
        raise ValueError(f"t must be one-dimensional, got shape {timestamps.shape}.")
    if timestamps.size == 0:
        raise ValueError("t must contain at least one sample.")
    if not np.all(np.diff(timestamps) >= 0.0):
        raise ValueError("t must be monotonically nondecreasing.")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in the interval (0, 1].")

    masks: list[np.ndarray] = []
    for start, stop in intervals:
        if stop <= start:
            raise ValueError(f"Invalid interval [{start}, {stop}).")
        terminal_start = stop - fraction * (stop - start)
        mask = np.logical_and(timestamps >= terminal_start, timestamps < stop)
        if not np.any(mask):
            raise ValueError(
                "No samples fall in terminal window "
                f"[{terminal_start}, {stop}) for interval [{start}, {stop})."
            )
        masks.append(mask)

    if not masks:
        raise ValueError("At least one setpoint interval is required.")
    return np.stack(masks, axis=0)


def steady_state_mae(
    t: np.ndarray,
    errors: np.ndarray,
    intervals: Sequence[tuple[float, float]] = REGULATION_SETPOINT_INTERVALS,
    *,
    fraction: float = STEADY_STATE_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Return equal-weight terminal-window MAE and its per-setpoint values.

    The initial zero-reference plateau is intentionally absent from the default
    intervals. Each supplied plateau contributes equally, independently of its
    sample count.
    """
    error_array = _sample_matrix(errors, "errors")
    masks = terminal_window_masks(t, intervals, fraction=fraction)
    if error_array.shape[0] != masks.shape[1]:
        raise ValueError(
            "errors and t must have the same number of samples, got "
            f"{error_array.shape[0]} and {masks.shape[1]}."
        )

    per_setpoint = np.stack(
        [np.mean(np.abs(error_array[mask]), axis=0) for mask in masks],
        axis=0,
    )
    return np.mean(per_setpoint, axis=0), per_setpoint


def relative_reduction_percent(
    baseline: np.ndarray | float,
    improved: np.ndarray | float,
) -> np.ndarray:
    """Return the percentage reduction from ``baseline`` to ``improved``."""
    baseline_array = np.asarray(baseline, dtype=float)
    improved_array = np.asarray(improved, dtype=float)
    baseline_array, improved_array = np.broadcast_arrays(baseline_array, improved_array)
    reduction = np.full(baseline_array.shape, np.nan, dtype=float)
    np.divide(
        baseline_array - improved_array,
        baseline_array,
        out=reduction,
        where=baseline_array != 0.0,
    )
    return 100.0 * reduction


def euclidean_reference_span(reference: np.ndarray) -> float:
    """Return the maximum pairwise Euclidean distance in a reference path."""
    reference_array = _sample_matrix(reference, "reference")
    if reference_array.shape[0] < 2:
        return 0.0

    maximum_squared_distance = 0.0
    for index, point in enumerate(reference_array[:-1]):
        deltas = reference_array[index + 1 :] - point
        squared_distances = np.einsum("ij,ij->i", deltas, deltas)
        maximum_squared_distance = max(
            maximum_squared_distance,
            float(np.max(squared_distances)),
        )
    return float(np.sqrt(maximum_squared_distance))


def so3_reference_span(rotation_vectors: np.ndarray) -> float:
    """Return the maximum pairwise geodesic distance in an SO(3) path."""
    rotations = _rotation_vectors(rotation_vectors, "rotation_vectors")
    if rotations.shape[0] < 2:
        return 0.0
    quaternions = Rotation.from_rotvec(rotations).as_quat()
    minimum_inner_product = 1.0
    for index, quaternion in enumerate(quaternions[:-1]):
        inner_products = np.abs(quaternions[index + 1 :] @ quaternion)
        minimum_inner_product = min(
            minimum_inner_product,
            float(np.min(inner_products)),
        )
    return float(2.0 * np.arccos(np.clip(minimum_inner_product, -1.0, 1.0)))


def normalized_rmse_percent(rmse: float, reference_span: float) -> float:
    """Normalize RMSE by a reference span and express the result in percent."""
    if reference_span < 0.0:
        raise ValueError("reference_span must be nonnegative.")
    if reference_span == 0.0:
        return float("nan")
    return 100.0 * float(rmse) / float(reference_span)


def operational_space_metrics(
    position: np.ndarray,
    position_reference: np.ndarray,
    orientation: np.ndarray,
    orientation_reference: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Compute the geometric position/orientation metrics reported in V.C."""
    position_array = _sample_matrix(position, "position")
    position_reference_array = _sample_matrix(position_reference, "position_reference")
    orientation_array = _rotation_vectors(orientation, "orientation")
    orientation_reference_array = _rotation_vectors(
        orientation_reference, "orientation_reference"
    )
    sample_count = position_array.shape[0]
    arrays = (
        position_reference_array,
        orientation_array,
        orientation_reference_array,
    )
    if any(array.shape[0] != sample_count for array in arrays):
        raise ValueError("All operational-space trajectories must share sample count.")
    if position_array.shape != position_reference_array.shape:
        raise ValueError("position and position_reference must have matching shapes.")

    position_error = position_reference_array - position_array
    current_rotations = Rotation.from_rotvec(orientation_array)
    desired_rotations = Rotation.from_rotvec(orientation_reference_array)
    orientation_error = (desired_rotations * current_rotations.inv()).as_rotvec()

    position_rmse = float(np.sqrt(np.mean(np.sum(np.square(position_error), axis=1))))
    orientation_rmse = float(
        np.sqrt(np.mean(np.sum(np.square(orientation_error), axis=1)))
    )
    position_span = euclidean_reference_span(position_reference_array)
    orientation_span = so3_reference_span(orientation_reference_array)

    return {
        "position_error": position_error,
        "orientation_error": orientation_error,
        "position_rmse": position_rmse,
        "orientation_rmse": orientation_rmse,
        "position_reference_span": position_span,
        "orientation_reference_span": orientation_span,
        "position_nrmse_percent": normalized_rmse_percent(position_rmse, position_span),
        "orientation_nrmse_percent": normalized_rmse_percent(
            orientation_rmse, orientation_span
        ),
    }


def _sample_matrix(values: np.ndarray, name: str) -> np.ndarray:
    # SciPy's Cython rotation backend requires writable, C-contiguous buffers;
    # arrays originating from JAX or memory-mapped NPZ data can be read-only.
    array = np.array(values, dtype=float, copy=True, order="C")
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (samples, coordinates).")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _rotation_vectors(values: np.ndarray, name: str) -> np.ndarray:
    array = _sample_matrix(values, name)
    if array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (samples, 3).")
    return array
