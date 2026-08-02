import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SECTION_DIR = (
    Path(__file__).resolve().parents[3] / "paper_results" / "secVc_model_based_control"
)
MODULE_DIR = SECTION_DIR / "code"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from evaluation_metrics import (  # noqa: E402
    REGULATION_SETPOINT_INTERVALS,
    componentwise_rmse,
    euclidean_reference_span,
    operational_space_metrics,
    relative_reduction_percent,
    so3_reference_span,
    steady_state_mae,
    terminal_window_masks,
)


def test_componentwise_rmse_retains_all_six_coordinates():
    errors = np.vstack((np.arange(1.0, 7.0), -np.arange(1.0, 7.0)))

    np.testing.assert_allclose(componentwise_rmse(errors), np.arange(1.0, 7.0))


def test_relative_reduction_is_derived_from_metric_values():
    reduction = relative_reduction_percent([4.0, 2.0, 0.0], [1.0, 1.0, 0.0])

    np.testing.assert_allclose(reduction[:2], [75.0, 50.0])
    assert np.isnan(reduction[2])


def test_terminal_windows_use_final_ten_percent_of_post_change_plateaus():
    t = np.arange(0.0, 15.01, 0.01)
    masks = terminal_window_masks(t, REGULATION_SETPOINT_INTERVALS)

    np.testing.assert_array_equal(masks.sum(axis=1), np.full(4, 30))
    for mask, (_, stop) in zip(masks, REGULATION_SETPOINT_INTERVALS):
        assert np.all(t[mask] < stop)
        assert not mask[np.flatnonzero(np.isclose(t, stop))[0]]
    assert not np.any(masks[:, t < 3.0])


def test_steady_state_mae_is_componentwise_and_equal_weights_setpoints():
    t = np.arange(0.0, 15.01, 0.01)
    errors = np.zeros((t.size, 6))
    masks = terminal_window_masks(t, REGULATION_SETPOINT_INTERVALS)
    expected_per_setpoint = np.stack(
        [setpoint * np.arange(1.0, 7.0) for setpoint in range(1, 5)]
    )
    for mask, values in zip(masks, expected_per_setpoint):
        errors[mask] = values

    aggregate, per_setpoint = steady_state_mae(t, errors)

    np.testing.assert_allclose(per_setpoint, expected_per_setpoint)
    np.testing.assert_allclose(aggregate, expected_per_setpoint.mean(axis=0))


def test_operational_metrics_are_geometric_across_rotation_vector_boundary():
    degrees = np.pi / 180.0
    position = np.zeros((2, 3))
    position_reference = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    orientation = np.array([[0.0, 0.0, 179.0 * degrees]] * 2)
    orientation_reference = np.array(
        [[0.0, 0.0, -179.0 * degrees], [0.0, 0.0, 179.0 * degrees]]
    )

    metrics = operational_space_metrics(
        position,
        position_reference,
        orientation,
        orientation_reference,
    )

    assert np.isclose(metrics["position_rmse"], np.sqrt(12.5))
    assert np.isclose(metrics["position_reference_span"], 5.0)
    assert np.isclose(metrics["position_nrmse_percent"], 100.0 * np.sqrt(0.5))
    assert np.isclose(metrics["orientation_rmse"], np.deg2rad(2.0) / np.sqrt(2.0))
    assert np.isclose(metrics["orientation_reference_span"], np.deg2rad(2.0))
    assert np.isclose(metrics["orientation_nrmse_percent"], 100.0 / np.sqrt(2.0))
    np.testing.assert_allclose(
        np.linalg.norm(metrics["orientation_error"], axis=1),
        np.deg2rad([2.0, 0.0]),
        atol=1e-12,
    )


def test_reference_spans_match_full_pairwise_calculation():
    rng = np.random.default_rng(42)
    positions = rng.normal(size=(25, 3))
    rotation_vectors = rng.normal(size=(25, 3))

    expected_position_span = np.max(
        np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    )
    quaternions = Rotation.from_rotvec(rotation_vectors).as_quat()
    expected_orientation_span = np.max(
        2.0 * np.arccos(np.clip(np.abs(quaternions @ quaternions.T), -1.0, 1.0))
    )

    assert np.isclose(euclidean_reference_span(positions), expected_position_span)
    assert np.isclose(so3_reference_span(rotation_vectors), expected_orientation_span)
