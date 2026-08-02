import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SECTION_DIR = (
    Path(__file__).resolve().parents[3] / "paper_results" / "secVc_model_based_control"
)
MODULE_DIR = SECTION_DIR / "code"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import report_metrics  # noqa: E402


def test_report_matches_canonical_operational_space_regression_values():
    report = report_metrics.build_metrics_report()
    position = report["operational_space"]["position"]
    orientation = report["operational_space"]["orientation"]

    assert np.isclose(position["rmse_m"], 2.835925238412752e-3)
    assert np.isclose(position["reference_span_m"], 0.13463535756926345)
    assert np.isclose(position["nrmse_percent"], 2.1063747960514787)
    assert np.isclose(np.rad2deg(orientation["rmse_rad"]), 2.2902277791469605)
    assert np.isclose(np.rad2deg(orientation["reference_span_rad"]), 114.59155902616467)
    assert np.isclose(orientation["nrmse_percent"], 1.998600768337599)


def test_report_exposes_six_coordinate_rmse_and_steady_state_metrics():
    report = report_metrics.build_metrics_report()
    configuration = report["configuration_space"]

    assert configuration["coordinate_order"] == [
        "kappa_x",
        "kappa_y",
        "kappa_z",
        "sigma_x",
        "sigma_y",
        "sigma_z",
    ]
    for controller in configuration["regulation"].values():
        assert len(controller["rmse"]) == 6
        assert len(controller["steady_state_mae"]) == 6
        assert len(controller["steady_state_mae_per_setpoint"]) == 4
    for controller in configuration["tracking"].values():
        assert len(controller["rmse"]) == 6


def test_report_matches_canonical_configuration_space_regression_values():
    report = report_metrics.build_metrics_report()
    configuration = report["configuration_space"]
    coordinates = configuration["coordinate_order"]

    computed_torque_regulation = configuration["regulation"]["Computed Torque"]
    computed_torque_tracking = configuration["tracking"]["Computed Torque"]
    np.testing.assert_allclose(
        [computed_torque_regulation["rmse"][name] for name in coordinates],
        [
            3.793044827207431e-14,
            2.0805802493204735,
            1.0211109565430658,
            0.012018448658962311,
            2.2963814990390372e-17,
            8.321348405938057e-17,
        ],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        [computed_torque_tracking["rmse"][name] for name in coordinates],
        [
            8.214812358747936e-15,
            0.0018663431823862612,
            0.0020940708429073736,
            5.082774073959551e-05,
            3.3568591057239607e-18,
            4.3969409724691505e-17,
        ],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        [computed_torque_regulation["steady_state_mae"][name] for name in coordinates],
        [
            3.333284661879546e-14,
            0.01248387270654943,
            0.011333863018661027,
            0.000530402449408513,
            4.485569604383307e-18,
            6.868158099256708e-18,
        ],
        rtol=1e-12,
    )


def test_report_falls_back_when_stored_coordinate_names_have_wrong_length(tmp_path):
    configuration_input = report_metrics.DEFAULT_CONFIGURATION_INPUT
    stale_metadata_input = tmp_path / "stale-metadata.npz"
    with np.load(configuration_input, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    metadata = json.loads(str(arrays["metadata"].item()))
    metadata["evaluation"]["strain_names"] = ["legacy_0", "legacy_1", "legacy_2"]
    arrays["metadata"] = np.array(json.dumps(metadata))
    np.savez_compressed(stale_metadata_input, **arrays)

    report = report_metrics.build_metrics_report(
        configuration_input=stale_metadata_input
    )

    assert report["configuration_space"]["coordinate_order"] == [
        "kappa_x",
        "kappa_y",
        "kappa_z",
        "sigma_x",
        "sigma_y",
        "sigma_z",
    ]


def test_report_cli_writes_machine_readable_json_from_unrelated_directory(tmp_path):
    output = tmp_path / "nested" / "metrics.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_DIR / "report_metrics.py"),
            "--json-output",
            str(output),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert "Section V.C evaluation metrics" in completed.stdout
    assert "configuration_space" in report
    assert "operational_space" in report
