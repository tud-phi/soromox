"""Report the reproducible evaluation metrics for the Section V.C results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from evaluation_metrics import (
    REGULATION_SETPOINT_INTERVALS,
    STEADY_STATE_FRACTION,
    componentwise_rmse,
    operational_space_metrics,
    relative_reduction_percent,
    steady_state_mae,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SECTION_DIR = SCRIPT_DIR.parent
DEFAULT_CONFIGURATION_INPUT = (
    SECTION_DIR
    / "configuration_space_comparison"
    / "data"
    / "regulation_tracking_comparison.npz"
)
DEFAULT_OPERATIONAL_INPUT = (
    SECTION_DIR
    / "operational_space_impedance_control"
    / "data"
    / "control_pcs_with_impedance_trajectory.npz"
)
DEFAULT_JSON_OUTPUT = SECTION_DIR / "data" / "metrics.json"


def build_metrics_report(
    configuration_input: str | Path = DEFAULT_CONFIGURATION_INPUT,
    operational_input: str | Path = DEFAULT_OPERATIONAL_INPUT,
) -> dict[str, Any]:
    """Build the complete report directly from the two canonical NPZ files."""
    configuration = _load_configuration_artifact(configuration_input)
    operational = _load_operational_artifact(operational_input)
    coordinate_names = configuration["coordinate_names"]

    regulation = _configuration_phase(
        configuration,
        "regulation",
        coordinate_names,
        include_steady_state=True,
    )
    tracking = _configuration_phase(
        configuration,
        "tracking",
        coordinate_names,
        include_steady_state=False,
    )
    comparisons = {
        "feedforward_vs_potential_tracking": _rmse_reduction(
            configuration,
            "tracking",
            baseline="Potential Compensation",
            improved="Feedforward Compensation",
            coordinate_names=coordinate_names,
        ),
        "computed_torque_vs_feedforward_tracking": _rmse_reduction(
            configuration,
            "tracking",
            baseline="Feedforward Compensation",
            improved="Computed Torque",
            coordinate_names=coordinate_names,
        ),
        "computed_torque_vs_potential_regulation": _rmse_reduction(
            configuration,
            "regulation",
            baseline="Potential Compensation",
            improved="Computed Torque",
            coordinate_names=coordinate_names,
        ),
    }

    required_operational_metrics = {
        "position_rmse",
        "orientation_rmse",
        "position_reference_span",
        "orientation_reference_span",
        "position_nrmse_percent",
        "orientation_nrmse_percent",
    }
    missing = sorted(required_operational_metrics - operational.keys())
    if missing:
        raise ValueError(f"Operational artifact is missing metrics: {missing}")

    return {
        "configuration_space": {
            "coordinate_order": list(coordinate_names),
            "regulation": regulation,
            "tracking": tracking,
            "relative_reductions_percent": comparisons,
            "steady_state_definition": {
                "fraction": configuration["steady_state_fraction"],
                "setpoint_intervals_s": [
                    list(interval)
                    for interval in configuration["steady_state_intervals"]
                ],
                "initial_interval_excluded": True,
                "aggregation": "equal mean across setpoints",
            },
        },
        "operational_space": {
            "position": {
                "rmse_m": operational["position_rmse"],
                "reference_span_m": operational["position_reference_span"],
                "nrmse_percent": operational["position_nrmse_percent"],
            },
            "orientation": {
                "rmse_rad": operational["orientation_rmse"],
                "reference_span_rad": operational["orientation_reference_span"],
                "nrmse_percent": operational["orientation_nrmse_percent"],
            },
        },
    }


def _load_configuration_artifact(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        evaluation = metadata.get("evaluation", {})
        results: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for scenario_index, scenario in enumerate(metadata["scenarios"]):
            scenario_results: dict[str, dict[str, np.ndarray]] = {}
            for controller_index, controller in enumerate(scenario["controller_names"]):
                prefix = f"s{scenario_index}_c{controller_index}"
                tracking_error = np.asarray(data[f"{prefix}_metrics_tracking_error"])
                rmse = np.asarray(data[f"{prefix}_metrics_rmse"])
                if rmse.shape != (tracking_error.shape[1],):
                    rmse = componentwise_rmse(tracking_error)
                metrics = {"rmse": rmse, "tracking_error": tracking_error}
                steady_state_key = f"{prefix}_metrics_steady_state_mae"
                per_setpoint_key = f"{prefix}_metrics_steady_state_mae_per_setpoint"
                if steady_state_key in data:
                    metrics["steady_state_mae"] = np.asarray(data[steady_state_key])
                    metrics["steady_state_mae_per_setpoint"] = np.asarray(
                        data[per_setpoint_key]
                    )
                elif scenario["key"] == "regulation":
                    intervals = tuple(
                        tuple(float(value) for value in interval)
                        for interval in evaluation.get(
                            "steady_state_setpoint_intervals",
                            REGULATION_SETPOINT_INTERVALS,
                        )
                    )
                    aggregate, per_setpoint = steady_state_mae(
                        np.asarray(data[f"{prefix}_t"]),
                        tracking_error,
                        intervals,
                        fraction=float(
                            evaluation.get(
                                "steady_state_fraction", STEADY_STATE_FRACTION
                            )
                        ),
                    )
                    metrics["steady_state_mae"] = aggregate
                    metrics["steady_state_mae_per_setpoint"] = per_setpoint
                scenario_results[controller] = metrics
            results[scenario["key"]] = scenario_results

    first_metrics = next(iter(next(iter(results.values())).values()))
    coordinate_count = first_metrics["tracking_error"].shape[1]
    default_names = (
        "kappa_x",
        "kappa_y",
        "kappa_z",
        "sigma_x",
        "sigma_y",
        "sigma_z",
    )
    if coordinate_count != len(default_names):
        default_names = tuple(f"q_{index}" for index in range(coordinate_count))
    stored_names = tuple(evaluation.get("strain_names", ()))
    coordinate_names = (
        stored_names if len(stored_names) == coordinate_count else default_names
    )
    return {
        "results": results,
        "coordinate_names": coordinate_names,
        "steady_state_intervals": tuple(
            tuple(float(value) for value in interval)
            for interval in evaluation.get(
                "steady_state_setpoint_intervals",
                REGULATION_SETPOINT_INTERVALS,
            )
        ),
        "steady_state_fraction": float(
            evaluation.get("steady_state_fraction", STEADY_STATE_FRACTION)
        ),
    }


def _load_operational_artifact(path: str | Path) -> dict[str, float]:
    with np.load(path, allow_pickle=False) as data:
        metrics = {
            key.removeprefix("metrics_"): float(data[key])
            for key in data.files
            if key.startswith("metrics_")
        }
        if metrics:
            return metrics
        x = np.asarray(data["x_traj_full"])
        x_des = np.asarray(data["x_des_traj_full"])
    computed = operational_space_metrics(
        position=x[:, 3:6],
        position_reference=x_des[:, 3:6],
        orientation=x[:, :3],
        orientation_reference=x_des[:, :3],
    )
    return {
        name: float(value)
        for name, value in computed.items()
        if name not in {"position_error", "orientation_error"}
    }


def _configuration_phase(
    artifact: dict[str, Any],
    phase: str,
    coordinate_names: tuple[str, ...],
    *,
    include_steady_state: bool,
) -> dict[str, Any]:
    phase_results = artifact["results"][phase]
    output: dict[str, Any] = {}
    for controller, metrics in phase_results.items():
        controller_output: dict[str, Any] = {
            "rmse": _coordinate_mapping(metrics["rmse"], coordinate_names)
        }
        if include_steady_state:
            controller_output["steady_state_mae"] = _coordinate_mapping(
                metrics["steady_state_mae"], coordinate_names
            )
            controller_output["steady_state_mae_per_setpoint"] = [
                _coordinate_mapping(values, coordinate_names)
                for values in np.asarray(
                    metrics["steady_state_mae_per_setpoint"], dtype=float
                )
            ]
        output[controller] = controller_output
    return output


def _rmse_reduction(
    artifact: dict[str, Any],
    phase: str,
    *,
    baseline: str,
    improved: str,
    coordinate_names: tuple[str, ...],
) -> dict[str, float]:
    results = artifact["results"][phase]
    reduction = relative_reduction_percent(
        results[baseline]["rmse"],
        results[improved]["rmse"],
    )
    return _coordinate_mapping(reduction, coordinate_names)


def _coordinate_mapping(
    values: np.ndarray,
    coordinate_names: tuple[str, ...],
) -> dict[str, float]:
    value_array = np.asarray(values, dtype=float)
    if value_array.shape != (len(coordinate_names),):
        raise ValueError(
            f"Metric has shape {value_array.shape}, expected "
            f"({len(coordinate_names)},)."
        )
    return {
        coordinate: float(value)
        for coordinate, value in zip(coordinate_names, value_array)
    }


def format_human_readable(report: dict[str, Any]) -> str:
    """Format the complete report for terminal inspection."""
    configuration = report["configuration_space"]
    coordinates = configuration["coordinate_order"]
    lines = ["Section V.C evaluation metrics", ""]
    for phase in ("regulation", "tracking"):
        lines.extend([f"Configuration-space {phase} RMSE", _table_header(coordinates)])
        for controller, metrics in configuration[phase].items():
            lines.append(_table_row(controller, metrics["rmse"], coordinates))
        if phase == "regulation":
            lines.extend(
                [
                    "",
                    "Configuration-space regulation steady-state MAE",
                    _table_header(coordinates),
                ]
            )
            for controller, metrics in configuration[phase].items():
                lines.append(
                    _table_row(controller, metrics["steady_state_mae"], coordinates)
                )
        lines.append("")

    lines.append("Relative RMSE reductions")
    for comparison, reductions in configuration["relative_reductions_percent"].items():
        formatted = ", ".join(
            f"{coordinate}={reductions[coordinate]:.3f}%" for coordinate in coordinates
        )
        lines.append(f"  {comparison}: {formatted}")

    position = report["operational_space"]["position"]
    orientation = report["operational_space"]["orientation"]
    lines.extend(
        [
            "",
            "Operational-space pose tracking",
            f"  position RMSE: {1e3 * position['rmse_m']:.6f} mm",
            f"  position reference span: {1e3 * position['reference_span_m']:.6f} mm",
            f"  position NRMSE: {position['nrmse_percent']:.6f}%",
            f"  orientation RMSE: {np.rad2deg(orientation['rmse_rad']):.6f} deg",
            "  orientation reference span: "
            f"{np.rad2deg(orientation['reference_span_rad']):.6f} deg",
            f"  orientation NRMSE: {orientation['nrmse_percent']:.6f}%",
        ]
    )
    return "\n".join(lines)


def _table_header(coordinates: list[str]) -> str:
    return f"{'Controller':<30} " + " ".join(
        f"{coordinate:>13}" for coordinate in coordinates
    )


def _table_row(
    controller: str,
    values: dict[str, float],
    coordinates: list[str],
) -> str:
    return f"{controller:<30} " + " ".join(
        f"{values[coordinate]:>13.6g}" for coordinate in coordinates
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configuration-input", type=Path, default=DEFAULT_CONFIGURATION_INPUT
    )
    parser.add_argument(
        "--operational-input", type=Path, default=DEFAULT_OPERATIONAL_INPUT
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        nargs="?",
        const=DEFAULT_JSON_OUTPUT,
        default=None,
        help=(
            "Write the machine-readable report. If no path is supplied, write "
            f"{DEFAULT_JSON_OUTPUT}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_metrics_report(args.configuration_input, args.operational_input)
    print(format_human_readable(report))
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON report written to: {args.json_output}")


if __name__ == "__main__":
    main()
