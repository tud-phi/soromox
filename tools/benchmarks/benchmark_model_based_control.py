#!/usr/bin/env python3
"""Benchmark the default model-based controller evaluation paths.

Run this script on two revisions to quantify controller performance changes. It
intentionally benchmarks only the implementation provided by the checked-out
revision instead of keeping legacy and current strategies in one benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from soromox.control import PIDControl, ReferenceTrajectory  # noqa: E402
from soromox.control.actuation_space import (  # noqa: E402
    FeedforwardCompensationTracker as ActuationFeedforwardCompensationTracker,
)
from soromox.control.configuration_space import (  # noqa: E402
    ComputedTorqueTracker,
    FeedforwardCompensationTracker,
)
from soromox.coordinate_transformations import (  # noqa: E402
    ActuationSpaceDynamics,
    OperationalSpaceDynamics,
)
from soromox.systems import SystemState  # noqa: E402
from tools.benchmarks._benchmark_common import (  # noqa: E402
    add_gauss_point_args,
    add_system_selection_args,
    block_until_ready,
    build_system_with_gauss_points,
    gauss_point_sweep_values,
    get_system_registry,
    normalize_gauss_point_values,
    normalize_system_names,
    system_gauss_point_metadata,
)

Array = jax.Array
Tree = Any
OUTPUT_FIELDS = (
    "system",
    "method",
    "size_label",
    "size_value",
    "gauss_points",
    "integration_points",
    "jit_compile_time_s",
    "jit_execution_time_s",
    "execution_repeats",
)


@dataclass(frozen=True)
class BenchmarkCase:
    """One default controller or controller-facing dynamics evaluation."""

    name: str
    builder: Callable[[Any, Mapping[str, Array]], tuple[Callable[..., Tree], tuple]]


def _reference_trajectory(context: Mapping[str, Array]) -> ReferenceTrajectory:
    """Build a deterministic smooth-state reference for controller timing."""
    q_des = 0.5 * context["q"]
    qd_des = 0.5 * context["qd"]
    qdd_des = -0.5 * context["qd"]
    return ReferenceTrajectory(
        ts=jnp.array([0.0, 1.0]),
        x_des_fn=lambda t: q_des,
        xd_des_fn=lambda t: qd_des,
        xdd_des_fn=lambda t: qdd_des,
    )


def _system_state(context: Mapping[str, Array]) -> SystemState:
    """Return the deterministic state shared by controller cases."""
    return SystemState(t=context["t"], y=context["y"])


def _build_computed_torque_case(system: Any, context: Mapping[str, Array]):
    controller = ComputedTorqueTracker(
        system,
        _reference_trajectory(context),
        PIDControl(0.0, 0.0, 0.0),
    )
    return controller.model_based_term, (_system_state(context),)


def _build_configuration_feedforward_case(
    system: Any, context: Mapping[str, Array]
):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FeedforwardCompensationTracker: For full theoretical guarantees",
            category=UserWarning,
        )
        controller = FeedforwardCompensationTracker(
            system,
            _reference_trajectory(context),
            PIDControl(0.0, 0.0, 0.0),
        )
    return controller.model_based_term, (_system_state(context),)


def _build_actuation_feedforward_case(system: Any, context: Mapping[str, Array]):
    dynamics = ActuationSpaceDynamics(system)
    controller = ActuationFeedforwardCompensationTracker(
        dynamics,
        _reference_trajectory(context),
        PIDControl(0.0, 0.0, 0.0),
    )
    return controller.model_based_term, (_system_state(context),)


def _build_operational_dynamics_case(system: Any, context: Mapping[str, Array]):
    dynamics = OperationalSpaceDynamics(
        system, jnp.atleast_1d(context["s_tip"])
    )
    return dynamics.dynamics_terms, (context["q"], context["qd"])


def _benchmark_cases() -> tuple[BenchmarkCase, ...]:
    """Return controller paths whose implementation is tracked over revisions."""
    return (
        BenchmarkCase(
            "configuration_space_computed_torque", _build_computed_torque_case
        ),
        BenchmarkCase(
            "configuration_space_feedforward_compensation",
            _build_configuration_feedforward_case,
        ),
        BenchmarkCase(
            "actuation_space_feedforward_compensation",
            _build_actuation_feedforward_case,
        ),
        BenchmarkCase(
            "operational_space_dynamics_terms",
            _build_operational_dynamics_case,
        ),
    )


def _measure_jitted_call(
    fn: Callable[..., Tree], args: tuple[Any, ...], repeats: int
) -> tuple[float, float]:
    """Return derived compile time and mean synchronized warm runtime."""
    jitted_fn = jax.jit(lambda *call_args: fn(*call_args))

    start = time.perf_counter()
    first = jitted_fn(*args)
    block_until_ready(first)
    first_time = time.perf_counter() - start

    execution_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = jitted_fn(*args)
        block_until_ready(output)
        execution_times.append(time.perf_counter() - start)

    execution_time = sum(execution_times) / len(execution_times)
    return max(0.0, first_time - execution_time), execution_time


def _write_json(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _system_registry() -> Mapping[str, Any]:
    shared = get_system_registry()
    return {name: shared[name] for name in ("pcs", "gvs")}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    registry = _system_registry()
    cases = _benchmark_cases()
    parser = argparse.ArgumentParser(
        description="Benchmark default model-based controller paths"
    )
    add_system_selection_args(
        parser,
        registry,
        default_segment_counts=[1, 4],
        default_systems=["pcs", "gvs"],
    )
    add_gauss_point_args(parser)
    parser.add_argument(
        "--methods",
        nargs="*",
        choices=[case.name for case in cases],
        default=None,
        help="Controller paths to benchmark (default: all)",
    )
    parser.add_argument(
        "--execution-repeats",
        type=int,
        default=10,
        help="Number of synchronized warm executions to average",
    )
    parser.add_argument("--json", type=Path, help="Optional JSON output path")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path")
    args = parser.parse_args(argv)

    try:
        args.systems = normalize_system_names(args.systems, registry)
    except ValueError as error:
        parser.error(str(error))
    if any(size < 1 for size in args.segment_counts):
        parser.error("All --segment-counts entries must be >= 1.")
    if args.execution_repeats < 1:
        parser.error("--execution-repeats must be >= 1.")
    if args.gauss_points is not None:
        if any(value < 1 for value in args.gauss_points):
            parser.error("All --gauss-points entries must be >= 1.")
        args.gauss_points = normalize_gauss_point_values(args.gauss_points)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    registry = _system_registry()
    selected_cases = [
        case
        for case in _benchmark_cases()
        if args.methods is None or case.name in args.methods
    ]
    rows: list[dict[str, Any]] = []

    for system_name in args.systems:
        config = registry[system_name]
        for size in args.segment_counts:
            for requested_gauss_points in gauss_point_sweep_values(
                config, args.gauss_points
            ):
                system = build_system_with_gauss_points(
                    config, size, requested_gauss_points
                )
                context = config.build_context(system)
                gauss_points, integration_points = system_gauss_point_metadata(system)
                print(
                    f"\n=== {system_name}: {config.size_label}={size}, "
                    f"gauss_points={gauss_points} ==="
                )

                for case in selected_cases:
                    fn, call_args = case.builder(system, context)
                    compile_time, execution_time = _measure_jitted_call(
                        fn, call_args, args.execution_repeats
                    )
                    print(
                        f"  {case.name}: compile {compile_time:.4f}s | "
                        f"exec {execution_time:.6f}s"
                    )
                    rows.append(
                        {
                            "system": system_name,
                            "method": case.name,
                            "size_label": config.size_label,
                            "size_value": size,
                            "gauss_points": gauss_points,
                            "integration_points": integration_points,
                            "jit_compile_time_s": compile_time,
                            "jit_execution_time_s": execution_time,
                            "execution_repeats": args.execution_repeats,
                        }
                    )

    if args.json:
        _write_json(rows, args.json)
        print(f"Wrote JSON results to {args.json.resolve()}")
    if args.csv:
        _write_csv(rows, args.csv)
        print(f"Wrote CSV results to {args.csv.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
