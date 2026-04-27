#!/usr/bin/env python3
"""Compare analytical PCS/GVS methods with SoftRobot autograd defaults.

This benchmark times paired analytical and autograd-backed implementations for
PCS and GVS models over a sweep of segment counts. The autograd path explicitly
calls the default implementations from ``SoftRobot`` so it remains comparable
even when subclasses override the public method names analytically.

uv run python tools/benchmarks/benchmark_autograd_vs_analytic.py \
  --systems pcs gvs \
  --segment-counts 1 2 4 8 12 20 \
  --execution-repeats 5 \
  --csv benchmarks/autograd-vs-analytic.csv \
  --json benchmarks/autograd-vs-analytic.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import equinox as eqx
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from soromox.systems import SoftRobot  # noqa: E402
from tools.benchmarks._benchmark_common import (  # noqa: E402
    add_gauss_point_args,
    add_system_selection_args,
    block_until_ready,
    build_system_with_gauss_points,
    gauss_point_sweep_values,
    get_system_registry,
    normalize_gauss_point_values,
    system_gauss_point_metadata,
)

Array = jax.Array
Tree = Any


def _analytical_jacobian(system: SoftRobot, q: Array, s: Array) -> Array:
    return system.jacobian(q, s)


def _autograd_jacobian(system: SoftRobot, q: Array, s: Array) -> Array:
    return SoftRobot.jacobian(system, q, s)


def _analytical_jacobian_and_time_derivative(
    system: SoftRobot, q: Array, qd: Array, s: Array
) -> tuple[Array, Array]:
    return system.jacobian_and_time_derivative(q, qd, s)


def _autograd_jacobian_and_time_derivative(
    system: SoftRobot, q: Array, qd: Array, s: Array
) -> tuple[Array, Array]:
    # Do not call SoftRobot.jacobian_and_time_derivative(system, ...): the base method
    # dispatches through self.jacobian, which would resolve to PCS/GVS analytical
    # overrides. Pin the inner Jacobian to SoftRobot.jacobian so this benchmark
    # measures the default autograd-from-forward-kinematics path.
    # Therefore, this is actually an autograd-based Hessian of the forward kinematics.
    return jax.jvp(lambda q_: SoftRobot.jacobian(system, q_, s), (q,), (qd,))


def _analytical_gravitational_force(system: SoftRobot, q: Array) -> Array:
    return system.gravitational_force(q)


def _autograd_gravitational_force(system: SoftRobot, q: Array) -> Array:
    return SoftRobot.gravitational_force(system, q)


def _case_registry() -> Mapping[
    str,
    tuple[
        Callable[..., Tree],
        Callable[..., Tree],
        Callable[[Mapping[str, Array]], tuple[Any, ...]],
    ],
]:
    return {
        "jacobian": (
            _analytical_jacobian,
            _autograd_jacobian,
            lambda ctx: (ctx["q"], ctx["s_tip"]),
        ),
        "jacobian_and_time_derivative": (
            _analytical_jacobian_and_time_derivative,
            _autograd_jacobian_and_time_derivative,
            lambda ctx: (ctx["q"], ctx["qd"], ctx["s_tip"]),
        ),
        "gravitational_force": (
            _analytical_gravitational_force,
            _autograd_gravitational_force,
            lambda ctx: (ctx["q"],),
        ),
    }


def _measure_jitted_call(
    fn: Callable[..., Tree],
    args: tuple[Any, ...],
    repeats: int,
    warmup_runs: int,
) -> tuple[float, float, Tree]:
    """Return ``(estimated_compile_time, mean_execution_time, last_output)``."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    start = time.perf_counter()
    first = fn(*args)
    block_until_ready(first)
    first_time = time.perf_counter() - start

    last_output = first
    for _ in range(max(0, warmup_runs)):
        last_output = fn(*args)
        block_until_ready(last_output)

    exec_times: list[float] = []
    for _ in range(repeats):
        loop_start = time.perf_counter()
        last_output = fn(*args)
        block_until_ready(last_output)
        exec_times.append(time.perf_counter() - loop_start)

    exec_time = sum(exec_times) / len(exec_times)
    compile_time = max(0.0, first_time - exec_time)
    return compile_time, exec_time, last_output


def _tree_difference_metrics(reference: Tree, candidate: Tree) -> tuple[float, float]:
    leaves_ref, treedef_ref = jax.tree_util.tree_flatten(reference)
    leaves_cand, treedef_cand = jax.tree_util.tree_flatten(candidate)
    if treedef_ref != treedef_cand:
        return float("nan"), float("nan")
    if not leaves_ref:
        return 0.0, 0.0

    max_abs = 0.0
    max_rel = 0.0
    for ref, cand in zip(leaves_ref, leaves_cand, strict=True):
        ref_arr = jnp.asarray(ref)
        cand_arr = jnp.asarray(cand)
        abs_diff = jnp.max(jnp.abs(ref_arr - cand_arr))
        denom = jnp.maximum(jnp.max(jnp.abs(ref_arr)), jnp.array(1e-12))
        rel_diff = abs_diff / denom
        max_abs = max(max_abs, float(abs_diff))
        max_rel = max(max_rel, float(rel_diff))
    return max_abs, max_rel


def _write_json(results: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"[+] Wrote JSON results to {path}")


def _write_csv(results: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "system",
        "method",
        "size_label",
        "segment_count",
        "dof",
        "gauss_points",
        "integration_points",
        "analytical_compile_time_s",
        "analytical_execution_time_s",
        "autograd_compile_time_s",
        "autograd_execution_time_s",
        "execution_ratio_autograd_over_analytical",
        "max_abs_diff",
        "max_rel_diff",
        "execution_repeats",
        "warmup_runs",
        "device",
        "device_id",
    ]
    with path.open("w", encoding="utf-8") as fp:
        fp.write(",".join(headers) + "\n")
        for row in results:
            fp.write(",".join(str(row.get(col, "")) for col in headers) + "\n")
    print(f"[+] Wrote CSV results to {path}")


def _format_ratio(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.2f}x"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    system_registry = {
        name: config
        for name, config in get_system_registry().items()
        if name in {"pcs", "gvs"}
    }
    case_registry = _case_registry()
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark analytical PCS/GVS methods against SoftRobot autograd "
            "default implementations."
        )
    )
    add_system_selection_args(parser, system_registry, default_segment_counts=[1, 2, 4])
    add_gauss_point_args(parser)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=list(case_registry.keys()),
        choices=sorted(case_registry.keys()),
        help="Methods to benchmark (default: all)",
    )
    parser.add_argument(
        "--execution-repeats",
        type=int,
        default=5,
        help="Number of post-compilation executions to average",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warm executions between the cold compile call and measured repeats",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path to store benchmark results as JSON",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path to store benchmark results as CSV",
    )
    args = parser.parse_args(argv)
    if args.gauss_points is not None:
        if any(value < 1 for value in args.gauss_points):
            parser.error("All --gauss-points entries must be >= 1.")
        args.gauss_points = normalize_gauss_point_values(args.gauss_points)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    system_registry = {
        name: config
        for name, config in get_system_registry().items()
        if name in {"pcs", "gvs"}
    }
    case_registry = _case_registry()
    device = jax.devices()[0]
    results: list[dict[str, Any]] = []

    for system_name in args.systems:
        system_cfg = system_registry[system_name]
        gauss_values = gauss_point_sweep_values(system_cfg, args.gauss_points)
        if not gauss_values:
            print(f"[!] Skipping {system_name}: no applicable Gauss-point values.")
            continue

        for segment_count in args.segment_counts:
            for requested_gauss_points in gauss_values:
                system = build_system_with_gauss_points(
                    system_cfg, segment_count, requested_gauss_points
                )
                ctx = system_cfg.build_context(system)
                gauss_points, integration_points = system_gauss_point_metadata(system)
                print(
                    f"\n=== {system_name}: {system_cfg.size_label}="
                    f"{segment_count}, dof={system.num_dofs}, "
                    f"gauss_points={gauss_points} ==="
                )

                for method_name in args.methods:
                    analytical_fn, autograd_fn, arg_builder = case_registry[method_name]
                    method_args = arg_builder(ctx)
                    analytical_jit = eqx.filter_jit(analytical_fn)
                    autograd_jit = eqx.filter_jit(autograd_fn)
                    analytical_args = (system, *method_args)
                    autograd_args = (system, *method_args)

                    print(f"  -> {method_name} ...", end=" ", flush=True)
                    (
                        analytical_compile,
                        analytical_execution,
                        analytical_output,
                    ) = _measure_jitted_call(
                        analytical_jit,
                        analytical_args,
                        args.execution_repeats,
                        args.warmup_runs,
                    )
                    (
                        autograd_compile,
                        autograd_execution,
                        autograd_output,
                    ) = _measure_jitted_call(
                        autograd_jit,
                        autograd_args,
                        args.execution_repeats,
                        args.warmup_runs,
                    )

                    max_abs_diff, max_rel_diff = _tree_difference_metrics(
                        analytical_output, autograd_output
                    )
                    ratio = autograd_execution / analytical_execution
                    print(
                        f"analytical={analytical_execution:.6f}s, "
                        f"autograd={autograd_execution:.6f}s, "
                        f"ratio={_format_ratio(ratio)}, "
                        f"max_abs_diff={max_abs_diff:.3e}"
                    )

                    results.append(
                        {
                            "system": system_name,
                            "method": method_name,
                            "size_label": system_cfg.size_label,
                            "segment_count": segment_count,
                            "dof": int(system.num_dofs),
                            "gauss_points": gauss_points,
                            "integration_points": integration_points,
                            "analytical_compile_time_s": analytical_compile,
                            "analytical_execution_time_s": analytical_execution,
                            "autograd_compile_time_s": autograd_compile,
                            "autograd_execution_time_s": autograd_execution,
                            "execution_ratio_autograd_over_analytical": ratio,
                            "max_abs_diff": max_abs_diff,
                            "max_rel_diff": max_rel_diff,
                            "execution_repeats": args.execution_repeats,
                            "warmup_runs": args.warmup_runs,
                            "device": device.platform,
                            "device_id": device.id,
                        }
                    )

    if args.json is not None:
        _write_json(results, args.json)
    if args.csv is not None:
        _write_csv(results, args.csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
