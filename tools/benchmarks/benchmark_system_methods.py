#!/usr/bin/env python3

"""Benchmark the core dynamics routines for Soromox soft robot systems.

This script measures both the ahead-of-time JIT compilation cost and the steady-state
execution time (after compilation) for selected systems and core methods. It supports
benchmarking the Pendulum, PlanarPCS, and PCS implementations over a sweep of link or
segment counts, and can optionally visualise the results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
except ImportError as exc:  # pragma: no cover - this is a runtime guard
    raise ImportError(
        "JAX is required to run the benchmarking tooling. Install it via `pip install jax` "
        "or consult the Soromox documentation for setup instructions."
    ) from exc

import matplotlib.pyplot as plt

from tools.benchmarks._benchmark_common import (
    add_integration_args,
    add_system_selection_args,
    block_until_ready,
    get_system_registry,
)

Array = jax.Array
Tree = Any


@dataclass
class RuntimeConfig:
    """Holds runtime controls shared across benchmark cases."""

    duration: float
    dt: float
    save_dt: int
    execution_repeats: int


@dataclass
class BenchmarkCase:
    """A single benchmarked callable for a system."""

    name: str
    builder: Callable[[Any, Mapping[str, Array], RuntimeConfig], Tuple[Callable[..., Tree], Tuple[Any, ...]]]
    description: str = ""


@dataclass
class SystemBenchmark:
    """Groups benchmark cases and helpers for one system type."""

    factory: Callable[[int], Any]
    size_label: str
    build_context: Callable[[Any], MutableMapping[str, Array]]
    cases: Sequence[BenchmarkCase]


def _measure_jitted_call(
    fn: Callable[..., Tree],
    args: Tuple[Any, ...],
    repeats: int,
) -> Tuple[float, float]:
    """Measure compile (first-call) and averaged execution time for a callable.

    The first invocation is treated as "compile + execute" time. Subsequent invocations
    (``repeats`` of them) are averaged to estimate the steady-state execution cost.
    """

    if repeats < 1:
        raise ValueError("execution repeats must be at least 1")

    start = time.perf_counter()
    first = fn(*args)
    block_until_ready(first)
    first_time = time.perf_counter() - start

    exec_times: List[float] = []
    for _ in range(repeats):
        loop_start = time.perf_counter()
        out = fn(*args)
        block_until_ready(out)
        exec_times.append(time.perf_counter() - loop_start)

    exec_time = sum(exec_times) / len(exec_times)
    compile_time = max(0.0, first_time - exec_time)
    return compile_time, exec_time


def _build_system_registry() -> Mapping[str, SystemBenchmark]:
    shared = get_system_registry()
    pendulum_cases = (
        BenchmarkCase(
            name="forward_kinematics",
            description="Tip pose forward kinematics",
            builder=lambda sys, ctx, _: (sys.forward_kinematics_tips, (ctx["q"],)),
        ),
        BenchmarkCase(
            name="jacobian",
            description="Tip Jacobians",
            builder=lambda sys, ctx, _: (sys.jacobians_tips, (ctx["q"],)),
        ),
        BenchmarkCase(
            name="jacobian_and_derivative",
            description="Tip Jacobians and their derivatives",
            builder=lambda sys, ctx, _: (
                sys.jacobians_and_derivatives_tips,
                (ctx["q"], ctx["qd"]),
            ),
        ),
        BenchmarkCase(
            name="forward_dynamics",
            builder=lambda sys, ctx, _: (
                lambda t, y, args: sys.forward_dynamics(t, y, args),
                (ctx["t"], ctx["y"], (ctx["u"], ctx["tau_ext"])),
            ),
        ),
        BenchmarkCase(
            name="total_energy",
            builder=lambda sys, ctx, _: (sys.total_energy, (ctx["q"], ctx["qd"])),
        ),
        BenchmarkCase(
            name="resolve_upon_time",
            builder=lambda sys, ctx, runtime: (
                lambda q0, qd0, u, tau, t0, t1, dt, save_dt: sys.resolve_upon_time(
                    q0=q0,
                    qd0=qd0,
                    u=u,
                    tau_ext=tau,
                    t0=t0,
                    t1=t1,
                    dt=dt,
                    save_dt=save_dt,
                ),
                (
                    ctx["q"],
                    ctx["qd"],
                    ctx["u"],
                    ctx["tau_ext"],
                    jnp.array(0.0),
                    jnp.array(runtime.duration),
                    jnp.array(runtime.dt),
                    runtime.save_dt,
                ),
            ),
        ),
    )

    planar_cases = (
        BenchmarkCase(
            name="forward_kinematics",
            builder=lambda sys, ctx, _: (sys.forward_kinematics, (ctx["q"], ctx["s_tip"])),
        ),
        BenchmarkCase(
            name="inverse_kinematics",
            description="Recover strains from planar tip poses",
            builder=lambda sys, ctx, _: (sys.inverse_kinematics, (ctx["chi_tips"],)),
        ),
        BenchmarkCase(
            name="jacobian",
            builder=lambda sys, ctx, _: (sys.jacobian, (ctx["q"], ctx["s_tip"])),
        ),
        BenchmarkCase(
            name="jacobian_and_derivative",
            builder=lambda sys, ctx, _: (
                sys.jacobian_and_derivative,
                (ctx["q"], ctx["qd"], ctx["s_tip"]),
            ),
        ),
        BenchmarkCase(
            name="forward_dynamics",
            builder=lambda sys, ctx, _: (
                lambda t, y, args: sys.forward_dynamics(t, y, args),
                (ctx["t"], ctx["y"], (ctx["u"], ctx["tau_ext"])),
            ),
        ),
        BenchmarkCase(
            name="total_energy",
            builder=lambda sys, ctx, _: (sys.total_energy, (ctx["q"], ctx["qd"])),
        ),
        BenchmarkCase(
            name="resolve_upon_time",
            builder=lambda sys, ctx, runtime: (
                lambda q0, qd0, u, tau, t0, t1, dt, save_dt: sys.resolve_upon_time(
                    q0=q0,
                    qd0=qd0,
                    u=u,
                    tau_ext=tau,
                    t0=t0,
                    t1=t1,
                    dt=dt,
                    save_dt=save_dt,
                ),
                (
                    ctx["q"],
                    ctx["qd"],
                    ctx["u"],
                    ctx["tau_ext"],
                    jnp.array(0.0),
                    jnp.array(runtime.duration),
                    jnp.array(runtime.dt),
                    runtime.save_dt,
                ),
            ),
        ),
    )

    pcs_cases = (
        BenchmarkCase(
            name="forward_kinematics",
            builder=lambda sys, ctx, _: (sys.forward_kinematics, (ctx["q"], ctx["s_tip"])),
        ),
        BenchmarkCase(
            name="forward_kinematics_batched",
            builder=lambda sys, ctx, _: (
                sys.forward_kinematics_batched,
                (ctx["q"], ctx["s_points"]),
            ),
        ),
        BenchmarkCase(
            name="inverse_kinematics",
            description="Recover strains from spatial tip transforms",
            builder=lambda sys, ctx, _: (sys.inverse_kinematics, (ctx["g_tips"],)),
        ),
        BenchmarkCase(
            name="jacobian",
            builder=lambda sys, ctx, _: (sys.jacobian, (ctx["q"], ctx["s_tip"])),
        ),
        BenchmarkCase(
            name="jacobian_and_derivative",
            builder=lambda sys, ctx, _: (
                sys.jacobian_and_derivative,
                (ctx["q"], ctx["qd"], ctx["s_tip"]),
            ),
        ),
        BenchmarkCase(
            name="forward_dynamics",
            builder=lambda sys, ctx, _: (
                lambda t, y, args: sys.forward_dynamics(t, y, args),
                (ctx["t"], ctx["y"], (ctx["u"], ctx["tau_ext"])),
            ),
        ),
        BenchmarkCase(
            name="total_energy",
            builder=lambda sys, ctx, _: (sys.total_energy, (ctx["q"], ctx["qd"])),
        ),
        BenchmarkCase(
            name="resolve_upon_time",
            builder=lambda sys, ctx, runtime: (
                lambda q0, qd0, u, tau, t0, t1, dt, save_n: sys.resolve_upon_time(
                    q0=q0,
                    qd0=qd0,
                    u=u,
                    tau_ext=tau,
                    t0=t0,
                    t1=t1,
                    dt=dt,
                    save_dt=save_n,
                ),
                (
                    ctx["q"],
                    ctx["qd"],
                    ctx["u"],
                    ctx["tau_ext"],
                    jnp.array(0.0),
                    jnp.array(runtime.duration),
                    jnp.array(runtime.dt),
                    runtime.save_dt,
                ),
            ),
        ),
    )

    return {
        "pendulum": SystemBenchmark(
            factory=shared["pendulum"].factory,
            size_label=shared["pendulum"].size_label,
            build_context=shared["pendulum"].build_context,
            cases=pendulum_cases,
        ),
        "planar_pcs": SystemBenchmark(
            factory=shared["planar_pcs"].factory,
            size_label=shared["planar_pcs"].size_label,
            build_context=shared["planar_pcs"].build_context,
            cases=planar_cases,
        ),
        "pcs": SystemBenchmark(
            factory=shared["pcs"].factory,
            size_label=shared["pcs"].size_label,
            build_context=shared["pcs"].build_context,
            cases=pcs_cases,
        ),
    }


def _plot_results(results: Sequence[Mapping[str, Any]], output: Path | None, show: bool) -> None:
    if not results:
        print("No results to plot.")
        return

    systems = sorted({res["system"] for res in results})
    fig, axes = plt.subplots(len(systems), 2, figsize=(12, 4 * len(systems)), squeeze=False, sharex="col")

    for row, system in enumerate(systems):
        subset = [res for res in results if res["system"] == system]
        size_label = subset[0]["size_label"]
        functions = sorted({res["function"] for res in subset})
        ax_compile = axes[row, 0]
        ax_exec = axes[row, 1]
        for fn_name in functions:
            fn_results = sorted(
                (res for res in subset if res["function"] == fn_name),
                key=lambda item: item["size_value"],
            )
            sizes = [item["size_value"] for item in fn_results]
            compile_times = [item["jit_compile_time_s"] for item in fn_results]
            exec_times = [item["jit_execution_time_s"] for item in fn_results]

            ax_compile.plot(sizes, compile_times, marker="o", label=fn_name)
            ax_exec.plot(sizes, exec_times, marker="o", label=fn_name)

        ax_compile.set_title(f"{system} – compile")
        ax_exec.set_title(f"{system} – execution")
        ax_compile.set_ylabel("time [s]")
        ax_exec.set_ylabel("time [s]")
        ax_compile.set_xlabel(size_label)
        ax_exec.set_xlabel(size_label)
        ax_compile.grid(True, linestyle=":", linewidth=0.5)
        ax_exec.grid(True, linestyle=":", linewidth=0.5)
        ax_exec.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    fig.tight_layout()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        print(f"Saved plot to {output}")
    if show:
        plt.show()
    plt.close(fig)


def _write_json(results: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"Wrote benchmark data to {path}")


def _write_csv(results: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "system",
        "function",
        "size_label",
        "size_value",
        "jit_compile_time_s",
        "jit_execution_time_s",
        "duration",
        "dt",
        "save_dt",
        "execution_repeats",
    ]
    with path.open("w", encoding="utf-8") as fp:
        fp.write(",".join(headers) + "\n")
        for row in results:
            fp.write(",".join(str(row[key]) for key in headers) + "\n")
    print(f"Wrote benchmark data to {path}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    registry = _build_system_registry()
    parser = argparse.ArgumentParser(description="Benchmark Soromox system implementations")
    add_system_selection_args(parser, registry, default_segment_counts=[1, 2, 4, 8])
    add_integration_args(parser)
    parser.add_argument(
        "--execution-repeats",
        type=int,
        default=3,
        help="Number of post-compilation executions to average",
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
    default_plot = Path("benchmarks/system_methods_results.pdf")
    parser.add_argument(
        "--plot",
        type=Path,
        default=default_plot,
        help=(
            "Path to save a Matplotlib summary figure (default: %(default)s). "
            "Use --show-plot as well if you want an interactive window."
        ),
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the plot window after benchmarking",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    runtime = RuntimeConfig(
        duration=args.duration,
        dt=args.dt,
        save_dt=max(0.0, args.save_dt),
        execution_repeats=args.execution_repeats,
    )

    registry = _build_system_registry()
    results: List[Dict[str, Any]] = []

    for system_name in args.systems:
        system_cfg = registry[system_name]
        for size in args.segment_counts:
            print(f"\n=== Benchmarking {system_name} with {system_cfg.size_label}={size} ===")
            system = system_cfg.factory(size)
            ctx = system_cfg.build_context(system)
            for case in system_cfg.cases:
                fn, call_args = case.builder(system, ctx, runtime)
                print(f"  -> {case.name} ...", end=" ", flush=True)
                compile_time, exec_time = _measure_jitted_call(fn, call_args, runtime.execution_repeats)

                per_point_note = ""
                if case.name == "forward_kinematics_batched" and len(call_args) >= 2:
                    s_arg = call_args[1]
                    num_points = None
                    if hasattr(s_arg, "shape") and s_arg.shape:
                        num_points = int(s_arg.shape[0])
                    else:
                        try:
                            num_points = int(len(s_arg))
                        except TypeError:
                            num_points = None
                    if num_points and num_points > 0:
                        compile_time /= num_points
                        exec_time /= num_points
                        per_point_note = f" (per point over {num_points})"

                print(f"compile {compile_time:.4f}s | exec {exec_time:.4f}s{per_point_note}")
                results.append(
                    {
                        "system": system_name,
                        "function": case.name,
                        "size_label": system_cfg.size_label,
                        "size_value": size,
                        "jit_compile_time_s": compile_time,
                        "jit_execution_time_s": exec_time,
                        "duration": runtime.duration,
                        "dt": runtime.dt,
                        "save_dt": runtime.save_dt,
                        "execution_repeats": runtime.execution_repeats,
                    }
                )

    artifacts: List[str] = []

    if args.json:
        _write_json(results, args.json)
        artifacts.append(str(args.json.resolve()))
    if args.csv:
        _write_csv(results, args.csv)
        artifacts.append(str(args.csv.resolve()))
    if args.plot or args.show_plot:
        _plot_results(results, args.plot, args.show_plot)
        if args.plot:
            artifacts.append(str(args.plot.resolve()))

    if artifacts:
        print("\nGenerated benchmark artifacts:")
        for path in artifacts:
            print(f" - {path}")
    else:
        print("\nNo benchmark artifacts were written. Pass --json/--csv/--plot to export data.")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
