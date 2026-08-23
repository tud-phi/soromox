#!/usr/bin/env python3

"""Benchmark the core dynamics routines for Soromox soft robot systems.

This script measures both first-call JIT compilation cost and steady-state
execution time after compilation for selected systems and core methods. It supports
benchmarking articulated, Pendulum, PlanarPCS, PCS, and GVS implementations over a sweep
of link or segment counts, and can optionally visualise the results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

from soromox.systems import SystemState
from tools.benchmarks._benchmark_common import (
    add_gauss_point_args,
    add_integration_args,
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


@dataclass
class RuntimeConfig:
    """Holds runtime controls shared across benchmark cases."""

    duration: float
    solver_dt: float
    save_dt: float
    execution_repeats: int


@dataclass
class BenchmarkCase:
    """A single benchmarked callable for a system."""

    name: str
    builder: Callable[
        [Any, Mapping[str, Array], RuntimeConfig],
        tuple[Callable[..., Tree], tuple[Any, ...]],
    ]
    description: str = ""


@dataclass
class SystemBenchmark:
    """Groups benchmark cases and helpers for one system type."""

    factory: Callable[[int], Any]
    size_label: str
    build_context: Callable[[Any], MutableMapping[str, Array]]
    cases: Sequence[BenchmarkCase]
    gauss_factory: Callable[[int, int], Any] | None = None
    default_gauss_points: int | None = None
    min_gauss_points: int = 1

    @property
    def supports_gauss_points(self) -> bool:
        return self.gauss_factory is not None


def _dense_forward_dynamics(
    system: Any, t: Array, y: Array, actuation_args: tuple | None
) -> Array:
    """Compute forward dynamics by explicitly assembling dense matrix terms."""
    del t
    q, qd = jnp.split(y, 2)
    if actuation_args is None:
        u, tau_ext = None, None
    elif len(actuation_args) == 1:
        u, tau_ext = actuation_args[0], None
    elif len(actuation_args) == 2:
        u, tau_ext = actuation_args
    else:
        raise ValueError("actuation_args must be None, (u,), or (u, tau_ext).")

    if u is None:
        u = jnp.zeros((system.num_actuators,), dtype=y.dtype)
    if tau_ext is None:
        tau_ext = jnp.zeros((system.num_dofs,), dtype=y.dtype)

    M = system.inertia_matrix(q)
    C = system.coriolis_matrix(q, qd)
    G = system.gravitational_force(q)
    D = system.damping_matrix(q)
    rhs = (
        system.actuation_force(q, u)
        + tau_ext
        - C @ qd
        - G
        - system.elastic_force(q)
        - D @ qd
    )
    qdd = jnp.linalg.solve(M, rhs)
    return jnp.concatenate([qd, qdd])


def _measure_jitted_call(
    fn: Callable[..., Tree],
    args: tuple[Any, ...],
    repeats: int,
) -> tuple[float, float]:
    """Measure compile (first-call) and averaged execution time for a callable.

    The first invocation is treated as "compile + execute" time. Subsequent invocations
    (``repeats`` of them) are averaged to estimate the steady-state execution cost.
    """

    if repeats < 1:
        raise ValueError("execution repeats must be at least 1")

    # JIT a closure around the benchmark callable instead of jitting bound
    # methods directly. Bound Equinox methods carry ``self`` as part of the
    # callable object, and JAX may try to hash the module if the bound method is
    # passed straight to ``jax.jit``.
    static_argnums = tuple(
        index
        for index, arg in enumerate(args)
        if arg is None or isinstance(arg, (bool, int, float, str))
    )
    jitted_fn = jax.jit(
        lambda *call_args: fn(*call_args), static_argnums=static_argnums
    )

    start = time.perf_counter()
    first = jitted_fn(*args)
    block_until_ready(first)
    first_time = time.perf_counter() - start

    exec_times: list[float] = []
    for _ in range(repeats):
        loop_start = time.perf_counter()
        out = jitted_fn(*args)
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
            name="jacobian_and_time_derivative",
            description="Tip Jacobians and their time derivatives",
            builder=lambda sys, ctx, _: (
                sys.jacobian_and_time_derivatives_tips,
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
            name="rollout_to",
            builder=lambda sys, ctx, runtime: (
                lambda q0, qd0, u, tau, t0, t1, solver_dt, save_dt: sys.rollout_to(
                    initial_state=SystemState(t=t0, y=jnp.concatenate([q0, qd0])),
                    u=u,
                    tau_ext=tau,
                    t1=t1,
                    solver_dt=solver_dt,
                    save_dt=save_dt,
                ),
                (
                    ctx["q"],
                    ctx["qd"],
                    ctx["u"],
                    ctx["tau_ext"],
                    0.0,
                    runtime.duration,
                    runtime.solver_dt,
                    runtime.save_dt,
                ),
            ),
        ),
    )

    articulated_cases = (
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
            name="jacobian_and_time_derivative",
            description="Tip Jacobians and their time derivatives",
            builder=lambda sys, ctx, _: (
                sys.jacobian_and_time_derivatives_tips,
                (ctx["q"], ctx["qd"]),
            ),
        ),
        BenchmarkCase(
            name="forward_dynamics",
            description="ABA-backed forward dynamics",
            builder=lambda sys, ctx, _: (
                lambda t, y, args: sys.forward_dynamics(t, y, args),
                (ctx["t"], ctx["y"], (ctx["u"], ctx["tau_ext"])),
            ),
        ),
        BenchmarkCase(
            name="forward_dynamics_dense",
            description="Dense Jacobian-energy matrix solve",
            builder=lambda sys, ctx, _: (
                lambda t, y, args: _dense_forward_dynamics(sys, t, y, args),
                (ctx["t"], ctx["y"], (ctx["u"], ctx["tau_ext"])),
            ),
        ),
        BenchmarkCase(
            name="total_energy",
            builder=lambda sys, ctx, _: (sys.total_energy, (ctx["q"], ctx["qd"])),
        ),
        BenchmarkCase(
            name="rollout_to",
            builder=lambda sys, ctx, runtime: (
                lambda q0, qd0, u, tau, t0, t1, solver_dt, save_dt: sys.rollout_to(
                    initial_state=SystemState(t=t0, y=jnp.concatenate([q0, qd0])),
                    u=u,
                    tau_ext=tau,
                    t1=t1,
                    solver_dt=solver_dt,
                    save_dt=save_dt,
                ),
                (
                    ctx["q"],
                    ctx["qd"],
                    ctx["u"],
                    ctx["tau_ext"],
                    0.0,
                    runtime.duration,
                    runtime.solver_dt,
                    runtime.save_dt,
                ),
            ),
        ),
    )

    planar_cases = (
        BenchmarkCase(
            name="forward_kinematics",
            builder=lambda sys, ctx, _: (
                sys.forward_kinematics,
                (ctx["q"], ctx["s_tip"]),
            ),
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
            name="jacobian_and_time_derivative",
            builder=lambda sys, ctx, _: (
                sys.jacobian_and_time_derivative,
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
            name="rollout_to",
            builder=lambda sys, ctx, runtime: (
                lambda q0, qd0, u, tau, t0, t1, solver_dt, save_dt: sys.rollout_to(
                    initial_state=SystemState(t=t0, y=jnp.concatenate([q0, qd0])),
                    u=u,
                    tau_ext=tau,
                    t1=t1,
                    solver_dt=solver_dt,
                    save_dt=save_dt,
                ),
                (
                    ctx["q"],
                    ctx["qd"],
                    ctx["u"],
                    ctx["tau_ext"],
                    0.0,
                    runtime.duration,
                    runtime.solver_dt,
                    runtime.save_dt,
                ),
            ),
        ),
    )

    pcs_cases = (
        BenchmarkCase(
            name="forward_kinematics",
            builder=lambda sys, ctx, _: (
                sys.forward_kinematics,
                (ctx["q"], ctx["s_tip"]),
            ),
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
            name="jacobian_and_time_derivative",
            builder=lambda sys, ctx, _: (
                sys.jacobian_and_time_derivative,
                (ctx["q"], ctx["qd"], ctx["s_tip"]),
            ),
        ),
        BenchmarkCase(
            name="dynamics_terms",
            description="Fused M, C @ qd, and G path",
            builder=lambda sys, ctx, _: (
                sys.dynamics_terms,
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
            name="rollout_to",
            builder=lambda sys, ctx, runtime: (
                lambda q0, qd0, u, tau, t0, t1, solver_dt, save_n: sys.rollout_to(
                    initial_state=SystemState(t=t0, y=jnp.concatenate([q0, qd0])),
                    u=u,
                    tau_ext=tau,
                    t1=t1,
                    solver_dt=solver_dt,
                    save_dt=save_n,
                ),
                (
                    ctx["q"],
                    ctx["qd"],
                    ctx["u"],
                    ctx["tau_ext"],
                    0.0,
                    runtime.duration,
                    runtime.solver_dt,
                    runtime.save_dt,
                ),
            ),
        ),
    )

    gvs_cases = (
        BenchmarkCase(
            name="forward_kinematics",
            builder=lambda sys, ctx, _: (
                sys.forward_kinematics,
                (ctx["q"], ctx["s_tip"]),
            ),
        ),
        BenchmarkCase(
            name="jacobian_bodyframe",
            builder=lambda sys, ctx, _: (
                sys.jacobian_bodyframe,
                (ctx["q"], ctx["s_tip"]),
            ),
        ),
        BenchmarkCase(
            name="jacobian_inertialframe",
            builder=lambda sys, ctx, _: (
                sys.jacobian_inertialframe,
                (ctx["q"], ctx["s_tip"]),
            ),
        ),
        BenchmarkCase(
            name="jacobian_and_time_derivative_bodyframe",
            builder=lambda sys, ctx, _: (
                sys.jacobian_and_time_derivative_bodyframe,
                (ctx["q"], ctx["qd"], ctx["s_tip"]),
            ),
        ),
        BenchmarkCase(
            name="jacobian_and_time_derivative_inertialframe",
            builder=lambda sys, ctx, _: (
                sys.jacobian_and_time_derivative_inertialframe,
                (ctx["q"], ctx["qd"], ctx["s_tip"]),
            ),
        ),
        BenchmarkCase(
            name="inertia_matrix",
            builder=lambda sys, ctx, _: (sys.inertia_matrix, (ctx["q"],)),
        ),
        BenchmarkCase(
            name="coriolis_matrix",
            builder=lambda sys, ctx, _: (sys.coriolis_matrix, (ctx["q"], ctx["qd"])),
        ),
        BenchmarkCase(
            name="gravitational_force",
            builder=lambda sys, ctx, _: (sys.gravitational_force, (ctx["q"],)),
        ),
        BenchmarkCase(
            name="dynamics_terms",
            description="Fused M, C @ qd, and G path",
            builder=lambda sys, ctx, _: (
                sys.dynamics_terms,
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
            name="rollout_to",
            builder=lambda sys, ctx, runtime: (
                lambda q0, qd0, u, tau, t0, t1, solver_dt, save_dt: sys.rollout_to(
                    initial_state=SystemState(t=t0, y=jnp.concatenate([q0, qd0])),
                    u=u,
                    tau_ext=tau,
                    t1=t1,
                    solver_dt=solver_dt,
                    save_dt=save_dt,
                ),
                (
                    ctx["q"],
                    ctx["qd"],
                    ctx["u"],
                    ctx["tau_ext"],
                    0.0,
                    runtime.duration,
                    runtime.solver_dt,
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
        "articulated_soft_robot": SystemBenchmark(
            factory=shared["articulated_soft_robot"].factory,
            size_label=shared["articulated_soft_robot"].size_label,
            build_context=shared["articulated_soft_robot"].build_context,
            cases=articulated_cases,
        ),
        "planar_pcs": SystemBenchmark(
            factory=shared["planar_pcs"].factory,
            size_label=shared["planar_pcs"].size_label,
            build_context=shared["planar_pcs"].build_context,
            cases=planar_cases,
            gauss_factory=shared["planar_pcs"].gauss_factory,
            default_gauss_points=shared["planar_pcs"].default_gauss_points,
            min_gauss_points=shared["planar_pcs"].min_gauss_points,
        ),
        "pcs": SystemBenchmark(
            factory=shared["pcs"].factory,
            size_label=shared["pcs"].size_label,
            build_context=shared["pcs"].build_context,
            cases=pcs_cases,
            gauss_factory=shared["pcs"].gauss_factory,
            default_gauss_points=shared["pcs"].default_gauss_points,
            min_gauss_points=shared["pcs"].min_gauss_points,
        ),
        "gvs": SystemBenchmark(
            factory=shared["gvs"].factory,
            size_label=shared["gvs"].size_label,
            build_context=shared["gvs"].build_context,
            cases=gvs_cases,
            gauss_factory=shared["gvs"].gauss_factory,
            default_gauss_points=shared["gvs"].default_gauss_points,
            min_gauss_points=shared["gvs"].min_gauss_points,
        ),
    }


def _plot_results(
    results: Sequence[Mapping[str, Any]], output: Path | None, show: bool
) -> None:
    if not results:
        print("No results to plot.")
        return

    systems = sorted({res["system"] for res in results})
    fig, axes = plt.subplots(
        len(systems), 2, figsize=(12, 4 * len(systems)), squeeze=False, sharex="col"
    )

    for row, system in enumerate(systems):
        subset = [res for res in results if res["system"] == system]
        size_label = subset[0]["size_label"]
        functions = sorted({res["function"] for res in subset})
        ax_compile = axes[row, 0]
        ax_exec = axes[row, 1]
        gauss_values = {
            res.get("gauss_points")
            for res in subset
            if res.get("gauss_points") not in (None, "")
        }
        plot_gauss_axis = len(gauss_values) > 1

        if plot_gauss_axis:
            sizes = sorted({res["size_value"] for res in subset})
            for fn_name in functions:
                for size in sizes:
                    fn_results = sorted(
                        (
                            res
                            for res in subset
                            if res["function"] == fn_name
                            and res["size_value"] == size
                            and res.get("gauss_points") not in (None, "")
                        ),
                        key=lambda item: item["gauss_points"],
                    )
                    if not fn_results:
                        continue
                    x_values = [item["gauss_points"] for item in fn_results]
                    compile_times = [item["jit_compile_time_s"] for item in fn_results]
                    exec_times = [item["jit_execution_time_s"] for item in fn_results]
                    label = (
                        f"{fn_name}, {size_label}={size}" if len(sizes) > 1 else fn_name
                    )
                    ax_compile.plot(x_values, compile_times, marker="o", label=label)
                    ax_exec.plot(x_values, exec_times, marker="o", label=label)
        else:
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
        x_label = "Gauss-Legendre points" if plot_gauss_axis else size_label
        ax_compile.set_xlabel(x_label)
        ax_exec.set_xlabel(x_label)
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
        "gauss_points",
        "integration_points",
        "jit_compile_time_s",
        "jit_execution_time_s",
        "duration",
        "solver_dt",
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
    parser = argparse.ArgumentParser(
        description="Benchmark Soromox system implementations"
    )
    add_system_selection_args(parser, registry, default_segment_counts=[1, 2, 4, 8])
    add_gauss_point_args(parser)
    add_integration_args(parser)
    # Collect all unique method names across all systems
    all_methods = sorted({case.name for cfg in registry.values() for case in cfg.cases})
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        choices=all_methods,
        metavar="METHOD",
        help=(
            f"Methods to benchmark (default: all). Available: {', '.join(all_methods)}"
        ),
    )
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
    args = parser.parse_args(argv)
    try:
        args.systems = normalize_system_names(args.systems, registry)
    except ValueError as exc:
        parser.error(str(exc))
    if args.gauss_points is not None:
        if any(value < 1 for value in args.gauss_points):
            parser.error("All --gauss-points entries must be >= 1.")
        args.gauss_points = normalize_gauss_point_values(args.gauss_points)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    runtime = RuntimeConfig(
        duration=args.duration,
        solver_dt=args.solver_dt,
        save_dt=max(0.0, args.save_dt),
        execution_repeats=args.execution_repeats,
    )

    registry = _build_system_registry()
    results: list[dict[str, Any]] = []

    for system_name in args.systems:
        system_cfg = registry[system_name]
        gauss_values = gauss_point_sweep_values(system_cfg, args.gauss_points)
        if not gauss_values:
            print(
                f"\n[!] Skipping {system_name}: --gauss-points is not applicable "
                "or all requested values are below the system minimum."
            )
            continue
        # Filter cases if --methods is specified
        cases_to_run = (
            [c for c in system_cfg.cases if c.name in args.methods]
            if args.methods
            else system_cfg.cases
        )
        for size in args.segment_counts:
            for requested_gauss_points in gauss_values:
                system = build_system_with_gauss_points(
                    system_cfg, size, requested_gauss_points
                )
                gauss_points, integration_points = system_gauss_point_metadata(system)
                gauss_note = ""
                if requested_gauss_points is not None:
                    gauss_note = (
                        f", gauss_points={gauss_points}, "
                        f"integration_points={integration_points}"
                    )

                print(
                    f"\n=== Benchmarking {system_name} with "
                    f"{system_cfg.size_label}={size}{gauss_note} ==="
                )
                ctx = system_cfg.build_context(system)
                for case in cases_to_run:
                    fn, call_args = case.builder(system, ctx, runtime)
                    print(f"  -> {case.name} ...", end=" ", flush=True)
                    compile_time, exec_time = _measure_jitted_call(
                        fn, call_args, runtime.execution_repeats
                    )

                    per_point_note = ""
                    if (
                        case.name == "forward_kinematics_batched"
                        and len(call_args) >= 2
                    ):
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

                    print(
                        f"compile {compile_time:.4f}s | exec {exec_time:.4f}s"
                        f"{per_point_note}"
                    )
                    results.append(
                        {
                            "system": system_name,
                            "function": case.name,
                            "size_label": system_cfg.size_label,
                            "size_value": size,
                            "gauss_points": gauss_points,
                            "integration_points": integration_points,
                            "jit_compile_time_s": compile_time,
                            "jit_execution_time_s": exec_time,
                            "duration": runtime.duration,
                            "solver_dt": runtime.solver_dt,
                            "save_dt": runtime.save_dt,
                            "execution_repeats": runtime.execution_repeats,
                        }
                    )

    artifacts: list[str] = []

    if args.json:
        _write_json(results, args.json)
        artifacts.append(str(args.json.resolve()))
    if args.csv:
        _write_csv(results, args.csv)
        artifacts.append(str(args.csv.resolve()))
    if args.plot or args.show_plot:
        _plot_results(results, args.plot, args.show_plot)
        if args.plot and results:
            artifacts.append(str(args.plot.resolve()))

    if artifacts:
        print("\nGenerated benchmark artifacts:")
        for path in artifacts:
            print(f" - {path}")
    elif not results:
        print(
            "\nNo benchmark artifacts were written because no benchmark rows were produced."
        )
    else:
        print(
            "\nNo benchmark artifacts were written. Pass --json/--csv/--plot to export data."
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
