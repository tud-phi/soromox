#!/usr/bin/env python3
"""Generate the Section IVb parallel-rollout benchmark data.

The available systems are provided by `_benchmark_common.get_system_registry`,
including articulated soft robots, pendulums, PCS, planar PCS, and GVS models.

GPU execution is the default because Section IVb studies accelerator batch
scaling. Pass ``--device cpu`` to run the same workload on the CPU. A requested
GPU must be visible to JAX; otherwise the script emits an actionable warning
and exits without writing misleading CPU measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _configure_device_before_jax_import(argv: list[str]) -> str:
    """Apply an explicit CPU selection before JAX initializes its backends."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args, _ = parser.parse_known_args(argv)
    if args.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    return str(args.device)


_EARLY_DEVICE_CHOICE = (
    _configure_device_before_jax_import(sys.argv[1:])
    if __name__ == "__main__"
    else None
)

import jax
import jax.numpy as jnp
from diffrax import Bosh3, Dopri5, Euler, Heun, Tsit5

jax.config.update("jax_enable_x64", True)

from soromox.simulation import SemiImplicitEuler  # noqa: E402
from soromox.systems import DynamicalSystem, SystemState  # noqa: E402
from tools.benchmarks._benchmark_common import (  # noqa: E402
    add_backend_arg,
    add_gauss_point_args,
    add_integration_args,
    add_system_selection_args,
    benchmark_environment_metadata,
    block_until_ready,
    build_system_with_gauss_points,
    gauss_point_sweep_values,
    get_gvs_basis_order_system_config,
    get_system_registry,
    normalize_gauss_point_values,
    normalize_system_names,
    resolve_execution_backend,
    system_gauss_point_metadata,
)

Array = jax.Array

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DEFAULT_CSV_PATH = DATA_DIR / "benchmark_results.csv"

DEFAULT_SYSTEMS = ["articulated_soft_robot", "planar_pcs", "pcs", "gvs"]
DEFAULT_SEGMENT_COUNTS = [1, 2, 4, 8, 16, 32]
DEFAULT_GVS_BASIS_ORDERS = [0, 1, 2, 3, 4, 5]
DEFAULT_SOLVER = "tsit5"
SOLVER_NAMES = (
    "euler",
    "semi-implicit-euler",
    "heun",
    "bosh3",
    "tsit5",
    "dopri5",
)


def _make_solver(name: str, system: DynamicalSystem | None = None):
    """Construct one of the fixed-step solvers supported by this benchmark."""

    if name == "semi-implicit-euler":
        if system is None:
            raise ValueError("semi-implicit-euler requires a Soromox system.")
        return SemiImplicitEuler(system)

    factories = {
        "euler": Euler,
        "heun": Heun,
        "bosh3": Bosh3,
        "tsit5": Tsit5,
        "dopri5": Dopri5,
    }
    try:
        return factories[name]()
    except KeyError as exc:  # pragma: no cover - argparse validates CLI input
        raise ValueError(f"Unknown solver {name!r}.") from exc


@dataclass
class RuntimeConfig:
    duration: float
    solver_dt: float
    save_dt: float
    solver: str = DEFAULT_SOLVER
    t0: float = 0.0


@dataclass(frozen=True)
class BenchmarkCase:
    """One process-isolated point in the requested benchmark sweep."""

    system: str
    size_label: str
    size_value: int
    gauss_points: int | None
    batch_size: int
    seed: int


def _select_device(requested_device: str) -> jax.Device | None:
    """Select the requested JAX device without silently changing platforms.

    Args:
        requested_device: JAX platform name. The CLI accepts ``"gpu"`` and
            ``"cpu"``.

    Returns:
        The first visible device on the requested platform. ``None`` is
        returned when GPU execution was requested but JAX cannot initialize a
        compatible accelerator.

    Raises:
        RuntimeError: If CPU execution was requested but JAX cannot initialize
            its CPU backend.
    """

    try:
        devices = jax.devices(requested_device)
    except RuntimeError as error:
        if requested_device != "gpu":
            raise
        warnings.warn(
            "GPU benchmark requested, but JAX could not initialize a compatible "
            "GPU. Check the CUDA-capable device, NVIDIA driver, and CUDA-enabled "
            "jaxlib installation, or pass '--device cpu' to run an explicit CPU "
            "benchmark. No benchmark data was written. "
            f"JAX reported: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if devices:
        return devices[0]
    if requested_device != "gpu":
        raise RuntimeError("JAX reported no CPU devices.")
    warnings.warn(
        "GPU benchmark requested, but JAX reported no compatible GPU devices. "
        "Check the NVIDIA driver and CUDA-enabled jaxlib installation, or pass "
        "'--device cpu' to run an explicit CPU benchmark. No benchmark data was "
        "written.",
        RuntimeWarning,
        stacklevel=2,
    )
    return None


def _repeat_with_noise(
    vec: Array,
    batch_size: int,
    noise_scale: float,
    key: jax.random.PRNGKeyArray,
) -> tuple[Array, jax.random.PRNGKeyArray]:
    arr = jnp.asarray(vec)
    batched = jnp.repeat(arr[None, ...], batch_size, axis=0)
    if batch_size <= 1 or noise_scale <= 0.0:
        return batched, key
    key, subkey = jax.random.split(key)
    noise = noise_scale * jax.random.normal(
        subkey, shape=batched.shape, dtype=batched.dtype
    )
    return batched + noise, key


def _repeat(vec: Array, batch_size: int) -> Array:
    arr = jnp.asarray(vec)
    return jnp.repeat(arr[None, ...], batch_size, axis=0)


def _build_batched_solver(
    system: DynamicalSystem, runtime: RuntimeConfig
) -> Callable[..., tuple[Array, Array, Array]]:
    t1 = runtime.t0 + runtime.duration
    solver = _make_solver(runtime.solver, system)

    def single_env(
        q0: Array, qd0: Array, u: Array, tau_ext: Array
    ) -> tuple[Array, Array, Array]:
        initial_state = SystemState(t=runtime.t0, y=jnp.concatenate([q0, qd0]))
        trajectory = system.rollout_to(
            initial_state=initial_state,
            u=u,
            tau_ext=tau_ext,
            t1=t1,
            solver_dt=runtime.solver_dt,
            save_dt=runtime.save_dt,
            solver=solver,
        )
        qs, qds = jnp.split(trajectory.y, 2, axis=1)
        return trajectory.t[-1], qs[-1], qds[-1]

    vmapped = jax.jit(jax.vmap(single_env, in_axes=(0, 0, 0, 0)))
    return vmapped


def _measure_wall_time(
    fn: Callable[..., tuple[Array, Array, Array]],
    inputs: tuple[Array, Array, Array, Array],
    warmup_runs: int,
    repeats: int,
) -> tuple[float, tuple[Array, Array, Array]]:
    if repeats < 1:
        raise ValueError("timing repeats must be at least 1")

    for _ in range(max(0, warmup_runs)):
        warm = fn(*inputs)
        block_until_ready(warm)

    timings: list[float] = []
    last_output: tuple[Array, Array, Array] | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        output = fn(*inputs)
        block_until_ready(output)
        timings.append(time.perf_counter() - start)
        last_output = output

    assert last_output is not None
    avg_time = sum(timings) / len(timings)
    return avg_time, last_output


def _rollout_state_diagnostics(
    state_outputs: Sequence[Array],
) -> tuple[bool, float]:
    """Reduce rollout stability metrics on-device before one host transfer."""

    rollout_finite_device = jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(value)) for value in state_outputs])
    )
    max_abs_state_device = jnp.max(
        jnp.stack([jnp.max(jnp.abs(value)) for value in state_outputs])
    )
    rollout_finite, max_abs_state = jax.device_get(
        (rollout_finite_device, max_abs_state_device)
    )
    return bool(rollout_finite), float(max_abs_state)


def _write_csv(
    results: Sequence[Mapping[str, Any]], path: Path, *, announce: bool = True
) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "system",
        "gvs_scaling",
        "size_label",
        "size_value",
        "segment_count",
        "gauss_points",
        "integration_points",
        "dof",
        "batch_size",
        "duration_s",
        "solver",
        "solver_dt",
        "save_dt",
        "wall_time_s",
        "per_env_wall_time_s",
        "simulated_time_s",
        "total_simulated_time_s",
        "per_env_speed_ratio",
        "total_speed_ratio",
        "speed_ratio",
        "noise_scale",
        "warmup_runs",
        "timing_repeats",
        "rollout_finite",
        "max_abs_state",
        "backend",
        "resolved_backend",
        "backend_applies",
        "device",
        "device_id",
        "timestamp_utc",
        "git_revision",
        "git_dirty",
        "python_version",
        "soromox_version",
        "jax_version",
        "jaxlib_version",
        "warp_version",
        "x64_enabled",
        "device_kind",
        "platform_version",
        "device_count",
    ]
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    if announce:
        print(f"[+] Wrote CSV results to {path}")


def _write_failure_csv(failures: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write the complete case-failure report, including an empty report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "system",
        "gvs_scaling",
        "size_label",
        "size_value",
        "gauss_points",
        "batch_size",
        "device",
        "backend",
        "solver",
        "failure_kind",
        "error_type",
        "error_message",
        "worker_exit_code",
    ]
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(failures)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    """Write a small worker outcome file consumed by the parent process."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _size_value(row: Mapping[str, Any]) -> int:
    return int(row.get("size_value", row["segment_count"]))


def _plot_results(
    results: Sequence[Mapping[str, Any]],
    path: Path | None,
    show: bool,
    log_x: bool,
    log_y: bool,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not results:
        print("[!] No results available for plotting.")
        return

    systems = sorted({row["system"] for row in results})
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(
        2,
        len(systems),
        figsize=(6 * len(systems), 8),
        squeeze=False,
        sharex="col",
    )

    for col, system in enumerate(systems):
        ax_per_env = axes[0, col]
        ax_total = axes[1, col]
        subset = [row for row in results if row["system"] == system]
        size_label = subset[0]["size_label"]
        segment_values = sorted({_size_value(row) for row in subset})
        gauss_values = sorted(
            {
                row.get("gauss_points")
                for row in subset
                if row.get("gauss_points") not in (None, "")
            }
        )
        gauss_groups: Sequence[int | None] = gauss_values if gauss_values else [None]
        for seg in segment_values:
            for gauss_points in gauss_groups:
                seg_rows = sorted(
                    (
                        row
                        for row in subset
                        if _size_value(row) == seg
                        and (
                            gauss_points is None
                            or row.get("gauss_points") == gauss_points
                        )
                    ),
                    key=lambda r: r["batch_size"],
                )
                if not seg_rows:
                    continue
                x = [row["batch_size"] for row in seg_rows]
                y_per_env = [
                    row.get("per_env_speed_ratio", row.get("speed_ratio"))
                    for row in seg_rows
                ]
                y_total = [
                    row.get("total_speed_ratio", row.get("throughput_ratio"))
                    for row in seg_rows
                ]
                label = f"{size_label}={seg}"
                if len(gauss_values) > 1:
                    label = f"{label}, gauss={gauss_points}"
                ax_per_env.plot(x, y_per_env, marker="o", label=label)
                ax_total.plot(x, y_total, marker="o", label=label)

        ax_per_env.set_title(f"{system} – per-env speed")
        ax_total.set_title(f"{system} – total throughput")
        ax_total.set_xlabel("number of environments (parallel simulations)")
        for axis in (ax_per_env, ax_total):
            axis.grid(True, linestyle=":", linewidth=0.6)
            if log_x:
                axis.set_xscale("log")
            if log_y:
                axis.set_yscale("log")
        # Only show legends on top row to reduce clutter
        ax_per_env.legend()

    axes[0, 0].set_ylabel("simulated time / wall time (per env)")
    axes[1, 0].set_ylabel("total simulated time / wall time")
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)
        print(f"[+] Saved plot to {path}")
    if show:
        plt.show()
    plt.close(fig)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    registry = get_system_registry()
    parser = argparse.ArgumentParser(
        description="Measure Soromox simulation throughput vs. batched environments."
    )
    add_system_selection_args(
        parser,
        registry,
        default_segment_counts=DEFAULT_SEGMENT_COUNTS,
        default_systems=DEFAULT_SYSTEMS,
    )
    parser.add_argument(
        "--device",
        choices=("gpu", "cpu"),
        default="gpu",
        help=(
            "JAX platform on which to run the benchmark. GPU is the default; "
            "CPU execution must be requested explicitly."
        ),
    )
    add_backend_arg(parser)
    add_gauss_point_args(parser)
    add_integration_args(parser)
    parser.add_argument(
        "--solver",
        choices=SOLVER_NAMES,
        default=DEFAULT_SOLVER,
        help=(
            "Fixed-step numerical integrator used for every rollout "
            f"(default: {DEFAULT_SOLVER})."
        ),
    )
    parser.add_argument(
        "--gvs-scaling",
        "--gvs-variant",
        choices=("segments", "basis-order"),
        default="segments",
        help=(
            "GVS sweep to benchmark. 'segments' scales the number of zero-order "
            "GVS segments. 'basis-order' keeps one GVS segment and sweeps the "
            "Legendre polynomial strain-basis order from --gvs-basis-orders."
        ),
    )
    parser.add_argument(
        "--gvs-basis-orders",
        nargs="+",
        type=int,
        default=DEFAULT_GVS_BASIS_ORDERS,
        help=(
            "Strain-basis orders used when --gvs-scaling=basis-order "
            f"(default: {' '.join(str(v) for v in DEFAULT_GVS_BASIS_ORDERS)})"
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        help="Parallel environment counts to benchmark",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed executions averaged per configuration",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="How many runs to execute (and discard) before timing",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1e-3,
        help="Standard deviation of per-environment perturbations applied to q/qd",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed for perturbations",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Path to write the result table as CSV (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--failures-csv",
        type=Path,
        default=None,
        help=(
            "Path to write the failed-case report. By default, '_failures' is "
            "appended to the --csv filename."
        ),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="Optional path for a quick diagnostic plot",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the plot window on completion",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use a logarithmic scale for the number-of-environments axis",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a logarithmic scale for the speed ratio axis",
    )
    parser.add_argument(
        "--_worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_outcome-json",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    try:
        args.systems = normalize_system_names(args.systems, registry)
    except ValueError as exc:
        parser.error(str(exc))

    if args.duration <= 0.0:
        parser.error("--duration must be positive.")
    if args.solver_dt <= 0.0:
        parser.error("--solver-dt/--dt must be positive.")
    if args.save_dt < args.solver_dt:
        parser.error("--save-dt must be greater than or equal to --solver-dt/--dt.")
    if any(bs < 1 for bs in args.batch_sizes):
        parser.error("All --batch-sizes entries must be >= 1.")
    if any(seg < 1 for seg in args.segment_counts):
        parser.error("All --segment-counts entries must be >= 1.")
    if args.gauss_points is not None and any(value < 1 for value in args.gauss_points):
        parser.error("All --gauss-points entries must be >= 1.")
    if any(order < 0 for order in args.gvs_basis_orders):
        parser.error("All --gvs-basis-orders entries must be >= 0.")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1.")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be non-negative.")

    args.batch_sizes = sorted(dict.fromkeys(args.batch_sizes))
    args.segment_counts = sorted(dict.fromkeys(args.segment_counts))
    args.gvs_basis_orders = sorted(dict.fromkeys(args.gvs_basis_orders))
    args.gauss_points = normalize_gauss_point_values(args.gauss_points)
    if args.failures_csv is None:
        args.failures_csv = args.csv.with_name(f"{args.csv.stem}_failures.csv")
    if args._worker and args._outcome_json is None:
        parser.error("Internal benchmark workers require --_outcome-json.")
    return args


def _run_benchmarks(
    args: argparse.Namespace,
    device: jax.Device,
    *,
    write_artifacts: bool = True,
) -> list[dict[str, Any]]:
    """Run all requested benchmark cases on one explicitly selected device.

    Args:
        args: Validated command-line arguments.
        device: JAX device selected from the requested platform.

    Returns:
        The successful benchmark rows. Artifacts are written when
        ``write_artifacts`` is true.
    """

    runtime = RuntimeConfig(
        duration=args.duration,
        solver_dt=args.solver_dt,
        save_dt=args.save_dt,
        solver=args.solver,
    )
    registry = dict(get_system_registry())
    if args.gvs_scaling == "basis-order":
        registry["gvs"] = get_gvs_basis_order_system_config()
    key = jax.random.PRNGKey(args.seed)
    environment_metadata = benchmark_environment_metadata(device)

    results: list[dict[str, Any]] = []
    for system_name in args.systems:
        config = registry[system_name]
        size_values = (
            args.gvs_basis_orders
            if system_name == "gvs" and args.gvs_scaling == "basis-order"
            else args.segment_counts
        )
        gauss_values = gauss_point_sweep_values(config, args.gauss_points)
        if not gauss_values:
            print(
                f"\n[!] Skipping {system_name}: --gauss-points is not applicable "
                "or all requested values are below the system minimum."
            )
            continue
        print(f"\n=== {system_name} ({config.size_label} sweep) ===")
        for size in size_values:
            for requested_gauss_points in gauss_values:
                system = build_system_with_gauss_points(
                    config,
                    size,
                    requested_gauss_points,
                    backend=args.backend,
                )
                resolved_backend = resolve_execution_backend(
                    system, platform=device.platform
                )
                ctx = config.build_context(system)
                dof = int(ctx["q"].shape[0])
                gauss_points, integration_points = system_gauss_point_metadata(system)

                gauss_note = ""
                if requested_gauss_points is not None:
                    gauss_note = (
                        f", gauss_points={gauss_points}, "
                        f"integration_points={integration_points}"
                    )

                print(
                    f"  -> {config.size_label}={size}, dof={dof}{gauss_note}, "
                    f"backend={resolved_backend}"
                )
                for batch in args.batch_sizes:
                    q_batch, key = _repeat_with_noise(
                        ctx["q"], batch, args.noise_scale, key
                    )
                    qd_batch, key = _repeat_with_noise(
                        ctx["qd"], batch, args.noise_scale, key
                    )
                    u_batch = _repeat(ctx["u"], batch)
                    tau_batch = _repeat(ctx["tau_ext"], batch)

                    solver = _build_batched_solver(system, runtime)
                    wall_time, outputs = _measure_wall_time(
                        solver,
                        (q_batch, qd_batch, u_batch, tau_batch),
                        warmup_runs=args.warmup_runs,
                        repeats=args.repeats,
                    )
                    ts_last, *_ = outputs
                    rollout_finite, max_abs_state = _rollout_state_diagnostics(
                        outputs[1:]
                    )
                    sim_time = float(jnp.mean(ts_last) - runtime.t0)
                    total_sim_time = sim_time * batch
                    per_env_speed = (
                        sim_time / wall_time if wall_time > 0 else float("inf")
                    )
                    total_speed = (
                        total_sim_time / wall_time if wall_time > 0 else float("inf")
                    )
                    per_env_wall = wall_time / batch

                    stability_note = "" if rollout_finite else " | NON-FINITE STATE"
                    print(
                        f"     batch={batch:>4d} | wall={wall_time:.4f}s | "
                        f"per-env sim/wall={per_env_speed:.2f}x | "
                        f"total sim/wall={total_speed:.2f}x | "
                        f"per-env wall={per_env_wall:.5f}s{stability_note}"
                    )

                    results.append(
                        {
                            "system": system_name,
                            "gvs_scaling": (
                                args.gvs_scaling if system_name == "gvs" else ""
                            ),
                            "size_label": config.size_label,
                            "size_value": size,
                            "segment_count": int(getattr(system, "num_segments", size)),
                            "gauss_points": gauss_points,
                            "integration_points": integration_points,
                            "dof": dof,
                            "batch_size": batch,
                            "duration_s": args.duration,
                            "solver": args.solver,
                            "solver_dt": args.solver_dt,
                            "save_dt": args.save_dt,
                            "wall_time_s": wall_time,
                            "per_env_wall_time_s": per_env_wall,
                            "simulated_time_s": sim_time,
                            "total_simulated_time_s": total_sim_time,
                            "per_env_speed_ratio": per_env_speed,
                            "total_speed_ratio": total_speed,
                            "speed_ratio": per_env_speed,
                            "noise_scale": args.noise_scale,
                            "warmup_runs": args.warmup_runs,
                            "timing_repeats": args.repeats,
                            "rollout_finite": rollout_finite,
                            "max_abs_state": max_abs_state,
                            "backend": args.backend,
                            "resolved_backend": resolved_backend,
                            "backend_applies": config.supports_backend,
                            "device": device.platform,
                            **environment_metadata,
                        }
                    )

    if write_artifacts:
        _write_csv(results, args.csv)
        if args.plot or args.show_plot:
            _plot_results(results, args.plot, args.show_plot, args.log_x, args.log_y)

    return results


def _build_benchmark_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    """Expand CLI sweep arguments into independently executable cases."""

    registry = dict(get_system_registry())
    if args.gvs_scaling == "basis-order":
        registry["gvs"] = get_gvs_basis_order_system_config()

    cases: list[BenchmarkCase] = []
    for system_name in args.systems:
        config = registry[system_name]
        size_values = (
            args.gvs_basis_orders
            if system_name == "gvs" and args.gvs_scaling == "basis-order"
            else args.segment_counts
        )
        gauss_values = gauss_point_sweep_values(config, args.gauss_points)
        if not gauss_values:
            print(
                f"\n[!] Skipping {system_name}: --gauss-points is not applicable "
                "or all requested values are below the system minimum."
            )
            continue

        for size in size_values:
            for gauss_points in gauss_values:
                for batch_size in args.batch_sizes:
                    # Independent deterministic streams avoid coupling a case's
                    # inputs to whether any preceding case succeeded.
                    case_seed = (int(args.seed) + len(cases)) % (2**32)
                    cases.append(
                        BenchmarkCase(
                            system=system_name,
                            size_label=config.size_label,
                            size_value=int(size),
                            gauss_points=gauss_points,
                            batch_size=int(batch_size),
                            seed=case_seed,
                        )
                    )
    return cases


def _worker_command(
    args: argparse.Namespace, case: BenchmarkCase, outcome_path: Path
) -> list[str]:
    """Build the fresh-interpreter command for one benchmark case."""

    gvs_basis_case = case.system == "gvs" and args.gvs_scaling == "basis-order"
    segment_count = 1 if gvs_basis_case else case.size_value
    basis_order = case.size_value if gvs_basis_case else args.gvs_basis_orders[0]
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--device",
        args.device,
        "--backend",
        args.backend,
        "--systems",
        case.system,
        "--segment-counts",
        str(segment_count),
        "--gvs-scaling",
        args.gvs_scaling,
        "--gvs-basis-orders",
        str(basis_order),
        "--batch-sizes",
        str(case.batch_size),
        "--duration",
        repr(args.duration),
        "--solver",
        args.solver,
        "--solver-dt",
        repr(args.solver_dt),
        "--save-dt",
        repr(args.save_dt),
        "--repeats",
        str(args.repeats),
        "--warmup-runs",
        str(args.warmup_runs),
        "--noise-scale",
        repr(args.noise_scale),
        "--seed",
        str(case.seed),
        "--_worker",
        "--_outcome-json",
        str(outcome_path),
    ]
    if case.gauss_points is not None:
        command.extend(["--gauss-points", str(case.gauss_points)])
    return command


def _is_out_of_memory_failure(error_type: str, error_message: str) -> bool:
    """Recognize common JAX, XLA, CUDA, ROCm, and Python OOM failures."""

    if error_type == "MemoryError":
        return True
    error_text = f"{error_type}: {error_message}".lower()
    markers = (
        "out of memory",
        "resource_exhausted",
        "cuda_error_out_of_memory",
        "cudaerroroutofmemory",
        "hiperroroutofmemory",
        "failed to allocate",
    )
    return any(marker in error_text for marker in markers)


def _failure_row(
    args: argparse.Namespace,
    case: BenchmarkCase,
    outcome: Mapping[str, Any],
    worker_exit_code: int,
) -> dict[str, Any]:
    """Combine a worker's error details with its benchmark coordinates."""

    error_type = str(outcome.get("error_type", "WorkerProcessError"))
    error_message = str(
        outcome.get(
            "error_message",
            "Worker exited without producing a valid outcome.",
        )
    )
    if error_type == "DeviceUnavailableError":
        failure_kind = "device_unavailable"
    elif _is_out_of_memory_failure(error_type, error_message):
        failure_kind = "out_of_memory"
    else:
        failure_kind = "worker_error"
    return {
        "system": case.system,
        "gvs_scaling": args.gvs_scaling if case.system == "gvs" else "",
        "size_label": case.size_label,
        "size_value": case.size_value,
        "gauss_points": case.gauss_points,
        "batch_size": case.batch_size,
        "device": args.device,
        "backend": args.backend,
        "solver": args.solver,
        "failure_kind": failure_kind,
        "error_type": error_type,
        "error_message": error_message,
        "worker_exit_code": worker_exit_code,
    }


def _print_failure_report(failures: Sequence[Mapping[str, Any]]) -> None:
    """Print a compact end-of-run summary of every failed case."""

    if not failures:
        print("\n[+] All benchmark cases completed; no failed runs.")
        return

    print(f"\n=== Failed benchmark cases ({len(failures)}) ===")
    for failure in failures:
        gauss_note = ""
        if failure.get("gauss_points") is not None:
            gauss_note = f", gauss_points={failure['gauss_points']}"
        message = " ".join(str(failure["error_message"]).split())
        print(
            f"  [{str(failure['failure_kind']).upper()}] "
            f"{failure['system']} {failure['size_label']}="
            f"{failure['size_value']}{gauss_note}, "
            f"batch={failure['batch_size']}: {failure['error_type']}: {message}"
        )


def _run_benchmark_worker(args: argparse.Namespace) -> int:
    """Execute one case and serialize its result or caught exception."""

    assert args._outcome_json is not None
    try:
        device = _select_device(args.device)
        if device is None:
            _write_json(
                {
                    "status": "failed",
                    "error_type": "DeviceUnavailableError",
                    "error_message": f"No compatible {args.device} device is visible.",
                },
                args._outcome_json,
            )
            return 2
        if jax.default_backend() != args.device:
            raise RuntimeError(
                f"Requested {args.device!r}, but JAX selected "
                f"{jax.default_backend()!r}."
            )
        with jax.default_device(device):
            results = _run_benchmarks(args, device, write_artifacts=False)
        if len(results) != 1:
            raise RuntimeError(
                f"An isolated worker produced {len(results)} rows instead of one."
            )
        _write_json({"status": "success", "results": results}, args._outcome_json)
        return 0
    except Exception as error:
        traceback.print_exc()
        _write_json(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            },
            args._outcome_json,
        )
        return 1


def _run_isolated_benchmarks(args: argparse.Namespace) -> int:
    """Run every case in a disposable process and checkpoint successes."""

    cases = _build_benchmark_cases(args)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    device_unavailable = False

    print(
        f"\n=== Running {len(cases)} process-isolated benchmark cases ===",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="soromox-rollout-benchmark-") as temp_dir:
        temp_path = Path(temp_dir)
        for case_index, case in enumerate(cases, start=1):
            gauss_note = (
                ""
                if case.gauss_points is None
                else f", gauss_points={case.gauss_points}"
            )
            print(
                f"\n[{case_index}/{len(cases)}] {case.system} "
                f"{case.size_label}={case.size_value}{gauss_note}, "
                f"batch={case.batch_size}",
                flush=True,
            )
            outcome_path = temp_path / f"case-{case_index}.json"
            try:
                completed = subprocess.run(
                    _worker_command(args, case, outcome_path),
                    check=False,
                )
            except OSError as error:
                outcome: Mapping[str, Any] = {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
                worker_exit_code = 1
            else:
                worker_exit_code = completed.returncode
                try:
                    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    outcome = {
                        "error_type": "WorkerProcessError",
                        "error_message": (
                            f"Worker exited with code {worker_exit_code} and did not "
                            f"produce a readable outcome: {error}"
                        ),
                    }

            worker_results = outcome.get("results")
            if (
                worker_exit_code == 0
                and outcome.get("status") == "success"
                and isinstance(worker_results, list)
                and len(worker_results) == 1
                and isinstance(worker_results[0], dict)
            ):
                results.append(worker_results[0])
                # Checkpoint after every success so a later OOM cannot discard it.
                _write_csv(results, args.csv, announce=False)
                continue

            failure = _failure_row(args, case, outcome, worker_exit_code)
            failures.append(failure)
            _write_failure_csv(failures, args.failures_csv)
            if failure["failure_kind"] == "device_unavailable":
                device_unavailable = True
                break

    _write_failure_csv(failures, args.failures_csv)
    if results:
        print(f"\n[+] Wrote {len(results)} successful rows to {args.csv}")
    else:
        print("\n[!] No benchmark cases completed successfully.")
    print(f"[+] Wrote failed-case report to {args.failures_csv}")
    _print_failure_report(failures)

    if args.plot or args.show_plot:
        _plot_results(results, args.plot, args.show_plot, args.log_x, args.log_y)

    if device_unavailable:
        return 2
    if any(row["failure_kind"] == "worker_error" for row in failures):
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, validate the requested platform, and run the benchmark."""

    args = parse_args(sys.argv[1:] if argv is None else argv)
    if _EARLY_DEVICE_CHOICE is not None and args.device != _EARLY_DEVICE_CHOICE:
        raise RuntimeError(
            "The --device choice changed after JAX was imported. Pass it on the "
            "process command line so CPU selection can occur before initialization."
        )
    if args._worker:
        return _run_benchmark_worker(args)
    return _run_isolated_benchmarks(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
