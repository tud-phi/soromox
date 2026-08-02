#!/usr/bin/env python3
"""Generate the Section IVb batched-rollout benchmark data.

The available systems are provided by `_benchmark_common.get_system_registry`,
including articulated soft robots, pendulums, PCS, planar PCS, and GVS models.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import jax
import jax.numpy as jnp
import numpy as onp
import seaborn as sns

jax.config.update("jax_enable_x64", True)

from soromox.systems import DynamicalSystem, SystemState  # noqa: E402
from tools.benchmarks._benchmark_common import (  # noqa: E402
    add_gauss_point_args,
    add_integration_args,
    add_system_selection_args,
    block_until_ready,
    build_system_with_gauss_points,
    gauss_point_sweep_values,
    get_gvs_basis_order_system_config,
    get_system_registry,
    normalize_gauss_point_values,
    normalize_system_names,
    system_gauss_point_metadata,
)

Array = jax.Array

DEFAULT_SYSTEMS = ["articulated_soft_robot", "planar_pcs", "pcs", "gvs"]
DEFAULT_SEGMENT_COUNTS = [1, 2, 4, 8, 16, 32]
DEFAULT_GVS_BASIS_ORDERS = [0, 1, 2, 3, 4, 5]


@dataclass
class RuntimeConfig:
    duration: float
    solver_dt: float
    save_dt: float
    t0: float = 0.0


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


def _write_csv(results: Sequence[Mapping[str, Any]], path: Path) -> None:
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
        "device",
        "device_id",
    ]
    with path.open("w", encoding="utf-8") as fp:
        fp.write(",".join(headers) + "\n")
        for row in results:
            fp.write(",".join(str(row.get(col, "")) for col in headers) + "\n")
    print(f"[+] Wrote CSV results to {path}")


def _write_npz(results: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, list[Any]] = {}
    for row in results:
        for key, value in row.items():
            columns.setdefault(key, []).append(value)
    payload = {key: onp.array(vals) for key, vals in columns.items()}
    onp.savez(path, **payload)
    print(f"[+] Wrote NPZ results to {path}")


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

    if not results:
        print("[!] No results available for plotting.")
        return

    systems = sorted({row["system"] for row in results})
    if sns is not None:
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
    add_gauss_point_args(parser)
    add_integration_args(parser)
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
        default=[1, 2, 4, 8, 16, 32, 64],
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
        help="Standard deviation of per-environment perturbations applied to q/qdot",
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
        help="Optional path to write the result table as CSV",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        help="Optional path to store the result table as a NumPy NPZ archive",
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
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    runtime = RuntimeConfig(
        duration=args.duration,
        solver_dt=args.solver_dt,
        save_dt=args.save_dt,
    )
    registry = dict(get_system_registry())
    if args.gvs_scaling == "basis-order":
        registry["gvs"] = get_gvs_basis_order_system_config()
    key = jax.random.PRNGKey(args.seed)
    device = jax.devices()[0]
    device_name = getattr(device, "device_kind", getattr(device, "device_type", ""))
    device_id = f"{device.platform}:{device_name or 'unknown'}:{device.id}"

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
                    config, size, requested_gauss_points
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

                print(f"  -> {config.size_label}={size}, dof={dof}{gauss_note}")
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
                    sim_time = float(jnp.mean(ts_last) - runtime.t0)
                    total_sim_time = sim_time * batch
                    per_env_speed = (
                        sim_time / wall_time if wall_time > 0 else float("inf")
                    )
                    total_speed = (
                        total_sim_time / wall_time if wall_time > 0 else float("inf")
                    )
                    per_env_wall = wall_time / batch

                    print(
                        f"     batch={batch:>4d} | wall={wall_time:.4f}s | "
                        f"per-env sim/wall={per_env_speed:.2f}x | "
                        f"total sim/wall={total_speed:.2f}x | "
                        f"per-env wall={per_env_wall:.5f}s"
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
                            "device": device.platform,
                            "device_id": device_id,
                        }
                    )

    if args.csv:
        _write_csv(results, args.csv)
    if args.npz:
        _write_npz(results, args.npz)
    if args.plot or args.show_plot:
        _plot_results(results, args.plot, args.show_plot, args.log_x, args.log_y)

    if not any((args.csv, args.npz, args.plot, args.show_plot)):
        print("\n[!] No output artifacts were requested (use --csv / --npz / --plot).")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
