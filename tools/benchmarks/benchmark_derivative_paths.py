#!/usr/bin/env python3
"""Benchmark derivative paths across analytical, autograd, and custom-JVP APIs.

This is the canonical derivative benchmark for Soromox systems. It compares:

- direct analytical hooks and force/tangent helpers,
- protected-primal autograd paths that bypass public custom-JVP wrappers,
- public APIs with custom JVPs enabled,
- public APIs with custom JVPs disabled.

Example:

uv run python tools/benchmarks/benchmark_derivative_paths.py \
  --systems planar_pcs pcs gvs \
  --segment-counts 1 2 4 8 16 32 \
  --execution-repeats 3 \
  --warmup-runs 1 \
  --csv benchmarks/derivative-paths.csv \
  --json benchmarks/derivative-paths.json \
  --markdown-summary benchmarks/derivative-paths.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

from soromox.autodiff import set_custom_jvp_enabled  # noqa: E402
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


StrategyFn = Callable[[SoftRobot, Mapping[str, Array]], Tree]

SYSTEM_SUMMARY_EXCLUDED_CASES = frozenset(
    {
        # Plain Jacobian evaluation is a useful primal wrapper-overhead control,
        # but it is not a derivative path. Keep it in the per-case table while
        # excluding it from system-level derivative speedup aggregates.
        "jacobian",
    }
)


def _total_length(system: SoftRobot) -> Array:
    if hasattr(system, "segment_lengths"):
        return jnp.sum(system.segment_lengths)
    if hasattr(system, "L"):
        return jnp.sum(system.L)
    return jnp.sum(jnp.atleast_1d(jnp.asarray(system.segment_length)))


def _basis_like(vector: Array) -> Array:
    return jnp.eye(vector.shape[0], dtype=vector.dtype)


def _kinetic_energy_gradient_from_tangent(
    system: SoftRobot, q: Array, qd: Array
) -> tuple[Array, Array]:
    q_basis = _basis_like(q)
    qd_basis = _basis_like(qd)
    dT_dq = jax.vmap(
        lambda q_tangent: system._kinetic_energy_jvp_tangent(
            q, qd, q_tangent, None
        )
    )(q_basis)
    dT_dqd = jax.vmap(
        lambda qdd: system._kinetic_energy_jvp_tangent(q, qd, None, qdd)
    )(qd_basis)
    return dT_dq, dT_dqd


def _total_energy_gradient_from_tangent(
    system: SoftRobot, q: Array, qd: Array
) -> tuple[Array, Array]:
    q_basis = _basis_like(q)
    qd_basis = _basis_like(qd)
    dE_dq = jax.vmap(
        lambda q_tangent: system._total_energy_jvp_tangent(q, qd, q_tangent, None)
    )(q_basis)
    dE_dqd = jax.vmap(
        lambda qdd: system._total_energy_jvp_tangent(q, qd, None, qdd)
    )(qd_basis)
    return dE_dq, dE_dqd


def _direct_jacobian(system: SoftRobot, ctx: Mapping[str, Array]) -> Array:
    return system._jacobian(ctx["q"], ctx["s"])


def _protected_autograd_jacobian(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> Array:
    return SoftRobot._jacobian(system, ctx["q"], ctx["s"])


def _public_jacobian(system: SoftRobot, ctx: Mapping[str, Array]) -> Array:
    return system.jacobian(ctx["q"], ctx["s"])


def _direct_forward_kinematics_jvp_q(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> tuple[Array, Array]:
    q = ctx["q"]
    qd = ctx["qd"]
    s = ctx["s"]
    pose = system._forward_kinematics(q, s)
    eta = system._jacobian(q, s) @ qd
    tangent = system._pose_tangent_from_inertial_velocity(pose, eta)
    return pose, tangent


def _protected_autograd_forward_kinematics_jvp_q(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> tuple[Array, Array]:
    return jax.jvp(
        lambda q_: system._forward_kinematics(q_, ctx["s"]),
        (ctx["q"],),
        (ctx["qd"],),
    )


def _public_forward_kinematics_jvp_q(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> tuple[Array, Array]:
    return jax.jvp(
        lambda q_: system.forward_kinematics(q_, ctx["s"]),
        (ctx["q"],),
        (ctx["qd"],),
    )


def _direct_forward_kinematics_jvp_s(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> tuple[Array, Array]:
    return system._forward_kinematics_and_arc_length_derivative(ctx["q"], ctx["s"])


def _protected_autograd_forward_kinematics_jvp_s(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> tuple[Array, Array]:
    return jax.jvp(
        lambda s_: system._forward_kinematics(ctx["q"], s_),
        (ctx["s"],),
        (jnp.ones_like(jnp.asarray(ctx["s"])),),
    )


def _public_forward_kinematics_jvp_s(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> tuple[Array, Array]:
    return jax.jvp(
        lambda s_: system.forward_kinematics(ctx["q"], s_),
        (ctx["s"],),
        (jnp.ones_like(jnp.asarray(ctx["s"])),),
    )


def _direct_jacobian_jvp_q(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> tuple[Array, Array]:
    return system._jacobian_and_time_derivative(ctx["q"], ctx["qd"], ctx["s"])


def _protected_autograd_jacobian_jvp_q(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> tuple[Array, Array]:
    return jax.jvp(
        lambda q_: system._jacobian(q_, ctx["s"]),
        (ctx["q"],),
        (ctx["qd"],),
    )


def _public_jacobian_jvp_q(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> tuple[Array, Array]:
    return jax.jvp(
        lambda q_: system.jacobian(q_, ctx["s"]),
        (ctx["q"],),
        (ctx["qd"],),
    )


def _direct_jacobian_jvp_s(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> tuple[Array, Array]:
    return system._jacobian_and_arc_length_derivative(ctx["q"], ctx["s"])


def _protected_autograd_jacobian_jvp_s(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> tuple[Array, Array]:
    return jax.jvp(
        lambda s_: system._jacobian(ctx["q"], s_),
        (ctx["s"],),
        (jnp.ones_like(jnp.asarray(ctx["s"])),),
    )


def _public_jacobian_jvp_s(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> tuple[Array, Array]:
    return jax.jvp(
        lambda s_: system.jacobian(ctx["q"], s_),
        (ctx["s"],),
        (jnp.ones_like(jnp.asarray(ctx["s"])),),
    )


def _direct_gravitational_energy_gradient(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> Array:
    return system.gravitational_force(ctx["q"])


def _protected_autograd_gravitational_energy_gradient(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> Array:
    return jax.grad(lambda q_: system._gravitational_energy(q_))(ctx["q"])


def _public_gravitational_energy_gradient(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> Array:
    return jax.grad(lambda q_: system.gravitational_energy(q_))(ctx["q"])


def _direct_elastic_energy_gradient(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> Array:
    return system.elastic_force(ctx["q"])


def _protected_autograd_elastic_energy_gradient(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> Array:
    return jax.grad(lambda q_: system._elastic_energy(q_))(ctx["q"])


def _public_elastic_energy_gradient(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> Array:
    return jax.grad(lambda q_: system.elastic_energy(q_))(ctx["q"])


def _direct_potential_energy_gradient(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> Array:
    return system.gravitational_force(ctx["q"]) + system.elastic_force(ctx["q"])


def _protected_autograd_potential_energy_gradient(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> Array:
    return jax.grad(
        lambda q_: system._gravitational_energy(q_) + system._elastic_energy(q_)
    )(ctx["q"])


def _public_potential_energy_gradient(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> Array:
    return jax.grad(lambda q_: system.potential_energy(q_))(ctx["q"])


def _direct_kinetic_energy_gradient(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> tuple[Array, Array]:
    return _kinetic_energy_gradient_from_tangent(system, ctx["q"], ctx["qd"])


def _protected_autograd_kinetic_energy_gradient(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> tuple[Array, Array]:
    return jax.grad(lambda q_, qd_: system._kinetic_energy(q_, qd_), argnums=(0, 1))(
        ctx["q"], ctx["qd"]
    )


def _public_kinetic_energy_gradient(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> tuple[Array, Array]:
    return jax.grad(lambda q_, qd_: system.kinetic_energy(q_, qd_), argnums=(0, 1))(
        ctx["q"], ctx["qd"]
    )


def _direct_total_energy_gradient(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> tuple[Array, Array]:
    return _total_energy_gradient_from_tangent(system, ctx["q"], ctx["qd"])


def _protected_autograd_total_energy_gradient(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> tuple[Array, Array]:
    return jax.grad(
        lambda q_, qd_: (
            system._kinetic_energy(q_, qd_)
            + system._gravitational_energy(q_)
            + system._elastic_energy(q_)
        ),
        argnums=(0, 1),
    )(ctx["q"], ctx["qd"])


def _public_total_energy_gradient(
    system: SoftRobot, ctx: Mapping[str, Array]
) -> tuple[Array, Array]:
    return jax.grad(lambda q_, qd_: system.total_energy(q_, qd_), argnums=(0, 1))(
        ctx["q"], ctx["qd"]
    )


def _case_registry() -> Mapping[str, Mapping[str, StrategyFn]]:
    return {
        "jacobian": {
            "direct_analytic": _direct_jacobian,
            "protected_autograd": _protected_autograd_jacobian,
            "public": _public_jacobian,
        },
        "forward_kinematics_jvp_q": {
            "direct_analytic": _direct_forward_kinematics_jvp_q,
            "protected_autograd": _protected_autograd_forward_kinematics_jvp_q,
            "public": _public_forward_kinematics_jvp_q,
        },
        "forward_kinematics_jvp_s": {
            "direct_analytic": _direct_forward_kinematics_jvp_s,
            "protected_autograd": _protected_autograd_forward_kinematics_jvp_s,
            "public": _public_forward_kinematics_jvp_s,
        },
        "jacobian_jvp_q": {
            "direct_analytic": _direct_jacobian_jvp_q,
            "protected_autograd": _protected_autograd_jacobian_jvp_q,
            "public": _public_jacobian_jvp_q,
        },
        "jacobian_jvp_s": {
            "direct_analytic": _direct_jacobian_jvp_s,
            "protected_autograd": _protected_autograd_jacobian_jvp_s,
            "public": _public_jacobian_jvp_s,
        },
        "gravitational_energy_gradient": {
            "direct_analytic": _direct_gravitational_energy_gradient,
            "protected_autograd": _protected_autograd_gravitational_energy_gradient,
            "public": _public_gravitational_energy_gradient,
        },
        "elastic_energy_gradient": {
            "direct_analytic": _direct_elastic_energy_gradient,
            "protected_autograd": _protected_autograd_elastic_energy_gradient,
            "public": _public_elastic_energy_gradient,
        },
        "potential_energy_gradient": {
            "direct_analytic": _direct_potential_energy_gradient,
            "protected_autograd": _protected_autograd_potential_energy_gradient,
            "public": _public_potential_energy_gradient,
        },
        "kinetic_energy_gradient": {
            "direct_analytic": _direct_kinetic_energy_gradient,
            "protected_autograd": _protected_autograd_kinetic_energy_gradient,
            "public": _public_kinetic_energy_gradient,
        },
        "total_energy_gradient": {
            "direct_analytic": _direct_total_energy_gradient,
            "protected_autograd": _protected_autograd_total_energy_gradient,
            "public": _public_total_energy_gradient,
        },
    }


def _strategy_names(custom_jvp_modes: Sequence[str]) -> list[str]:
    names = ["direct_analytic", "protected_autograd"]
    if "enabled" in custom_jvp_modes:
        names.append("public_custom_jvp_enabled")
    if "disabled" in custom_jvp_modes:
        names.append("public_custom_jvp_disabled")
    return names


def _strategy_custom_jvp_mode(strategy: str) -> str:
    if strategy == "public_custom_jvp_enabled":
        return "enabled"
    if strategy == "public_custom_jvp_disabled":
        return "disabled"
    return "n/a"


def _case_fn(case_fns: Mapping[str, StrategyFn], strategy: str) -> StrategyFn:
    if strategy in {"direct_analytic", "protected_autograd"}:
        return case_fns[strategy]
    return case_fns["public"]


def _configure_strategy(strategy: str) -> None:
    set_custom_jvp_enabled(strategy == "public_custom_jvp_enabled")
    if hasattr(jax, "clear_caches"):
        jax.clear_caches()


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


def _geomean(values: Sequence[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value) and value > 0.0]
    if not finite_values:
        return float("nan")
    return math.exp(sum(math.log(value) for value in finite_values) / len(finite_values))


def _format_ratio(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.2f}x"


def _format_seconds(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def _format_scientific(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.3e}"


def _write_json(results: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"[+] Wrote JSON results to {path}")


def _result_headers() -> list[str]:
    return [
        "system",
        "case",
        "strategy",
        "custom_jvp",
        "size_label",
        "segment_count",
        "dof",
        "gauss_points",
        "integration_points",
        "compile_time_s",
        "execution_time_s",
        "ratio_to_direct_analytic",
        "ratio_to_protected_autograd",
        "max_abs_diff",
        "max_rel_diff",
        "execution_repeats",
        "warmup_runs",
        "arc_length_fraction",
        "device",
        "device_id",
    ]


def _write_csv(results: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=_result_headers(), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"[+] Wrote CSV results to {path}")


def _rows_by_key(
    results: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int, str, str], Mapping[str, Any]]:
    rows: dict[tuple[str, str, int, str, str], Mapping[str, Any]] = {}
    for row in results:
        key = (
            str(row["system"]),
            str(row["case"]),
            int(row["segment_count"]),
            str(row.get("gauss_points", "")),
            str(row["strategy"]),
        )
        rows[key] = row
    return rows


def _paired_ratio(
    rows: Mapping[tuple[str, str, int, str, str], Mapping[str, Any]],
    system: str,
    case: str,
    segment_count: int,
    gauss_points: str,
    numerator_strategy: str,
    denominator_strategy: str,
) -> float:
    numerator = rows.get(
        (system, case, segment_count, gauss_points, numerator_strategy)
    )
    denominator = rows.get(
        (system, case, segment_count, gauss_points, denominator_strategy)
    )
    if numerator is None or denominator is None:
        return float("nan")
    denominator_time = float(denominator["execution_time_s"])
    if denominator_time <= 0.0:
        return float("nan")
    return float(numerator["execution_time_s"]) / denominator_time


def _write_markdown_summary(results: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _rows_by_key(results)
    systems = sorted({str(row["system"]) for row in results})
    groups = sorted({(str(row["system"]), str(row["case"])) for row in results})
    segment_counts = sorted({int(row["segment_count"]) for row in results})
    largest_segment_count = max(segment_counts) if segment_counts else None
    segment_label = (
        f"segments={largest_segment_count}"
        if largest_segment_count is not None
        else "largest segment"
    )

    lines = [
        "# Derivative Path Benchmark Summary",
        "",
        "## How to read this report",
        "",
        "- `custom-JVP enabled speedup` is "
        "`public_custom_jvp_disabled / public_custom_jvp_enabled`. Values above "
        "`1.0x` mean enabling custom JVPs is faster.",
        "- `direct analytical speedup` is "
        "`protected_autograd / direct_analytic`. Values above `1.0x` mean the "
        "direct analytical helper is faster than differentiating protected "
        "primal hooks.",
        "- `geomean` aggregates all requested segment counts for that system/case.",
        "- The system summary excludes primal control cases such as `jacobian`; "
        "those rows remain visible in the per-case table.",
        f"- `{segment_label}` shows the largest requested segment count only.",
        "",
        "## System Summary",
        "",
        "| system | custom-JVP enabled speedup geomean | "
        f"custom-JVP enabled speedup {segment_label} | "
        "direct analytical speedup geomean | "
        f"direct analytical speedup {segment_label} | max abs diff |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for system in systems:
        system_rows = [
            row
            for row in results
            if str(row["system"]) == system
            and str(row["case"]) not in SYSTEM_SUMMARY_EXCLUDED_CASES
        ]
        system_groups = sorted(
            {(str(row["case"]), str(row.get("gauss_points", ""))) for row in system_rows}
        )
        public_ratios: list[float] = []
        public_ratios_largest: list[float] = []
        direct_ratios: list[float] = []
        direct_ratios_largest: list[float] = []

        for case, gauss_points in system_groups:
            for segment_count in segment_counts:
                public_ratio = _paired_ratio(
                    rows,
                    system,
                    case,
                    segment_count,
                    gauss_points,
                    "public_custom_jvp_disabled",
                    "public_custom_jvp_enabled",
                )
                direct_ratio = _paired_ratio(
                    rows,
                    system,
                    case,
                    segment_count,
                    gauss_points,
                    "protected_autograd",
                    "direct_analytic",
                )
                public_ratios.append(public_ratio)
                direct_ratios.append(direct_ratio)
                if segment_count == largest_segment_count:
                    public_ratios_largest.append(public_ratio)
                    direct_ratios_largest.append(direct_ratio)

        max_abs_diff = max(float(row["max_abs_diff"]) for row in system_rows)
        lines.append(
            "| "
            f"{system} | "
            f"{_format_ratio(_geomean(public_ratios))} | "
            f"{_format_ratio(_geomean(public_ratios_largest))} | "
            f"{_format_ratio(_geomean(direct_ratios))} | "
            f"{_format_ratio(_geomean(direct_ratios_largest))} | "
            f"{_format_scientific(max_abs_diff)} |"
        )

    lines.extend(
        [
            "",
            "## Per-Case Details",
            "",
            "| system | case | custom-JVP enabled speedup geomean | "
            f"custom-JVP enabled speedup {segment_label} | "
            "direct analytical speedup geomean | "
            f"direct analytical speedup {segment_label} | max abs diff | max rel diff |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for system, case in groups:
        group_rows = [
            row
            for row in results
            if str(row["system"]) == system and str(row["case"]) == case
        ]
        gauss_values = sorted({str(row.get("gauss_points", "")) for row in group_rows})
        public_ratios: list[float] = []
        direct_ratios: list[float] = []
        public_ratio_largest = float("nan")
        direct_ratio_largest = float("nan")

        for row in group_rows:
            segment_count = int(row["segment_count"])
            gauss_points = str(row.get("gauss_points", ""))
            if row["strategy"] == "public_custom_jvp_enabled":
                ratio = _paired_ratio(
                    rows,
                    system,
                    case,
                    segment_count,
                    gauss_points,
                    "public_custom_jvp_disabled",
                    "public_custom_jvp_enabled",
                )
                public_ratios.append(ratio)
                if segment_count == largest_segment_count:
                    public_ratio_largest = ratio
            if row["strategy"] == "direct_analytic":
                ratio = _paired_ratio(
                    rows,
                    system,
                    case,
                    segment_count,
                    gauss_points,
                    "protected_autograd",
                    "direct_analytic",
                )
                direct_ratios.append(ratio)
                if segment_count == largest_segment_count:
                    direct_ratio_largest = ratio

        max_abs_diff = max(float(row["max_abs_diff"]) for row in group_rows)
        max_rel_diff = max(float(row["max_rel_diff"]) for row in group_rows)
        gauss_suffix = "" if gauss_values == ["None"] else f" ({','.join(gauss_values)})"
        lines.append(
            "| "
            f"{system}{gauss_suffix} | {case} | "
            f"{_format_ratio(_geomean(public_ratios))} | "
            f"{_format_ratio(public_ratio_largest)} | "
            f"{_format_ratio(_geomean(direct_ratios))} | "
            f"{_format_ratio(direct_ratio_largest)} | "
            f"{_format_scientific(max_abs_diff)} | "
            f"{_format_scientific(max_rel_diff)} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[+] Wrote Markdown summary to {path}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    system_registry = {
        name: config
        for name, config in get_system_registry().items()
        if name in {"planar_pcs", "pcs", "gvs"}
    }
    case_registry = _case_registry()
    default_strategies = _strategy_names(("enabled", "disabled"))
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark derivative paths across direct analytical, protected "
            "autograd, and public custom-JVP APIs."
        )
    )
    add_system_selection_args(
        parser, system_registry, default_segment_counts=[1, 2, 4, 8, 16, 32]
    )
    add_gauss_point_args(parser)
    parser.add_argument(
        "--cases",
        nargs="*",
        default=list(case_registry.keys()),
        choices=sorted(case_registry.keys()),
        help="Derivative cases to benchmark (default: all).",
    )
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=default_strategies,
        choices=default_strategies,
        help="Strategies to benchmark (default: all).",
    )
    parser.add_argument(
        "--custom-jvp-modes",
        nargs="*",
        default=["enabled", "disabled"],
        choices=["enabled", "disabled"],
        help="Public custom-JVP modes to include (default: enabled disabled).",
    )
    parser.add_argument(
        "--arc-length-fraction",
        type=float,
        default=0.37,
        help="Interior fraction of total arc length used for derivative evaluation.",
    )
    parser.add_argument(
        "--execution-repeats",
        type=int,
        default=3,
        help="Number of post-compilation executions to average.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warm executions between the cold compile call and measured repeats.",
    )
    parser.add_argument("--json", type=Path, help="Optional JSON output path.")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    parser.add_argument(
        "--markdown-summary",
        type=Path,
        help="Optional Markdown summary output path.",
    )
    args = parser.parse_args(argv)
    if args.gauss_points is not None:
        if any(value < 1 for value in args.gauss_points):
            parser.error("All --gauss-points entries must be >= 1.")
        args.gauss_points = normalize_gauss_point_values(args.gauss_points)
    if not 0.0 < args.arc_length_fraction < 1.0:
        parser.error("--arc-length-fraction must be strictly between 0 and 1.")

    allowed_public = {
        f"public_custom_jvp_{mode}" for mode in args.custom_jvp_modes
    }
    args.strategies = [
        strategy
        for strategy in args.strategies
        if not strategy.startswith("public_custom_jvp_") or strategy in allowed_public
    ]
    if not args.strategies:
        parser.error("No strategies selected after applying --custom-jvp-modes.")
    return args


def _benchmark_context(system: SoftRobot, system_ctx: Mapping[str, Array], args: Any):
    q = system_ctx["q"]
    s = jnp.asarray(args.arc_length_fraction, dtype=q.dtype) * _total_length(system)
    ctx = dict(system_ctx)
    ctx["s"] = s
    return ctx


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    system_registry = {
        name: config
        for name, config in get_system_registry().items()
        if name in {"planar_pcs", "pcs", "gvs"}
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
                ctx = _benchmark_context(system, system_cfg.build_context(system), args)
                gauss_points, integration_points = system_gauss_point_metadata(system)
                print(
                    f"\n=== {system_name}: {system_cfg.size_label}="
                    f"{segment_count}, dof={system.num_dofs}, "
                    f"gauss_points={gauss_points}, s={float(ctx['s']):.6f} ==="
                )

                for case_name in args.cases:
                    case_fns = case_registry[case_name]
                    print(f"  -> {case_name}")
                    temporary_rows: list[dict[str, Any]] = []
                    outputs: dict[str, Tree] = {}
                    executions: dict[str, float] = {}

                    for strategy in args.strategies:
                        _configure_strategy(strategy)
                        strategy_fn = _case_fn(case_fns, strategy)
                        strategy_jit = eqx.filter_jit(strategy_fn)
                        print(f"     {strategy} ...", end=" ", flush=True)
                        compile_time, execution_time, output = _measure_jitted_call(
                            strategy_jit,
                            (system, ctx),
                            args.execution_repeats,
                            args.warmup_runs,
                        )
                        outputs[strategy] = output
                        executions[strategy] = execution_time
                        temporary_rows.append(
                            {
                                "system": system_name,
                                "case": case_name,
                                "strategy": strategy,
                                "custom_jvp": _strategy_custom_jvp_mode(strategy),
                                "size_label": system_cfg.size_label,
                                "segment_count": segment_count,
                                "dof": int(system.num_dofs),
                                "gauss_points": gauss_points,
                                "integration_points": integration_points,
                                "compile_time_s": compile_time,
                                "execution_time_s": execution_time,
                                "execution_repeats": args.execution_repeats,
                                "warmup_runs": args.warmup_runs,
                                "arc_length_fraction": args.arc_length_fraction,
                                "device": device.platform,
                                "device_id": device.id,
                            }
                        )
                        print(f"exec={_format_seconds(execution_time)}s")

                    direct_output = outputs.get("direct_analytic")
                    direct_execution = executions.get("direct_analytic")
                    protected_execution = executions.get("protected_autograd")
                    for row in temporary_rows:
                        strategy = str(row["strategy"])
                        output = outputs[strategy]
                        if direct_output is None:
                            max_abs_diff, max_rel_diff = float("nan"), float("nan")
                        else:
                            max_abs_diff, max_rel_diff = _tree_difference_metrics(
                                direct_output, output
                            )
                        execution_time = float(row["execution_time_s"])
                        row["ratio_to_direct_analytic"] = (
                            execution_time / direct_execution
                            if direct_execution is not None and direct_execution > 0.0
                            else float("nan")
                        )
                        row["ratio_to_protected_autograd"] = (
                            execution_time / protected_execution
                            if protected_execution is not None
                            and protected_execution > 0.0
                            else float("nan")
                        )
                        row["max_abs_diff"] = max_abs_diff
                        row["max_rel_diff"] = max_rel_diff
                        results.append(row)

    set_custom_jvp_enabled(True)

    if args.json is not None:
        _write_json(results, args.json)
    if args.csv is not None:
        _write_csv(results, args.csv)
    if args.markdown_summary is not None:
        _write_markdown_summary(results, args.markdown_summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
