#!/usr/bin/env python3
"""Compare PCS forward-dynamics derivatives with custom JVPs on and off.

The benchmark separates the first-call compilation cost from steady-state
execution time. Every timed call is synchronized with the active JAX device.
Custom-JVP and pure-autodiff outputs are also compared for correctness.

Example:
    python tools/benchmarks/pcs_custom_jvp_speed_test.py \
        --segment-counts 1 2 4 \
        --setups full bending_extension axial threadlike \
        --workloads state_jvp state_jacobian \
        --repeats 10
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting  # noqa: E402
from soromox.autodiff import set_custom_jvp_enabled  # noqa: E402
from soromox.systems import PCS, LinkSpec, PCSStructure  # noqa: E402

Array = jax.Array
Tree = Any
Workload = Callable[[Array, Array, Array], Tree]

SETUPS = ("full", "bending_extension", "axial", "threadlike")
WORKLOADS = ("primal", "state_jvp", "state_jacobian", "all_jacobians")


def _strain_selector(setup: str, num_segments: int) -> Array | None:
    if setup == "full":
        return None
    if setup in {"bending_extension", "threadlike"}:
        per_segment = jnp.array([False, False, True, True, False, False])
    elif setup == "axial":
        per_segment = jnp.array([False, False, False, True, False, False])
    else:
        raise ValueError(f"Unknown setup: {setup}")
    return jnp.tile(per_segment, num_segments)


def _threadlike_actuator(num_segments: int) -> ThreadlikeActuator:
    offsets = jnp.array([-0.01, 0.01])
    routing = ThreadlikeRouting.linear(
        intercept=jnp.stack(
            [jnp.zeros_like(offsets), offsets, jnp.zeros_like(offsets)], axis=1
        ),
        slope=jnp.zeros((2, 3)),
        start_segment_index=(0, 0),
        end_segment_index=(num_segments - 1, num_segments - 1),
    )
    return ThreadlikeActuator.tendons(routing)


def build_model(setup: str, num_segments: int, gauss_points: int) -> PCS:
    """Build one benchmark PCS configuration."""
    actuators = _threadlike_actuator(num_segments) if setup == "threadlike" else None
    return PCS.from_links(
        [
            LinkSpec.circular(
                length=0.1,
                radius=0.02,
                density=1070.0,
                young_modulus=2.0e3,
                shear_modulus=1.0e3,
                material_damping_coefficient=1.0e-3,
                reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            )
            for _ in range(num_segments)
        ],
        gravity=jnp.array([0.0, 0.0, -9.81]),
        structure=PCSStructure(
            num_gauss_points=gauss_points,
            strain_selector=_strain_selector(setup, num_segments),
        ),
        actuators=actuators,
    )


def build_inputs(model: PCS) -> tuple[Array, Array, Array, Array]:
    """Return state, state direction, actuator input, and external force."""
    q = jnp.linspace(-0.025, 0.035, model.num_dofs)
    qd = jnp.linspace(0.04, -0.03, model.num_dofs)
    y = jnp.concatenate([q, qd])
    y_direction = jnp.linspace(-0.2, 0.25, 2 * model.num_dofs)
    u = jnp.linspace(0.05, 0.1, model.num_actuators)
    tau_ext = jnp.linspace(-0.01, 0.015, model.num_dofs)
    return y, y_direction, u, tau_ext


def build_workload(model: PCS, workload: str, y_direction: Array) -> Workload:
    """Create one public forward-dynamics derivative workload."""
    t = jnp.array(0.0)

    def dynamics(y: Array, u: Array, tau_ext: Array) -> Array:
        return model.forward_dynamics(t, y, (u, tau_ext))

    if workload == "primal":
        return dynamics
    if workload == "state_jvp":
        return lambda y, u, tau_ext: jax.jvp(
            lambda y_value: dynamics(y_value, u, tau_ext),
            (y,),
            (y_direction,),
        )
    if workload == "state_jacobian":
        return lambda y, u, tau_ext: jax.jacfwd(
            lambda y_value: dynamics(y_value, u, tau_ext)
        )(y)
    if workload == "all_jacobians":
        return lambda y, u, tau_ext: jax.jacfwd(
            dynamics,
            argnums=(0, 1, 2),
        )(y, u, tau_ext)
    raise ValueError(f"Unknown workload: {workload}")


def block_until_ready(tree: Tree) -> None:
    """Synchronize every array leaf in a result PyTree."""
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def measure(
    fn: Workload,
    args: tuple[Array, Array, Array],
    warmup_runs: int,
    repeats: int,
) -> tuple[float, float, Tree]:
    """Return estimated compilation seconds, median execution seconds, output."""
    compiled = jax.jit(fn)
    start = time.perf_counter()
    output = compiled(*args)
    block_until_ready(output)
    first_call = time.perf_counter() - start

    for _ in range(warmup_runs):
        output = compiled(*args)
        block_until_ready(output)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = compiled(*args)
        block_until_ready(output)
        samples.append(time.perf_counter() - start)

    execution = statistics.median(samples)
    compilation = max(0.0, first_call - execution)
    return compilation, execution, output


def difference(reference: Tree, candidate: Tree) -> tuple[float, float]:
    """Return maximum absolute and scale-normalized PyTree differences."""
    reference_leaves, reference_tree = jax.tree_util.tree_flatten(reference)
    candidate_leaves, candidate_tree = jax.tree_util.tree_flatten(candidate)
    if reference_tree != candidate_tree:
        return float("inf"), float("inf")

    max_absolute = 0.0
    max_relative = 0.0
    for reference_leaf, candidate_leaf in zip(
        reference_leaves, candidate_leaves, strict=True
    ):
        reference_array = jnp.asarray(reference_leaf)
        candidate_array = jnp.asarray(candidate_leaf)
        if reference_array.size == 0:
            continue
        absolute = float(jnp.max(jnp.abs(reference_array - candidate_array)))
        scale = max(float(jnp.max(jnp.abs(reference_array))), 1.0e-12)
        max_absolute = max(max_absolute, absolute)
        max_relative = max(max_relative, absolute / scale)
    return max_absolute, max_relative


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Execute every requested PCS custom-JVP benchmark pair."""
    rows: list[dict[str, Any]] = []
    device = jax.devices()[0]
    print(
        f"device={device.platform}:{getattr(device, 'device_kind', 'unknown')} "
        f"x64={jax.config.x64_enabled} repeats={args.repeats}"
    )
    print(
        "setup                 links dof workload          "
        "off_ms    on_ms speedup compile_off compile_on  rel_error"
    )

    for setup in args.setups:
        for num_segments in args.segment_counts:
            model = build_model(setup, num_segments, args.gauss_points)
            y, y_direction, u, tau_ext = build_inputs(model)
            inputs = (y, u, tau_ext)

            for workload in args.workloads:
                measurements: dict[bool, tuple[float, float, Tree]] = {}
                for enabled in (False, True):
                    set_custom_jvp_enabled(enabled)
                    jax.clear_caches()
                    workload_fn = build_workload(model, workload, y_direction)
                    measurements[enabled] = measure(
                        workload_fn,
                        inputs,
                        args.warmup_runs,
                        args.repeats,
                    )

                compile_off, execution_off, output_off = measurements[False]
                compile_on, execution_on, output_on = measurements[True]
                max_absolute, max_relative = difference(output_off, output_on)
                if max_relative > args.correctness_rtol:
                    raise AssertionError(
                        "Custom-JVP output does not match pure autodiff: "
                        f"setup={setup}, segments={num_segments}, "
                        f"workload={workload}, max_relative={max_relative:.3e}, "
                        f"limit={args.correctness_rtol:.3e}"
                    )
                speedup = execution_off / execution_on
                row = {
                    "setup": setup,
                    "segment_count": num_segments,
                    "dof": model.num_dofs,
                    "gauss_points": args.gauss_points,
                    "workload": workload,
                    "custom_jvp_disabled_compile_s": compile_off,
                    "custom_jvp_enabled_compile_s": compile_on,
                    "custom_jvp_disabled_execution_s": execution_off,
                    "custom_jvp_enabled_execution_s": execution_on,
                    "speedup": speedup,
                    "max_absolute_difference": max_absolute,
                    "max_relative_difference": max_relative,
                    "device": str(device),
                }
                rows.append(row)
                print(
                    f"{setup:21s} {num_segments:5d} {model.num_dofs:3d} "
                    f"{workload:17s} {1e3 * execution_off:8.3f} "
                    f"{1e3 * execution_on:8.3f} {speedup:7.2f}x "
                    f"{compile_off:11.3f} {compile_on:10.3f} "
                    f"{max_relative:10.3e}"
                )

    set_custom_jvp_enabled(True)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line benchmark options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-counts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--setups", nargs="+", choices=SETUPS, default=list(SETUPS))
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=WORKLOADS,
        default=["state_jvp", "state_jacobian"],
    )
    parser.add_argument("--gauss-points", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--correctness-rtol", type=float, default=1.0e-5)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    if any(count < 1 for count in args.segment_counts):
        parser.error("--segment-counts values must be positive")
    if args.gauss_points < 1:
        parser.error("--gauss-points must be positive")
    if args.warmup_runs < 0 or args.repeats < 1:
        parser.error("warmup runs must be non-negative and repeats must be positive")
    if args.correctness_rtol <= 0.0:
        parser.error("--correctness-rtol must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Run the benchmark and optionally write its rows as CSV."""
    args = parse_args(argv)
    rows = run(args)
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
