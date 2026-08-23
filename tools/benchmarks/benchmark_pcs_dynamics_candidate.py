#!/usr/bin/env python3

"""Compare optimized PCS dynamics with the implementation on ``origin/main``."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))
    sys.path.insert(0, str(repository_root / "src"))

import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_enable_x64", True)

from soromox.systems import PCS  # noqa: E402
from soromox.utils.lie_algebra import se3  # noqa: E402
from tools.benchmarks._benchmark_common import (  # noqa: E402
    block_until_ready,
    get_system_registry,
)

Tree = Any


def reference_dynamics_terms(
    system: PCS, q: Array, qd: Array
) -> tuple[Array, Array, Array]:
    """Reproduce the PCS dynamics assembly on ``origin/main``."""
    weights, poses, jacobians, jacobian_dot_qd = system._integration_kinematics(
        q, qd, convective_only_jd=True
    )

    def segment_terms(i: Array) -> tuple[Array, Array, Array]:
        local_mass = system.M_segments[i]

        def point_terms(j: Array) -> tuple[Array, Array, Array]:
            weight = weights[i, j]
            pose = poses[i, j]
            jacobian = jacobians[i, j]
            jacobian_dot_qd_ij = jacobian_dot_qd[i, j]
            eta = jacobian @ qd
            adjoint_pose_inverse = se3.adjoint_inverse(pose)
            inertia = weight * jacobian.T @ local_mass @ jacobian
            coriolis_qd = weight * (
                jacobian.T
                @ (
                    local_mass @ jacobian_dot_qd_ij
                    + se3.coadjoint(eta) @ local_mass @ eta
                )
            )
            gravity = (
                -weight * jacobian.T @ local_mass @ adjoint_pose_inverse @ system.g
            )
            return inertia, coriolis_qd, gravity

        return jax.vmap(point_terms)(jnp.arange(system.num_gauss_points))

    inertia, coriolis_qd, gravity = jax.vmap(segment_terms)(
        jnp.arange(system.num_segments)
    )
    return (
        jnp.sum(inertia, axis=(0, 1)),
        jnp.sum(coriolis_qd, axis=(0, 1)),
        jnp.sum(gravity, axis=(0, 1)),
    )


def reference_forward_dynamics(system: PCS, y: Array) -> Array:
    """Reproduce PCS forward dynamics on ``origin/main``."""
    q, qd = jnp.split(y, 2)
    inertia, coriolis_qd, gravity = reference_dynamics_terms(system, q, qd)
    actuation = system.actuation_force(q, jnp.zeros((system.num_actuators,)), qd=qd)
    rhs = (
        actuation
        - coriolis_qd
        - gravity
        - system.elastic_force(q)
        - system.damping_matrix(q) @ qd
    )
    return jnp.concatenate([qd, jnp.linalg.solve(inertia, rhs)])


def _compile(fn: Callable[..., Tree], *inputs: Array) -> tuple[Any, float]:
    start = time.perf_counter()
    compiled = jax.jit(fn).lower(*inputs).compile()
    return compiled, time.perf_counter() - start


def _memory_analysis(compiled: Any) -> dict[str, int]:
    stats = compiled.memory_analysis()
    if stats is None:
        return {}
    return {
        name: int(value)
        for name in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "temp_size_in_bytes",
            "alias_size_in_bytes",
            "generated_code_size_in_bytes",
        )
        if (value := getattr(stats, name, None)) is not None
    }


def _accuracy(actual: Tree, expected: Tree) -> dict[str, float]:
    actual_leaves = jax.tree.leaves(actual)
    expected_leaves = jax.tree.leaves(expected)
    differences = [
        jnp.ravel(actual_leaf - expected_leaf)
        for actual_leaf, expected_leaf in zip(
            actual_leaves, expected_leaves, strict=True
        )
    ]
    references = [jnp.ravel(leaf) for leaf in expected_leaves]
    difference = jnp.concatenate(differences)
    reference = jnp.concatenate(references)
    return {
        "max_abs": float(jnp.max(jnp.abs(difference))),
        "relative_l2": float(
            jnp.linalg.norm(difference) / jnp.maximum(jnp.linalg.norm(reference), 1e-15)
        ),
    }


def _timings(
    functions: dict[str, Any],
    inputs: tuple[Array, ...],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, dict[str, float]]:
    for _ in range(warmup):
        for function in functions.values():
            block_until_ready(function(*inputs))

    names = tuple(functions)
    samples = {name: [] for name in names}
    for repeat in range(repeats):
        offset = repeat % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            start = time.perf_counter_ns()
            block_until_ready(functions[name](*inputs))
            samples[name].append((time.perf_counter_ns() - start) * 1e-6)

    return {
        name: {
            "mean_ms": statistics.fmean(values),
            "median_ms": statistics.median(values),
            "minimum_ms": min(values),
            "maximum_ms": max(values),
        }
        for name, values in samples.items()
    }


def benchmark_case(
    *,
    operation: str,
    num_segments: int,
    batch_size: int,
    num_gauss_points: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    config = get_system_registry()["pcs"]
    system = config.gauss_factory(num_segments, num_gauss_points)
    context = config.build_context(system)

    if operation == "dynamics_terms":
        q, qd = context["q"], context["qd"]

        def reference(q_: Array, qd_: Array) -> Tree:
            return reference_dynamics_terms(system, q_, qd_)

        def candidate(q_: Array, qd_: Array) -> Tree:
            return system.dynamics_terms(q_, qd_)

        if batch_size > 1:
            offsets = jnp.linspace(-1e-3, 1e-3, batch_size)[:, None]
            q = jnp.broadcast_to(q, (batch_size, q.shape[0])) + offsets
            qd = jnp.broadcast_to(qd, (batch_size, qd.shape[0])) - offsets
            reference = jax.vmap(reference)
            candidate = jax.vmap(candidate)
        inputs = (q, qd)
    else:
        y = context["y"]

        def reference(y_: Array) -> Tree:
            return reference_forward_dynamics(system, y_)

        def candidate(y_: Array) -> Tree:
            return system.forward_dynamics(jnp.asarray(0.0), y_)

        if batch_size > 1:
            offsets = jnp.linspace(-1e-3, 1e-3, batch_size)[:, None]
            half_width = y.shape[0] // 2
            state_offsets = jnp.broadcast_to(offsets, (batch_size, half_width))
            y = jnp.broadcast_to(y, (batch_size, y.shape[0])) + jnp.concatenate(
                [state_offsets, -state_offsets], axis=1
            )
            reference = jax.vmap(reference)
            candidate = jax.vmap(candidate)
        inputs = (y,)

    jax.clear_caches()
    reference_compiled, reference_compile_s = _compile(reference, *inputs)
    jax.clear_caches()
    candidate_compiled, candidate_compile_s = _compile(candidate, *inputs)
    expected = reference_compiled(*inputs)
    actual = candidate_compiled(*inputs)
    block_until_ready(expected)
    block_until_ready(actual)
    accuracy = _accuracy(actual, expected)
    if not all(
        bool(jnp.all(jnp.isfinite(actual_leaf)))
        for actual_leaf in jax.tree.leaves(actual)
    ):
        raise AssertionError("candidate PCS dynamics produced non-finite output")
    if accuracy["relative_l2"] > 1e-8:
        raise AssertionError(
            "candidate PCS dynamics exceeded the relative L2 error tolerance"
        )

    timings = _timings(
        {"reference": reference_compiled, "candidate": candidate_compiled},
        inputs,
        warmup=warmup,
        repeats=repeats,
    )
    return {
        "operation": operation,
        "num_segments": num_segments,
        "num_dofs": system.num_dofs,
        "num_gauss_points": num_gauss_points,
        "batch_size": batch_size,
        "accuracy": accuracy,
        "reference": {
            "compile_s": reference_compile_s,
            "timing": timings["reference"],
            "memory": _memory_analysis(reference_compiled),
        },
        "candidate": {
            "compile_s": candidate_compile_s,
            "timing": timings["candidate"],
            "memory": _memory_analysis(candidate_compiled),
        },
        "median_speedup": (
            timings["reference"]["median_ms"] / timings["candidate"]["median_ms"]
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation",
        choices=("dynamics_terms", "forward_dynamics"),
        default="forward_dynamics",
    )
    parser.add_argument("--segments", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--gauss-points", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    if min(args.segments) < 1 or min(args.batch_sizes) < 1:
        parser.error("segments and batch sizes must be positive")
    if args.gauss_points < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error(
            "gauss points and repeats must be positive; warmup cannot be negative"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    results = [
        benchmark_case(
            operation=args.operation,
            num_segments=num_segments,
            batch_size=batch_size,
            num_gauss_points=args.gauss_points,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for num_segments in args.segments
        for batch_size in args.batch_sizes
    ]
    report = {
        "device": str(jax.devices()[0]),
        "jax_version": jax.__version__,
        "x64_enabled": jax.config.x64_enabled,
        "results": results,
    }
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
