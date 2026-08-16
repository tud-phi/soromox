#!/usr/bin/env python3
"""Benchmark Lie-algebra primitives and their differentiated call paths.

The benchmark intentionally uses only public Lie-algebra functions so the same
driver can compare two source revisions through ``PYTHONPATH``. It reports JIT
compile time, median steady-state execution time, interquartile spread, and
whether the compiled result is finite.

Example comparing two checked-out or archived source trees::

    PYTHONPATH=/tmp/soromox-main/src uv run python \
        tools/benchmarks/benchmark_lie_algebra.py \
        --label main --json /tmp/lie-main.json
    PYTHONPATH=/tmp/soromox-pr/src uv run python \
        tools/benchmarks/benchmark_lie_algebra.py \
        --label pr --json /tmp/lie-pr.json
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from soromox.utils.lie_algebra import constant_strain, se2, se3, so3  # noqa: E402

Array = jax.Array
Tree = Any


@dataclass(frozen=True)
class BenchmarkCase:
    """One JIT-compiled Lie-algebra workload."""

    name: str
    family: str
    operation: str
    scenario: str
    fn: Callable[..., Tree]
    args: tuple[Any, ...]


def _constant_strain_function(algebra: str, operation: str) -> Callable[..., Array]:
    """Resolve one constant-strain function across the compared revisions.

    The PR uses algebra-specific package modules, while ``main`` exposes
    suffixed functions from a single module. Keeping this adaptation local to
    the comparison driver lets both revisions run the same benchmark without
    adding legacy aliases to the library API.
    """
    algebra_module = getattr(constant_strain, algebra)
    if hasattr(algebra_module, "tangent"):
        return getattr(algebra_module, operation)
    return getattr(constant_strain, f"{operation}_{algebra}")


def _block_until_ready(tree: Tree) -> None:
    """Block until every array leaf in ``tree`` is ready."""
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _tree_is_finite(tree: Tree) -> bool:
    """Return whether every array leaf contains only finite values."""
    leaves = [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tree)]
    return bool(jnp.all(jnp.stack(leaves))) if leaves else True


def _rotation_matrix(omega: Array) -> Array:
    """Construct an independent Rodrigues rotation for benchmark inputs."""
    theta = jnp.linalg.norm(omega)
    axis = omega / theta
    x, y, z = axis
    skew = jnp.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return jnp.eye(3) + jnp.sin(theta) * skew + (1.0 - jnp.cos(theta)) * (skew @ skew)


def _se2_transform(xi: Array) -> Array:
    """Construct a regular ``SE(2)`` transform for logarithm benchmarks."""
    theta = xi[0]
    sinc = jnp.sin(theta) / theta
    cosc = (1.0 - jnp.cos(theta)) / theta**2
    p = jnp.array(
        [
            sinc * xi[1] - theta * cosc * xi[2],
            theta * cosc * xi[1] + sinc * xi[2],
        ]
    )
    return jnp.array(
        [
            [jnp.cos(theta), -jnp.sin(theta), p[0]],
            [jnp.sin(theta), jnp.cos(theta), p[1]],
            [0.0, 0.0, 1.0],
        ]
    )


def _se3_transform(xi: Array) -> Array:
    """Construct a regular ``SE(3)`` transform for logarithm benchmarks."""
    omega = xi[:3]
    theta = jnp.linalg.norm(omega)
    omega_hat = so3.skew(omega)
    omega_hat_sq = omega_hat @ omega_hat
    V = (
        jnp.eye(3)
        + ((1.0 - jnp.cos(theta)) / theta**2) * omega_hat
        + ((theta - jnp.sin(theta)) / theta**3) * omega_hat_sq
    )
    R = _rotation_matrix(omega)
    p = V @ xi[3:]
    return jnp.block([[R, p[:, None]], [jnp.zeros((1, 3)), jnp.ones((1, 1))]])


def _weighted_sum(value: Array) -> Array:
    """Reduce an array without making rotation objectives constant."""
    weights = jnp.linspace(0.5, 1.5, value.size, dtype=value.dtype).reshape(value.shape)
    return jnp.sum(weights * value)


def _cases() -> tuple[BenchmarkCase, ...]:
    eps = jnp.asarray(1e-10)
    s = jnp.asarray(0.2)
    omega_regular = jnp.array([0.2, -0.3, 0.1])
    omega_tiny = jnp.array([1e-10, -2e-10, 3e-10])
    xi2_regular = jnp.array([0.3, 0.2, -0.1])
    xi2_tiny = jnp.array([1e-10, 0.2, -0.1])
    xi3_regular = jnp.concatenate([omega_regular, jnp.array([0.2, -0.1, 0.3])])
    xi3_tiny = jnp.concatenate([omega_tiny, jnp.array([0.2, -0.1, 0.3])])
    xid2 = jnp.array([-0.1, 0.3, -0.2])
    xid3 = jnp.array([-0.1, 0.2, 0.05, 0.3, -0.2, 0.1])
    R_regular = _rotation_matrix(omega_regular)
    R_tiny = jnp.eye(3) + so3.skew(omega_tiny)
    g2_regular = _se2_transform(xi2_regular)
    g2_tiny = jnp.array([[1.0, -1e-10, 0.2], [1e-10, 1.0, -0.1], [0.0, 0.0, 1.0]])
    g3_regular = _se3_transform(xi3_regular)
    g3_tiny = jnp.block(
        [
            [R_tiny, xi3_tiny[3:, None]],
            [jnp.zeros((1, 3)), jnp.ones((1, 1))],
        ]
    )
    constant_se2_adjoint = _constant_strain_function("se2", "adjoint")
    constant_se2_tangent = _constant_strain_function("se2", "tangent")
    constant_se2_tangent_derivative = _constant_strain_function(
        "se2", "tangent_derivative"
    )
    constant_se3_adjoint = _constant_strain_function("se3", "adjoint")
    constant_se3_tangent = _constant_strain_function("se3", "tangent")
    constant_se3_tangent_derivative = _constant_strain_function(
        "se3", "tangent_derivative"
    )

    cases: list[BenchmarkCase] = []

    def add(
        family: str,
        operation: str,
        scenario: str,
        fn: Callable[..., Tree],
        *args: Any,
    ) -> None:
        cases.append(
            BenchmarkCase(
                name=f"{family}.{operation}.{scenario}",
                family=family,
                operation=operation,
                scenario=scenario,
                fn=fn,
                args=args,
            )
        )

    for scenario, omega, rotation in (
        ("regular", omega_regular, R_regular),
        ("tiny", omega_tiny, R_tiny),
    ):
        add("so3", "exp", scenario, so3.exp, omega, eps)
        add("so3", "log", scenario, so3.log, rotation, eps)
        add("so3", "exp_jacrev", scenario, jax.jacrev(so3.exp), omega, eps)
        add(
            "so3",
            "exp_hessian",
            scenario,
            jax.hessian(
                lambda value, threshold: _weighted_sum(so3.exp(value, threshold))
            ),
            omega,
            eps,
        )

    for scenario, xi, transform in (
        ("regular", xi2_regular, g2_regular),
        ("tiny", xi2_tiny, g2_tiny),
    ):
        add("se2", "exp", scenario, se2.exp, xi, eps)
        add("se2", "log", scenario, se2.log, transform, eps)
        add("se2", "exp_jacrev", scenario, jax.jacrev(se2.exp), xi, eps)
        add(
            "se2",
            "exp_hessian",
            scenario,
            jax.hessian(
                lambda value, threshold: _weighted_sum(se2.exp(value, threshold))
            ),
            xi,
            eps,
        )
        add("constant_se2", "adjoint", scenario, constant_se2_adjoint, xi, s, eps)
        add("constant_se2", "tangent", scenario, constant_se2_tangent, xi, s, eps)
        add(
            "constant_se2",
            "tangent_derivative",
            scenario,
            constant_se2_tangent_derivative,
            xi,
            xid2,
            s,
            eps,
        )
        add(
            "constant_se2",
            "tangent_hessian",
            scenario,
            jax.hessian(
                lambda value, arc, threshold: _weighted_sum(
                    constant_se2_tangent(value, arc, threshold)
                )
            ),
            xi,
            s,
            eps,
        )

    for scenario, xi, transform in (
        ("regular", xi3_regular, g3_regular),
        ("tiny", xi3_tiny, g3_tiny),
    ):
        add("se3", "exp", scenario, se3.exp, xi, eps)
        add("se3", "log", scenario, se3.log, transform, eps)
        add("se3", "exp_jacrev", scenario, jax.jacrev(se3.exp), xi, eps)
        add(
            "se3",
            "exp_hessian",
            scenario,
            jax.hessian(
                lambda value, threshold: _weighted_sum(se3.exp(value, threshold))
            ),
            xi,
            eps,
        )
        add("constant_se3", "adjoint", scenario, constant_se3_adjoint, xi, s, eps)
        add("constant_se3", "tangent", scenario, constant_se3_tangent, xi, s, eps)
        add(
            "constant_se3",
            "tangent_derivative",
            scenario,
            constant_se3_tangent_derivative,
            xi,
            xid3,
            s,
            eps,
        )
        add(
            "constant_se3",
            "tangent_hessian",
            scenario,
            jax.hessian(
                lambda value, arc, threshold: _weighted_sum(
                    constant_se3_tangent(value, arc, threshold)
                )
            ),
            xi,
            s,
            eps,
        )

    return tuple(cases)


def _measure(
    case: BenchmarkCase, execution_repeats: int, warmup_runs: int
) -> dict[str, Any]:
    """Compile and measure one case with synchronized executions."""
    jitted = jax.jit(lambda *args: case.fn(*args))
    start = time.perf_counter_ns()
    result = jitted(*case.args)
    _block_until_ready(result)
    first_ns = time.perf_counter_ns() - start

    for _ in range(warmup_runs):
        result = jitted(*case.args)
        _block_until_ready(result)

    samples_ns: list[int] = []
    for _ in range(execution_repeats):
        start = time.perf_counter_ns()
        result = jitted(*case.args)
        _block_until_ready(result)
        samples_ns.append(time.perf_counter_ns() - start)

    quartiles = statistics.quantiles(samples_ns, n=4, method="inclusive")
    median_ns = statistics.median(samples_ns)
    return {
        "name": case.name,
        "family": case.family,
        "operation": case.operation,
        "scenario": case.scenario,
        "jit_compile_time_s": max(0.0, (first_ns - median_ns) * 1e-9),
        "jit_execution_median_s": median_ns * 1e-9,
        "jit_execution_q1_s": quartiles[0] * 1e-9,
        "jit_execution_q3_s": quartiles[2] * 1e-9,
        "finite": _tree_is_finite(result),
        "execution_repeats": execution_repeats,
        "warmup_runs": warmup_runs,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Lie-algebra values and differentiated call paths."
    )
    parser.add_argument("--label", default="unlabeled", help="Revision label.")
    parser.add_argument(
        "--execution-repeats",
        type=int,
        default=101,
        help="Synchronized samples per case.",
    )
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run all benchmark cases and optionally write JSON/CSV artifacts."""
    args = _parser().parse_args(argv)
    if args.execution_repeats < 4:
        raise ValueError("execution repeats must be at least 4")
    if args.warmup_runs < 0:
        raise ValueError("warmup runs must be non-negative")

    rows: list[dict[str, Any]] = []
    for case in _cases():
        print(f"{case.name:<44}", end="", flush=True)
        row = _measure(case, args.execution_repeats, args.warmup_runs)
        row["label"] = args.label
        rows.append(row)
        print(
            f" compile={1e3 * row['jit_compile_time_s']:8.3f} ms"
            f" exec={1e6 * row['jit_execution_median_s']:8.3f} us"
            f" finite={row['finite']}"
        )

    output = {
        "metadata": {
            "label": args.label,
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "platform": platform.platform(),
            "device": str(jax.devices()[0]),
        },
        "results": rows,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(output, indent=2) + "\n")
        print(f"Wrote {args.json}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
