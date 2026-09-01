#!/usr/bin/env python3
"""Compare PCS forward-dynamics derivatives with custom JVPs on and off.

The benchmark separates the first-call compilation cost from steady-state
execution time. Every timed call is synchronized with the active JAX device.
Custom-JVP and pure-autodiff outputs are also compared for correctness.

For ``state_jvp``, the disabled path is JAX's direct JVP and the enabled path
is the analytical directional recurrence. For ``state_jacobian_matvec``, both
paths first materialize a full state Jacobian and then multiply it by the same
direction: JAX ``jacfwd`` when disabled and the analytical PCS Jacobian when
enabled. Each workload returns the primal and tangent so their timed work and
correctness checks are directly comparable.

Rollout workloads apply a constant actuation over a short fixed-step trajectory
and return the final generalized position ``q(T)``. They benchmark either one
directional JVP or the full Jacobian with respect to the actuation ``u``, the
initial generalized position ``q(0)``, or the per-segment parameter vector
``[length, density, Young's modulus]``.

Example:
    python tools/benchmarks/pcs_custom_jvp_speed_test.py \
        --segment-counts 1 2 4 8 16 32 \
        --setups full bending_extension axial threadlike \
        --workloads state_jvp state_jacobian_matvec state_jacobian \
            rollout_actuation_jvp rollout_actuation_jacobian \
            rollout_initial_position_jvp rollout_initial_position_jacobian \
            rollout_parameters_jvp rollout_parameters_jacobian \
        --repeats 10
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting  # noqa: E402
from soromox.autodiff import set_custom_jvp_enabled  # noqa: E402
from soromox.systems import (  # noqa: E402
    PCS,
    IsotropicMaterialParams,
    LinkSpec,
    PCSStructure,
)

Array = jax.Array
Tree = Any
Workload = Callable[[Array, Array, Array, Array], Tree]

LINK_LENGTH = 0.1
LINK_DENSITY = 1070.0
YOUNG_MODULUS = 2.0e3
SHEAR_MODULUS = 1.0e3
MATERIAL_DAMPING_COEFFICIENT = 1.0e-3
DEFAULT_SEGMENT_COUNTS = (1, 2, 4, 8, 16, 32)

SETUPS = ("full", "bending_extension", "axial", "threadlike")
WORKLOADS = (
    "primal",
    "state_jvp",
    "state_jacobian_matvec",
    "state_jacobian",
    "all_jacobians",
    "rollout_actuation_jvp",
    "rollout_actuation_jacobian",
    "rollout_initial_position_jvp",
    "rollout_initial_position_jacobian",
    "rollout_parameters_jvp",
    "rollout_parameters_jacobian",
)


class BenchmarkInputs(NamedTuple):
    """Dynamic inputs and directions shared by the benchmark workloads."""

    y: Array
    y_direction: Array
    u: Array
    u_direction: Array
    tau_ext: Array
    system_parameters: Array
    system_parameter_direction: Array


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
                length=LINK_LENGTH,
                radius=0.02,
                density=LINK_DENSITY,
                young_modulus=YOUNG_MODULUS,
                shear_modulus=SHEAR_MODULUS,
                material_damping_coefficient=MATERIAL_DAMPING_COEFFICIENT,
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


def build_inputs(model: PCS) -> BenchmarkInputs:
    """Return dynamic benchmark inputs and directional-JVP tangents."""
    q = jnp.linspace(-0.025, 0.035, model.num_dofs)
    qd = jnp.linspace(0.04, -0.03, model.num_dofs)
    y = jnp.concatenate([q, qd])
    y_direction = jnp.linspace(-0.2, 0.25, 2 * model.num_dofs)
    u = jnp.linspace(0.05, 0.1, model.num_actuators)
    u_direction = jnp.linspace(-0.15, 0.2, model.num_actuators)
    tau_ext = jnp.linspace(-0.01, 0.015, model.num_dofs)
    system_parameters = jnp.concatenate(
        [
            model.L,
            model.rho,
            jnp.full((model.num_segments,), YOUNG_MODULUS, dtype=y.dtype),
        ]
    )
    system_parameter_direction = 0.01 * system_parameters
    return BenchmarkInputs(
        y=y,
        y_direction=y_direction,
        u=u,
        u_direction=u_direction,
        tau_ext=tau_ext,
        system_parameters=system_parameters,
        system_parameter_direction=system_parameter_direction,
    )


def model_with_system_parameters(model: PCS, parameters: Array) -> PCS:
    """Update per-segment length, density, and Young's modulus."""
    length, density, young_modulus = jnp.split(parameters, 3)
    updated = model.update_link_params(length=length, density=density)
    material = IsotropicMaterialParams(
        young_modulus=young_modulus,
        shear_modulus=jnp.full_like(young_modulus, SHEAR_MODULUS),
        material_damping_coefficient=jnp.full_like(
            young_modulus,
            MATERIAL_DAMPING_COEFFICIENT,
        ),
    )
    return updated.with_isotropic_material(material)


def rollout_final_position(
    model: PCS,
    y: Array,
    u: Array,
    tau_ext: Array,
    *,
    rollout_steps: int,
    rollout_dt: float,
    use_custom_jvp: bool,
) -> Array:
    """Return ``q(T)`` from a short fixed-step symplectic-Euler rollout."""
    dt = jnp.asarray(rollout_dt, dtype=y.dtype)

    def step(y_value: Array, step_index: Array) -> tuple[Array, None]:
        t = dt * step_index
        if use_custom_jvp:
            yd = model.forward_dynamics(t, y_value, (u, tau_ext))
        else:
            yd = model._forward_dynamics(t, y_value, (u, tau_ext))
        q, qd = jnp.split(y_value, 2)
        _, qdd = jnp.split(yd, 2)
        qd_next = qd + dt * qdd
        q_next = q + dt * qd_next
        return jnp.concatenate([q_next, qd_next]), None

    final_y, _ = jax.lax.scan(
        step,
        y,
        jnp.arange(rollout_steps, dtype=jnp.int32),
    )
    final_q, _ = jnp.split(final_y, 2)
    return final_q


def build_workload(
    model: PCS,
    workload: str,
    y_direction: Array,
    u_direction: Array,
    system_parameter_direction: Array,
    *,
    rollout_steps: int,
    rollout_dt: float,
    use_custom_jvp: bool,
) -> Workload:
    """Create a custom-JVP or direct-JAX-autodiff derivative workload.

    The reference path calls the protected primal directly. Consequently its
    derivatives always come from JAX autodiff and cannot accidentally capture
    the global custom-JVP toggle at trace time.
    """
    t = jnp.array(0.0)

    def dynamics(
        y: Array,
        u: Array,
        tau_ext: Array,
        system_parameters: Array,
    ) -> Array:
        del system_parameters
        if use_custom_jvp:
            return model.forward_dynamics(t, y, (u, tau_ext))
        return model._forward_dynamics(t, y, (u, tau_ext))

    def rollout(
        rollout_model: PCS,
        y: Array,
        u: Array,
        tau_ext: Array,
    ) -> Array:
        return rollout_final_position(
            rollout_model,
            y,
            u,
            tau_ext,
            rollout_steps=rollout_steps,
            rollout_dt=rollout_dt,
            use_custom_jvp=use_custom_jvp,
        )

    if workload == "primal":
        return dynamics
    if workload == "state_jvp":
        return lambda y, u, tau_ext, parameters: jax.jvp(
            lambda y_value: dynamics(y_value, u, tau_ext, parameters),
            (y,),
            (y_direction,),
        )
    if workload == "state_jacobian_matvec":
        if use_custom_jvp:
            return lambda y, u, tau_ext, parameters: (
                dynamics(y, u, tau_ext, parameters),
                model.forward_dynamics_state_jacobian(
                    t,
                    y,
                    (u, tau_ext),
                )
                @ y_direction,
            )
        return lambda y, u, tau_ext, parameters: (
            dynamics(y, u, tau_ext, parameters),
            jax.jacfwd(lambda y_value: dynamics(y_value, u, tau_ext, parameters))(y)
            @ y_direction,
        )
    if workload == "state_jacobian":
        return lambda y, u, tau_ext, parameters: jax.jacfwd(
            lambda y_value: dynamics(y_value, u, tau_ext, parameters)
        )(y)
    if workload == "all_jacobians":
        return lambda y, u, tau_ext, parameters: jax.jacfwd(
            dynamics,
            argnums=(0, 1, 2),
        )(y, u, tau_ext, parameters)
    if workload == "rollout_actuation_jvp":
        return lambda y, u, tau_ext, parameters: jax.jvp(
            lambda u_value: rollout(model, y, u_value, tau_ext),
            (u,),
            (u_direction,),
        )
    if workload == "rollout_actuation_jacobian":
        return lambda y, u, tau_ext, parameters: jax.jacfwd(
            lambda u_value: rollout(model, y, u_value, tau_ext)
        )(u)
    if workload == "rollout_initial_position_jvp":
        return lambda y, u, tau_ext, parameters: jax.jvp(
            lambda q_value: rollout(
                model,
                jnp.concatenate([q_value, y[model.num_dofs :]]),
                u,
                tau_ext,
            ),
            (y[: model.num_dofs],),
            (y_direction[: model.num_dofs],),
        )
    if workload == "rollout_initial_position_jacobian":
        return lambda y, u, tau_ext, parameters: jax.jacfwd(
            lambda q_value: rollout(
                model,
                jnp.concatenate([q_value, y[model.num_dofs :]]),
                u,
                tau_ext,
            )
        )(y[: model.num_dofs])
    if workload == "rollout_parameters_jvp":
        return lambda y, u, tau_ext, parameters: jax.jvp(
            lambda parameter_values: rollout(
                model_with_system_parameters(model, parameter_values),
                y,
                u,
                tau_ext,
            ),
            (parameters,),
            (system_parameter_direction,),
        )
    if workload == "rollout_parameters_jacobian":
        return lambda y, u, tau_ext, parameters: jax.jacfwd(
            lambda parameter_values: rollout(
                model_with_system_parameters(model, parameter_values),
                y,
                u,
                tau_ext,
            )
        )(parameters)
    raise ValueError(f"Unknown workload: {workload}")


def block_until_ready(tree: Tree) -> None:
    """Synchronize every array leaf in a result PyTree."""
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def measure(
    fn: Workload,
    args: tuple[Array, Array, Array, Array],
    warmup_runs: int,
    repeats: int,
    calls_per_sample: int,
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
        for _ in range(calls_per_sample):
            output = compiled(*args)
        block_until_ready(output)
        samples.append((time.perf_counter() - start) / calls_per_sample)

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
        f"x64={jax.config.x64_enabled} repeats={args.repeats} "
        f"calls_per_sample={args.calls_per_sample}"
    )
    print(
        "setup                 links dof workload                            "
        "off_ms    on_ms speedup compile_off compile_on  rel_error"
    )

    for setup in args.setups:
        for num_segments in args.segment_counts:
            model = build_model(setup, num_segments, args.gauss_points)
            benchmark_inputs = build_inputs(model)
            inputs = (
                benchmark_inputs.y,
                benchmark_inputs.u,
                benchmark_inputs.tau_ext,
                benchmark_inputs.system_parameters,
            )

            for workload in args.workloads:
                measurements: dict[bool, tuple[float, float, Tree]] = {}
                for enabled in (False, True):
                    set_custom_jvp_enabled(enabled)
                    jax.clear_caches()
                    workload_fn = build_workload(
                        model,
                        workload,
                        benchmark_inputs.y_direction,
                        benchmark_inputs.u_direction,
                        benchmark_inputs.system_parameter_direction,
                        rollout_steps=args.rollout_steps,
                        rollout_dt=args.rollout_dt,
                        use_custom_jvp=enabled,
                    )
                    measurements[enabled] = measure(
                        workload_fn,
                        inputs,
                        args.warmup_runs,
                        args.repeats,
                        args.calls_per_sample,
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
                    "rollout_steps": args.rollout_steps,
                    "rollout_dt": args.rollout_dt,
                    "calls_per_sample": args.calls_per_sample,
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
                    f"{workload:35s} {1e3 * execution_off:8.3f} "
                    f"{1e3 * execution_on:8.3f} {speedup:7.2f}x "
                    f"{compile_off:11.3f} {compile_on:10.3f} "
                    f"{max_relative:10.3e}"
                )

    set_custom_jvp_enabled(True)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line benchmark options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segment-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEGMENT_COUNTS),
    )
    parser.add_argument("--setups", nargs="+", choices=SETUPS, default=list(SETUPS))
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=WORKLOADS,
        default=["state_jvp", "state_jacobian_matvec", "state_jacobian"],
    )
    parser.add_argument("--gauss-points", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--rollout-dt", type=float, default=1.0e-4)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--calls-per-sample", type=int, default=1)
    parser.add_argument("--correctness-rtol", type=float, default=1.0e-5)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    if any(count < 1 for count in args.segment_counts):
        parser.error("--segment-counts values must be positive")
    if args.gauss_points < 1:
        parser.error("--gauss-points must be positive")
    if args.rollout_steps < 1:
        parser.error("--rollout-steps must be positive")
    if args.rollout_dt <= 0.0:
        parser.error("--rollout-dt must be positive")
    if args.warmup_runs < 0 or args.repeats < 1:
        parser.error("warmup runs must be non-negative and repeats must be positive")
    if args.calls_per_sample < 1:
        parser.error("--calls-per-sample must be positive")
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
