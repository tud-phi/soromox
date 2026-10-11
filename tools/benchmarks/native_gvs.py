"""Measure GVS Warp terms/forward dynamics and optional fixed-joint caching.

Use the same script with PYTHONPATH pointing at baseline and candidate sources.
Construction/compilation are excluded. Save outputs for cross-revision validation.
The cache trial is valid only for immutable, fixed-joint operands; it is not a
change to the public differentiable executor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import warp as wp

from soromox.execution.warp.gvs.operands import GVSOperands
from soromox.systems import GVS, GVSSegment, JointSpec, LinkSpec, StrainBasisSpec


def robot(links: int, order: int, all_strains: bool, moving: bool) -> GVS:
    """Build a fixed-base GVS model for the backend comparison.

    Args:
        links: Number of serial links.
        order: Polynomial order for each selected strain.
        all_strains: Select all six strains; otherwise select two bending strains.
        moving: Alternate revolute and prismatic joints when true.

    Returns:
        A GVS model with the Warp backend selected.
    """
    return GVS.from_segments(
        tuple(
            GVSSegment(
                link=LinkSpec.circular(
                    length=0.12 - 0.01 * min(i, 2),
                    radius=0.018,
                    reference_strain=[0, 0, 0, 1, 0, 0],
                    density=1040,
                    young_modulus=250000,
                    shear_modulus=90000,
                    material_damping_coefficient=0.02,
                ),
                joint=(
                    JointSpec.revolute(axis="y", stiffness=[[0.1]], damping=[[0.01]])
                    if moving and i % 2 == 0
                    else JointSpec.prismatic(
                        axis="x", stiffness=[[10.0]], damping=[[0.1]]
                    )
                    if moving
                    else JointSpec.fixed()
                ),
                basis=StrainBasisSpec(
                    type="legendre",
                    strain_selector=[True] * 6
                    if all_strains
                    else [False, True, True, False, False, False],
                    basis_order=[order] * 6,
                ),
                num_gauss_points=5 + 2 * (i % 2),
            )
            for i in range(links)
        ),
        backend="warp",
    )


def time_call(fn: Callable, args: tuple, repeats: int) -> dict[str, Any]:
    """Measure synchronized steady-state JAX execution time.

    Args:
        fn: Compiled function to warm up and measure.
        args: Positional function arguments.
        repeats: Number of timed groups of ten calls.

    Returns:
        Median and individual group durations per call in microseconds.
    """
    jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(10):
            out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - start) * 1e6 / 10)
    return {"us": float(np.median(samples)), "samples_us": samples}


def main() -> None:
    """Run the benchmark command described by the module.

    Returns:
        None. Writes the requested report and prints validation or timing results.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, nargs="+", default=[32, 128, 512])
    p.add_argument(
        "--cases", nargs="+", default=["small", "moving", "large8", "large16"]
    )
    p.add_argument("--cache-fixed-joints", action="store_true")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import soromox.execution.warp.gvs.cell as cell_module

    source = Path(cell_module.__file__).parents[1]
    report = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob("*.py")
        },
        "gpu": wp.get_device("cuda:0").name,
        "jax": jax.__version__,
        "warp": wp.__version__,
        "cache_trial": args.cache_fixed_joints,
        "cases": [],
    }
    for case in args.cases:
        r = robot(
            8 if case == "large8" else 16 if case == "large16" else 2,
            2 if case.startswith("large") else 1,
            case.startswith("large"),
            case == "moving",
        )
        for w in args.worlds:
            q = jnp.broadcast_to(
                jnp.linspace(-0.002, 0.002, r.num_coordinates), (w, r.num_coordinates)
            )
            v = jnp.broadcast_to(
                jnp.linspace(-0.003, 0.003, r.num_velocities), (w, r.num_velocities)
            )
            if args.cache_fixed_joints:
                from soromox.execution.warp.gvs.prepared import prepare_dynamics_terms

                op = GVSOperands.from_model(
                    r, block_dim=r.backend_params.warp_block_dim
                )
                terms = jax.jit(prepare_dynamics_terms(op))
            else:
                terms = jax.jit(
                    jax.vmap(lambda x, y, r=r: r.dynamics_terms(x, y, backend="warp"))
                )

            def forward_fn(
                q: jax.Array, v: jax.Array, terms: Callable = terms, r: GVS = r
            ) -> jax.Array:
                """Evaluate unforced forward dynamics for a batch of states.

                Args:
                    q: Batched generalized coordinates.
                    v: Batched generalized velocities.
                    terms: Dynamics-term callable captured for this benchmark case.
                    r: Robot supplying stiffness and damping.

                Returns:
                    Batched generalized accelerations from the JAX linear solve.
                """
                m, c, g = terms(q, v)
                rhs = (
                    -c
                    - g
                    - jnp.einsum("ij,bj->bi", r.K, q)
                    - jnp.einsum("ij,bj->bi", r.D_active, v)
                )
                return jax.vmap(r._solve_inertia)(m, rhs)

            forward = jax.jit(forward_fn)
            tt = time_call(terms, (q, v), args.repeats)
            ft = time_call(forward, (q, v), args.repeats)
            vals = terms(q, v)
            acceleration = forward(q, v)
            output = args.output.with_name(args.output.stem + f"-{case}-{w}.npz")
            np.savez(
                output,
                M=np.asarray(vals[0]),
                C=np.asarray(vals[1]),
                G=np.asarray(vals[2]),
                a=np.asarray(acceleration),
            )
            entry = {
                "case": case,
                "worlds": w,
                "dofs": r.num_velocities,
                "terms": tt,
                "forward": ft,
                "outputs": str(output),
            }
            report["cases"].append(entry)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(case, w, r.num_velocities, tt["us"], ft["us"], flush=True)
            jax.clear_caches()


if __name__ == "__main__":
    main()
