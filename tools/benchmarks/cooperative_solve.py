"""Compare an optional native SPD solve against JAX on actual GVS matrices.

Consumes NPZ outputs written by native_gvs.py. JAX includes its normal dispatch;
Warp graph timing measures native integration potential, not a JAX replacement.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import warp as wp

from soromox.execution.warp.common.linear_solve import cooperative_cholesky_kernel


def main() -> None:
    """Run the benchmark command described by the module.

    Returns:
        None. Writes the requested report and prints validation or timing results.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--column-factor", action="store_true")
    args = parser.parse_args()
    report = []
    for entry in json.loads(args.inputs.read_text())["cases"]:
        m = np.load(entry["outputs"])["M"]
        worlds, n, _ = m.shape
        rhs = np.broadcast_to(np.linspace(-0.1, 0.1, n), (worlds, n)).copy()
        matrix, b = [wp.array(a, dtype=wp.float64, device="cuda:0") for a in (m, rhs)]
        factor, workspace, x = wp.zeros_like(matrix), wp.zeros_like(b), wp.zeros_like(b)
        if args.column_factor:
            factor = factor.transpose((0, 2, 1))
        status = wp.zeros(worlds, dtype=wp.int32, device="cuda:0")

        def solve(
            matrix: wp.array = matrix,
            b: wp.array = b,
            factor: wp.array = factor,
            workspace: wp.array = workspace,
            x: wp.array = x,
            status: wp.array = status,
            worlds: int = worlds,
        ) -> None:
            """Launch the optional Cholesky solve for one benchmark batch.

            Args:
                matrix: Batched inertia matrices.
                b: Batched right-hand sides.
                factor: Caller-owned factor workspace.
                workspace: Caller-owned forward-substitution workspace.
                x: Caller-owned solution output.
                status: Caller-owned pivot-floor status output.
                worlds: Number of systems in the batch.

            Returns:
                None. Writes the solve outputs and workspaces in place.
            """
            wp.launch_tiled(
                cooperative_cholesky_kernel,
                dim=worlds,
                inputs=[matrix, b, wp.float64(1e-12)],
                outputs=[factor, workspace, x, status],
                block_dim=32,
                device="cuda:0",
            )

        solve()
        with wp.ScopedCapture(device="cuda:0") as capture:
            for _ in range(30):
                solve()
        wp.capture_launch(capture.graph)
        wp.synchronize_device("cuda:0")
        native = []
        for _ in range(5):
            start = time.perf_counter()
            wp.capture_launch(capture.graph)
            wp.synchronize_device("cuda:0")
            native.append((time.perf_counter() - start) * 1e6 / 30)
        fn = jax.jit(lambda a, b: jax.vmap(jnp.linalg.solve)(a, b))
        jm, jb = jnp.asarray(m), jnp.asarray(rhs)
        expected = np.asarray(fn(jm, jb))
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(30):
                out = fn(jm, jb)
            jax.block_until_ready(out)
            samples.append((time.perf_counter() - start) * 1e6 / 30)
        actual = x.numpy()
        assert not status.numpy().any()
        denominator = np.linalg.norm(m, ord=np.inf, axis=(1, 2)) * np.linalg.norm(
            actual, ord=np.inf, axis=1
        ) + np.linalg.norm(rhs, ord=np.inf, axis=1)
        residual = float(
            np.max(
                np.linalg.norm(
                    np.einsum("wij,wj->wi", m, actual) - rhs, ord=np.inf, axis=1
                )
                / denominator
            )
        )
        assert residual < 2e-14
        error = float(np.max(abs(actual - expected)) / np.max(abs(expected)))
        condition = float(np.linalg.cond(m[0]))
        assert error < 10 * condition * np.finfo(np.float64).eps
        result = {
            "case": entry["case"],
            "factor_layout": "column" if args.column_factor else "row",
            "worlds": worlds,
            "dofs": n,
            "native_us": float(np.median(native)),
            "jax_us": float(np.median(samples)),
            "native_samples_us": native,
            "jax_samples_us": samples,
            "relative_error": error,
            "normalized_residual": residual,
            "condition_number": condition,
        }
        report.append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(result, flush=True)


if __name__ == "__main__":
    main()
