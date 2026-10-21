"""Compare diagonal/full GVS chain assembly on a warmed CUDA graph.

Run this same script against the base and candidate checkouts, using PYTHONPATH
or their editable environments. Timings exclude model construction, JAX local
terms, allocation and compilation. Each sample synchronizes after graph replay.
Example: python tools/benchmarks/gvs_inertia.py --output /tmp/inertia.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

from soromox.execution.warp.gvs import chain, floating_chain
from soromox.execution.warp.gvs.executor import _local_dynamics_terms
from soromox.execution.warp.gvs.operands import (
    GVSFloatingOperands,
    GVSFloatingPipelineShapes,
    GVSOperands,
    GVSPipelineShapes,
)
from soromox.systems import GVS, GVSSegment, JointSpec, LinkSpec, StrainBasisSpec
from soromox.systems.components.inertia import spatial_inertia


def make_model(floating: bool) -> GVS:
    """Build three four-strain links with one internal revolute connector."""
    return GVS.from_segments(
        tuple(
            GVSSegment(
                link=LinkSpec.circular(
                    length=0.4 / 3,
                    radius=0.03,
                    density=1220,
                    reference_strain=[0, 0, 0, 1, 0, 0],
                    young_modulus=2e5,
                    shear_modulus=8e4,
                ),
                joint=JointSpec.revolute(axis="x") if i == 1 else JointSpec.fixed(),
                basis=StrainBasisSpec(
                    type="legendre",
                    strain_selector=[True, True, True, True, False, False],
                    basis_order=[1] * 6,
                ),
                num_gauss_points=5,
            )
            for i in range(3)
        ),
        floating_base=floating,
        scale_rotational_basis_by_length=True,
        gravity=jnp.array([0.0, 0.0, -9.81]),
    )


def make_data(model: GVS, worlds: int, block: int, device: wp.Device):
    """Prepare homogeneous chain inputs and retain every referenced allocation."""
    floating = model.floating_base
    op = (GVSFloatingOperands if floating else GVSOperands).from_model(
        model, block_dim=block
    )
    q = jnp.linspace(-0.003, 0.004, model.num_internal_dofs)[None]
    qd = q * 3
    local = _local_dynamics_terms(op, q, qd)
    names = (
        "joint_adjoint",
        "joint_adjoint_dot",
        "joint_tangent",
        "joint_tangent_dot_qd",
        "joint_velocity",
        "cell_adjoint",
        "cell_tangent_local",
        "cell_link_velocity",
        "cell_step_velocity",
        "cell_tangent_velocity_dot",
    )
    widths = (6, 6, model.num_internal_dofs, 1, 1, 6, op.max_dof, 1, 1, 1)
    values = {
        name: np.tile(np.asarray(value).reshape(-1, width), (worlds, 1))
        for name, value, width in zip(names, local, widths, strict=True)
    }
    values.update(
        global_to_local=op.link_global_to_local,
        active_dofs=op.active_dofs_per_segment,
        inertia_upper_rows=op.inertia_upper_rows,
        inertia_upper_columns=op.inertia_upper_columns,
        weighted_masses=np.asarray(op.weighted_inertias).reshape(
            -1, op.weighted_inertias.shape[-1]
        ),
        num_cells_array=np.array([op.num_cells], np.int32),
        num_quadrature_array=np.array([op.num_quadrature], np.int32),
        lanes_per_block=np.array([block], np.int32),
    )
    if floating:
        pose = np.array([0.1, -0.2, 0.05, 0.97, 0.02, -0.01, 0.03])
        pose[:4] /= np.linalg.norm(pose[:4])
        values.update(
            base_pose=np.tile(pose[None], (worlds, 1)),
            velocity=np.tile(
                np.concatenate((np.full((1, 6), 0.02), qd), axis=1), (worlds, 1)
            ),
            gravity_world=op.gravity_world,
        )
        data = floating_chain.GVSFloatingChainData()
        kernel = floating_chain.homogeneous_floating_persistent_chain_kernel
        shapes = GVSFloatingPipelineShapes.from_operands(op, batch_size=worlds)
    else:
        values.update(qd=np.tile(qd, (worlds, 1)), gravity_base=op.gravity_base)
        data = chain.GVSChainData()
        kernel = chain.homogeneous_persistent_chain_kernel
        shapes = GVSPipelineShapes.from_operands(op, batch_size=worlds)
    owners = {}
    for name, value in values.items():
        array = np.asarray(value)
        dtype = wp.int32 if np.issubdtype(array.dtype, np.integer) else wp.float64
        owners[name] = wp.array(array, dtype=dtype, device=device)
        setattr(data, name, owners[name])
    for name, shape in shapes.chain_outputs().items():
        owners[name] = wp.zeros(shape, dtype=wp.float64, device=device)
        setattr(data, name, owners[name])
    return data, kernel, owners


def main() -> None:
    """Report median batch latency for interleaved inertia layouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", nargs="+", type=int, default=[1, 64, 1024])
    parser.add_argument("--block-dim", type=int, default=64)
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.repeats < 20
        or args.repeats % 20
        or args.samples < 1
        or min(args.worlds) < 1
    ):
        parser.error(
            "positive world/sample counts and repeats divisible by 20 are required"
        )
    if args.block_dim < 32 or args.block_dim > 1024 or args.block_dim % 32:
        parser.error("block-dim must be a multiple of 32 between 32 and 1024")
    jax.config.update("jax_enable_x64", True)
    wp.init()
    device = wp.get_device("cuda:0")
    robot = make_model(not args.fixed)
    # Identical sectional physics for diagonal and full/zero-tip storage.
    sections = robot.inner_mass_matrices
    full = robot.update_link_params(
        sectional_inertia=sections, tip_inertia=jnp.zeros((3, 6, 6))
    )
    tip = spatial_inertia(
        jnp.array([0.0, 0.0, 0.04]),
        jnp.tile(jnp.array([0.01, -0.012, 0.003]), (3, 1)),
        jnp.zeros((3, 3, 3)).at[2].set(jnp.diag(jnp.array([1e-5, 2e-5, 3e-5]))),
    )
    models = {
        "diagonal": robot,
        "full_zero_tip": full,
        "full_nonzero_tip": full.update_link_params(tip_inertia=tip),
    }
    result = {
        "device": device.name,
        "warp": wp.__version__,
        "floating": not args.fixed,
        "dofs": robot.num_velocities,
        "block_dim": args.block_dim,
        "samples": args.samples,
        "evaluations_per_sample": args.repeats,
        "jax": jax.__version__,
        "timings": [],
    }
    for worlds in args.worlds:
        graphs, owners, outputs = {}, [], {}
        for name, model in models.items():
            data, kernel, allocations = make_data(model, worlds, args.block_dim, device)
            owners.append((data, allocations))

            def launch(kernel=kernel, worlds=worlds, data=data) -> None:
                wp.launch_tiled(
                    kernel,
                    dim=worlds,
                    inputs=[data],
                    block_dim=args.block_dim,
                    device=device,
                )

            launch()
            wp.synchronize_device(device)
            outputs[name] = [
                getattr(data, key).numpy()[0]
                for key in ("inertia", "coriolis_qd", "gravity_force")
            ]
            with wp.ScopedCapture(device=device) as capture:
                for _ in range(20):
                    launch()
            graphs[name] = capture.graph
        for actual, expected in zip(
            outputs["full_zero_tip"], outputs["diagonal"], strict=True
        ):
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
        for graph in graphs.values():
            wp.capture_launch(graph)
        wp.synchronize_device(device)
        timings = {name: [] for name in graphs}
        for sample in range(args.samples):
            for name in list(graphs) if sample % 2 == 0 else list(graphs)[::-1]:
                wp.synchronize_device(device)
                start = time.perf_counter()
                for _ in range(args.repeats // 20):
                    wp.capture_launch(graphs[name])
                wp.synchronize_device(device)
                timings[name].append((time.perf_counter() - start) * 1e6 / args.repeats)
        for name, samples in timings.items():
            row = {
                "worlds": worlds,
                "layout": name,
                "median_us": float(np.median(samples)),
                "samples_us": samples,
            }
            result["timings"].append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
