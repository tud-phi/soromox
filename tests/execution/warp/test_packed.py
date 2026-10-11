"""Native packed indexing and cooperative solve numerical contracts."""

import os
from typing import Any

import numpy as np
import pytest
import warp as wp

from soromox.execution.warp.gvs.cell import (
    GVSCellTermsData,
    cell_terms_kernel,
    packed_cell_terms_kernel,
    packed_cooperative_cell_terms_kernel,
)


def device() -> Any:
    """Select the explicitly requested test device.

    Returns:
        CUDA device when SOROMOX_TEST_CUDA=1, otherwise CPU.
    """
    return wp.get_device(
        "cuda:0" if os.environ.get("SOROMOX_TEST_CUDA") == "1" else "cpu"
    )


def test_packed_cells() -> None:
    """Different segment/cell/basis dimensions retain scalar cell arithmetic."""
    rng = np.random.default_rng(847)
    fields = (
        "q_link",
        "qd_link",
        "basis_z1",
        "basis_z2",
        "basis_rows",
        "reference_z1",
        "reference_z2",
        "segment_lengths",
        "cell_widths",
        "num_cells",
        "order_zero",
        "adjoint",
        "tangent_local",
        "link_velocity",
        "step_velocity",
        "tangent_velocity_dot",
    )
    descriptors = []
    references = []
    work = []
    for w, (segments, cells, width) in enumerate([(1, 3, 3), (2, 5, 7), (3, 4, 18)]):
        nc = segments * cells
        arrays = [
            rng.normal(0, 0.01, (segments, width)),
            rng.normal(0, 0.02, (segments, width)),
            rng.normal(0, 0.1, (nc, width)),
            rng.normal(0, 0.1, (nc, width)),
            np.broadcast_to(np.arange(width) % 6, (segments, width)).astype(np.int32),
            np.tile([0.0, 0, 0, 1, 0, 0], (nc, 1)),
            np.tile([0.0, 0, 0, 1, 0, 0], (nc, 1)),
            np.full(segments, 0.12),
            np.full(nc, 1 / cells),
            np.array([cells], np.int32),
            np.array([0], np.int32),
            np.zeros((nc * 6, 6)),
            np.zeros((nc * 6, width)),
            np.zeros((nc * 6, 1)),
            np.zeros((nc * 6, 1)),
            np.zeros((nc * 6, 1)),
        ]
        arrays = [
            wp.array(
                a,
                dtype=wp.int32 if a.dtype == np.int32 else wp.float64,
                device=device(),
            )
            for a in arrays
        ]
        wp.launch(cell_terms_kernel, dim=nc, inputs=arrays, device=device())
        references.append([a.numpy() for a in arrays[11:]])
        d = GVSCellTermsData()
        for name, a in zip(fields, arrays, strict=True):
            setattr(d, name, a)
        descriptors.append(d)
        work.extend((w, c, 0) for c in range(nc))
    ds = wp.array(descriptors, dtype=GVSCellTermsData, device=device())
    work = wp.array(work, dtype=wp.vec3i, device=device())
    for kernel, dim in [
        (packed_cell_terms_kernel, work.size),
        (packed_cooperative_cell_terms_kernel, (work.size, 2)),
    ]:
        for d in descriptors:
            for name in fields[11:]:
                getattr(d, name).fill_(np.nan)
        wp.launch(kernel, dim=dim, inputs=[ds, work], block_dim=128, device=device())
        for d, expected in zip(descriptors, references, strict=True):
            for name, ref in zip(fields[11:], expected, strict=True):
                np.testing.assert_array_equal(getattr(d, name).numpy(), ref)


@pytest.mark.skipif(
    os.environ.get("SOROMOX_TEST_CUDA") != "1", reason="opt-in GPU factorization"
)
@pytest.mark.parametrize("n", [1, 8, 37, 65, 144, 288])
@pytest.mark.parametrize("column_factor", [False, True])
def test_cooperative_cholesky(n: int, column_factor: bool) -> None:
    """Check Cholesky accuracy with row-major and transposed factor storage.

    Args:
        n: Linear-system dimension.
        column_factor: Transpose the factor workspace when true.

    Returns:
        None. Asserts solution accuracy, residuals and unfloored pivots.
    """
    from soromox.execution.warp.common.linear_solve import cooperative_cholesky_kernel

    rng = np.random.default_rng(n)
    a = rng.normal(size=(3, n, n))
    m = a @ np.swapaxes(a, 1, 2) + np.eye(n)[None] * 0.2
    b = rng.normal(size=(3, n))
    matrix = wp.array(m, dtype=wp.float64, device=device())
    rhs = wp.array(b, dtype=wp.float64, device=device())
    factor = wp.zeros_like(matrix)
    if column_factor:
        factor = factor.transpose((0, 2, 1))
    x, workspace = wp.zeros_like(rhs), wp.zeros_like(rhs)
    status = wp.zeros(3, dtype=wp.int32, device=device())
    wp.launch_tiled(
        cooperative_cholesky_kernel,
        dim=3,
        inputs=[matrix, rhs, wp.float64(1e-12)],
        outputs=[factor, workspace, x, status],
        block_dim=32,
        device=device(),
    )
    np.testing.assert_allclose(
        x.numpy(), np.linalg.solve(m, b[..., None])[..., 0], rtol=2e-11, atol=2e-11
    )
    assert not status.numpy().any()
    np.testing.assert_allclose(
        m @ x.numpy()[..., None], b[..., None], rtol=3e-11, atol=3e-11
    )


@pytest.mark.parametrize("floating", [False, True])
@pytest.mark.parametrize("moving", [False, True])
def test_prepared_fixed_joint_terms(floating: bool, moving: bool) -> None:
    """Check cached and uncached joint evaluation across model variants.

    Args:
        floating: Use a floating base when true.
        moving: Include a prismatic joint when true.

    Returns:
        None. Asserts bitwise equality of all dynamics terms.
    """
    import jax
    import jax.numpy as jnp

    from soromox.execution.warp.gvs.executor import execute_dynamics_terms
    from soromox.execution.warp.gvs.operands import GVSFloatingOperands, GVSOperands
    from soromox.execution.warp.gvs.prepared import prepare_dynamics_terms
    from soromox.systems import GVS, GVSSegment, JointSpec, LinkSpec, StrainBasisSpec

    r = GVS.from_segments(
        (
            GVSSegment(
                link=LinkSpec.circular(
                    length=0.12,
                    radius=0.015,
                    density=1000,
                    reference_strain=[0, 0, 0, 1, 0, 0],
                    young_modulus=2e5,
                    shear_modulus=8e4,
                ),
                joint=JointSpec.prismatic(axis="x") if moving else JointSpec.fixed(),
                basis=StrainBasisSpec(
                    type="legendre", strain_selector=[True] * 6, basis_order=[2] * 6
                ),
                num_gauss_points=5,
            ),
        ),
        backend="warp",
        floating_base=floating,
    )
    operand_type = GVSFloatingOperands if floating else GVSOperands
    op = operand_type.from_model(r, block_dim=1 if device().is_cpu else 64)
    evaluate = jax.jit(prepare_dynamics_terms(op))
    reference = jax.jit(lambda q, v: execute_dynamics_terms(op, q, v))
    for worlds, scale in [(1, 0.001), (3, 0.002)]:
        q = jnp.full((worlds, r.num_coordinates), scale)
        if floating:
            q = q.at[:, :4].set(jnp.array([0.0, 0.0, 0.0, 1.0]))
        v = jnp.full((worlds, r.num_velocities), scale * 0.3)
        for x, y in zip(evaluate(q, v), reference(q, v), strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


@pytest.mark.skipif(
    os.environ.get("SOROMOX_TEST_CUDA") != "1", reason="opt-in packed GPU factorization"
)
def test_packed_cholesky_and_pivot_status() -> None:
    """Check different solve dimensions and report floored pivots.

    Returns:
        None. Asserts packed solve results and per-world pivot status.
    """
    from soromox.execution.warp.common.linear_solve import (
        CholeskySolveData,
        packed_cholesky_kernel,
    )

    descriptors = []
    for n, pivot in [(3, 0.2), (9, 0.4), (7, -0.1)]:
        d = CholeskySolveData()
        matrix = np.eye(n)[None] * 2.0
        matrix[0, -1, -1] = pivot
        d.matrix = wp.array(matrix, dtype=wp.float64, device=device())
        d.rhs = wp.ones((1, n), dtype=wp.float64, device=device())
        d.factor = wp.empty_like(d.matrix)
        d.workspace = wp.empty_like(d.rhs)
        d.solution = wp.empty_like(d.rhs)
        d.status = wp.zeros(1, dtype=wp.int32, device=device())
        descriptors.append(d)
    ds = wp.array(descriptors, dtype=CholeskySolveData, device=device())
    work = wp.array([(i, 0, 0) for i in range(3)], dtype=wp.vec3i, device=device())
    wp.launch_tiled(
        packed_cholesky_kernel,
        dim=3,
        inputs=[ds, work, wp.float64(1e-12)],
        block_dim=32,
        device=device(),
    )
    for i, d in enumerate(descriptors):
        assert d.status.numpy()[0] == int(i == 2)
        if i < 2:
            np.testing.assert_allclose(
                d.solution.numpy(), 1 / np.diag(d.matrix.numpy()[0])[None], rtol=3e-15
            )
        else:
            np.testing.assert_allclose(d.solution.numpy()[0, -1], 1e12, rtol=3e-15)
