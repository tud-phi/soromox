"""JAX/Warp equivalence tests for runtime-shaped planar PCS dynamics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import pytest
from numpy.testing import assert_allclose

from soromox.execution.warp.pcs.operands import (
    PCSKinematicsOperands,
    PCSKinematicsShapes,
    PCSOperands,
    PCSPipelineShapes,
)

from ._equivalence import assert_backend_equivalence
from ._kinematics_equivalence import (
    assert_inertial_jacobian_finite_difference,
    assert_kinematics_backend_equivalence,
    assert_warp_derivatives_use_jax,
)


def test_planar_pcs_operands_select_planar_dimension_data(
    make_planar_pcs_model: Callable[[str], Any],
) -> None:
    """Share PCS orchestration while retaining planar cached operands."""

    model = make_planar_pcs_model("jax")
    operands = PCSOperands.from_model(model)

    assert operands.is_planar is True
    assert operands.num_segments == model.num_segments
    assert operands.num_dofs == model.num_internal_dofs
    assert operands.num_gauss_points == model.num_gauss_points
    assert operands.block_dim == model.backend_params.warp_block_dim
    assert operands.active_strain_indices is model.active_strain_indices
    assert operands.reference_strain is model.xi_ref
    assert operands.weighted_mass_diagonals is model.weighted_mass_diagonals


def test_planar_pcs_pipeline_shapes_cover_workspace_and_results(
    make_planar_pcs_model: Callable[[str], Any],
) -> None:
    """Describe every external allocation needed by the planar PCS pipeline."""

    model = make_planar_pcs_model("jax")
    operands = PCSOperands.from_model(model)
    shapes = PCSPipelineShapes.from_operands(operands, batch_size=3)

    assert shapes.spatial_dim == 3
    assert shapes.operator_outputs()["adjoint_inverse"] == (
        3 * model.num_segments * (model.num_gauss_points + 1) * 3,
        3,
    )
    assert "velocity_first" not in shapes.chain_outputs()
    assert shapes.chain_outputs()["inertia"] == (
        3,
        model.num_internal_dofs,
        model.num_internal_dofs,
    )


def test_planar_pcs_kinematics_shapes_cover_reduced_and_fused_paths(
    make_planar_pcs_model: Callable[[str], Any],
) -> None:
    """Describe caller-owned PlanarPCS kinematics workspaces and outputs."""

    model = make_planar_pcs_model("jax")
    operands = PCSKinematicsOperands.from_model(model)
    shapes = PCSKinematicsShapes.from_operands(operands, batch_size=3, num_samples=7)

    assert shapes.pose_workspace() == {
        "segment_strain": (3, model.num_segments, 3),
        "segment_pose": (3, model.num_segments + 1, 3),
    }
    assert shapes.jacobian_workspace() == shapes.workspace()
    assert shapes.pose_output() == (3, 7, 3)
    assert shapes.jacobian_output() == (3, 7, 3, model.num_internal_dofs)


def test_planar_pcs_public_kinematics_match_jax_on_cpu_and_gpu(
    make_planar_pcs_model: Callable[[str], Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Match planar PCS kinematics for every public vectorization form."""

    pytest.importorskip("warp")
    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "planar-pcs-kinematics-cache"))
    model = make_planar_pcs_model("jax")

    assert_kinematics_backend_equivalence(model)
    assert_inertial_jacobian_finite_difference(model)
    assert_warp_derivatives_use_jax(model)


def test_planar_pcs_public_dynamics_apis_match_jax_on_gpu(
    make_planar_pcs_model: Callable[[str], Any],
    state_batch: Callable[[Any], tuple[Any, Any, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Match PlanarPCS terms and forward dynamics for every public input form."""

    if jax.default_backend() != "gpu":
        pytest.skip("PlanarPCS Warp equivalence requires a JAX GPU backend.")
    pytest.importorskip("warp")
    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "planar-pcs-warp-cache"))
    jax_model = make_planar_pcs_model("jax")
    warp_model = make_planar_pcs_model("warp")
    q, qd, y = state_batch(jax_model)

    assert_backend_equivalence(jax_model, warp_model, q, qd, y)


@pytest.mark.parametrize("num_gauss_points", [1, 3, 7, 9])
def test_planar_pcs_runtime_quadrature_counts_match_jax_on_gpu(
    num_gauss_points: int,
    make_planar_pcs_model: Callable[..., Any],
    state_batch: Callable[[Any], tuple[Any, Any, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Match public Warp dynamics for arbitrary positive quadrature counts."""

    if jax.default_backend() != "gpu":
        pytest.skip("PlanarPCS Warp equivalence requires a JAX GPU backend.")
    pytest.importorskip("warp")
    monkeypatch.setenv(
        "WARP_CACHE_PATH",
        str(tmp_path / f"planar-pcs-quadrature-{num_gauss_points}"),
    )
    jax_model = make_planar_pcs_model("jax", num_gauss_points=num_gauss_points)
    warp_model = make_planar_pcs_model("warp", num_gauss_points=num_gauss_points)
    q, qd, _y = state_batch(jax_model)

    expected = jax.vmap(jax_model._assemble_dynamics_terms)(q, qd)
    actual = warp_model.dynamics_terms(q, qd)

    for actual_term, expected_term in zip(actual, expected, strict=True):
        assert_allclose(actual_term, expected_term, rtol=2e-8, atol=2e-10)
