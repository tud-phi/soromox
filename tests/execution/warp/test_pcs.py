"""JAX/Warp equivalence tests for runtime-shaped spatial PCS dynamics."""

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


def test_pcs_operands_are_views_over_precomputed_model_data(
    make_pcs_model: Callable[[str], Any],
) -> None:
    """Expose only the constant-strain arrays required by Warp execution."""

    model = make_pcs_model("jax")
    operands = PCSOperands.from_model(model)

    assert operands.is_planar is False
    assert operands.num_segments == model.num_segments
    assert operands.num_dofs == model.num_internal_dofs
    assert operands.num_gauss_points == model.num_gauss_points
    assert operands.block_dim == model.backend_params.warp_block_dim
    assert operands.active_strain_indices is model.active_strain_indices
    assert operands.reference_strain is model.xi_ref
    assert operands.weighted_mass_diagonals is model.weighted_mass_diagonals


def test_spatial_pcs_pipeline_shapes_cover_workspace_and_results(
    make_pcs_model: Callable[[str], Any],
) -> None:
    """Describe every external allocation needed by the spatial PCS pipeline."""

    model = make_pcs_model("jax")
    operands = PCSOperands.from_model(model)
    shapes = PCSPipelineShapes.from_operands(operands, batch_size=3)

    assert shapes.spatial_dim == 6
    assert shapes.operator_outputs()["adjoint_inverse"] == (
        3 * model.num_segments * (model.num_gauss_points + 1) * 6,
        6,
    )
    assert shapes.chain_outputs()["velocity_first"] == (3 * 6, 1)
    assert shapes.chain_outputs()["inertia"] == (
        3,
        model.num_internal_dofs,
        model.num_internal_dofs,
    )


@pytest.mark.parametrize(
    ("batch_size", "error_type"),
    [(True, TypeError), (1.5, TypeError), (0, ValueError), (-1, ValueError)],
)
def test_pcs_pipeline_shapes_require_positive_integer_batch(
    make_pcs_model: Callable[[str], Any],
    batch_size: Any,
    error_type: type[Exception],
) -> None:
    """Reject allocation contracts that cannot represent an environment batch."""

    model = make_pcs_model("jax")
    operands = PCSOperands.from_model(model)
    with pytest.raises(error_type, match="batch_size"):
        PCSPipelineShapes.from_operands(operands, batch_size=batch_size)


def test_spatial_pcs_kinematics_shapes_cover_reduced_and_fused_paths(
    make_pcs_model: Callable[[str], Any],
) -> None:
    """Describe caller-owned PCS kinematics workspaces and outputs."""

    model = make_pcs_model("jax")
    operands = PCSKinematicsOperands.from_model(model)
    shapes = PCSKinematicsShapes.from_operands(operands, batch_size=3, num_samples=7)

    assert shapes.pose_workspace() == {
        "segment_strain": (3, model.num_segments, 6),
        "segment_pose": (3, model.num_segments + 1, 4, 4),
    }
    assert shapes.jacobian_workspace() == shapes.workspace()
    assert shapes.jvp_workspace() == {
        "segment_strain": (3, model.num_segments, 6),
        "segment_pose": (3, model.num_segments + 1, 4, 4),
        "segment_twist": (3, model.num_segments + 1, 6),
    }
    assert shapes.vjp_workspace() == {
        "segment_strain": (3, model.num_segments, 6),
        "segment_pose": (3, model.num_segments + 1, 4, 4),
        "segment_wrench": (3, model.num_segments + 1, 6),
    }
    assert shapes.pose_output() == (3, 7, 4, 4)
    assert shapes.jacobian_output() == (3, 7, 6, model.num_internal_dofs)
    assert shapes.twist_output() == (3, 7, 6)
    assert shapes.generalized_force_output() == (3, model.num_internal_dofs)


def test_spatial_pcs_public_kinematics_match_jax_on_cpu_and_gpu(
    make_pcs_model: Callable[[str], Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Match spatial PCS kinematics for every public vectorization form."""

    pytest.importorskip("warp")
    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "pcs-kinematics-cache"))
    model = make_pcs_model("jax")

    assert_kinematics_backend_equivalence(model)
    assert_inertial_jacobian_finite_difference(model)
    assert_warp_derivatives_use_jax(model)


def test_pcs_public_dynamics_apis_match_jax_on_gpu(
    make_pcs_model: Callable[[str], Any],
    state_batch: Callable[[Any], tuple[Any, Any, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Match PCS terms and forward dynamics for every public input form."""

    if jax.default_backend() != "gpu":
        pytest.skip("PCS Warp equivalence requires a JAX GPU backend.")
    pytest.importorskip("warp")
    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "pcs-warp-cache"))
    jax_model = make_pcs_model("jax")
    warp_model = make_pcs_model("warp")
    q, qd, y = state_batch(jax_model)

    assert_backend_equivalence(jax_model, warp_model, q, qd, y)


@pytest.mark.parametrize("num_gauss_points", [1, 3, 7, 9])
def test_spatial_pcs_runtime_quadrature_counts_match_jax_on_gpu(
    num_gauss_points: int,
    make_pcs_model: Callable[..., Any],
    state_batch: Callable[[Any], tuple[Any, Any, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Match public Warp dynamics for arbitrary positive quadrature counts."""

    if jax.default_backend() != "gpu":
        pytest.skip("PCS Warp equivalence requires a JAX GPU backend.")
    pytest.importorskip("warp")
    monkeypatch.setenv(
        "WARP_CACHE_PATH",
        str(tmp_path / f"pcs-quadrature-{num_gauss_points}"),
    )
    jax_model = make_pcs_model("jax", num_gauss_points=num_gauss_points)
    warp_model = make_pcs_model("warp", num_gauss_points=num_gauss_points)
    q, qd, _y = state_batch(jax_model)

    expected = jax.vmap(jax_model._assemble_dynamics_terms)(q, qd)
    actual = warp_model.dynamics_terms(q, qd)

    for actual_term, expected_term in zip(actual, expected, strict=True):
        assert_allclose(actual_term, expected_term, rtol=2e-8, atol=2e-10)
