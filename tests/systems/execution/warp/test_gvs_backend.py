"""JAX/Warp equivalence tests for shape-generic GVS dynamics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from soromox.systems import (
    GVS,
    GVSBackendParams,
    GVSSegment,
    JointSpec,
    LinkSpec,
    StrainBasisSpec,
)
from soromox.systems.execution.warp.gvs.operands import (
    GVSKinematicsOperands,
    GVSKinematicsShapes,
    GVSOperands,
    GVSPipelineShapes,
)

from ._equivalence import assert_backend_equivalence
from ._kinematics_equivalence import (
    assert_inertial_jacobian_finite_difference,
    assert_kinematics_backend_equivalence,
    assert_warp_derivatives_use_jax,
)


def _all_basis_family_model() -> GVS:
    """Build a high-order padded GVS chain containing every basis family."""

    basis_types = (
        "monomial",
        "legendre",
        "chebyshev",
        "fourier",
        "gaussian",
        "imq",
    )
    segments = tuple(
        GVSSegment(
            link=LinkSpec.circular(
                length=0.07 + 0.01 * index,
                radius=0.012,
                density=1020.0,
                reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                young_modulus=2.0e5,
                shear_modulus=7.5e4,
                material_damping_coefficient=0.01,
            ),
            joint=JointSpec(type="fixed"),
            basis=StrainBasisSpec(
                type=basis_type,
                strain_selector=[False, True, False, False, False, False],
                basis_order=[0, 6, 0, 0, 0, 0],
            ),
            num_gauss_points=5,
        )
        for index, basis_type in enumerate(basis_types)
    )
    return GVS.from_segments(
        segments,
        max_dof=13,
        scale_rotational_basis_by_length=True,
        backend="jax",
    )


def test_gvs_operands_are_views_over_precomputed_model_data(
    make_gvs_model: Callable[[str], Any],
) -> None:
    """Keep preprocessing in the model and the executor contract minimal."""

    model = make_gvs_model("jax")
    operands = GVSOperands.from_model(model, block_dim=128)

    assert operands.num_segments == model.num_segments
    assert operands.num_dofs == model.num_dofs
    assert operands.num_cells == model.max_num_integration_points - 1
    assert operands.num_quadrature == model.max_num_integration_points - 2
    assert operands.block_dim == 128
    assert operands.joint_basis is model.B_joint
    assert operands.link_basis_z1_values is model.scaled_B_Z1_values
    assert operands.weighted_mass_diagonals is model.inner_weighted_mass_diagonals
    assert_allclose(operands.gravity_base, model.gravity_base, rtol=0.0, atol=0.0)


def test_gvs_pipeline_shapes_cover_public_workspace_and_outputs(
    make_gvs_model: Callable[[str], Any],
) -> None:
    """Provide external Warp solvers one allocation contract for fixed topology."""

    model = make_gvs_model("jax")
    operands = GVSOperands.from_model(model, block_dim=128)
    shapes = GVSPipelineShapes.from_operands(operands, batch_size=3)

    assert shapes.local_state == (3 * model.num_segments, model.max_dof)
    assert shapes.joint_outputs()["adjoint"] == (3 * model.num_segments * 6, 6)
    assert shapes.cell_outputs()["tangent_local"] == (
        3 * model.num_segments * operands.num_cells * 6,
        model.max_dof,
    )
    assert shapes.chain_outputs()["inertia"] == (
        3,
        model.num_dofs,
        model.num_dofs,
    )
    assert shapes.chain_outputs()["gravity_force"] == (3, model.num_dofs)


@pytest.mark.parametrize(
    ("batch_size", "error_type"),
    [(True, TypeError), (1.5, TypeError), (0, ValueError), (-1, ValueError)],
)
def test_gvs_pipeline_shapes_require_positive_integer_batch(
    make_gvs_model: Callable[[str], Any],
    batch_size: Any,
    error_type: type[Exception],
) -> None:
    """Reject allocation contracts that cannot represent an environment batch."""

    model = make_gvs_model("jax")
    operands = GVSOperands.from_model(model, block_dim=128)
    with pytest.raises(error_type, match="batch_size"):
        GVSPipelineShapes.from_operands(operands, batch_size=batch_size)


def test_gvs_kinematics_shapes_cover_reduced_and_fused_paths(
    make_gvs_model: Callable[[str], Any],
) -> None:
    """Describe caller-owned GVS kinematics workspaces and outputs."""

    model = make_gvs_model("jax")
    operands = GVSKinematicsOperands.from_model(model, block_dim=1)
    shapes = GVSKinematicsShapes.from_operands(operands, batch_size=3, num_samples=7)
    node_count = 3 * model.num_segments * model.max_num_integration_points

    assert shapes.pose_workspace() == {"node_pose": (node_count * 4, 4)}
    assert shapes.workspace() == {
        "node_pose": (node_count * 4, 4),
        "node_jacobian": (node_count * 6, model.num_dofs),
    }
    assert shapes.pose_output() == (3, 7, 4, 4)
    assert shapes.jacobian_output() == (3, 7, 6, model.num_dofs)


def test_gvs_public_kinematics_match_jax_on_cpu_and_gpu(
    make_gvs_model: Callable[[str], Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Match GVS kinematics for every public vectorization form."""

    pytest.importorskip("warp")
    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "gvs-kinematics-cache"))
    model = make_gvs_model("jax")

    assert_kinematics_backend_equivalence(model)
    assert_inertial_jacobian_finite_difference(model)
    assert_warp_derivatives_use_jax(model)


def test_gvs_backend_params_override_reaches_warp_launches(
    make_gvs_model: Callable[[str], Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Use an explicit block size in dynamics and Jacobian recurrences."""

    pytest.importorskip("warp")
    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "gvs-block-override"))
    base = make_gvs_model("jax")
    model = GVS(
        params=base.params,
        structure=base.structure,
        backend="warp",
        backend_params=GVSBackendParams(warp_block_dim=192),
    )
    q = jnp.linspace(-0.018, 0.023, model.num_dofs, dtype=jnp.float64)
    qd = jnp.linspace(0.11, -0.09, model.num_dofs, dtype=jnp.float64)
    for actual, expected in zip(
        model.dynamics_terms(q, qd, backend="warp"),
        model.dynamics_terms(q, qd, backend="jax"),
        strict=True,
    ):
        assert_allclose(actual, expected, rtol=2e-8, atol=2e-10)

    s = 0.41 * jnp.sum(model.segment_lengths)
    assert_allclose(
        model.jacobian_inertialframe(q, s, backend="warp"),
        model.jacobian_inertialframe(q, s, backend="jax"),
        rtol=2e-9,
        atol=2e-10,
    )


def test_gvs_warp_kinematics_supports_every_basis_family_and_high_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Match all runtime basis evaluators with basis order greater than five."""

    pytest.importorskip("warp")
    monkeypatch.setenv(
        "WARP_CACHE_PATH", str(tmp_path / "gvs-all-bases-kinematics-cache")
    )
    model = _all_basis_family_model()
    q = jnp.linspace(-0.025, 0.031, model.num_dofs, dtype=jnp.float64)
    boundaries = model.segment_end_positions
    interiors = boundaries[:-1] + 0.43 * model.segment_lengths
    samples = jnp.concatenate((boundaries[::-1], interiors, interiors[2:3]), axis=0)

    assert_allclose(
        model.basis_type_index,
        jnp.arange(6, dtype=model.basis_type_index.dtype),
        rtol=0.0,
        atol=0.0,
    )
    for method_name in (
        "forward_kinematics_abscissa_batched",
        "jacobian_inertialframe_abscissa_batched",
        "forward_kinematics_and_jacobian_inertialframe_abscissa_batched",
    ):
        method = getattr(model, method_name)
        expected = method(q, samples, backend="jax")
        actual = method(q, samples, backend="warp")
        for actual_leaf, expected_leaf in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            assert_allclose(actual_leaf, expected_leaf, rtol=2e-9, atol=2e-10)


def test_gvs_public_dynamics_apis_match_jax_on_gpu(
    make_gvs_model: Callable[[str], Any],
    state_batch: Callable[[Any], tuple[Any, Any, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Match GVS terms and forward dynamics for every public input form."""

    if jax.default_backend() != "gpu":
        pytest.skip("GVS Warp equivalence requires a JAX GPU backend.")
    pytest.importorskip("warp")
    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "gvs-warp-cache"))
    jax_model = make_gvs_model("jax")
    warp_model = make_gvs_model("warp")
    q, qd, y = state_batch(jax_model)

    assert_backend_equivalence(jax_model, warp_model, q, qd, y)
