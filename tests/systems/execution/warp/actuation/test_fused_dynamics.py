"""Fused Warp dynamics and linear-threadlike actuation tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from soromox.systems.execution import (
    GVS_DYNAMICS,
    PCS_DYNAMICS,
    dispatch_fused_dynamics_actuation_force,
)
from soromox.systems.execution.warp import loader
from soromox.systems.execution.warp.config import gvs_block_dim
from soromox.systems.execution.warp.gvs.actuation.operands import (
    GVSThreadlikeOperands,
)
from soromox.systems.execution.warp.gvs.operands import GVSOperands
from soromox.systems.execution.warp.pcs.actuation.operands import (
    PCSThreadlikeOperands,
)
from soromox.systems.execution.warp.pcs.operands import PCSOperands

from .test_threadlike import _build_system


@pytest.mark.parametrize("family", ["planar_pcs", "pcs", "gvs"])
def test_fused_dynamics_actuation_executors_match_jax(
    family: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Match all four terms produced by each public fused Warp executor."""

    pytest.importorskip("warp")
    if family != "gvs" and jax.default_backend() != "gpu":
        pytest.skip("Fused PlanarPCS and PCS Warp dynamics require CUDA.")
    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / f"{family}-fused"))
    model = _build_system(family, 2, 16)
    q = jnp.stack(
        (
            jnp.linspace(-0.025, 0.018, model.num_dofs),
            jnp.linspace(0.021, -0.013, model.num_dofs),
        )
    )
    qd = jnp.stack(
        (
            jnp.linspace(-0.17, 0.23, model.num_dofs),
            jnp.linspace(0.14, -0.19, model.num_dofs),
        )
    )
    controls = jnp.reshape(
        jnp.linspace(-0.8, 1.2, q.shape[0] * model.num_actuators),
        (q.shape[0], model.num_actuators),
    )
    if family == "gvs":
        key = "gvs"
        dynamics = GVSOperands.from_model(
            model,
            block_dim=gvs_block_dim(
                model.num_dofs,
                gpu=jax.default_backend() == "gpu",
            ),
        )
        actuation = GVSThreadlikeOperands.from_model(model)
    else:
        key = "pcs"
        dynamics = PCSOperands.from_model(model)
        actuation = PCSThreadlikeOperands.from_model(model)

    actual = loader.execute_dynamics_and_threadlike_actuation_force(
        key,
        dynamics,
        actuation,
        q,
        qd,
        controls,
    )
    expected_dynamics = jax.vmap(model._assemble_dynamics_terms)(q, qd)
    expected_force = jax.vmap(model._actuation_force)(q, controls, qd)
    expected = (*expected_dynamics, expected_force)
    for actual_term, expected_term in zip(actual, expected, strict=True):
        assert_allclose(actual_term, expected_term, rtol=3e-8, atol=3e-10)


@pytest.mark.parametrize("family", ["planar_pcs", "pcs"])
@pytest.mark.parametrize("num_paths", [16, 64])
def test_cooperative_pcs_force_topologies_match_jax(
    family: str,
    num_paths: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Cover the quadrature-parallel and path-parallel PCS CUDA layouts."""

    pytest.importorskip("warp")
    monkeypatch.setenv(
        "WARP_CACHE_PATH",
        str(tmp_path / f"{family}-{num_paths}-cooperative"),
    )
    model = _build_system(family, 2, num_paths)
    q = jnp.stack(
        (
            jnp.linspace(-0.021, 0.017, model.num_dofs),
            jnp.linspace(0.013, -0.019, model.num_dofs),
        )
    )
    controls = jnp.reshape(
        jnp.linspace(-0.9, 1.1, q.shape[0] * num_paths),
        (q.shape[0], num_paths),
    )
    operands = PCSThreadlikeOperands.from_model(model)
    actual = loader.execute_actuation("pcs", "force", operands, q, controls)
    expected = jax.vmap(model._actuation_force)(q, controls, jnp.zeros_like(q))
    assert_allclose(actual, expected, rtol=3e-9, atol=3e-10)


@pytest.mark.parametrize(
    ("family", "device"),
    [("pcs", "gpu"), ("gvs", "gpu"), ("gvs", "cpu")],
)
def test_gvs_and_pcs_enable_fused_system_dispatch(
    family: str,
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Activate fusion on every device supported by each Warp family."""

    model = _build_system(family, 2, 4)
    q = jnp.linspace(-0.02, 0.03, model.num_dofs)
    qd = jnp.linspace(0.12, -0.08, model.num_dofs)
    controls = jnp.linspace(-0.4, 0.7, model.num_actuators)

    def fake_batch(model_, q_, qd_, controls_):
        del qd_, controls_
        batch_size = q_.shape[0]
        marker = jnp.asarray(batch_size, dtype=q_.dtype)
        return (
            jnp.full((batch_size, model_.num_dofs, model_.num_dofs), marker),
            jnp.full((batch_size, model_.num_dofs), marker + 1.0),
            jnp.full((batch_size, model_.num_dofs), marker + 2.0),
            jnp.full((batch_size, model_.num_dofs), marker + 3.0),
        )

    monkeypatch.setattr(
        loader,
        f"_execute_{family}_dynamics_actuation_batch",
        fake_batch,
    )
    monkeypatch.setattr(
        "soromox.systems.execution.dispatch.jax.default_backend",
        lambda: device,
    )

    result = dispatch_fused_dynamics_actuation_force(
        model,
        q,
        qd,
        controls,
        backend="warp",
        capabilities=GVS_DYNAMICS if family == "gvs" else PCS_DYNAMICS,
    )

    assert result is not None
    for term, marker in zip(result, (1.0, 2.0, 3.0, 4.0), strict=True):
        assert_allclose(term, marker)


def test_fused_gvs_vmap_uses_one_canonical_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consolidate mapped GVS states and controls into one fused invocation."""

    model = _build_system("gvs", 2, 4)
    q = jnp.zeros((5, model.num_dofs), dtype=jnp.float64)
    qd = jnp.zeros_like(q)
    controls = jnp.zeros((5, model.num_actuators), dtype=jnp.float64)
    observed_batches: list[tuple[int, ...]] = []

    def fake_gvs_batch(model_, q_, qd_, controls_):
        del qd_, controls_
        observed_batches.append(q_.shape)
        batch_size = q_.shape[0]
        return (
            jnp.zeros((batch_size, model_.num_dofs, model_.num_dofs)),
            jnp.zeros((batch_size, model_.num_dofs)),
            jnp.zeros((batch_size, model_.num_dofs)),
            jnp.zeros((batch_size, model_.num_dofs)),
        )

    monkeypatch.setattr(loader, "_execute_gvs_dynamics_actuation_batch", fake_gvs_batch)
    evaluator = loader.get_dynamics_actuation_evaluator("gvs")
    result = jax.vmap(evaluator, in_axes=(None, 0, 0, 0))(
        model,
        q,
        qd,
        controls,
    )

    assert result[0].shape == (5, model.num_dofs, model.num_dofs)
    assert observed_batches[-1] == (5, model.num_dofs)
