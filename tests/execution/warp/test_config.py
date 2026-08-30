"""Tests for user-visible Warp launch configuration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from soromox.execution import GVSBackendParams
from soromox.execution.warp import loader
from soromox.execution.warp.config import gvs_block_dim


@pytest.mark.parametrize(
    ("num_dofs", "expected"),
    [(1, 128), (64, 128), (65, 192), (256, 192)],
)
def test_gvs_retains_bounded_shape_rule(num_dofs: int, expected: int) -> None:
    """Document the currently benchmarked shape-generic GVS launch policy."""

    assert gvs_block_dim(num_dofs, gpu=True) == expected
    assert gvs_block_dim(num_dofs, gpu=False) == 1


def test_gvs_executor_uses_model_owned_gpu_block_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route GVS execution through the public backend parameters."""

    model = SimpleNamespace(
        backend_params=GVSBackendParams(warp_block_dim=256),
    )
    monkeypatch.setattr(loader.jax, "default_backend", lambda: "gpu")
    assert loader._gvs_model_block_dim(model) == 256

    monkeypatch.setattr(loader.jax, "default_backend", lambda: "cpu")
    assert loader._gvs_model_block_dim(model) == 1
