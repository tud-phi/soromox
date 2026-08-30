"""Tests for lazy resolution and validation of optional Warp executors."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from soromox.execution.warp import loader


@pytest.fixture(autouse=True)
def clear_executor_cache() -> None:
    """Isolate the process-global lazy-loader cache between tests."""

    loader.load_executor.cache_clear()
    yield
    loader.load_executor.cache_clear()


def test_load_executor_resolves_and_caches_family_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Import a family module once and return its canonical execute function."""

    imports: list[str] = []

    def execute(*args):
        return args

    def fake_import(name: str):
        imports.append(name)
        return SimpleNamespace(execute_dynamics_terms=execute)

    monkeypatch.setattr(loader, "import_module", fake_import)

    first = loader.load_executor("gvs")
    second = loader.load_executor("gvs")

    assert first is execute
    assert second is execute
    assert imports == ["soromox.execution.warp.gvs.executor"]


def test_load_executor_explains_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate a missing Warp package into an actionable installation error."""

    def fake_import(name: str):
        del name
        raise ModuleNotFoundError("No module named 'warp'", name="warp")

    monkeypatch.setattr(loader, "import_module", fake_import)

    with pytest.raises(ImportError, match=r"pip install soromox\[warp\]"):
        loader.load_executor("pcs")


def test_load_executor_preserves_unrelated_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not disguise a broken executor module as an optional-package issue."""

    failure = ModuleNotFoundError("No module named 'unexpected'", name="unexpected")

    def fake_import(name: str):
        del name
        raise failure

    monkeypatch.setattr(loader, "import_module", fake_import)

    with pytest.raises(ModuleNotFoundError) as error:
        loader.load_executor("gvs")
    assert error.value is failure


def test_batch_validation_rejects_empty_and_non_fp64_states() -> None:
    """Fail before an executor sees unsupported batch inputs."""

    with pytest.raises(ValueError, match="non-empty batch"):
        loader._validate_batch(
            jnp.zeros((0, 2), dtype=jnp.float64),
            jnp.zeros((0, 2), dtype=jnp.float64),
            "test",
        )
    with pytest.raises(TypeError, match="requires float64"):
        loader._validate_batch(
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.zeros((1, 2), dtype=jnp.float32),
            "test",
        )
