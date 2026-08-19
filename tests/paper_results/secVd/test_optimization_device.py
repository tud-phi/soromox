"""Device-selection regressions for the Section Vd optimizers."""

import sys
from pathlib import Path

import pytest

CODE_DIR = (
    Path(__file__).resolve().parents[3]
    / "paper_results"
    / "secVd_control_gain_optimization"
    / "code"
)
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from secvd_optimization import (  # noqa: E402
    configure_optimization_device,
    resolve_optimization_device,
)


@pytest.mark.parametrize(
    ("batch_size", "expected"),
    [(1, "cpu"), (2, "gpu"), (6, "gpu")],
)
def test_auto_device_depends_on_optimization_batch_size(batch_size, expected):
    assert (
        resolve_optimization_device(requested="auto", batch_size=batch_size) == expected
    )


@pytest.mark.parametrize("requested", ["cpu", "gpu"])
@pytest.mark.parametrize("batch_size", [1, 6])
def test_explicit_device_overrides_batch_default(requested, batch_size):
    assert (
        resolve_optimization_device(requested=requested, batch_size=batch_size)
        == requested
    )


def test_early_cli_device_configuration_updates_jax_environment(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "gpu")
    assert configure_optimization_device(batch_size=1, argv=[]) == "cpu"
    assert (
        configure_optimization_device(
            batch_size=1, argv=["--device", "gpu", "--num-iters", "5"]
        )
        == "gpu"
    )


@pytest.mark.parametrize("batch_size", [0, -1])
def test_optimization_batch_size_must_be_positive(batch_size):
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        resolve_optimization_device(requested="auto", batch_size=batch_size)
