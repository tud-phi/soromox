"""Device-selection regressions for the Section Vd optimizers."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
    batch_size_from_argv,
    configure_optimization_device,
    jax_platforms_for,
    prepare_result_dir,
    resolve_optimization_device,
)


@pytest.mark.parametrize(
    ("batch_size", "expected"),
    [(1, "cpu"), (2, "cpu"), (6, "cpu")],
)
def test_auto_device_is_cpu_at_every_batch_size(batch_size, expected):
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
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert (
        configure_optimization_device(
            batch_size=1, argv=["--device", "gpu", "--num-iters", "5"]
        )
        == "gpu"
    )
    assert os.environ["JAX_PLATFORMS"] == "cuda"


@pytest.mark.parametrize("batch_size", [0, -1])
def test_optimization_batch_size_must_be_positive(batch_size):
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        resolve_optimization_device(requested="auto", batch_size=batch_size)


@pytest.mark.parametrize(
    ("device", "allow_fallback", "expected"),
    [
        ("cpu", True, "cpu"),
        ("cpu", False, "cpu"),
        ("gpu", False, "cuda"),
        ("gpu", True, "cuda,cpu"),
    ],
)
def test_jax_platform_names_are_the_ones_jax_accepts(device, allow_fallback, expected):
    assert jax_platforms_for(device, allow_fallback=allow_fallback) == expected


def test_unknown_device_has_no_platform_mapping():
    with pytest.raises(ValueError, match="Unknown optimization device"):
        jax_platforms_for("tpu", allow_fallback=False)


@pytest.mark.parametrize(
    ("argv", "device", "platforms"),
    [
        ([], "cpu", "cpu"),
        (["--device", "gpu"], "gpu", "cuda"),
    ],
)
def test_the_selected_device_maps_to_a_platform_jax_accepts(
    monkeypatch, argv, device, platforms
):
    """An explicit GPU request must be strict, never silently landing on CPU."""
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    assert configure_optimization_device(batch_size=6, argv=argv) == device
    assert os.environ["JAX_PLATFORMS"] == platforms


def _prepare_args(tmp_path, *, requested, resolved, batch_size):
    return SimpleNamespace(
        num_iters=1,
        result_dir=tmp_path,
        force=False,
        debug_nans=False,
        device=requested,
        resolved_device=resolved,
        optimization_batch_size=batch_size,
    )


def _fake_jax(actual_device):
    return SimpleNamespace(
        config=SimpleNamespace(update=lambda *_args, **_kwargs: None),
        default_backend=lambda: actual_device,
    )


def test_automatic_gpu_choice_warns_when_it_falls_back_to_cpu(
    monkeypatch, tmp_path, capsys
):
    """Defensive: auto no longer picks the GPU, but the fallback must still warn
    rather than fail if that rule is ever changed back."""
    monkeypatch.setitem(sys.modules, "jax", _fake_jax("cpu"))
    args = _prepare_args(tmp_path, requested="auto", resolved="gpu", batch_size=6)

    assert prepare_result_dir(args) == tmp_path.resolve()
    assert args.resolved_device == "cpu"
    assert "[WARNING] Preferred 'gpu'" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("requested", "resolved", "actual", "batch_size"),
    [
        ("gpu", "gpu", "cpu", 1),
        ("auto", "cpu", "gpu", 1),
    ],
)
def test_non_fallback_backend_mismatch_is_rejected(
    monkeypatch, tmp_path, requested, resolved, actual, batch_size
):
    monkeypatch.setitem(sys.modules, "jax", _fake_jax(actual))
    args = _prepare_args(
        tmp_path,
        requested=requested,
        resolved=resolved,
        batch_size=batch_size,
    )

    with pytest.raises(RuntimeError, match="unexpected backend"):
        prepare_result_dir(args)


def test_cli_module_does_not_import_jax():
    """JAX ignores JAX_PLATFORMS after import, so the CLI layer must stay JAX-free."""
    probe = (
        "import sys; import secvd_optimization; "
        "sys.exit(1 if 'jax' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=CODE_DIR, capture_output=True, text=True
    )
    assert result.returncode == 0, "importing the CLI layer pulled in jax"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], 6),
        (["--batch-size", "1"], 1),
        (["--batch-size", "3", "--num-iters", "5"], 3),
        (["--num-iters", "5"], 6),
    ],
)
def test_batch_size_is_read_from_argv_before_jax_is_imported(argv, expected):
    """Device selection needs the batch size before JAX exists, so it is parsed early."""
    assert batch_size_from_argv(6, argv) == expected


@pytest.mark.parametrize("argv", [[], ["--batch-size", "1"], ["--batch-size", "6"]])
def test_auto_selects_cpu_at_every_requested_batch_size(monkeypatch, argv):
    """Batching adds width, not sequential depth, so it never justifies the GPU.

    A Section Vd rollout is 50000 sequential solver steps, which makes the GPU
    launch-latency bound: measured at B=1 it did not finish a single iteration in
    950 s, against 40.5 s on CPU.
    """
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    assert configure_optimization_device(batch_size=6, argv=argv) == "cpu"
