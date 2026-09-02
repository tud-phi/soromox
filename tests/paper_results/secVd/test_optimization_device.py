"""Device-selection regressions for the Section Vd optimizers.

The case deliberately holds no opinion about which backend is faster: that
depends on the machine's relative FP64 throughput and on how its GPU handles a
long chain of small sequential kernels, neither of which is a property of
Section Vd. So ``auto`` defers to JAX, and an explicit ``--device`` is a strict
pin rather than a preference.
"""

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
    configure_optimization_device,
    jax_platforms_for,
    prepare_result_dir,
)


@pytest.mark.parametrize("argv", [[], ["--num-iters", "5"], ["--batch-size", "6"]])
def test_auto_leaves_the_backend_choice_to_jax(monkeypatch, argv):
    """``auto`` must not pin a platform, at any batch size.

    An earlier rule sent ``B > 1`` to the GPU, and a later one sent everything to
    the CPU. Both encoded one machine's benchmark as if it were a property of
    the case.
    """
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    assert configure_optimization_device(argv) == "auto"
    assert "JAX_PLATFORMS" not in os.environ


def test_auto_does_not_disturb_an_existing_platform_setting(monkeypatch):
    """A caller's own JAX_PLATFORMS is a deliberate choice; auto must respect it."""
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    assert configure_optimization_device([]) == "auto"
    assert os.environ["JAX_PLATFORMS"] == "cuda"


@pytest.mark.parametrize(
    ("argv", "device", "platforms"),
    [
        (["--device", "cpu"], "cpu", "cpu"),
        (["--device", "gpu"], "gpu", "cuda"),
        (["--device", "gpu", "--num-iters", "5"], "gpu", "cuda"),
    ],
)
def test_an_explicit_device_pins_that_platform(monkeypatch, argv, device, platforms):
    """An explicit GPU request is strict: it must never silently land on CPU."""
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    assert configure_optimization_device(argv) == device
    assert os.environ["JAX_PLATFORMS"] == platforms


def test_jax_platform_names_are_the_ones_jax_accepts():
    # "gpu" is the user-facing name; JAX calls the platform "cuda".
    assert jax_platforms_for("cpu") == "cpu"
    assert jax_platforms_for("gpu") == "cuda"


def test_unknown_device_has_no_platform_mapping():
    with pytest.raises(ValueError, match="Unknown optimization device"):
        jax_platforms_for("tpu")


def _prepare_args(tmp_path, *, device):
    return SimpleNamespace(
        num_iters=1,
        result_dir=tmp_path,
        force=False,
        debug_nans=False,
        device=device,
        optimization_batch_size=6,
    )


def _fake_jax(actual_device):
    return SimpleNamespace(
        config=SimpleNamespace(update=lambda *_args, **_kwargs: None),
        default_backend=lambda: actual_device,
    )


@pytest.mark.parametrize("actual", ["cpu", "gpu"])
def test_auto_accepts_whatever_backend_jax_initialized(monkeypatch, tmp_path, actual):
    monkeypatch.setitem(sys.modules, "jax", _fake_jax(actual))
    args = _prepare_args(tmp_path, device="auto")

    assert prepare_result_dir(args) == tmp_path.resolve()
    # The archive records the backend that actually ran, not the one requested.
    assert args.resolved_device == actual


@pytest.mark.parametrize(("device", "actual"), [("gpu", "cpu"), ("cpu", "gpu")])
def test_an_explicit_device_rejects_a_backend_mismatch(
    monkeypatch, tmp_path, device, actual
):
    """Running elsewhere would attach a timing claim to the wrong hardware."""
    monkeypatch.setitem(sys.modules, "jax", _fake_jax(actual))
    args = _prepare_args(tmp_path, device=device)

    with pytest.raises(RuntimeError, match="Device selection must run before"):
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


@pytest.mark.parametrize("method", ["collocated", "synergistic"])
def test_both_entrypoints_run_the_same_number_of_starts(method):
    """Both methods must use the shared batch size, not a local literal.

    The collocated entrypoint once hardcoded ``OPTIMIZATION_BATCH_SIZE = 1``
    while the synergistic one used the shared constant, so a run that reported
    itself as restoring six starts silently optimized one, and the two archives
    were not comparable.
    """
    import ast

    source = (CODE_DIR / f"control_gain_optimization_with_{method}.py").read_text()
    assigned = [
        node.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "OPTIMIZATION_BATCH_SIZE"
            for t in node.targets
        )
    ]
    assert len(assigned) == 1, f"{method}: expected one OPTIMIZATION_BATCH_SIZE"
    assert isinstance(assigned[0], ast.Name) and assigned[0].id == "SECVD_BATCH_SIZE", (
        f"{method} sets OPTIMIZATION_BATCH_SIZE to a literal instead of "
        "SECVD_BATCH_SIZE, so the two methods can silently disagree"
    )
