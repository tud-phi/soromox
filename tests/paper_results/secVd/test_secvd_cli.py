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

import numpy as np  # noqa: E402
from secvd_cli import (  # noqa: E402
    configure_optimization_device,
    jax_platforms_for,
    parse_optimization_args,
    prepare_result_dir,
)
from secvd_init import SECVD_BATCH_SIZE  # noqa: E402
from secvd_results import METHODS  # noqa: E402


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
        batch_size=6,
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
    probe = "import sys; import secvd_cli; sys.exit(1 if 'jax' in sys.modules else 0)"
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=CODE_DIR, capture_output=True, text=True
    )
    assert result.returncode == 0, "importing the CLI layer pulled in jax"


def test_the_solver_step_default_is_the_step_the_archives_run_at(tmp_path):
    """A default recipe must reproduce what ships, not something ten times finer.

    ``--solver-dt`` used to default to the case's own ``1e-4`` -- the reference
    step the setpoint rollout uses -- while both committed archives record
    ``1e-3``. Every documented command therefore ran ten times finer and ten
    times slower than the numbers it claimed to reproduce, and schema v3 made
    that visible only after the run had finished.
    """
    from secvd_case import TUNED_SOLVER_DT

    for method in METHODS:
        args = parse_optimization_args(
            description="test",
            default_result_dir_for=lambda _method: tmp_path,
            argv=["--method", method],
        )
        assert args.solver_dt == TUNED_SOLVER_DT

    for method in METHODS:
        path = CODE_DIR.parent / "data" / method / "optimization_results.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            step = float(np.asarray(data["solver_dt"]).ravel()[0])
        assert step == TUNED_SOLVER_DT, (
            f"the committed {method} archive runs at {step}, but a default "
            f"generator run would use {TUNED_SOLVER_DT}"
        )


@pytest.mark.parametrize("method", METHODS)
def test_a_default_run_reproduces_the_committed_initialization(method, tmp_path):
    """The seed is per method, so the shared constant is not the whole story.

    Collocated moved to seed 2 to keep every start at or above nominal: a start
    drawn well below nominal takes the largest relative step at a fixed rate, and
    that is what made the loss band ragged. Synergistic kept seed 0. One shared
    default would silently stop reproducing one of the two archives.
    """
    from secvd_case import TUNED_INIT_SEED

    args = parse_optimization_args(
        description="test",
        default_result_dir_for=lambda _method: tmp_path,
        argv=["--method", method],
    )
    assert args.init_seed == TUNED_INIT_SEED[method]

    path = CODE_DIR.parent / "data" / method / "optimization_results.npz"
    if not path.exists():
        pytest.skip(f"no committed {method} archive to compare against")
    with np.load(path) as data:
        stored = int(np.asarray(data["init_seed"]).ravel()[0])
    assert stored == args.init_seed, (
        f"the committed {method} archive was drawn with seed {stored}, but a "
        f"default generator run would use {args.init_seed}"
    )


def test_the_batch_size_default_is_the_shared_constant(tmp_path):
    """A run must restore as many starts as it reports.

    The collocated entrypoint once hardcoded a batch size of one while the
    synergistic one used the shared constant, so a run that reported itself as
    optimizing six starts silently optimized one, and the two archives were not
    comparable. The parser now owns the default outright.
    """
    args = parse_optimization_args(
        description="test",
        default_result_dir_for=lambda _method: tmp_path,
        argv=[],
    )
    assert args.batch_size == SECVD_BATCH_SIZE
