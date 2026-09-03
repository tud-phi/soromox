import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_DIR = (
    Path(__file__).resolve().parents[3]
    / "paper_results"
    / "secIVb_parallel_rollouts_gpu"
    / "code"
)
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import generate_parallel_rollout_benchmark as benchmark  # noqa: E402


def test_cpu_device_is_configured_before_jax_import(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)

    selected = benchmark._configure_device_before_jax_import(["--device", "cpu"])

    assert selected == "cpu"
    assert benchmark.os.environ["JAX_PLATFORMS"] == "cpu"


def test_output_paths_default_to_case_data_directory():
    args = benchmark.parse_args([])

    assert args.csv == benchmark.DATA_DIR / "benchmark_results.csv"
    assert args.failures_csv == benchmark.DATA_DIR / "benchmark_results_failures.csv"
    assert args.device == "gpu"
    assert args.backend == "auto"
    assert args.solver == "tsit5"


def test_csv_output_path_can_be_overridden(tmp_path):
    csv_path = tmp_path / "custom.csv"

    args = benchmark.parse_args(["--csv", str(csv_path)])

    assert args.csv == csv_path
    assert args.failures_csv == tmp_path / "custom_failures.csv"


def test_failures_csv_output_path_can_be_overridden(tmp_path):
    failures_path = tmp_path / "failed-cases.csv"

    args = benchmark.parse_args(["--failures-csv", str(failures_path)])

    assert args.failures_csv == failures_path


def test_npz_output_option_is_not_supported():
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--npz", "results.npz"])


@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_device_option_is_parsed(device):
    args = benchmark.parse_args(["--device", device])

    assert args.device == device


def test_missing_gpu_emits_actionable_warning(monkeypatch):
    def unavailable_gpu(platform):
        assert platform == "gpu"
        raise RuntimeError("CUDA driver is unavailable")

    monkeypatch.setattr(benchmark.jax, "devices", unavailable_gpu)

    with pytest.warns(RuntimeWarning, match="--device cpu"):
        device = benchmark._select_device("gpu")

    assert device is None


@pytest.mark.parametrize("backend", ["auto", "jax", "warp"])
def test_backend_option_is_parsed(backend):
    args = benchmark.parse_args(["--backend", backend])

    assert args.backend == backend


def test_invalid_backend_is_rejected():
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--backend", "cuda"])


@pytest.mark.parametrize("solver", benchmark.SOLVER_NAMES)
def test_solver_option_is_parsed(solver):
    args = benchmark.parse_args(["--solver", solver])

    assert args.solver == solver
    if solver != "semi-implicit-euler":
        assert benchmark._make_solver(solver) is not None


def test_invalid_solver_is_rejected():
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--solver", "verlet"])


def test_semi_implicit_euler_applies_kick_before_drift():
    config = benchmark.get_system_registry()["articulated_soft_robot"]
    system = config.factory(1)
    ctx = config.build_context(system)
    dt = 1e-4
    initial_y = benchmark.jnp.concatenate([ctx["q"], ctx["qd"]])
    initial_state = benchmark.SystemState(t=0.0, y=initial_y)

    derivative = system.forward_dynamics(
        benchmark.jnp.asarray(0.0),
        initial_y,
        (ctx["u"], ctx["tau_ext"]),
    )
    _, qdd0 = benchmark.jnp.split(derivative, 2)
    qd1 = ctx["qd"] + dt * qdd0
    expected = benchmark.jnp.concatenate([ctx["q"] + dt * qd1, qd1])

    trajectory = system.rollout_to(
        initial_state=initial_state,
        u=ctx["u"],
        tau_ext=ctx["tau_ext"],
        t1=dt,
        solver_dt=dt,
        save_ts=benchmark.jnp.array([0.0, dt]),
        solver=benchmark.SemiImplicitEuler(system),
    )

    assert benchmark.jnp.allclose(trajectory.y[-1], expected)


def test_rollout_state_diagnostics_use_one_host_transfer(monkeypatch):
    device_get_calls = []
    original_device_get = benchmark.jax.device_get

    def record_device_get(value):
        device_get_calls.append(value)
        return original_device_get(value)

    monkeypatch.setattr(benchmark.jax, "device_get", record_device_get)

    rollout_finite, max_abs_state = benchmark._rollout_state_diagnostics(
        (
            benchmark.jnp.array([[1.0, -2.0]]),
            benchmark.jnp.array([[benchmark.jnp.inf, 3.0]]),
        )
    )

    assert rollout_finite is False
    assert max_abs_state == float("inf")
    assert len(device_get_calls) == 1


def test_csv_records_environment_metadata(tmp_path):
    path = tmp_path / "results.csv"
    benchmark._write_csv([{}], path)

    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")

    assert "git_revision" in header
    assert "jax_version" in header
    assert "device_id" in header
    assert "backend" in header
    assert "resolved_backend" in header
    assert "backend_applies" in header
    assert "warp_version" in header
    assert "solver" in header
    assert "rollout_finite" in header
    assert "max_abs_state" in header


@pytest.mark.parametrize(
    ("error_type", "error_message"),
    [
        ("XlaRuntimeError", "RESOURCE_EXHAUSTED: Out of memory allocating buffer"),
        ("RuntimeError", "CUDA_ERROR_OUT_OF_MEMORY"),
        ("MemoryError", ""),
    ],
)
def test_out_of_memory_failures_are_recognized(error_type, error_message):
    assert benchmark._is_out_of_memory_failure(error_type, error_message)


def test_basis_order_zero_worker_uses_a_valid_segment_count(tmp_path):
    args = benchmark.parse_args(
        [
            "--systems",
            "gvs",
            "--gvs-scaling",
            "basis-order",
            "--gvs-basis-orders",
            "0",
        ]
    )
    case = benchmark._build_benchmark_cases(args)[0]

    command = benchmark._worker_command(args, case, tmp_path / "outcome.json")

    assert command[command.index("--segment-counts") + 1] == "1"
    assert command[command.index("--gvs-basis-orders") + 1] == "0"


def test_isolated_runner_checkpoints_and_continues_after_oom(tmp_path, monkeypatch):
    csv_path = tmp_path / "results.csv"
    args = benchmark.parse_args(
        [
            "--device",
            "cpu",
            "--systems",
            "pendulum",
            "--segment-counts",
            "1",
            "--batch-sizes",
            "1",
            "2",
            "4",
            "--csv",
            str(csv_path),
        ]
    )
    attempted_batches = []

    def fake_worker(command, check):
        assert check is False
        batch = int(command[command.index("--batch-sizes") + 1])
        outcome_path = Path(command[command.index("--_outcome-json") + 1])
        attempted_batches.append(batch)

        if batch == 2:
            with csv_path.open(encoding="utf-8", newline="") as fp:
                checkpointed = list(csv.DictReader(fp))
            assert [int(row["batch_size"]) for row in checkpointed] == [1]
            outcome = {
                "status": "failed",
                "error_type": "XlaRuntimeError",
                "error_message": "RESOURCE_EXHAUSTED: Out of memory",
            }
            return_code = 1
        else:
            outcome = {
                "status": "success",
                "results": [{"system": "pendulum", "batch_size": batch}],
            }
            return_code = 0

        outcome_path.write_text(json.dumps(outcome), encoding="utf-8")
        return SimpleNamespace(returncode=return_code)

    monkeypatch.setattr(benchmark.subprocess, "run", fake_worker)

    return_code = benchmark._run_isolated_benchmarks(args)

    assert return_code == 0
    assert attempted_batches == [1, 2, 4]
    with csv_path.open(encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    assert [int(row["batch_size"]) for row in rows] == [1, 4]
    with args.failures_csv.open(encoding="utf-8", newline="") as fp:
        failures = list(csv.DictReader(fp))
    assert len(failures) == 1
    assert failures[0]["batch_size"] == "2"
    assert failures[0]["failure_kind"] == "out_of_memory"
