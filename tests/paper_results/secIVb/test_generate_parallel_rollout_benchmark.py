import sys
from pathlib import Path

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
    assert args.device == "gpu"


def test_csv_output_path_can_be_overridden(tmp_path):
    csv_path = tmp_path / "custom.csv"

    args = benchmark.parse_args(["--csv", str(csv_path)])

    assert args.csv == csv_path


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


def test_csv_records_environment_metadata(tmp_path):
    path = tmp_path / "results.csv"
    benchmark._write_csv([{}], path)

    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")

    assert "git_revision" in header
    assert "jax_version" in header
    assert "device_id" in header
