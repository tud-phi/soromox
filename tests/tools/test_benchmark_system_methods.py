from __future__ import annotations

import json
import os
from collections import defaultdict

import jax.numpy as jnp
import pytest

from tools.benchmarks import benchmark_system_methods as benchmark

KINEMATICS_METHODS = {
    "forward_kinematics",
    "forward_kinematics_abscissa_batched",
    "jacobian_inertialframe",
    "jacobian_inertialframe_abscissa_batched",
    "forward_kinematics_and_jacobian_inertialframe",
    "forward_kinematics_and_jacobian_inertialframe_abscissa_batched",
}


@pytest.mark.parametrize("system_name", ("planar_pcs", "pcs", "gvs"))
def test_warp_kinematics_methods_are_registered(system_name: str) -> None:
    registry = benchmark._build_system_registry()
    registered_methods = {case.name for case in registry[system_name].cases}

    assert registered_methods >= KINEMATICS_METHODS


def test_configure_device_before_jax_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)

    assert benchmark._configure_device_before_jax_import(["--device", "cpu"]) == "cpu"
    assert os.environ["JAX_PLATFORMS"] == "cpu"

    assert benchmark._configure_device_before_jax_import(["--device", "gpu"]) == "gpu"
    assert os.environ["JAX_PLATFORMS"] == "cuda"


def test_active_cpu_affinity_is_compact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        benchmark.os,
        "sched_getaffinity",
        lambda _pid: {0, 1, 2, 4, 7, 8},
        raising=False,
    )

    assert benchmark._active_cpu_affinity() == "0-2;4;7-8"


@pytest.mark.parametrize(
    ("option", "value"),
    [("--execution-repeats", "0"), ("--warmup-duration", "-0.1")],
)
def test_invalid_timing_options_are_rejected(option: str, value: str) -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args([option, value])


@pytest.mark.parametrize("backend", ["auto", "jax", "warp"])
def test_backend_option_is_parsed(backend: str) -> None:
    args = benchmark.parse_args(["--backend", backend])

    assert args.backend == backend


def test_invalid_backend_is_rejected() -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--backend", "cuda"])


@pytest.mark.parametrize(
    "method",
    [
        "dynamics_terms",
        "forward_dynamics",
        "forward_kinematics",
        "forward_kinematics_abscissa_batched",
        "jacobian_inertialframe",
        "jacobian_inertialframe_abscissa_batched",
        "forward_kinematics_and_jacobian_inertialframe",
        "forward_kinematics_and_jacobian_inertialframe_abscissa_batched",
        "rollout_to",
        "rollout_closed_loop_to",
        "rollout_discrete_closed_loop_to",
    ],
)
def test_backend_metadata_includes_accelerated_methods(method: str) -> None:
    registry = benchmark._build_system_registry()

    assert benchmark._execution_backend_applies(registry["gvs"], method)


def test_backend_metadata_excludes_jax_only_methods_and_systems() -> None:
    registry = benchmark._build_system_registry()

    assert not benchmark._execution_backend_applies(registry["gvs"], "inertia_matrix")
    assert not benchmark._execution_backend_applies(
        registry["pendulum"], "forward_dynamics"
    )


def test_csv_records_execution_backend_metadata(tmp_path) -> None:
    path = tmp_path / "results.csv"
    benchmark._write_csv([defaultdict(str)], path)

    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")

    assert "device" in header
    assert "backend" in header
    assert "resolved_backend" in header
    assert "backend_applies" in header
    assert "warp_version" in header


def test_json_records_resolved_backend(tmp_path) -> None:
    path = tmp_path / "results.json"
    benchmark._write_json(
        [{"device": "gpu", "backend": "auto", "resolved_backend": "warp"}],
        path,
    )

    rows = json.loads(path.read_text(encoding="utf-8"))

    assert rows[0]["resolved_backend"] == "warp"


def test_measure_jitted_call_supports_warmup_duration() -> None:
    compile_time, execution_time = benchmark._measure_jitted_call(
        lambda value: value + 1.0,
        (jnp.array(1.0),),
        repeats=3,
        warmup_duration=0.001,
    )

    assert compile_time >= 0.0
    assert execution_time > 0.0


def test_csv_records_environment_metadata(tmp_path) -> None:
    path = tmp_path / "results.csv"
    benchmark._write_csv([{}], path)

    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")

    assert "git_revision" in header
    assert "jax_version" in header
    assert "device_id" in header
