from __future__ import annotations

import os

import jax.numpy as jnp
import pytest

from tools.benchmarks import benchmark_system_methods as benchmark


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
    )

    assert benchmark._active_cpu_affinity() == "0-2;4;7-8"


@pytest.mark.parametrize(
    ("option", "value"),
    [("--execution-repeats", "0"), ("--warmup-duration", "-0.1")],
)
def test_invalid_timing_options_are_rejected(option: str, value: str) -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args([option, value])


def test_measure_jitted_call_supports_warmup_duration() -> None:
    compile_time, execution_time = benchmark._measure_jitted_call(
        lambda value: value + 1.0,
        (jnp.array(1.0),),
        repeats=3,
        warmup_duration=0.001,
    )

    assert compile_time >= 0.0
    assert execution_time > 0.0
