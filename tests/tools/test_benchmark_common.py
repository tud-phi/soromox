import jax
import pytest

from tools.benchmarks._benchmark_common import (
    benchmark_environment_metadata,
    build_system_with_gauss_points,
    get_gvs_basis_order_system_config,
    get_system_registry,
)

jax.config.update("jax_enable_x64", True)


def test_benchmark_environment_metadata_identifies_runtime_and_source() -> None:
    metadata = benchmark_environment_metadata(jax.devices("cpu")[0])

    assert metadata["git_revision"]
    assert metadata["python_version"]
    assert metadata["soromox_version"]
    assert metadata["jax_version"] == jax.__version__
    assert metadata["jaxlib_version"]
    assert metadata["x64_enabled"] is True
    assert metadata["device_kind"]
    assert metadata["device_id"].startswith("cpu:")
    assert metadata["device_count"] >= 1


@pytest.mark.parametrize("system_name", list(get_system_registry()))
def test_benchmark_registry_builds_current_system_api(system_name: str) -> None:
    config = get_system_registry()[system_name]

    system = build_system_with_gauss_points(config, size=1, gauss_points=None)
    context = config.build_context(system)

    assert context["q"].shape == context["qd"].shape
    assert context["u"].shape == (system.num_actuators,)
    assert context["tau_ext"].shape == context["q"].shape
    assert context["y"].shape == (2 * context["q"].shape[0],)


def test_gvs_basis_order_registry_builds_current_system_api() -> None:
    config = get_gvs_basis_order_system_config()

    system = build_system_with_gauss_points(config, size=0, gauss_points=5)
    context = config.build_context(system)

    assert context["q"].shape == (system.num_dofs,)
    assert context["u"].shape == (system.num_actuators,)
