import jax
import pytest

from tools.benchmarks._benchmark_common import (
    SystemConfig,
    benchmark_environment_metadata,
    build_system_with_gauss_points,
    get_gvs_basis_order_system_config,
    get_system_registry,
    resolve_execution_backend,
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

    assert context["q"].shape == (system.num_internal_dofs,)
    assert context["u"].shape == (system.num_actuators,)


@pytest.mark.parametrize("system_name", ["planar_pcs", "pcs", "gvs"])
def test_backend_is_forwarded_to_supported_default_factories(
    system_name: str,
) -> None:
    config = get_system_registry()[system_name]

    system = build_system_with_gauss_points(
        config,
        size=1,
        gauss_points=None,
        backend="jax",
    )

    assert config.supports_backend
    assert system.backend == "jax"


def test_backend_is_forwarded_to_gvs_basis_order_gauss_factory() -> None:
    config = get_gvs_basis_order_system_config()

    system = build_system_with_gauss_points(
        config,
        size=2,
        gauss_points=7,
        backend="warp",
    )

    assert system.backend == "warp"
    assert system.max_num_gauss_points == 7


def test_backend_is_forwarded_to_normal_and_gauss_spy_factories() -> None:
    calls: list[tuple[int, int | None, str]] = []

    def factory(size: int, *, backend: str):
        calls.append((size, None, backend))
        return object()

    def gauss_factory(size: int, gauss_points: int, *, backend: str):
        calls.append((size, gauss_points, backend))
        return object()

    config = SystemConfig(
        factory=factory,
        size_label="size",
        build_context=lambda _system: {},
        gauss_factory=gauss_factory,
        supports_backend=True,
    )

    build_system_with_gauss_points(config, 3, None, backend="warp")
    build_system_with_gauss_points(config, 4, 9, backend="jax")

    assert calls == [(3, None, "warp"), (4, 9, "jax")]


def test_jax_only_factory_does_not_receive_backend_keyword() -> None:
    calls: list[int] = []

    def factory(size: int):
        calls.append(size)
        return object()

    config = SystemConfig(
        factory=factory,
        size_label="size",
        build_context=lambda _system: {},
    )

    system = build_system_with_gauss_points(
        config,
        size=2,
        gauss_points=None,
        backend="warp",
    )

    assert calls == [2]
    assert resolve_execution_backend(system, platform="gpu") == "jax"


@pytest.mark.parametrize(
    ("configured", "platform", "expected"),
    [
        ("auto", "cpu", "jax"),
        ("auto", "gpu", "warp"),
        ("jax", "gpu", "jax"),
        ("warp", "cpu", "warp"),
    ],
)
def test_resolve_execution_backend(
    configured: str,
    platform: str,
    expected: str,
) -> None:
    system = type("System", (), {"backend": configured})()

    assert resolve_execution_backend(system, platform=platform) == expected
