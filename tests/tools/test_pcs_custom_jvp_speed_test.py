from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from soromox.autodiff import custom_jvp_mode
from soromox.systems import PCS
from tools.benchmarks import pcs_custom_jvp_speed_test as benchmark

jax.config.update("jax_enable_x64", True)


def _assert_trees_allclose(reference, candidate) -> None:
    reference_leaves, reference_tree = jax.tree_util.tree_flatten(reference)
    candidate_leaves, candidate_tree = jax.tree_util.tree_flatten(candidate)

    assert candidate_tree == reference_tree
    for reference_leaf, candidate_leaf in zip(
        reference_leaves,
        candidate_leaves,
        strict=True,
    ):
        assert_allclose(candidate_leaf, reference_leaf, rtol=1.0e-5, atol=1.0e-8)


def _build_workload(model: PCS, workload: str, *, use_custom_jvp: bool):
    benchmark_inputs = benchmark.build_inputs(model)
    workload_fn = benchmark.build_workload(
        model,
        workload,
        benchmark_inputs.y_direction,
        benchmark_inputs.u_direction,
        benchmark_inputs.system_parameter_direction,
        rollout_steps=3,
        rollout_dt=1.0e-4,
        use_custom_jvp=use_custom_jvp,
    )
    args = (
        benchmark_inputs.y,
        benchmark_inputs.u,
        benchmark_inputs.tau_ext,
        benchmark_inputs.system_parameters,
    )
    return benchmark_inputs, workload_fn, args


def test_benchmark_defaults_cover_standard_quadrature_and_larger_models() -> None:
    args = benchmark.parse_args([])

    assert args.gauss_points == 5
    assert args.segment_counts == [1, 2, 4, 8, 16, 32]
    assert args.workloads == [
        "state_jvp",
        "state_jacobian_matvec",
        "state_jacobian",
    ]
    assert args.rollout_steps == 8
    assert args.rollout_dt == 1.0e-4
    assert args.calls_per_sample == 1


def test_benchmark_custom_jvp_matches_direct_jax_autodiff() -> None:
    """The timed baseline must bypass the mutable custom-JVP dispatch toggle."""
    model = benchmark.build_model("threadlike", num_segments=2, gauss_points=5)
    _, reference_fn, inputs = _build_workload(
        model, "all_jacobians", use_custom_jvp=False
    )
    _, candidate_fn, _ = _build_workload(model, "all_jacobians", use_custom_jvp=True)

    reference = reference_fn(*inputs)
    with custom_jvp_mode(True):
        candidate = candidate_fn(*inputs)

    _assert_trees_allclose(reference, candidate)


def test_custom_jvp_input_only_tangent_matches_direct_jax_autodiff() -> None:
    """Cover the custom rule's no-state-tangent inertia-matrix branch."""
    model = benchmark.build_model("threadlike", num_segments=2, gauss_points=5)
    benchmark_inputs = benchmark.build_inputs(model)
    y = benchmark_inputs.y
    u = benchmark_inputs.u
    tau_ext = benchmark_inputs.tau_ext
    t = jnp.array(0.0)

    reference = jax.jacfwd(
        lambda u_value: model._forward_dynamics(t, y, (u_value, tau_ext))
    )(u)
    candidate = jax.jacfwd(
        lambda u_value: PCS._forward_dynamics_custom_jvp(
            model,
            t,
            y,
            (u_value, tau_ext),
        )
    )(u)

    assert_allclose(candidate, reference, rtol=1.0e-5, atol=1.0e-8)


@pytest.mark.parametrize("num_segments", [2, 16])
def test_directional_jvp_matches_full_analytical_jacobian_matvec(
    num_segments: int,
) -> None:
    """Cover short and scan-based analytical state-JVP recurrences."""
    model = benchmark.build_model(
        "threadlike",
        num_segments=num_segments,
        gauss_points=5,
    )
    _, directional_fn, inputs = _build_workload(
        model,
        "state_jvp",
        use_custom_jvp=True,
    )
    _, full_jacobian_fn, _ = _build_workload(
        model,
        "state_jacobian_matvec",
        use_custom_jvp=True,
    )
    _, autodiff_full_jacobian_fn, _ = _build_workload(
        model,
        "state_jacobian_matvec",
        use_custom_jvp=False,
    )

    directional = directional_fn(*inputs)
    full_jacobian = full_jacobian_fn(*inputs)
    autodiff_full_jacobian = autodiff_full_jacobian_fn(*inputs)

    _assert_trees_allclose(directional, full_jacobian)
    _assert_trees_allclose(directional, autodiff_full_jacobian)


@pytest.mark.parametrize(
    "actuation_args",
    [
        pytest.param(None, id="none"),
        pytest.param((None,), id="default-input"),
    ],
)
def test_custom_jvp_state_tangent_accepts_optional_inputs(
    actuation_args: tuple | None,
) -> None:
    model = benchmark.build_model("axial", num_segments=1, gauss_points=5)
    y = benchmark.build_inputs(model).y
    t = jnp.array(0.0)

    reference = jax.jacfwd(
        lambda y_value: model._forward_dynamics(t, y_value, actuation_args)
    )(y)
    candidate = jax.jacfwd(
        lambda y_value: PCS._forward_dynamics_custom_jvp(
            model,
            t,
            y_value,
            actuation_args,
        )
    )(y)

    assert_allclose(candidate, reference, rtol=1.0e-5, atol=1.0e-8)


@pytest.mark.parametrize(
    "workload",
    [
        "rollout_actuation_jvp",
        "rollout_actuation_jacobian",
        "rollout_initial_position_jvp",
        "rollout_initial_position_jacobian",
        "rollout_parameters_jvp",
        "rollout_parameters_jacobian",
    ],
)
def test_rollout_derivative_workloads_match_direct_jax_autodiff(workload: str) -> None:
    model = benchmark.build_model("axial", num_segments=1, gauss_points=5)
    _, reference_fn, inputs = _build_workload(
        model,
        workload,
        use_custom_jvp=False,
    )
    _, candidate_fn, _ = _build_workload(
        model,
        workload,
        use_custom_jvp=True,
    )

    reference = reference_fn(*inputs)
    candidate = candidate_fn(*inputs)

    _assert_trees_allclose(reference, candidate)
    derivative = reference[1] if workload.endswith("_jvp") else reference
    assert jnp.linalg.norm(derivative) > 0.0


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--rollout-steps", "0"),
        ("--rollout-dt", "0"),
        ("--calls-per-sample", "0"),
    ],
)
def test_invalid_rollout_options_are_rejected(option: str, value: str) -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args([option, value])
