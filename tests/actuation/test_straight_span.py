"""Closed-form guide routing, geometry sampling, and analytical forces."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from system_param_builders import planar_pcs_params

from soromox.actuation import (
    ConfigurationDependentThreadlikeRouting,
    RoutingSpanParams,
    StraightSpanRouting,
)
from soromox.actuation.threadlike import _solver as _routing_solver
from soromox.actuation.threadlike import straight_span as straight_module
from soromox.systems import PlanarPCS, PlanarPCSStructure

jax.config.update("jax_enable_x64", True)


def _model(offset=0.0, backend="auto"):
    """Build a planar section and a straight span at the requested guide offset.

    Args:
        offset: Common material y guide offset in metres.
        backend: Model execution preference.

    Returns:
        A PlanarPCS host and StraightSpanRouting instance.
    """
    robot = PlanarPCS(
        planar_pcs_params(
            length=jnp.array([0.1]),
            radius=jnp.array([0.02]),
            density=jnp.array([1000.0]),
            young_modulus=jnp.array([1e3]),
            shear_modulus=jnp.array([5e2]),
            gravity=jnp.zeros(2),
            damping_matrix=jnp.zeros((3, 3)),
        ),
        PlanarPCSStructure(num_gauss_points=3),
        base_pose=jnp.zeros(3),
        backend=backend,
    )
    offsets = jnp.array([[0.0, offset, 0.0]])
    return robot, StraightSpanRouting(
        RoutingSpanParams(offsets, offsets, (0,), (0,), (0,))
    )


@pytest.mark.parametrize("offset", [0.0, 0.004])
def test_straight_span_chord_and_analytical_derivatives(offset, straight_backend):
    """Check curved geometry against a circular chord and differentiated lengths.

    Args:
        offset: Centered or offset guide placement.
    """
    robot, routing = _model(offset)
    configuration = jnp.array([2.0, 0.0, 0.0])
    solve = eqx.filter_jit(
        lambda host, values: routing.solve_routing(
            host, values, backend=straight_backend
        )
    )
    solution = solve(robot, configuration)
    assert isinstance(routing, ConfigurationDependentThreadlikeRouting)
    assert bool(solution.converged)
    expected = 2 * (0.5 - offset) * np.sin(0.1)
    assert_allclose(solution.path_lengths, [expected], atol=1e-12)
    analytical = routing.path_length_jacobian(robot, configuration, solution)

    def length(values):
        """Return resolved lengths at the requested configuration."""
        return solve(robot, values).path_lengths

    assert_allclose(analytical, jax.jacfwd(length)(configuration), atol=1e-10)
    assert_allclose(
        analytical[0],
        jax.grad(lambda values: length(values).sum())(configuration),
        atol=1e-10,
    )
    points = routing.span_positions(
        robot, configuration, solution, 0, jnp.array([0.0, 0.5, 1.0])
    )
    assert_allclose(points[1], (points[0] + points[2]) / 2, atol=1e-12)
    assert_allclose(jnp.linalg.norm(points[-1] - points[0]), expected, atol=1e-12)
    batched = jax.jit(
        jax.vmap(
            lambda values: (
                routing.solve_routing(
                    robot, values, backend=straight_backend
                ).path_lengths
            )
        )
    )
    configurations = jnp.stack([configuration, configuration / 2])
    reference = jax.vmap(
        lambda values: routing.solve_routing(robot, values, backend="jax").path_lengths
    )(configurations)
    assert_allclose(batched(configurations), reference, atol=1e-12)


def test_straight_span_needs_no_optimizer_or_runtime_autodiff(
    monkeypatch, straight_backend
):
    """Resolving spans and forces uses analytical host kinematics.

    Args:
        monkeypatch: Fixture disabling optimization and autodiff transforms.
    """
    robot, routing = _model(0.002)

    def forbidden(*args, **kwargs):
        """Reject a call to an optimizer or derivative transform."""
        raise AssertionError("Unexpected numerical solve or autodiff")

    monkeypatch.setattr(_routing_solver, "_forward_solve", forbidden)
    for name in ["jacfwd", "jacrev", "grad"]:
        monkeypatch.setattr(jax, name, forbidden)
    configuration = jnp.array([0.2, 0.0, 0.0])
    solution = eqx.filter_jit(routing.solve_routing)(
        robot, configuration, backend=straight_backend
    )
    assert bool(
        jnp.all(
            jnp.isfinite(routing.path_length_jacobian(robot, configuration, solution))
        )
    )


def test_straight_span_parameter_gradients_and_empty_paths(straight_backend):
    """Support numerical guide updates, geometry gradients, and empty topology."""
    robot, routing = _model()

    def length(offset):
        """Return the span length after a numerical guide update."""
        return (
            routing.with_params(
                routing.params.replace(end_offset=jnp.array([[0.0, offset, 0.0]]))
            )
            .solve_routing(robot, jnp.zeros(3), backend=straight_backend)
            .path_lengths[0]
        )

    assert_allclose(jax.grad(length)(0.004), 0.004 / np.hypot(0.1, 0.004), atol=1e-10)
    empty = StraightSpanRouting(
        RoutingSpanParams(jnp.empty((0, 3)), jnp.empty((0, 3)), (), (), ())
    )
    solution = empty.solve_routing(robot, jnp.zeros(3), backend=straight_backend)
    assert bool(solution.converged)
    assert solution.path_lengths.shape == (0,)
    assert empty.path_velocity_jacobian(robot, jnp.zeros(3), solution).shape == (0, 3)
    with pytest.raises(ValueError, match="configuration"):
        routing.solve_routing(robot, jnp.zeros(2))


def test_collapsed_straight_span_rejects_derivatives(straight_backend):
    """Coincident endpoints of a physical span do not define a cable direction."""
    robot, routing = _model()
    configuration = jnp.array([0.0, -1.0, 0.0])
    solution = routing.solve_routing(robot, configuration, backend=straight_backend)
    assert not bool(solution.converged)
    assert bool(
        jnp.all(jnp.isnan(routing.path_length_jacobian(robot, configuration, solution)))
    )
    length = eqx.filter_jit(
        lambda values: routing.solve_routing(
            robot, values, backend=straight_backend
        ).path_lengths.sum()
    )
    assert bool(jnp.all(jnp.isnan(jax.grad(length)(configuration))))
    assert bool(jnp.all(jnp.isnan(jax.jacfwd(length)(configuration))))


def test_straight_span_rejects_unknown_backend():
    """Report unsupported routing backends before evaluating any geometry."""
    robot, routing = _model()
    with pytest.raises(ValueError, match="backend"):
        routing.solve_routing(robot, jnp.zeros(3), backend="unknown")


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("configured", ["jax", "warp", "auto"])
@pytest.mark.parametrize("override", [None, "jax", "warp", "auto"])
def test_straight_span_dispatch_inherits_model_and_overrides(
    monkeypatch, device, configured, override
):
    """Use the common backend policy and enter the selected evaluator exactly once."""
    from soromox.execution.warp import loader

    robot, routing = _model(backend=configured)
    calls = []

    def resolve(strategy, host, configuration):
        """Record Warp dispatch using a deterministic reference result."""
        calls.append("warp")
        return strategy.solve_routing(host, configuration, backend="jax")

    monkeypatch.setattr(loader, "load_straight_span_evaluator", lambda: resolve)
    monkeypatch.setattr(jax, "default_backend", lambda: device)
    solution = routing.solve_routing(robot, jnp.zeros(3), backend=override)
    preference = configured if override is None else override
    expected = preference == "warp" or (preference == "auto" and device == "gpu")
    assert calls == (["warp"] if expected else [])
    assert_allclose(solution.path_lengths, [0.1], atol=1e-12)


def test_model_warp_default_preserves_forward_and_reverse_derivatives():
    """Differentiation falls back explicitly to JAX without dispatch recursion."""
    pytest.importorskip("warp")
    robot, routing = _model(0.004, backend="warp")
    configuration = jnp.array([0.2, 0.001, -0.002])
    solve = jax.jit(lambda values: routing.solve_routing(robot, values))
    solution = solve(configuration)
    expected = routing.path_length_jacobian(robot, configuration, solution)

    def lengths(values):
        """Return lengths through the model's configured backend."""
        return solve(values).path_lengths

    assert_allclose(jax.jacfwd(lengths)(configuration), expected, atol=1e-12)
    assert_allclose(jax.jacrev(lengths)(configuration), expected, atol=1e-12)


@pytest.mark.parametrize("shape", [(0, 3), (2, 3, 3), (2, 0, 3)])
def test_warp_nested_and_empty_batches(shape):
    """Flatten environment axes while preserving every solution field."""
    pytest.importorskip("warp")
    robot, routing = _model(0.004)
    configurations = jnp.linspace(-0.1, 0.2, np.prod(shape)).reshape(shape)

    def evaluate(values, backend):
        """Evaluate one configuration with explicit backend selection."""
        return routing.solve_routing(robot, values, backend=backend)

    for _ in shape[:-1]:
        evaluate = jax.vmap(evaluate, in_axes=(0, None))
    actual = eqx.filter_jit(evaluate)(configurations, "warp")
    expected = eqx.filter_jit(evaluate)(configurations, "jax")
    for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert_allclose(value, reference, atol=1e-12)


def test_warp_rejects_batched_guide_parameters():
    """Report the shared-parameter batching contract before a kernel launch."""
    pytest.importorskip("warp")
    robot, routing = _model()

    def evaluate(offset):
        """Build numerical guide parameters from a mapped value."""
        updated = routing.with_params(
            routing.params.replace(end_offset=jnp.array([[0, offset, 0]]))
        )
        return updated.solve_routing(robot, jnp.zeros(3), backend="warp")

    with pytest.raises(ValueError, match="shared parameters"):
        jax.jit(jax.vmap(evaluate))(jnp.array([0.001, 0.002]))


def test_warp_batches_rigid_hosts():
    """Support configuration batches with no active strain coordinates."""
    pytest.importorskip("warp")
    template, routing = _model()
    robot = PlanarPCS(
        template.params,
        PlanarPCSStructure(
            strain_selector=jnp.zeros(3, dtype=bool), num_gauss_points=3
        ),
        base_pose=jnp.zeros(3),
    )
    solve = jax.jit(
        jax.vmap(lambda values: routing.solve_routing(robot, values, backend="warp"))
    )
    solution = solve(jnp.empty((2, 0)))
    assert_allclose(solution.path_lengths, [[0.1], [0.1]], atol=1e-12)
    assert solution.spans[0].length_jacobian.shape == (2, 0)


def test_warp_requires_fp64():
    """Fail clearly before passing FP32 inputs to FP64 kernels."""
    pytest.importorskip("warp")
    robot, routing = _model()
    with jax.enable_x64(False), pytest.raises(ValueError, match="64-bit"):
        routing.solve_routing(robot, jnp.zeros(3), backend="warp")


@pytest.mark.parametrize("curvature", [np.nan, np.inf])
def test_nonfinite_straight_spans_are_rejected(curvature, straight_backend):
    """Reject nonfinite local geometry consistently across backends."""
    robot, routing = _model()
    solution = eqx.filter_jit(routing.solve_routing)(
        robot, jnp.array([curvature, 0, 0]), backend=straight_backend
    )
    assert not bool(solution.converged)
    assert bool(jnp.all(jnp.isnan(solution.spans[0].length_jacobian)))
    assert_allclose(solution.path_lengths, [solution.spans[0].length], equal_nan=True)


def test_guides_share_segment_and_boundary_evaluations(monkeypatch):
    """Reuse each endpoint frame across paths with different offsets.

    Args:
        monkeypatch: Fixture recording analytical endpoint evaluations.
    """
    robot, _ = _model()
    transverse = jnp.array([-0.004, 0.0, 0.004])
    offsets = jnp.stack((jnp.zeros(3), transverse, jnp.zeros(3)), axis=-1)
    routing = StraightSpanRouting(
        RoutingSpanParams(offsets, offsets, (0, 1, 2), (0, 0, 0), (0, 0, 0))
    )
    segments = PlanarPCS.relative_segment_kinematics
    boundaries = straight_module.segment_boundary_poses
    calls = []

    def evaluate(host, configuration):
        """Record shared segment kinematics."""
        calls.append("segments")
        return segments(host, configuration)

    def compose(base, links, joints):
        """Record shared world boundary composition."""
        calls.append("boundaries")
        return boundaries(base, links, joints)

    monkeypatch.setattr(PlanarPCS, "relative_segment_kinematics", evaluate)
    monkeypatch.setattr(straight_module, "segment_boundary_poses", compose)
    configuration = jnp.array([2.0, 0.0, 0.0])
    solution = routing.solve_routing(robot, configuration, backend="jax")
    assert calls == ["segments", "boundaries"]
    assert_allclose(solution.path_lengths, 2 * (0.5 - transverse) * np.sin(0.1))
    numerical = jax.jacfwd(
        lambda values: routing.solve_routing(robot, values, backend="jax").path_lengths
    )(configuration)
    assert_allclose(
        routing.path_length_jacobian(robot, configuration, solution),
        numerical,
        atol=1e-10,
    )
