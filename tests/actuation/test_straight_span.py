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
    _routing_solver,
)
from soromox.systems import PlanarPCS, PlanarPCSStructure

jax.config.update("jax_enable_x64", True)


def _model(offset=0.0):
    """Build a planar section and a straight span at the requested guide offset.

    Args:
        offset: Common material y guide offset in metres.

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
    )
    offsets = jnp.array([[0.0, offset, 0.0]])
    return robot, StraightSpanRouting(
        RoutingSpanParams(offsets, offsets, (0,), (0,), (0,))
    )


@pytest.mark.parametrize("offset", [0.0, 0.004])
def test_straight_span_chord_and_analytical_derivatives(offset):
    """Check curved geometry against a circular chord and differentiated lengths.

    Args:
        offset: Centered or offset guide placement.
    """
    robot, routing = _model(offset)
    configuration = jnp.array([2.0, 0.0, 0.0])
    solve = eqx.filter_jit(routing.solve_routing)
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
        jax.vmap(lambda values: routing.solve_routing(robot, values).path_lengths)
    )
    assert_allclose(
        batched(jnp.stack([configuration] * 2)), [[expected]] * 2, atol=1e-12
    )


def test_straight_span_needs_no_optimizer_or_runtime_autodiff(monkeypatch):
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
    solution = eqx.filter_jit(routing.solve_routing)(robot, configuration)
    assert bool(
        jnp.all(
            jnp.isfinite(routing.path_length_jacobian(robot, configuration, solution))
        )
    )


def test_straight_span_parameter_gradients_and_empty_paths():
    """Support numerical guide updates, geometry gradients, and empty topology."""
    robot, routing = _model()

    def length(offset):
        """Return the span length after a numerical guide update."""
        return (
            routing.with_params(
                routing.params.replace(end_offset=jnp.array([[0.0, offset, 0.0]]))
            )
            .solve_routing(robot, jnp.zeros(3))
            .path_lengths[0]
        )

    assert_allclose(jax.grad(length)(0.004), 0.004 / np.hypot(0.1, 0.004), atol=1e-10)
    empty = StraightSpanRouting(
        RoutingSpanParams(jnp.empty((0, 3)), jnp.empty((0, 3)), (), (), ())
    )
    solution = empty.solve_routing(robot, jnp.zeros(3))
    assert bool(solution.converged)
    assert solution.path_lengths.shape == (0,)
    assert empty.path_velocity_jacobian(robot, jnp.zeros(3), solution).shape == (0, 3)
    with pytest.raises(ValueError, match="configuration"):
        routing.solve_routing(robot, jnp.zeros(2))


def test_collapsed_straight_span_rejects_derivatives():
    """Coincident endpoints of a physical span do not define a cable direction."""
    robot, routing = _model()
    configuration = jnp.array([0.0, -1.0, 0.0])
    solution = routing.solve_routing(robot, configuration)
    assert not bool(solution.converged)
    assert bool(
        jnp.all(jnp.isnan(routing.path_length_jacobian(robot, configuration, solution)))
    )


def test_guides_share_endpoint_frame_evaluations(monkeypatch):
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
    endpoint = PlanarPCS.routing_endpoint_kinematics
    calls = []

    def evaluate(host, configuration, segment, at_end, offset):
        """Record a frame evaluation and return its guide kinematics."""
        calls.append((segment, at_end))
        return endpoint(host, configuration, segment, at_end, offset)

    monkeypatch.setattr(PlanarPCS, "routing_endpoint_kinematics", evaluate)
    configuration = jnp.array([2.0, 0.0, 0.0])
    solution = routing.solve_routing(robot, configuration)
    assert calls == [(0, False), (0, True)]
    assert_allclose(solution.path_lengths, 2 * (0.5 - transverse) * np.sin(0.1))
    numerical = jax.jacfwd(
        lambda values: routing.solve_routing(robot, values).path_lengths
    )(configuration)
    assert_allclose(
        routing.path_length_jacobian(robot, configuration, solution),
        numerical,
        atol=1e-10,
    )
