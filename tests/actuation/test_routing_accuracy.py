"""Independent physical feasibility and optimized sensitivity acceptance."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from soromox.actuation import RoutingSolverOptions
from soromox.actuation.threadlike import _solver as solver

jax.config.update("jax_enable_x64", True)


def _problem(target=0.2):
    """Return a scalar routing problem with a wall at one half metre."""
    return solver.RoutingProblem(
        jnp.array([[[0.0], [1.0], [0.0]]]),
        jnp.array([[1.0, -target, 0.0]]),
        jnp.ones(1),
        jnp.empty((0, 1)),
        jnp.empty(0),
        jnp.ones((1, 1, 1)),
        jnp.zeros((1, 1)),
        jnp.array([0.25]),
    )


def test_physical_limits_use_geometry_scale():
    """Endpoint and wall budgets use metres, independently of solver stopping."""
    problem = eqx.tree_at(
        lambda value: (value.equality_matrix, value.target, value.length_scale),
        _problem(),
        (jnp.ones((1, 1)), jnp.array([0.2 - 1e-6]), 0.014),
    )
    state = jnp.array([0.2, 0.0, 0.0])
    assert not bool(solver.verify_solution(state, problem, RoutingSolverOptions()))
    assert bool(
        solver.verify_solution(
            state, problem, RoutingSolverOptions(endpoint_tolerance=2e-8)
        )
    )
    wall = eqx.tree_at(lambda value: value.length_scale, _problem(0.500001), 0.014)
    assert bool(
        solver.verify_solution(jnp.array([0.500001, 0.0]), wall, RoutingSolverOptions())
    )
    assert not bool(
        solver.verify_solution(
            jnp.array([0.500001, 0.0]), wall, RoutingSolverOptions(wall_tolerance=1e-9)
        )
    )


def test_forward_acceptance_does_not_certify_sensitivities(monkeypatch):
    """A usable forward result can still have unreliable coefficient derivatives.

    Args:
        monkeypatch: Fixture replacing only the primal solver for this test.
    """
    options = RoutingSolverOptions()
    state = jnp.array([0.200001, 0.0])
    assert bool(solver.verify_solution(state, _problem(), options))
    assert not bool(solver.verify_solution(jnp.array([0.3, 0.0]), _problem(), options))
    monkeypatch.setattr(
        solver,
        "_forward_solve",
        lambda problem, initial, options: (
            state,
            jnp.array([solver.verify_solution(state, problem, options), 1.0]),
        ),
    )
    jax.clear_caches()

    def position(target):
        """Return the coefficient for a perturbed scalar target."""
        return solver.solve_problem(_problem(target), jnp.zeros(1), options)[0][0]

    assert_allclose(position(0.2), 0.200001)
    assert jnp.isnan(jax.jacfwd(position)(0.2))
    assert jnp.isnan(jax.grad(position)(0.2))
    jax.clear_caches()


def test_endpoint_tolerance_limits_position_norm():
    """A diagonal residual must meet the Euclidean position error budget."""
    problem = eqx.tree_at(
        lambda value: (value.equality_matrix, value.target),
        _problem(),
        (jnp.ones((2, 1)), jnp.full(2, 0.2 - 0.8e-8)),
    )
    assert not bool(
        solver.verify_solution(
            jnp.array([0.2, 0.0, 0.0, 0.0]), problem, RoutingSolverOptions()
        )
    )


@pytest.mark.parametrize(
    "name",
    [
        "endpoint_tolerance",
        "wall_tolerance",
        "optimality_tolerance",
        "complementarity_tolerance",
        "dual_tolerance",
        "sensitivity_tolerance",
    ],
)
def test_accuracy_options_require_positive_finite_values(name):
    """Reject invalid acceptance budgets.

    Args:
        name: Independent numerical or physical tolerance field.
    """
    for value in [0.0, -1.0, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            RoutingSolverOptions(**{name: value})
