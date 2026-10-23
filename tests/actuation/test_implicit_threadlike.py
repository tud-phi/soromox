"""Routing geometry, constrained solves, and implicit sensitivity regressions."""

from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize
from system_param_builders import (
    pcs_params,
    planar_base_pose,
    planar_pcs_params,
    spatial_base_pose,
)

from soromox.actuation import (
    CircularLumenParams,
    RoutingDiscretization,
    RoutingSolverOptions,
    RoutingSpanParams,
    ShortestPathRouting,
    ShortestPathRoutingParams,
)
from soromox.actuation._routing_solver import (
    RoutingProblem,
    _forward_solve,
    equalities,
    inequalities,
    objective,
    solve_problem,
)
from soromox.systems import (
    GVS,
    PCS,
    GVSSegment,
    JointSpec,
    LinkSpec,
    PCSStructure,
    PlanarPCS,
    StrainBasisSpec,
)
from soromox.systems.pcs import PlanarPCSStructure

jax.config.update("jax_enable_x64", True)


def _spans(**updates):
    fields = {
        "start_offset": [[0.0, 0.0, 0.0]],
        "end_offset": [[0.0, 0.0, 0.0]],
        "span_path_index": (0,),
        "start_segment_index": (0,),
        "end_segment_index": (0,),
    }
    fields.update(updates)
    return RoutingSpanParams(**fields)


def _robot(lengths=(0.1,)):
    count = len(lengths)
    params = planar_pcs_params(
        length=jnp.array(lengths),
        radius=jnp.full(count, 0.05),
        density=jnp.full(count, 1000.0),
        gravity=jnp.zeros(2),
        young_modulus=jnp.full(count, 1e3),
        shear_modulus=jnp.full(count, 5e2),
        damping_matrix=jnp.zeros((3 * count, 3 * count)),
    )
    return PlanarPCS(
        params, PlanarPCSStructure(num_gauss_points=5), base_pose=planar_base_pose()
    )


def _routing(spans=None, radii=(0.02,), elements=3, max_steps=200):
    return ShortestPathRouting(
        ShortestPathRoutingParams(
            _spans() if spans is None else spans, CircularLumenParams(radii)
        ),
        RoutingDiscretization(elements_per_segment=elements, quadrature_order=3),
        RoutingSolverOptions(max_steps=max_steps, tolerance=1e-8),
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"span_path_index": (1,)},
        {"start_segment_index": (-1,)},
        {"end_segment_index": ()},
        {"start_offset": [[1.0, 0.0, 0.0]]},
        {"end_offset": [[0.0, float("nan"), 0.0]]},
    ],
)
def test_span_validation(updates):
    with pytest.raises(ValueError):
        _spans(**updates)


def test_parameter_composition_and_topology_preserving_updates():
    params = _routing().params
    updated = eqx.filter_jit(
        lambda values: params.replace(spans=params.spans.replace(end_offset=values))
    )(jnp.array([[0.0, 0.001, 0.0]]))
    assert_allclose(updated.spans.end_offset, [[0, 0.001, 0]])
    with pytest.raises(ValueError, match="topology"):
        params.spans.replace(end_segment_index=(1,))
    with pytest.raises(ValueError, match="segment count"):
        params.replace(domain=CircularLumenParams([0.02, 0.02]))
    with pytest.raises(ValueError, match="inside"):
        params.replace(spans=params.spans.replace(end_offset=[[0, 0.03, 0]]))
    with pytest.raises(ValueError, match="positive"):
        CircularLumenParams([0.0])


def test_guides_are_explicit_and_contiguous():
    spans = _spans(
        start_offset=[[0, 0, 0], [0, 0.001, 0]],
        end_offset=[[0, 0.001, 0], [0, 0, 0]],
        span_path_index=(0, 0),
        start_segment_index=(0, 1),
        end_segment_index=(0, 1),
    )
    assert spans.num_paths == 1
    assert spans.num_spans == 2
    with pytest.raises(ValueError, match="guide"):
        spans.replace(start_offset=[[0, 0, 0], [0, 0.002, 0]])
    with pytest.raises(ValueError, match="contiguous"):
        _spans(
            start_offset=jnp.zeros((2, 3)),
            end_offset=jnp.zeros((2, 3)),
            span_path_index=(0, 0),
            start_segment_index=(0, 2),
            end_segment_index=(0, 2),
        )


def test_straight_path_matches_analytic_length_and_endpoints():
    robot = _robot()
    routing = _routing(_spans(end_offset=[[0, 0.004, 0]]))
    result = eqx.filter_jit(routing.solve_routing)(robot, jnp.zeros(3))
    assert bool(result.converged)
    assert_allclose(result.path_lengths, [np.hypot(0.1, 0.004)], atol=1e-10)
    assert_allclose(
        routing.span_offsets(result, 0, jnp.array([0, 0.1])),
        [[0, 0, 0], [0, 0.004, 0]],
        atol=1e-9,
    )
    assert_allclose(result.spans[0].coefficients, 0.04, atol=1e-8)
    closed_over_length = jax.jit(
        lambda configuration: routing.solve_routing(robot, configuration).path_lengths
    )
    assert_allclose(closed_over_length(jnp.zeros(3)), result.path_lengths, atol=1e-10)


def test_curved_routing_shortens_relative_to_fixed_centerline():
    robot = _robot()
    routing = _routing(elements=6)
    result = routing.solve_routing(robot, jnp.array([3.0, 0, 0]))
    assert bool(result.converged)
    expected = 2 * np.sin(3.0 * 0.1 / 2) / 3.0
    assert_allclose(result.path_lengths, [expected], atol=2e-7)
    assert float(result.path_lengths[0]) < 0.1
    assert float(result.spans[0].dense_wall_violation) == 0
    warm = routing.solve_routing(robot, jnp.array([3.0, 0, 0]), initial_guess=result)
    assert bool(warm.converged)
    assert_allclose(warm.path_lengths, result.path_lengths, atol=1e-9)


def _wall_problem(target):
    return RoutingProblem(
        tangent_matrix=jnp.array([[[0.0], [1.0], [0.0]]]),
        tangent_offset=jnp.array([[1.0, -target, 0.0]]),
        weights=jnp.ones(1),
        equality_matrix=jnp.empty((0, 1)),
        target=jnp.empty(0),
        wall_matrix=jnp.array([[[1.0]]]),
        wall_offset=jnp.zeros((1, 1)),
        radius_squared=jnp.array([0.25]),
    )


def test_solver_matches_scipy_and_recovers_wall_multiplier():
    problem = _wall_problem(1.0)
    options = RoutingSolverOptions()
    state, diagnostics = solve_problem(problem, jnp.zeros(1), options)
    reference = minimize(
        lambda coefficients: float(objective(jnp.array(coefficients), problem)[0]),
        np.zeros(1),
        method="SLSQP",
        constraints={
            "type": "ineq",
            "fun": lambda coefficients: (
                -np.asarray(inequalities(jnp.array(coefficients), problem))
            ),
        },
        options={"ftol": 1e-12, "maxiter": 200},
    )
    assert reference.success
    assert diagnostics[0] == 1
    assert_allclose(state[0], reference.x[0], atol=1e-8)
    assert_allclose(state[1], 1 / np.sqrt(5), atol=1e-7)
    assert_allclose(equalities(state[:1], problem), [])


@pytest.mark.parametrize("coefficient, expected", [(0.2, True), (0.1, False)])
def test_solver_checks_optimality_independently_of_termination_code(
    monkeypatch, coefficient, expected
):
    import optimistix as optx

    def terminated_solve(*args, **kwargs):
        return SimpleNamespace(
            value=jnp.array([coefficient]),
            result=optx.RESULTS.nonlinear_max_steps_reached
            if expected
            else optx.RESULTS.successful,
            stats={
                "multipliers_eq": jnp.empty(0),
                "multipliers_ineq": jnp.zeros(1),
                "num_steps": 21,
            },
        )

    monkeypatch.setattr(optx, "minimise", terminated_solve)
    _, diagnostics = _forward_solve(
        _wall_problem(0.2), jnp.zeros(1), RoutingSolverOptions()
    )
    assert bool(diagnostics[0]) == expected


def test_solver_implicit_derivatives_and_batching():
    options = RoutingSolverOptions()

    def position(target):
        return solve_problem(_wall_problem(target), jnp.zeros(1), options)[0][0]

    assert_allclose(
        jax.jit(jax.vmap(position))(jnp.array([0.2, 1.0])), [0.2, 0.5], atol=1e-7
    )
    assert_allclose(jax.jacfwd(position)(0.2), 1.0, atol=1e-7)
    assert_allclose(jax.grad(position)(0.2), 1.0, atol=1e-7)
    assert_allclose(jax.grad(position)(1.0), 0.0, atol=1e-7)

    def contact_position(radius):
        problem = eqx.tree_at(
            lambda problem: problem.radius_squared,
            _wall_problem(1.0),
            jnp.array([radius**2]),
        )
        return solve_problem(problem, jnp.zeros(1), options)[0][0]

    assert_allclose(jax.grad(contact_position)(0.5), 1.0, atol=1e-7)


def test_shortest_path_length_derivative_matches_resolved_differences():
    robot = _robot()
    routing = _routing()

    def length(curvature):
        return routing.solve_routing(robot, jnp.array([curvature, 0, 0])).path_lengths[
            0
        ]

    finite_difference = (length(3.001) - length(2.999)) / 0.002
    assert_allclose(jax.grad(length)(3.0), finite_difference, rtol=2e-3, atol=1e-8)
    configuration = jnp.array([3.0, 0.0, 0.0])
    solution = routing.solve_routing(robot, configuration)
    envelope = eqx.filter_jit(routing.path_length_jacobian)(
        robot, configuration, solution
    )
    assert_allclose(envelope[0, 0], finite_difference, rtol=2e-3, atol=1e-8)


def test_envelope_gradient_at_wall_contact_matches_finite_differences():
    robot = _robot()
    routing = _routing(radii=(0.001,))
    configuration = jnp.array([3.0, 0.0, 0.0])
    solution = routing.solve_routing(robot, configuration)
    gradient = routing.path_length_jacobian(robot, configuration, solution)
    perturbation = jnp.array([1e-3, 0.0, 0.0])
    plus = routing.solve_routing(robot, configuration + perturbation)
    minus = routing.solve_routing(robot, configuration - perturbation)
    assert bool(solution.converged & plus.converged & minus.converged)
    finite_difference = (plus.path_lengths - minus.path_lengths) / 0.002
    assert_allclose(gradient[:, 0], finite_difference, rtol=2e-3, atol=1e-8)
    failed = eqx.tree_at(
        lambda value: value.spans[0].converged, solution, jnp.array(False)
    )
    assert bool(
        jnp.all(jnp.isnan(routing.path_length_jacobian(robot, configuration, failed)))
    )


def test_planar_rejects_spatial_offsets():
    routing = _routing(_spans(end_offset=[[0, 0, 0.001]]))
    with pytest.raises(ValueError, match="zero z"):
        routing.solve_routing(_robot(), jnp.zeros(3))


def test_gvs_routing_and_geometry_parameter_derivatives():
    segment = GVSSegment(
        link=LinkSpec.circular(
            young_modulus=1e3,
            shear_modulus=5e2,
            density=1000.0,
            material_damping_coefficient=0.0,
            length=0.1,
            radius=0.03,
            reference_strain=[0, 0, 0, 1, 0, 0],
        ),
        joint=JointSpec(type="fixed"),
        basis=StrainBasisSpec(
            type="monomial",
            strain_selector=[1, 1, 1, 1, 0, 0],
            basis_order=[0, 0, 0, 0, 0, 0],
        ),
        num_gauss_points=5,
    )
    robot = GVS.from_segments([segment], gravity=jnp.zeros(3))
    routing = _routing(elements=2)

    def length(endpoint):
        params = routing.params.replace(
            spans=routing.spans.replace(end_offset=jnp.array([[0.0, endpoint, 0.002]]))
        )
        result = routing.with_params(params).solve_routing(
            robot, jnp.zeros(robot.num_coordinates)
        )
        return result.path_lengths[0]

    expected = np.sqrt(0.1**2 + 0.004**2 + 0.002**2)
    assert_allclose(length(0.004), expected, atol=1e-10)
    assert_allclose(jax.grad(length)(0.004), 0.004 / expected, atol=1e-7)


def test_free_crossing_and_explicit_guide_have_different_lengths():
    robot = _robot((0.05, 0.05))
    free = _routing(_spans(end_segment_index=(1,)), radii=(0.02, 0.02), elements=2)
    guided_spans = _spans(
        start_offset=[[0, 0, 0], [0, 0.01, 0]],
        end_offset=[[0, 0.01, 0], [0, 0, 0]],
        span_path_index=(0, 0),
        start_segment_index=(0, 1),
        end_segment_index=(0, 1),
    )
    guided = _routing(guided_spans, radii=(0.02, 0.02), elements=2)
    free_result = free.solve_routing(robot, jnp.zeros(6))
    guided_result = guided.solve_routing(robot, jnp.zeros(6))
    assert bool(free_result.converged) and bool(guided_result.converged)
    assert_allclose(free_result.path_lengths, [0.1], atol=1e-10)
    assert_allclose(guided_result.path_lengths, [2 * np.hypot(0.05, 0.01)], atol=1e-10)


def test_wall_contact_limits_routing_and_reports_sampling_error():
    robot = _robot()
    routing = _routing(radii=(0.001,), elements=3)
    solution = routing.solve_routing(robot, jnp.array([3.0, 0.0, 0.0]))
    span = solution.spans[0]
    assert bool(solution.converged)
    assert bool(jnp.any(span.active_mask))
    assert bool(jnp.any(span.inequality_multipliers > 0))
    assert 2 * np.sin(0.15) / 3 < float(span.length) < 0.1
    assert float(span.primal_residual) < 1e-7
    assert float(span.dense_wall_violation) >= 0


def test_iteration_limit_and_contact_transition_do_not_return_valid_gradients():
    robot = _robot()
    routing = _routing(max_steps=1)

    def length(curvature):
        return routing.solve_routing(
            robot, jnp.array([curvature, 0.0, 0.0])
        ).path_lengths[0]

    solution = routing.solve_routing(robot, jnp.array([3.0, 0.0, 0.0]))
    assert not bool(solution.converged)
    assert jnp.isnan(jax.grad(length)(3.0))

    def position(target):
        return solve_problem(
            _wall_problem(target), jnp.zeros(1), RoutingSolverOptions()
        )[0][0]

    assert jnp.isnan(jax.jacfwd(position)(0.5))
    assert jnp.isnan(jax.grad(position)(0.5))


def test_spatial_pcs_multiple_paths():
    params = pcs_params(
        length=jnp.array([0.1]),
        radius=jnp.array([0.03]),
        density=jnp.array([1000.0]),
        young_modulus=jnp.array([1e3]),
        shear_modulus=jnp.array([5e2]),
        damping_matrix=jnp.zeros((6, 6)),
        gravity=jnp.zeros(3),
    )
    robot = PCS(params, PCSStructure(num_gauss_points=5), base_pose=spatial_base_pose())
    spans = _spans(
        start_offset=jnp.zeros((2, 3)),
        end_offset=[[0, 0.002, 0.004], [0, -0.003, 0.001]],
        span_path_index=(0, 1),
        start_segment_index=(0, 0),
        end_segment_index=(0, 0),
    )
    solution = _routing(spans, elements=2).solve_routing(robot, jnp.zeros(6))
    assert bool(solution.converged)
    assert_allclose(
        solution.path_lengths,
        np.sqrt([0.1**2 + 0.002**2 + 0.004**2, 0.1**2 + 0.003**2 + 0.001**2]),
        atol=1e-10,
    )


def test_empty_routing_has_no_paths_or_spans():
    spans = _spans(
        start_offset=jnp.empty((0, 3)),
        end_offset=jnp.empty((0, 3)),
        span_path_index=(),
        start_segment_index=(),
        end_segment_index=(),
    )
    solution = _routing(spans).solve_routing(_robot(), jnp.zeros(3))
    assert solution.path_lengths.shape == (0,)
    assert solution.spans == ()
    assert bool(solution.converged)
