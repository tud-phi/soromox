"""Installed routing strategies obey the common transmission and robot APIs."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from system_param_builders import pcs_params, planar_pcs_params

from soromox.actuation import (
    BaseThreadlikeRouting,
    CircularLumenParams,
    PiecewiseLinearRoutingDiscretization,
    RoutingSolverOptions,
    RoutingSpanParams,
    ShortestPathRouting,
    ShortestPathRoutingParams,
    StraightSpanRouting,
    ThreadlikeActuator,
    ThreadlikeFriction,
    ThreadlikeImpedance,
    ThreadlikeRouting,
    ThreadlikeTransmission,
    ThreadlikeTransmissionParams,
)
from soromox.execution.warp.actuation.threadlike.fixed_routing import (
    supports_linear_threadlike_matrix,
)
from soromox.systems import (
    GVS,
    PCS,
    GVSSegment,
    JointSpec,
    LinkSpec,
    PCSStructure,
    PlanarPCS,
    PlanarPCSStructure,
    StrainBasisSpec,
)

jax.config.update("jax_enable_x64", True)


def _routing(strategy, *, count=2, radius=0.02):
    offsets = jnp.zeros((count, 3)).at[:, 1].set(jnp.linspace(-0.004, 0.004, count))
    if strategy == "fixed":
        return ThreadlikeRouting.linear(intercept=offsets)
    spans = RoutingSpanParams(
        offsets, offsets, tuple(range(count)), (0,) * count, (0,) * count
    )
    if strategy == "straight":
        return StraightSpanRouting(spans)
    return ShortestPathRouting(
        ShortestPathRoutingParams(spans, CircularLumenParams.constant(radius, 1)),
        PiecewiseLinearRoutingDiscretization(
            elements_per_segment=3, quadrature_order=3
        ),
        RoutingSolverOptions(tolerance=1e-9),
    )


def _robot(kind="planar", *, actuators=(), passive_elements=(), floating=False):
    if kind == "gvs":
        return GVS.from_segments(
            [
                GVSSegment(
                    link=LinkSpec.circular(
                        young_modulus=1e3,
                        shear_modulus=500,
                        density=1000,
                        length=0.1,
                        radius=0.03,
                        reference_strain=[0, 0, 0, 1, 0, 0],
                    ),
                    joint=JointSpec.fixed(),
                    basis=StrainBasisSpec(
                        type="monomial",
                        strain_selector=[0, 0, 1, 1, 0, 0],
                        basis_order=[0] * 6,
                    ),
                    num_gauss_points=5,
                )
            ],
            gravity=jnp.zeros(3),
            actuators=actuators,
            passive_elements=passive_elements,
            floating_base=floating,
            backend="jax",
        )
    planar = kind == "planar"
    dimension = 3 if planar else 6
    params = (planar_pcs_params if planar else pcs_params)(
        length=jnp.array([0.1]),
        radius=jnp.array([0.03]),
        density=jnp.array([1000.0]),
        young_modulus=jnp.array([1000.0]),
        shear_modulus=jnp.array([500.0]),
        gravity=jnp.zeros(2 if planar else 3),
        damping_matrix=jnp.zeros((dimension, dimension)),
    )
    return (PlanarPCS if planar else PCS)(
        params,
        (PlanarPCSStructure if planar else PCSStructure)(num_gauss_points=5),
        actuators=actuators,
        passive_elements=passive_elements,
        floating_base=floating,
        backend="jax",
    )


def _configuration(robot, curvature=3.0):
    index = 0 if robot.is_planar or isinstance(robot, GVS) else 2
    internal = jnp.zeros(robot.num_internal_dofs).at[index].set(curvature)
    base = None
    if robot.floating_base:
        base = (
            jnp.array([0.3, 0.1, -0.2])
            if robot.is_planar
            else jnp.array([0.0, 0.0, np.sin(0.2), np.cos(0.2), 0.1, -0.2, 0.3])
        )
    return robot.pack_configuration(internal, base_pose=base)


@pytest.mark.parametrize("kind", ["planar", "spatial", "gvs"])
@pytest.mark.parametrize("strategy", ["fixed", "straight", "shortest"])
def test_installed_lengths_coordinates_rates_and_virtual_work(kind, strategy):
    routing = _routing(strategy)
    actuator = ThreadlikeActuator.tendons(routing)
    robot = _robot(kind, actuators=actuator)
    q = _configuration(robot)
    velocity = jnp.linspace(0.01, 0.03, robot.num_velocities)
    tension = jnp.array([0.2, 0.5])
    assert isinstance(routing, BaseThreadlikeRouting)
    assert actuator.num_channels == robot.num_actuators == 2

    transmission = actuator.transmission
    lengths = eqx.filter_jit(transmission.path_lengths)(robot, q)
    offsets = np.array([-0.004, 0.004])
    expected = (
        (1 - 3 * offsets) * 0.1
        if strategy == "fixed"
        else 2 * (1 / 3 - offsets) * np.sin(0.15)
    )
    assert_allclose(lengths, expected, atol=3e-7)
    assert_allclose(actuator.path_lengths(robot, q), lengths)
    assert_allclose(eqx.filter_jit(robot.actuator_coordinates)(q), -lengths)
    jacobian = eqx.filter_jit(robot.actuator_coordinate_jacobian)(q)
    assert_allclose(
        jacobian, jax.jacfwd(robot.actuator_coordinates)(q), atol=3e-7, rtol=2e-4
    )
    assert_allclose(transmission.path_length_jacobian(robot, q), -jacobian, atol=1e-12)
    assert_allclose(actuator.path_velocities(robot, q, velocity), -jacobian @ velocity)
    assert_allclose(robot.actuator_velocities(q, velocity), jacobian @ velocity)
    matrix = eqx.filter_jit(robot.actuation_matrix)(q)
    assert_allclose(matrix, jacobian.T, atol=1e-12)
    force = robot.actuation_force(q, tension)
    assert_allclose(force, matrix @ tension, atol=1e-12)
    assert_allclose(force @ velocity, tension @ robot.actuator_velocities(q, velocity))


@pytest.mark.parametrize("kind", ["planar", "spatial"])
@pytest.mark.parametrize("strategy", ["fixed", "straight", "shortest"])
def test_direct_transmission_on_floating_host_matches_robot_api(kind, strategy):
    actuator = ThreadlikeActuator.tendons(_routing(strategy, count=1))
    robot = _robot(kind, actuators=actuator, floating=True)
    q = _configuration(robot, curvature=0.5)
    transmission = actuator.transmission
    assert_allclose(transmission.coordinates(robot, q), robot.actuator_coordinates(q))
    jacobian = transmission.coordinate_jacobian(robot, q)
    assert jacobian.shape == (1, robot.num_velocities)
    assert_allclose(jacobian[:, : robot.num_base_velocities], 0)
    assert_allclose(jacobian, robot.actuator_coordinate_jacobian(q), atol=1e-10)
    assert_allclose(
        transmission.actuation_matrix(robot, q), robot.actuation_matrix(q), atol=1e-10
    )
    velocity = jnp.linspace(0.01, 0.03, robot.num_velocities)
    _, rate = jax.jvp(
        lambda values: transmission.coordinates(robot, values),
        (q,),
        (robot.configuration_derivative(q, velocity),),
    )
    assert_allclose(rate, jacobian @ velocity, atol=1e-9)
    points = actuator.path_poses(robot, q, jnp.asarray(0.05))
    assert points.shape == (1, 2 if kind == "planar" else 3)
    assert bool(jnp.all(jnp.isfinite(points)))


@pytest.mark.parametrize("strategy", ["straight", "shortest"])
def test_parameter_constructors_updates_and_gradients_preserve_strategy(strategy):
    routing = _routing(strategy, count=1)
    actuator = ThreadlikeActuator.tendons(routing)
    robot = _robot(actuators=actuator)
    q = _configuration(robot)
    from_params = ThreadlikeActuator.tendons(routing.params)
    assert isinstance(from_params.transmission.routing, type(routing))
    direct = ThreadlikeTransmission(
        ThreadlikeTransmissionParams(routing.params, jnp.array([2.0]))
    )
    assert_allclose(
        direct.coordinates(robot, q), 2 * actuator.path_lengths(robot, q), atol=3e-7
    )

    def updated_model(offset):
        spans = routing.spans.replace(end_offset=jnp.array([[0.0, offset, 0.0]]))
        params = (
            spans if strategy == "straight" else routing.params.replace(spans=spans)
        )
        transmission = actuator.params.transmission.replace(routing=params)
        return robot.update_actuator_params(0, transmission=transmission)

    @eqx.filter_jit
    def lengths(offset):
        return updated_model(offset).actuator_coordinates(q)

    updated = updated_model(0.002)
    actual = updated.actuators[0].transmission.routing
    if strategy == "shortest":
        assert actual.discretization == routing.discretization
        assert actual.solver_options == routing.solver_options
    assert_allclose(lengths(0.002), updated.actuator_coordinates(q), atol=1e-10)
    derivative = jax.jacfwd(lengths)(0.002)
    numerical = (lengths(0.00201) - lengths(0.00199)) / 2e-5
    assert_allclose(derivative, numerical, atol=2e-6, rtol=2e-3)
    with pytest.raises(ValueError, match="topology|type|reconstruction"):
        actuator.update_params(
            transmission=actuator.params.transmission.replace(
                routing=_routing("fixed", count=1).params
            )
        )
    with pytest.raises(ValueError, match="frictionless"):
        _robot(
            actuators=ThreadlikeActuator.tendons(
                routing, friction=ThreadlikeFriction.capstan(coefficient=0.1)
            )
        )


def test_mixed_strategy_channel_order_and_arbitrary_coordinate_scaling():
    routes = [_routing(name, count=1) for name in ("fixed", "straight", "shortest")]
    actuators = [ThreadlikeActuator.tendons(route) for route in routes]
    actuators[1] = ThreadlikeActuator.push_rods(routes[1])
    actuators[2] = ThreadlikeActuator.pressure_chambers(
        routes[2], effective_areas=jnp.array([2e-4])
    )
    robot = _robot(actuators=actuators)
    q = _configuration(robot)
    expected = jnp.concatenate(
        [actuator.coordinates(robot, q) for actuator in actuators]
    )
    assert_allclose(robot.actuator_coordinates(q), expected)
    assert expected[0] < 0 < expected[1]
    assert_allclose(expected[2], 2e-4 * routes[2].path_lengths(robot, q)[0])
    assert_allclose(robot.actuation_matrix(q), robot.actuator_coordinate_jacobian(q).T)
    assert not supports_linear_threadlike_matrix(robot)
    batched = jax.jit(jax.vmap(robot.actuator_coordinates))(jnp.stack([q, q / 2]))
    assert_allclose(batched[0], expected, atol=1e-10)
    assert_allclose(batched[1], robot.actuator_coordinates(q / 2), atol=1e-10)


@pytest.mark.parametrize("strategy", ["straight", "shortest"])
def test_failed_routes_do_not_expose_usable_actuator_coordinates(strategy, monkeypatch):
    routing = _routing(strategy, count=1)
    robot = _robot(actuators=ThreadlikeActuator.tendons(routing))
    q = _configuration(robot)
    solution = routing.solve_routing(robot, q)
    failed = eqx.tree_at(
        lambda value: value.spans[0].converged, solution, jnp.array(False)
    )
    monkeypatch.setattr(type(routing), "solve_routing", lambda *args, **kwargs: failed)
    assert bool(jnp.all(jnp.isnan(robot.actuator_coordinates(q))))
    assert bool(jnp.all(jnp.isnan(robot.actuator_coordinate_jacobian(q))))
    assert bool(jnp.all(jnp.isnan(robot.actuation_matrix(q))))


@pytest.mark.parametrize("strategy", ["straight", "shortest"])
def test_passive_paths_use_the_same_strategy_contract(strategy):
    routing = _routing(strategy, count=1)
    passive = ThreadlikeImpedance(
        routing=routing,
        stiffness=jnp.array([10.0]),
        damping=jnp.array([0.2]),
        rest_length=jnp.array([0.09]),
    )
    robot = _robot(passive_elements=(passive,))
    q = _configuration(robot)
    assert_allclose(
        passive.elastic_force(robot, q),
        jax.grad(lambda values: passive.elastic_energy(robot, values))(q),
        atol=1e-7,
    )
    jacobian = passive.path_length_jacobian(robot, q)
    assert_allclose(passive.damping_matrix(robot, q), 0.2 * jacobian.T @ jacobian)


def test_wall_contact_coordinate_gradient_matches_resolved_length_changes():
    spans = RoutingSpanParams([[0, 0, 0]], [[0, 0, 0]], (0,), (0,), (0,))
    routing = ShortestPathRouting(
        ShortestPathRoutingParams(spans, CircularLumenParams.constant(0.001, 1)),
        PiecewiseLinearRoutingDiscretization(
            elements_per_segment=3, quadrature_order=3
        ),
        RoutingSolverOptions(wall_tolerance=3e-5),
    )
    robot = _robot(actuators=ThreadlikeActuator.tendons(routing))
    q = _configuration(robot)
    delta = jnp.array([0.001, 0, 0])
    finite_difference = (
        robot.actuator_coordinates(q + delta) - robot.actuator_coordinates(q - delta)
    ) / 0.002
    assert_allclose(
        robot.actuator_coordinate_jacobian(q)[:, 0],
        finite_difference,
        atol=1e-8,
        rtol=2e-3,
    )


@pytest.fixture(scope="module", params=["planar", "gvs"])
def sampling_robot(request):
    """Create unequal links with a floating base and a moving joint on GVS."""
    lengths = jnp.array([0.1, 0.08, 0.12])
    if request.param == "planar":
        robot = PlanarPCS(
            planar_pcs_params(
                length=lengths,
                radius=jnp.full(3, 0.03),
                density=jnp.full(3, 1000.0),
                young_modulus=jnp.full(3, 1000.0),
                shear_modulus=jnp.full(3, 500.0),
                gravity=jnp.zeros(2),
                damping_matrix=jnp.zeros((9, 9)),
            ),
            PlanarPCSStructure(num_gauss_points=3),
            backend="jax",
        )
        q = jnp.array([0.2, 0, 0, -0.1, 0, 0, 0.3, 0, 0])
    else:
        robot = GVS.from_segments(
            [
                GVSSegment(
                    link=LinkSpec.circular(
                        young_modulus=1000.0,
                        shear_modulus=500.0,
                        density=1000.0,
                        length=length,
                        radius=0.03,
                        reference_strain=[0, 0, 0, 1, 0, 0],
                    ),
                    joint=JointSpec.revolute(axis="z")
                    if index == 1
                    else JointSpec.fixed(),
                    basis=StrainBasisSpec(
                        type="monomial",
                        strain_selector=[0, 0, 1, 0, 0, 0],
                        basis_order=[0] * 6,
                    ),
                    num_gauss_points=5,
                )
                for index, length in enumerate(lengths)
            ],
            floating_base=True,
            backend="jax",
        )
        q = robot.pack_configuration(
            jnp.array([0.2, 0.4, -0.1, 0.3]),
            base_pose=jnp.array([0, 0, np.sin(0.1), np.cos(0.1), 0.5, -0.2, 0.1]),
        )
    return robot, q


@pytest.fixture(scope="module", params=["straight", "shortest"])
def sampling_case(sampling_robot, request):
    """Resolve interleaved paths whose spans have different mesh sizes."""
    robot, q = sampling_robot
    offsets = jnp.array([[0, 0.003, 0], [0, -0.002, 0], [0, 0.003, 0]])
    spans = RoutingSpanParams(offsets, offsets, (0, 1, 0), (0, 1, 1), (0, 1, 2))
    routing = (
        StraightSpanRouting(spans)
        if request.param == "straight"
        else ShortestPathRouting(
            ShortestPathRoutingParams(spans, CircularLumenParams.constant(0.02, 3)),
            PiecewiseLinearRoutingDiscretization(
                elements_per_segment=2, quadrature_order=3
            ),
        )
    )
    solution = eqx.filter_jit(routing.solve_routing)(robot, q)
    assert bool(solution.converged)
    return robot, q, routing, solution


def _span_sampling_reference(robot, q, routing, solution, station):
    """Sample every span independently as a reference for selection and clamping."""
    boundaries = jnp.concatenate((jnp.zeros(1), jnp.cumsum(robot.segment_length)))
    paths = []
    for path in range(routing.num_paths):
        indices = [
            i for i, item in enumerate(routing.spans.span_path_index) if item == path
        ]
        point = jnp.zeros(3)
        for index in reversed(indices):
            start = boundaries[routing.spans.start_segment_index[index]]
            end = boundaries[routing.spans.end_segment_index[index] + 1]
            fraction = jnp.clip((station - start) / (end - start), 0, 1)
            candidate = routing.span_positions(robot, q, solution, index, fraction)[0]
            point = jnp.where(
                (station <= end) | (index == indices[-1]), candidate, point
            )
        paths.append(point[:2] if robot.is_planar else point)
    return jnp.stack(paths)


def test_selected_span_sampling_matches_endpoints_boundaries_and_batches(sampling_case):
    robot, q, routing, solution = sampling_case
    stations = jnp.array([-0.02, 0, 0.05, 0.1, 0.10001, 0.14, 0.18, 0.24, 0.3, 0.32])

    def sample(station):
        return routing.path_positions(robot, q, station, solution)

    def reference(station):
        return _span_sampling_reference(robot, q, routing, solution, station)

    assert_allclose(
        jax.jit(jax.vmap(sample))(stations),
        jax.jit(jax.vmap(reference))(stations),
        atol=1e-12,
    )
    for station in stations:
        assert_allclose(sample(station), reference(station), atol=1e-12)
    _, derivative = jax.jvp(sample, (jnp.array(0.143),), (jnp.array(1.0),))
    expected = (reference(jnp.array(0.143001)) - reference(jnp.array(0.142999))) / 2e-6
    assert_allclose(derivative, expected, atol=1e-8)


def test_selected_span_sampling_limits_kinematics_to_physical_paths(
    sampling_case, monkeypatch
):
    robot, q, routing, solution = sampling_case
    expected = _span_sampling_reference(robot, q, routing, solution, jnp.array(0.14))

    def unused_span(*args, **kwargs):
        raise AssertionError("Sampling selected paths must not evaluate every span")

    monkeypatch.setattr(type(routing), "span_positions", unused_span)
    calls = []
    original = type(robot).forward_kinematics_abscissa_batched

    def batched_positions(host, configuration, stations):
        calls.append(stations.shape)
        return original(host, configuration, stations)

    monkeypatch.setattr(
        type(robot), "forward_kinematics_abscissa_batched", batched_positions
    )
    actual = routing.path_positions(robot, q, jnp.array(0.14), solution)
    assert_allclose(actual, expected, atol=1e-12)
    assert calls == (
        [(routing.num_paths,)] if isinstance(routing, ShortestPathRouting) else []
    )


@pytest.mark.parametrize("strategy", ["straight", "shortest"])
def test_empty_paths_keep_sampling_shape(strategy):
    routing = _routing(strategy, count=0)
    robot = _robot()
    q = _configuration(robot)
    assert eqx.filter_jit(routing.path_positions)(robot, q, jnp.array(0.05)).shape == (
        0,
        2,
    )
