# ruff: noqa: E402
"""Continuum Capstan friction for threadlike actuator transmissions."""

import equinox as eqx
import jax
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import Array
from numpy.testing import assert_allclose
from system_param_builders import (
    pcs_params,
    planar_base_pose,
    planar_pcs_params,
    spatial_base_pose,
)

from soromox.actuation import (
    BaseThreadlikeFrictionParams,
    BaseThreadlikeRoutingParams,
    CapstanFrictionParams,
    ThreadlikeActuator,
    ThreadlikeFriction,
    ThreadlikeImpedance,
    ThreadlikeRouting,
    capstan_effort_ratio,
    threadlike_constant_strain_turn_rate,
    threadlike_tangent,
    threadlike_turn_rate,
)
from soromox.actuation.friction import threadlike_turning_angle
from soromox.coordinate_transformations import ActuationSpaceDynamics
from soromox.execution.warp.actuation.threadlike import (
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
    StrainBasisSpec,
)
from soromox.systems.pcs import PlanarPCSStructure
from soromox.utils.lie_algebra import so3


def _planar(*, num_segments: int = 1, actuators=None) -> PlanarPCS:
    """Construct a planar PCS fixture with all strain coordinates active."""
    body = planar_pcs_params(
        length=jnp.full((num_segments,), 0.1),
        radius=jnp.full((num_segments,), 0.02),
        density=jnp.full((num_segments,), 1000.0),
        gravity=jnp.zeros((2,)),
        young_modulus=jnp.full((num_segments,), 1e3),
        shear_modulus=jnp.full((num_segments,), 5e2),
        damping_matrix=1e-3 * jnp.eye(3 * num_segments),
    )
    return PlanarPCS(
        body,
        PlanarPCSStructure(num_gauss_points=7),
        base_pose=planar_base_pose(),
        actuators=actuators,
    )


def _spatial(
    *, num_segments: int = 1, actuators=None, num_gauss_points: int = 7
) -> PCS:
    """Construct a spatial PCS fixture with all strain coordinates active."""
    body = pcs_params(
        length=jnp.full((num_segments,), 0.1),
        radius=jnp.full((num_segments,), 0.02),
        density=jnp.full((num_segments,), 1000.0),
        gravity=jnp.zeros((3,)),
        young_modulus=jnp.full((num_segments,), 1e3),
        shear_modulus=jnp.full((num_segments,), 5e2),
        damping_matrix=1e-3 * jnp.eye(6 * num_segments),
    )
    return PCS(
        body,
        PCSStructure(num_gauss_points=num_gauss_points),
        base_pose=spatial_base_pose(),
        actuators=actuators,
    )


def _gvs(*, actuators=None, basis_order=0) -> GVS:
    """Construct a one-link GVS fixture with fixed base and joint."""
    segment = GVSSegment(
        link=LinkSpec.circular(
            young_modulus=3e5,
            shear_modulus=3e5 / 2.9,
            density=1300.0,
            material_damping_coefficient=1e4,
            length=0.1,
            radius=0.015,
            reference_strain=[0, 0, 0, 1, 0, 0],
        ),
        joint=JointSpec(type="fixed"),
        basis=StrainBasisSpec(
            type="monomial",
            strain_selector=[0, 0, 1, 1, 0, 0],
            basis_order=[0, 0, basis_order, 0, 0, 0],
        ),
        num_gauss_points=7,
    )
    return GVS.from_segments([segment], gravity=jnp.zeros((3,)), actuators=actuators)


def _routing(
    offset: float = 0.02,
    *,
    start_segment: int = 0,
    end_segment: int = 0,
) -> ThreadlikeRouting:
    """Construct one constant-offset material-frame path."""
    return ThreadlikeRouting.linear(
        intercept=jnp.array([0.0, offset, 0.0]),
        start_segment_index=(start_segment,),
        end_segment_index=(end_segment,),
    )


def _constant_bending_configuration(robot, curvature: float) -> Array:
    """Return a pure positive-z bending configuration for a fixture host."""
    if isinstance(robot, PlanarPCS):
        return jnp.array([curvature, 0.0, 0.0])
    if isinstance(robot, PCS):
        return jnp.array([0.0, 0.0, curvature, 0.0, 0.0, 0.0])
    return jnp.array([curvature, 0.0])


@pytest.mark.parametrize("factory", [_planar, _spatial, _gvs])
def test_zero_coefficient_matches_frictionless_transmission(factory):
    """Keep the zero-loss result identical for every continuum host."""
    routing = _routing()
    ideal = factory(actuators=ThreadlikeActuator.tendons(routing))
    zero_loss = factory(
        actuators=ThreadlikeActuator.tendons(
            routing, friction=ThreadlikeFriction.capstan(coefficient=0.0)
        )
    )
    q = _constant_bending_configuration(ideal, 3.0)

    assert jnp.array_equal(ideal.actuation_matrix(q), zero_loss.actuation_matrix(q))
    assert_allclose(
        ideal.actuator_coordinate_jacobian(q),
        zero_loss.actuator_coordinate_jacobian(q),
    )


@pytest.mark.parametrize("factory", [_planar, _spatial, _gvs])
def test_constant_curvature_capstan_matrix_matches_closed_form(factory):
    """Match the continuum integral for a constant-offset circular path."""
    coefficient = 0.6
    curvature = 4.0
    offset = 0.02
    routing = _routing(offset)
    robot = factory(
        actuators=ThreadlikeActuator.tendons(
            routing,
            friction=ThreadlikeFriction.capstan(coefficient=coefficient),
        )
    )
    q = _constant_bending_configuration(robot, curvature)
    length = 0.1
    weighted_length = -jnp.expm1(-coefficient * curvature * length) / (
        coefficient * curvature
    )

    expected = jnp.zeros((robot.num_velocities, 1))
    if isinstance(robot, (PlanarPCS, GVS)):
        expected = expected.at[0, 0].set(offset * weighted_length)
        expected = expected.at[1, 0].set(-weighted_length)
    else:
        expected = expected.at[2, 0].set(offset * weighted_length)
        expected = expected.at[3, 0].set(-weighted_length)

    assert_allclose(robot.actuation_matrix(q), expected, rtol=5e-8, atol=5e-10)


def test_turn_rate_formula_matches_autodiff_of_spatial_unit_tangent():
    """Verify the analytical production formula against a test-only JVP."""
    angular = jnp.array([0.7, -0.3, 1.1])
    linear = jnp.array([1.0, 0.2, -0.1])
    angular_derivative = jnp.array([0.2, 0.1, -0.4])
    linear_derivative = jnp.array([-0.1, 0.3, 0.2])
    offset = jnp.array([0.0, 0.03, -0.02])
    offset_derivative = jnp.array([0.0, -0.2, 0.1])
    offset_second_derivative = jnp.array([0.0, 0.4, -0.15])

    analytical = threadlike_turn_rate(
        angular,
        linear,
        angular_derivative,
        linear_derivative,
        offset,
        offset_derivative,
        offset_second_derivative,
        eps=1e-12,
    )

    def spatial_unit_tangent(delta: Array) -> Array:
        """Construct a local spatial tangent curve for test differentiation."""
        omega = angular + delta * angular_derivative
        velocity = linear + delta * linear_derivative
        current_offset = (
            offset
            + delta * offset_derivative
            + 0.5 * delta**2 * offset_second_derivative
        )
        current_offset_derivative = offset_derivative + delta * offset_second_derivative
        tangent = threadlike_tangent(
            omega, velocity, current_offset, current_offset_derivative
        )
        unit = tangent / jnp.linalg.norm(tangent)
        return so3.exp(delta * angular) @ unit

    _, derivative = jax.jvp(
        spatial_unit_tangent, (jnp.asarray(0.0),), (jnp.asarray(1.0),)
    )
    assert_allclose(analytical, jnp.linalg.norm(derivative), rtol=2e-12, atol=2e-12)


def test_constant_strain_turn_rate_matches_full_analytical_formula():
    """Verify the PCS specialization of the general tangent derivative."""
    angular = jnp.array([0.7, -0.3, 1.1])
    linear = jnp.array([1.0, 0.2, -0.1])
    offset = jnp.array([0.0, 0.03, -0.02])
    offset_derivative = jnp.array([0.0, -0.2, 0.1])
    offset_second_derivative = jnp.array([0.0, 0.4, -0.15])

    specialized = threadlike_constant_strain_turn_rate(
        angular,
        linear,
        offset,
        offset_derivative,
        offset_second_derivative,
        eps=1e-12,
    )
    general = threadlike_turn_rate(
        angular,
        linear,
        jnp.zeros((3,)),
        jnp.zeros((3,)),
        offset,
        offset_derivative,
        offset_second_derivative,
        eps=1e-12,
    )

    assert_allclose(specialized, general, rtol=2e-12, atol=2e-12)


class SinusoidalRoutingParams(BaseThreadlikeRoutingParams):
    """Parameters for an analytical sinusoidal test routing."""

    amplitude: Array
    frequency: Array
    start_segment_index: tuple[int, ...] = eqx.field(static=True)
    end_segment_index: tuple[int, ...] = eqx.field(static=True)

    @property
    def num_paths(self) -> int:
        """Return the number of sinusoidal paths."""
        return int(self.amplitude.shape[0])

    def validate_structure(self) -> None:
        """Validate amplitude and frequency shapes."""
        if self.frequency.shape != self.amplitude.shape:
            raise ValueError("frequency must match amplitude shape.")


def _sinusoidal_offset(params: SinusoidalRoutingParams, s: Array) -> Array:
    """Return a sinusoidal material-``y`` routing offset."""
    y = params.amplitude * jnp.sin(params.frequency * s)
    return jnp.stack((jnp.zeros_like(y), y, jnp.zeros_like(y)), axis=-1)


def _sinusoidal_derivative(params: SinusoidalRoutingParams, s: Array) -> Array:
    """Return the analytical first derivative of the sinusoidal offset."""
    y = params.amplitude * params.frequency * jnp.cos(params.frequency * s)
    return jnp.stack((jnp.zeros_like(y), y, jnp.zeros_like(y)), axis=-1)


def _sinusoidal_second_derivative(params: SinusoidalRoutingParams, s: Array) -> Array:
    """Return the analytical second derivative of the sinusoidal offset."""
    y = -(params.frequency**2) * params.amplitude * jnp.sin(params.frequency * s)
    return jnp.stack((jnp.zeros_like(y), y, jnp.zeros_like(y)), axis=-1)


def test_custom_routing_callbacks_preserve_the_path_axis():
    """Keep vectorized callback output shaped ``(num_paths, 3)``."""
    params = SinusoidalRoutingParams(
        amplitude=jnp.array([0.01, 0.02]),
        frequency=jnp.array([8.0, 12.0]),
        start_segment_index=(0, 0),
        end_segment_index=(0, 0),
    )

    for callback in (
        _sinusoidal_offset,
        _sinusoidal_derivative,
        _sinusoidal_second_derivative,
    ):
        assert callback(params, jnp.asarray(0.04)).shape == (2, 3)


def test_custom_routing_analytical_turn_matches_dense_numerical_reference():
    """Include material-tangent variation in accumulated continuum turn."""
    params = SinusoidalRoutingParams(
        amplitude=jnp.array([0.015]),
        frequency=jnp.array([18.0]),
        start_segment_index=(0,),
        end_segment_index=(0,),
    )
    routing = ThreadlikeRouting(
        params=params,
        offset_fn=_sinusoidal_offset,
        derivative_fn=_sinusoidal_derivative,
        second_derivative_fn=_sinusoidal_second_derivative,
    )
    robot = _spatial(num_gauss_points=17)
    q = jnp.array([0.2, -0.4, 2.0, 0.1, 0.05, -0.03])
    strains = robot.strain(q).reshape((1, 6))
    accumulated = robot._threadlike_accumulated_turn(strains, routing, robot.length)[0]

    points = jnp.linspace(0.0, robot.length, 20001)
    rates = robot._threadlike_turn_rates(strains, routing, jnp.asarray(0), points)[:, 0]
    reference = jnp.trapezoid(rates, points)
    assert_allclose(accumulated, reference, rtol=5e-5, atol=5e-8)


def test_gvs_varying_strain_turn_rate_uses_analytical_basis_derivative():
    """Match GVS turn rate to test-only differentiation of its strain field."""
    routing = _routing()
    robot = _gvs(basis_order=1)
    q = jnp.array([1.2, -0.7, 0.1])
    gathered = robot._min_size_gathered(q)
    point = jnp.asarray(0.063)
    actual = robot._threadlike_turn_rates(
        gathered, routing, jnp.asarray(0), point[None]
    )[0, 0]

    def spatial_unit_tangent(s: Array) -> Array:
        """Evaluate the world path tangent for test-only differentiation."""
        strain, _ = robot._threadlike_strain_and_derivative(
            gathered[0, 1], jnp.asarray(0), s[None]
        )
        current = strain[0]
        offset = routing.offset(routing.params, s)[0]
        derivative = routing.derivative(routing.params, s)[0]
        tangent = threadlike_tangent(current[:3], current[3:], offset, derivative)
        unit = tangent / jnp.linalg.norm(tangent)
        rotation = robot.forward_kinematics(q, s)[:3, :3]
        return rotation @ unit

    _, derivative = jax.jvp(spatial_unit_tangent, (point,), (jnp.asarray(1.0),))
    assert_allclose(actual, jnp.linalg.norm(derivative), rtol=5e-6, atol=2e-7)


@pytest.mark.parametrize("factory", [_planar, _spatial, _gvs])
def test_friction_separates_coordinate_jacobian_from_actuation_matrix(factory):
    """Keep kinematics geometric while attenuating generalized force."""
    routing = _routing()
    robot = factory(
        actuators=ThreadlikeActuator.tendons(
            routing, friction=ThreadlikeFriction.capstan(coefficient=0.8)
        )
    )
    q = _constant_bending_configuration(robot, 5.0)
    coordinate_jacobian = robot.actuator_coordinate_jacobian(q)

    assert_allclose(
        coordinate_jacobian,
        jax.jacrev(robot.actuator_coordinates)(q),
        rtol=3e-7,
        atol=3e-9,
    )
    assert not jnp.allclose(coordinate_jacobian, robot.actuation_matrix(q).T)


def test_actuation_space_preserves_the_lossy_force_map():
    """Transform effort covectors with ``J^-T A`` instead of assuming identity."""
    routing = _routing()
    robot = _planar(
        actuators=ThreadlikeActuator.tendons(
            routing, friction=ThreadlikeFriction.capstan(coefficient=0.8)
        )
    )
    transform = ActuationSpaceDynamics(robot)
    q = _constant_bending_configuration(robot, 5.0)
    expected = transform.jacobian_inverse(q).T @ robot.actuation_matrix(q)

    assert_allclose(transform.actuation_matrix(q), expected)
    assert not jnp.allclose(
        transform.actuation_matrix(q), jnp.array([[1.0], [0.0], [0.0]])
    )


class HyperbolicFrictionParams(BaseThreadlikeFrictionParams):
    """Parameter of a custom hyperbolic effort-loss law."""

    coefficient: Array

    def validate_values(self) -> None:
        """Reject negative custom coefficients."""
        if bool(jnp.any(self.coefficient < 0.0)):
            raise ValueError("coefficient must be non-negative.")


def _hyperbolic_effort_ratio(
    params: HyperbolicFrictionParams, accumulated_turn: Array
) -> Array:
    """Return a custom effort ratio using only explicit accumulated turn."""
    return 1.0 / (1.0 + params.coefficient * accumulated_turn)


def test_custom_effort_ratio_needs_no_quadrature_context():
    """Keep custom friction laws independent of host geometry containers."""
    routing = _routing()
    friction = ThreadlikeFriction(
        params=HyperbolicFrictionParams(coefficient=jnp.asarray(0.5)),
        effort_ratio_fn=_hyperbolic_effort_ratio,
        is_frictionless=False,
    )
    robot = _spatial(actuators=ThreadlikeActuator.tendons(routing, friction=friction))
    q = _constant_bending_configuration(robot, 4.0)

    capstan = _spatial(
        actuators=ThreadlikeActuator.tendons(
            routing, friction=ThreadlikeFriction.capstan(coefficient=0.5)
        )
    )
    assert jnp.isfinite(robot.actuation_matrix(q)).all()
    assert not jnp.allclose(robot.actuation_matrix(q), capstan.actuation_matrix(q))


def test_capstan_effort_ratio_name_and_values():
    """Expose a modality-neutral ratio name for all threadlike actuators."""
    params = CapstanFrictionParams(coefficient=jnp.array([0.0, 0.5]))
    turn = jnp.array([2.0, 2.0])
    assert_allclose(capstan_effort_ratio(params, turn), jnp.array([1.0, jnp.exp(-1.0)]))


def test_friction_is_measured_from_each_path_anchor():
    """Exclude curvature before a path enters its configured segment span."""
    routing = _routing(start_segment=1, end_segment=1)
    actuator = ThreadlikeActuator.tendons(
        routing, friction=ThreadlikeFriction.capstan(coefficient=0.7)
    )
    robot = _planar(num_segments=2, actuators=actuator)
    q = jnp.array([8.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    strains = robot.strain(q).reshape((2, 3))

    assert_allclose(
        robot._threadlike_accumulated_turn(strains, routing, jnp.asarray(0.1)),
        jnp.zeros((1,)),
        atol=1e-12,
    )
    assert_allclose(
        robot._threadlike_accumulated_turn(strains, routing, jnp.asarray(0.2)),
        jnp.array([0.2]),
        rtol=5e-8,
        atol=1e-10,
    )


def test_accumulated_turn_includes_piecewise_host_tangent_jumps():
    """Include genuine tangent jumps introduced by the host discretization."""
    routing = _routing(offset=0.0, end_segment=1)
    robot = _planar(num_segments=2)
    strains = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    assert_allclose(
        robot._threadlike_accumulated_turn(strains, routing, jnp.asarray(0.2)),
        jnp.array([0.5 * jnp.pi]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_custom_routing_requires_analytical_second_derivative_for_friction():
    """Reject production autodiff fallbacks for custom routing geometry."""
    params = SinusoidalRoutingParams(
        amplitude=jnp.array([0.01]),
        frequency=jnp.array([10.0]),
        start_segment_index=(0,),
        end_segment_index=(0,),
    )
    routing = ThreadlikeRouting(
        params=params,
        offset_fn=_sinusoidal_offset,
        derivative_fn=_sinusoidal_derivative,
    )
    actuator = ThreadlikeActuator.tendons(
        routing, friction=ThreadlikeFriction.capstan(coefficient=0.4)
    )

    with pytest.raises(ValueError, match="second_derivative_fn"):
        _spatial(actuators=actuator)


def test_custom_callbacks_with_linear_params_require_a_second_derivative():
    """Do not infer linear behavior from the parameter class alone."""
    params = ThreadlikeRouting.linear(
        intercept=jnp.array([0.0, 0.01, 0.0]),
        slope=jnp.array([0.0, 0.02, 0.0]),
    ).params

    def nonlinear_offset(path_params, s):
        """Return a nonlinear offset using linear-routing storage."""
        return path_params.intercept + s**2 * path_params.slope

    def nonlinear_derivative(path_params, s):
        """Return the first derivative of the nonlinear offset."""
        return 2.0 * s * path_params.slope

    routing = ThreadlikeRouting(
        params=params,
        offset_fn=nonlinear_offset,
        derivative_fn=nonlinear_derivative,
    )
    actuator = ThreadlikeActuator.tendons(
        routing, friction=ThreadlikeFriction.capstan(coefficient=0.4)
    )

    assert not routing.has_analytical_second_derivative
    with pytest.raises(ValueError, match="second_derivative_fn"):
        _spatial(actuators=actuator)


def test_builtin_linear_callbacks_supply_zero_second_derivative():
    """Retain the implicit analytical derivative for built-in linear routing."""
    linear = ThreadlikeRouting.linear(intercept=jnp.array([0.0, 0.01, 0.0]))
    routing = ThreadlikeRouting(params=linear.params)

    assert routing.has_analytical_second_derivative
    assert_allclose(
        routing.second_derivative(routing.params, jnp.asarray(0.04)),
        jnp.zeros((1, 3)),
    )


@pytest.mark.parametrize(
    "right_unit_tangent",
    [jnp.zeros((3,)), jnp.array([1.0, 0.0, 0.0]), jnp.array([-1.0, 0.0, 0.0])],
)
def test_turning_angle_has_finite_values_and_derivatives(right_unit_tangent):
    """Keep boundary-angle values and both differentiation modes finite."""

    def angle(tangents: Array) -> Array:
        """Evaluate the angle from packed left and right tangents."""
        return threadlike_turning_angle(tangents[:3], tangents[3:])

    tangents = jnp.concatenate((jnp.zeros((3,)), right_unit_tangent))
    assert jnp.isfinite(angle(tangents))
    assert jnp.isfinite(jax.jacfwd(angle)(tangents)).all()
    assert jnp.isfinite(jax.jacrev(angle)(tangents)).all()


@pytest.mark.parametrize(
    "turn_rate",
    [threadlike_turn_rate, threadlike_constant_strain_turn_rate],
)
def test_turn_rate_has_finite_values_and_derivatives_at_collapsed_tangent(turn_rate):
    """Keep analytical turn-rate singularity handling differentiable."""
    zeros = jnp.zeros((3,))

    def rate(linear_strain: Array) -> Array:
        """Evaluate either turn-rate formula at a collapsed path tangent."""
        if turn_rate is threadlike_turn_rate:
            return turn_rate(
                zeros,
                linear_strain,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                eps=1e-12,
            )
        return turn_rate(
            zeros,
            linear_strain,
            zeros,
            zeros,
            zeros,
            eps=1e-12,
        )

    assert jnp.isfinite(rate(zeros))
    assert jnp.isfinite(jax.jacfwd(rate)(zeros)).all()
    assert jnp.isfinite(jax.jacrev(rate)(zeros)).all()


@pytest.mark.parametrize("value", [-0.1, jnp.inf, jnp.nan])
def test_capstan_parameters_reject_invalid_values(value):
    """Reject coefficients that could create effort or poison numerics."""
    with pytest.raises(ValueError, match="coefficient"):
        CapstanFrictionParams(coefficient=jnp.asarray(value))


def test_capstan_parameters_reject_wrong_per_path_shape():
    """Require one coefficient per routed path when an array is supplied."""
    routing = ThreadlikeRouting.linear(
        intercept=jnp.array([[0.0, 0.01, 0.0], [0.0, -0.01, 0.0]])
    )
    friction = ThreadlikeFriction.capstan(coefficient=jnp.ones((3,)))
    with pytest.raises(ValueError, match="coefficient must be a scalar or"):
        ThreadlikeActuator.tendons(routing, friction=friction)


def test_friction_is_active_only_and_disables_warp_matrix_path():
    """Keep passive impedance conservative and gate unsupported Warp friction."""
    routing = _routing()
    friction = ThreadlikeFriction.capstan(coefficient=0.3)
    actuator = ThreadlikeActuator.tendons(routing, friction=friction)
    robot = _spatial(actuators=actuator)

    assert not supports_linear_threadlike_matrix(robot)
    with pytest.raises(TypeError, match="unexpected keyword argument 'friction'"):
        ThreadlikeImpedance(
            routing=routing,
            stiffness=jnp.ones((1,)),
            damping=jnp.ones((1,)),
            rest_length=jnp.ones((1,)),
            friction=friction,
        )
