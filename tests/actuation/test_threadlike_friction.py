# ruff: noqa: E402
"""Guide friction on threadlike transmissions, and the pluggable model."""

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
    CapstanFrictionParams,
    ExponentialLengthFrictionParams,
    FrictionlessParams,
    ThreadlikeActuator,
    ThreadlikeFriction,
    ThreadlikeImpedance,
    ThreadlikeRouting,
    capstan_transmission_ratio,
)
from soromox.autodiff import custom_jvp_mode
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

# Pull that bends the reference robot by a few radians per metre.
TENSION = 0.03


def _planar(
    *,
    num_segments=1,
    actuators=None,
    passive_elements=(),
    num_gauss_points=5,
    gravity=jnp.zeros((2,)),
):
    body = planar_pcs_params(
        length=jnp.full((num_segments,), 0.1),
        radius=jnp.full((num_segments,), 0.02),
        density=jnp.full((num_segments,), 1000.0),
        gravity=gravity,
        young_modulus=jnp.full((num_segments,), 1e3),
        shear_modulus=jnp.full((num_segments,), 5e2),
        damping_matrix=1e-3 * jnp.eye(3 * num_segments),
    )
    return PlanarPCS(
        body,
        PlanarPCSStructure(num_gauss_points=num_gauss_points),
        base_pose=planar_base_pose(),
        actuators=actuators,
        passive_elements=passive_elements,
    )


def _spatial(*, num_segments=1, actuators=None):
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
        PCSStructure(num_gauss_points=5),
        base_pose=spatial_base_pose(),
        actuators=actuators,
    )


def _gvs(*, actuators=None):
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
            strain_selector=[1, 1, 1, 1, 0, 0],
            basis_order=[0, 0, 0, 0, 0, 0],
        ),
        num_gauss_points=5,
    )
    return GVS.from_segments([segment], gravity=jnp.zeros((3,)), actuators=actuators)


def _routing(offsets, *, num_segments=1):
    count = len(offsets)
    zeros = jnp.zeros((count,))
    return ThreadlikeRouting.linear(
        intercept=jnp.stack((zeros, jnp.asarray(offsets), zeros), axis=-1),
        start_segment_index=(0,) * count,
        end_segment_index=(num_segments - 1,) * count,
    )


def _anchored_routing(offset, *, start, end):
    """One path anchored at segment ``start`` and ending at segment ``end``."""
    return ThreadlikeRouting.linear(
        intercept=jnp.array([[0.0, offset, 0.0]]),
        start_segment_index=(start,),
        end_segment_index=(end,),
    )


def _geometry(robot, path_routing, q, *, s, segment_index):
    """Evaluate the quadrature context of every path at one node."""
    strains = robot.strain(q).reshape((robot.num_segments, -1))
    params = path_routing.params
    return jax.vmap(
        robot._threadlike_local_geometry,
        in_axes=(None, None, None, None, 0, 0, 0),
        out_axes=0,
    )(
        jnp.asarray(segment_index),
        strains[segment_index],
        jnp.asarray(s),
        path_routing,
        params,
        params.start_segment_index_array,
        params.end_segment_index_array,
    )


def _bend_index(factory):
    """Index of the bending strain within one segment, per host."""
    return 0 if factory is _planar else 2


def _impedance(routing, coefficient=0.0):
    return ThreadlikeImpedance(
        routing=routing,
        stiffness=jnp.array([50.0, 30.0]),
        damping=jnp.array([0.3, 0.2]),
        rest_length=jnp.array([0.09, 0.1]),
        friction=ThreadlikeFriction.capstan(coefficient=coefficient),
    )


def _settle(robot, u, iterations=30):
    """Return the static equilibrium reached by Newton iteration from rest."""

    def residual(q):
        return (
            robot.elastic_force(q)
            + robot.gravitational_force(q)
            - robot.actuation_force(q, u)
        )

    q = jnp.zeros((robot.num_coordinates,))
    for _ in range(iterations):
        q = q - jnp.linalg.solve(jax.jacobian(residual)(q), residual(q))
    return q


def _with_friction(robot, coefficient):
    """Return a copy of ``robot`` whose Capstan coefficient is ``coefficient``."""
    transmission = robot.actuators[0].params.transmission
    return robot.update_actuator_params(
        0,
        transmission=transmission.replace(
            friction=transmission.friction.replace(coefficient=coefficient)
        ),
    )


def _with_eps(robot, eps):
    """Return a copy of ``robot`` whose Capstan curvature floor is ``eps``."""
    transmission = robot.actuators[0].params.transmission
    return robot.update_actuator_params(
        0,
        transmission=transmission.replace(
            friction=transmission.friction.replace(eps=eps)
        ),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [_planar, _spatial])
def test_zero_friction_is_bit_identical(factory):
    routing = _routing([0.02, -0.015])
    plain = factory(actuators=ThreadlikeActuator.tendons(routing))
    zero = factory(
        actuators=ThreadlikeActuator.tendons(
            routing, friction=ThreadlikeFriction.capstan(coefficient=0.0)
        )
    )
    q = jnp.linspace(-0.01, 0.04, plain.num_coordinates)
    assert jnp.array_equal(plain.actuation_matrix(q), zero.actuation_matrix(q))


def test_capstan_matches_closed_form():
    """Attenuation follows the Euler belt law and compounds along the arc."""
    mu, kappa, length = 0.3, 8.0, 0.1
    routing = _routing([0.02])
    frictionless = _planar(actuators=ThreadlikeActuator.tendons(routing))
    attenuated = _planar(
        actuators=ThreadlikeActuator.tendons(
            routing, friction=ThreadlikeFriction.capstan(coefficient=mu)
        )
    )
    q = jnp.array([kappa, 0.0, 0.0])
    wrap = mu * kappa * length
    expected = (1.0 - jnp.exp(-wrap)) / wrap
    axial_row = 1
    assert_allclose(
        attenuated.actuation_matrix(q)[axial_row],
        expected * frictionless.actuation_matrix(q)[axial_row],
        rtol=1e-8,
    )

    # Segment-averaged transmission decreases monotonically toward the tip.
    num_segments = 3
    span = _routing([0.02], num_segments=num_segments)
    chain_0 = _planar(
        num_segments=num_segments, actuators=ThreadlikeActuator.tendons(span)
    )
    chain_mu = _planar(
        num_segments=num_segments,
        actuators=ThreadlikeActuator.tendons(
            span, friction=ThreadlikeFriction.capstan(coefficient=0.4)
        ),
    )
    q_chain = (
        jnp.zeros((chain_0.num_coordinates,)).at[::3].set(jnp.array([6.0, 9.0, 12.0]))
    )
    rows = jnp.arange(num_segments) * 3 + axial_row
    ratio = (
        chain_mu.actuation_matrix(q_chain)[rows, 0]
        / chain_0.actuation_matrix(q_chain)[rows, 0]
    )
    assert jnp.all(ratio[1:] < ratio[:-1])
    assert jnp.all((ratio > 0.0) & (ratio <= 1.0))


def test_path_curvature_is_the_curvature_of_the_routed_path():
    """``|omega x u_hat|`` must be the curvature the routed path traces."""
    radius, torsion, bend = 0.02, 5.0, 7.0
    routing = _routing([radius])
    spatial = _spatial(actuators=ThreadlikeActuator.tendons(routing))
    path_routing = spatial.actuators[0].transmission.routing

    def spatial_curvature(q):
        strains = spatial.strain(q).reshape((spatial.num_segments, 6))
        return spatial._threadlike_path_curvature(strains, path_routing)

    # Under pure torsion an off-axis path traces a helix.
    helix = spatial_curvature(jnp.zeros((spatial.num_coordinates,)).at[0].set(torsion))
    chord = jnp.sqrt(1.0 + (torsion * radius) ** 2)
    assert_allclose(helix, jnp.full_like(helix, torsion**2 * radius / chord))

    # In a planar configuration it reduces to |kappa_z| for any offset, which
    # is what makes the planar and spatial hosts agree.
    reduced = spatial_curvature(jnp.zeros((spatial.num_coordinates,)).at[2].set(bend))
    planar = _planar(actuators=ThreadlikeActuator.tendons(routing))
    planar_q = jnp.zeros((planar.num_coordinates,)).at[0].set(bend)
    planar_curvature = planar._threadlike_path_curvature(
        planar.strain(planar_q).reshape((planar.num_segments, 3))
    )
    assert_allclose(reduced, jnp.full_like(reduced, bend), rtol=1e-9)
    assert_allclose(planar_curvature, jnp.full_like(planar_curvature, bend), rtol=1e-9)


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("actuator_friction", "passive_friction"),
    [(0.0, 0.0), (0.6, 0.0), (0.0, 0.6), (0.6, 0.6)],
)
def test_energy_rate_matches_actuation_and_damping_power(
    actuator_friction, passive_friction
):
    """The robot-side energy budget stays exact at any friction coefficient."""
    routing = _routing([0.02, -0.015])
    robot = _planar(
        actuators=ThreadlikeActuator.tendons(
            routing, friction=ThreadlikeFriction.capstan(coefficient=actuator_friction)
        ),
        passive_elements=(_impedance(routing, passive_friction),),
        gravity=jnp.array([0.0, -9.81]),
    )
    q = jnp.array([4.0, 0.02, -0.01])
    qd = jnp.array([0.7, -0.3, 0.5])
    u = jnp.array([3.0, 1.5])

    state_rate = robot.forward_dynamics(0.0, jnp.concatenate([q, qd]), (u,))
    qdd = state_rate[robot.num_coordinates :]

    def energy(q_, qd_):
        return robot.kinetic_energy(q_, qd_) + robot.potential_energy(q_)

    energy_rate = jax.grad(energy, 0)(q, qd) @ qd + jax.grad(energy, 1)(q, qd) @ qdd
    power = qd @ robot.actuation_force(q, u, qd=qd) - qd @ robot.damping_matrix(q) @ qd
    assert_allclose(energy_rate, power, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(
    ("preset", "u", "qd_sign"),
    [
        (ThreadlikeActuator.tendons, 4.0, 1.0),
        (ThreadlikeActuator.tendons, -4.0, -1.0),
        (ThreadlikeActuator.push_rods, 4.0, -1.0),
    ],
)
def test_driving_effort_dissipates_for_pull_and_push(preset, u, qd_sign):
    """Guides absorb power whenever the effort drives the path, either way."""
    routing = _routing([0.02])
    robot = _planar(
        actuators=preset(routing, friction=ThreadlikeFriction.capstan(coefficient=0.5))
    )
    transmission = robot.actuators[0].transmission
    q = jnp.array([7.0, 0.0, 0.0])
    qd = jnp.array([qd_sign, 0.0, 0.0])
    effort = jnp.array([u])

    scale = transmission.params.coordinate_scale
    supplied = jnp.sum(
        effort * scale * (transmission.kinematic_matrix(robot, q).T @ qd)
    )
    delivered = qd @ (transmission.moment_matrix(robot, q) @ effort)

    assert supplied > 0.0  # the effort drives the path in this direction
    assert delivered > 0.0
    assert delivered < supplied
    assert supplied - delivered > 0.0


def test_closed_loop_work_is_nonzero_and_orientation_dependent():
    """Friction admits no actuation potential, so loop work does not vanish."""
    routing = _routing([0.02])
    effort = jnp.array([2.0])

    def loop_work(robot, orientation, num_points=2000):
        angle = orientation * jnp.linspace(0.0, 2.0 * jnp.pi, num_points)
        path = jnp.stack(
            [
                5.0 + 4.0 * jnp.cos(angle),
                0.05 * jnp.sin(angle),
                jnp.zeros_like(angle),
            ],
            axis=-1,
        )
        force = jax.vmap(lambda q: robot.actuation_matrix(q) @ effort)(path)
        midpoint = 0.5 * (force[1:] + force[:-1])
        return jnp.sum(midpoint * jnp.diff(path, axis=0))

    attenuated = _planar(
        actuators=ThreadlikeActuator.tendons(
            routing, friction=ThreadlikeFriction.capstan(coefficient=0.8)
        )
    )
    forward = loop_work(attenuated, 1.0)
    assert jnp.abs(forward) > 1e-4
    assert_allclose(loop_work(attenuated, -1.0), -forward, rtol=1e-6)

    frictionless = _planar(actuators=ThreadlikeActuator.tendons(routing))
    assert jnp.abs(loop_work(frictionless, 1.0)) < 1e-12


@pytest.mark.parametrize("passive_friction", [0.0, 1.0])
def test_passive_friction_is_the_only_unactuated_energy_source(passive_friction):
    """An unactuated robot decays monotonically only at zero passive friction."""
    routing = _routing([0.02, -0.015], num_segments=2)
    robot = _planar(
        num_segments=2,
        passive_elements=(_impedance(routing, passive_friction),),
        gravity=jnp.array([0.0, -9.81]),
    )
    control = jnp.zeros((robot.num_actuators,))
    key = jax.random.PRNGKey(3)

    def energy_rate(index):
        keys = jax.random.split(jax.random.fold_in(key, index))
        q = 4.0 * jax.random.normal(keys[0], (robot.num_coordinates,))
        qd = jax.random.normal(keys[1], (robot.num_velocities,))
        return (
            qd @ robot.actuation_force(q, control, qd=qd)
            - qd @ robot.damping_matrix(q) @ qd
        )

    rates = jax.vmap(energy_rate)(jnp.arange(200))
    if passive_friction == 0.0:
        assert jnp.all(rates <= 0.0)
    else:
        assert jnp.any(rates > 0.0)


def test_floating_passive_nonconservative_force_has_zero_base_prefix():
    """Passive routing friction acts only on the internal material coordinates."""

    routing = _routing([0.02, -0.015])
    fixed = _planar(passive_elements=(_impedance(routing, coefficient=0.6),))
    floating = PlanarPCS(
        params=fixed.params,
        structure=PlanarPCSStructure(num_gauss_points=fixed.num_gauss_points),
        passive_elements=fixed.passive_elements,
        floating_base=True,
        backend="jax",
    )
    q_internal = jnp.array([4.0, 0.02, -0.01])
    q = floating.pack_configuration(
        q_internal,
        base_pose=jnp.array([0.3, 0.1, -0.2]),
    )
    expected = fixed.passive_nonconservative_force(q_internal)
    actual = floating.passive_nonconservative_force(q)

    assert_allclose(actual[: floating.num_base_velocities], 0.0, atol=0.0)
    assert_allclose(actual[floating.num_base_velocities :], expected)
    assert_allclose(floating.passive_nonconservative_force(q_internal), expected)


def test_passive_damping_is_psd_under_friction():
    """The chosen congruence stays dissipative where the rejected form does not."""
    routing = _routing([0.02, -0.015], num_segments=2)
    passive = _impedance(routing, 1.2)
    robot = _planar(num_segments=2, passive_elements=(passive,))
    q = jnp.array([6.0, 0.02, -0.01, 10.0, 0.02, -0.01])

    damping = passive.damping_matrix(robot, q)
    assert_allclose(damping, damping.T, atol=1e-18)
    assert jnp.min(jnp.linalg.eigvalsh(damping)) > -1e-15

    # The rejected A_mu D A_0^T reads the rate through the exact Jacobian.
    transmission = passive._transmission_matrix(robot, q)
    jacobian = passive._length_jacobian(robot, q).T
    rejected = transmission @ (passive.params.damping[:, None] * jacobian.T)
    symmetric = 0.5 * (rejected + rejected.T)
    direction = jnp.linalg.eigh(symmetric)[1][:, 0]

    assert direction @ rejected @ direction < 0.0  # would inject energy
    assert direction @ damping @ direction > 0.0


def test_passive_split_preserves_energy_gradient_and_total_dynamics():
    """Splitting the spring force changes bookkeeping, not physics."""
    routing = _routing([0.02, -0.015], num_segments=2)
    passive = _impedance(routing, 0.8)
    robot = _planar(num_segments=2, passive_elements=(passive,))
    q = jnp.array([5.0, 0.02, -0.01, -3.0, 0.01, 0.02])

    extension = passive.path_lengths(robot, q) - passive.params.rest_length
    transmitted = passive._transmission_matrix(robot, q) @ (
        passive.params.stiffness * extension
    )
    assert_allclose(
        transmitted,
        passive.elastic_force(robot, q) + passive.nonconservative_force(robot, q),
        atol=1e-15,
    )

    for enabled in (True, False):
        with custom_jvp_mode(enabled):
            assert_allclose(
                jax.grad(robot.elastic_energy)(q),
                robot.elastic_force(q),
                rtol=1e-9,
                atol=1e-14,
            )


def test_push_and_pull_share_the_same_transmission_ratio():
    """The ratio depends on the configuration only, never on the effort."""
    routing = _routing([0.02])
    tendon = _planar(
        actuators=ThreadlikeActuator.tendons(
            routing, friction=ThreadlikeFriction.capstan(coefficient=0.7)
        )
    )
    push_rod = _planar(
        actuators=ThreadlikeActuator.push_rods(
            routing, friction=ThreadlikeFriction.capstan(coefficient=0.7)
        )
    )
    q = jnp.array([6.0, 0.01, 0.0])
    assert_allclose(
        tendon.actuation_matrix(q), -push_rod.actuation_matrix(q), atol=1e-18
    )


# ---------------------------------------------------------------------------
# Planar routed-path offset sign
# ---------------------------------------------------------------------------


def test_pulled_tendon_shortens_and_bends_toward_its_own_side():
    """A pulled tendon must shorten and curve the backbone onto its own side."""
    robot = _planar(actuators=ThreadlikeActuator.tendons(_routing([0.02, -0.02])))
    actuator = robot.actuators[0]
    rest = 0.1

    q_first = _settle(robot, jnp.array([TENSION, 0.0]))
    first = actuator.path_lengths(robot, q_first)
    assert q_first[0] > 0.0
    assert first[0] < rest < first[1]

    q_second = _settle(robot, jnp.array([0.0, TENSION]))
    second = actuator.path_lengths(robot, q_second)
    assert_allclose(q_second[0], -q_first[0], rtol=1e-9)
    assert_allclose(second, first[::-1], rtol=1e-9)

    q_push = _settle(robot, jnp.array([-TENSION, 0.0]))
    pushed = actuator.path_lengths(robot, q_push)
    assert q_push[0] < 0.0
    assert pushed[0] > rest > pushed[1]


def test_planar_path_length_matches_rendered_path():
    """Path length must equal the arc length of the curve the renderer draws."""
    robot = _planar(actuators=ThreadlikeActuator.tendons(_routing([0.02])))
    actuator = robot.actuators[0]
    q = jnp.array([10.0, 0.0, 0.0])

    samples = jnp.linspace(0.0, float(jnp.sum(robot.L)), 20001)
    positions = jax.vmap(lambda s: actuator.path_poses(robot, q, s))(samples)
    polyline = jnp.sum(jnp.linalg.norm(jnp.diff(positions[:, 0], axis=0), axis=-1))

    assert_allclose(actuator.path_lengths(robot, q), jnp.array([0.08]), rtol=1e-6)
    assert_allclose(polyline, 0.08, rtol=1e-6)


# ---------------------------------------------------------------------------
# Invariants and autodiff
# ---------------------------------------------------------------------------


def test_friction_breaks_length_gradient_identity():
    """Friction deliberately severs moment_matrix from the length gradient."""
    routing = _routing([0.02, -0.015])
    q = jnp.array([4.0, 0.02, -0.01])

    frictionless = _planar(actuators=ThreadlikeActuator.tendons(routing))
    assert_allclose(
        jax.jacrev(frictionless.actuator_coordinates)(q),
        frictionless.actuation_matrix(q).T,
        rtol=2e-7,
        atol=2e-9,
    )

    attenuated = _planar(
        actuators=ThreadlikeActuator.tendons(
            routing, friction=ThreadlikeFriction.capstan(coefficient=0.6)
        )
    )
    gap = (
        jax.jacrev(attenuated.actuator_coordinates)(q)
        - attenuated.actuation_matrix(q).T
    )
    assert jnp.max(jnp.abs(gap)) > 1e-3


def test_capstan_coefficient_is_traceable_and_identifiable():
    """The coefficient stays a differentiable leaf under jit."""
    routing = _routing([0.02, -0.015])
    robot = _planar(
        actuators=ThreadlikeActuator.tendons(
            routing, friction=ThreadlikeFriction.capstan(coefficient=0.2)
        )
    )
    q = jnp.array([4.0, 0.02, -0.01])
    u = jnp.array([2.0, 1.0])

    @jax.jit
    def loss(coefficient):
        identified = _with_friction(robot, coefficient)
        return jnp.sum(identified.actuation_force(q, u) ** 2)

    for start in (0.0, 0.35):
        value, gradient = jax.value_and_grad(loss)(jnp.asarray(start))
        assert jnp.isfinite(value) and jnp.isfinite(gradient)
        assert jnp.abs(gradient) > 0.0

    per_path = jax.grad(loss)(jnp.array([0.1, 0.4]))
    assert per_path.shape == (2,)
    assert jnp.all(jnp.isfinite(per_path)) and jnp.all(per_path != 0.0)


@pytest.mark.parametrize("eps", [0.0, 0.1])
def test_friction_gradients_are_finite_at_singular_states(eps):
    """The |kappa| kink at a straight backbone must not produce NaNs."""
    robot = _planar(
        actuators=ThreadlikeActuator.tendons(
            _routing([0.02]),
            friction=ThreadlikeFriction.capstan(coefficient=0.3, eps=eps),
        ),
    )
    straight = jnp.zeros((robot.num_coordinates,))

    assert jnp.isfinite(jax.jacrev(robot.actuation_matrix)(straight)).all()
    assert jnp.isfinite(jax.jacrev(robot.actuator_coordinates)(straight)).all()

    def matrix_norm(coefficient):
        identified = _with_friction(robot, coefficient)
        return jnp.sum(identified.actuation_matrix(straight) ** 2)

    sensitivity = jax.grad(matrix_norm)(jnp.asarray(0.3))
    assert jnp.isfinite(sensitivity)
    # The curvature floor is what makes the coefficient identifiable straight.
    if eps == 0.0:
        assert sensitivity == 0.0
        return
    assert jnp.abs(sensitivity) > 0.0

    # The floor is itself a traced leaf, so it can be fitted alongside.
    def floor_norm(value):
        return jnp.sum(_with_eps(robot, value).actuation_matrix(straight) ** 2)

    floor_sensitivity = jax.grad(floor_norm)(jnp.asarray(eps))
    assert jnp.isfinite(floor_sensitivity) and jnp.abs(floor_sensitivity) > 0.0


def test_curvature_floor_bias_is_bounded_and_removes_the_kink():
    """The floor only ever over-estimates the wrap angle, by at most eps * L."""
    q = jnp.array([3.0, 0.0, 0.0, -2.0, 0.0, 0.0])
    robot = _planar(num_segments=2)
    total_length = float(jnp.sum(robot.L))
    starts = jnp.zeros((1,), dtype=jnp.int32)
    curvature = robot._threadlike_path_curvature(robot.strain(q).reshape((2, 3)))
    reference = robot._threadlike_safe_wrap_angle(
        curvature, total_length, starts, jnp.asarray(0.0)
    )

    for eps in (0.0, 0.05, 0.4):
        angle = robot._threadlike_safe_wrap_angle(
            curvature, total_length, starts, jnp.asarray(eps)
        )
        assert jnp.all(angle >= reference)
        assert jnp.all(angle - reference <= eps * total_length + 1e-12)

    routing = _routing([0.02])
    plain = _planar(
        actuators=ThreadlikeActuator.tendons(
            routing, friction=ThreadlikeFriction.capstan(coefficient=0.5)
        )
    )
    unsmoothed = _planar(
        actuators=ThreadlikeActuator.tendons(
            routing, friction=ThreadlikeFriction.capstan(coefficient=0.5, eps=0.0)
        ),
    )
    q_single = jnp.array([3.0, 0.0, 0.0])
    assert jnp.array_equal(
        plain.actuation_matrix(q_single), unsmoothed.actuation_matrix(q_single)
    )


def test_gvs_rejects_nonzero_friction():
    """GVS strain varies along s, so the piecewise-linear wrap angle is invalid."""
    routing = _routing([0.02])
    _gvs(actuators=ThreadlikeActuator.tendons(routing))  # zero stays allowed

    with pytest.raises(TypeError, match="wrap-angle model"):
        _gvs(
            actuators=ThreadlikeActuator.tendons(
                routing, friction=ThreadlikeFriction.capstan(coefficient=0.3)
            )
        )


# ---------------------------------------------------------------------------
# The pluggable friction model
# ---------------------------------------------------------------------------


class _HyperbolicFrictionParams(BaseThreadlikeFrictionParams):
    """Parameters of a law defined entirely outside soromox."""

    a: Array


def _hyperbolic_transmission_ratio(params, context):
    """A law defined entirely outside soromox, to pin the extension point."""
    return 1.0 / (1.0 + params.a * context.wrap_angle)


def _hyperbolic(a):
    return ThreadlikeFriction(
        params=_HyperbolicFrictionParams(a=jnp.asarray(a)),
        ratio_fn=_hyperbolic_transmission_ratio,
        requires_wrap_angle=True,
        is_frictionless=False,
    )


@pytest.mark.parametrize("factory", [_planar, _spatial])
def test_default_is_frictionless_and_bit_identical(factory):
    """The shipped default delivers full effort and costs nothing."""
    routing = _routing([0.02, -0.015])
    q = jnp.linspace(-0.01, 0.04, factory().num_coordinates)

    default = factory(actuators=ThreadlikeActuator.tendons(routing))
    explicit = factory(
        actuators=ThreadlikeActuator.tendons(
            routing, friction=ThreadlikeFriction.frictionless()
        )
    )
    assert ThreadlikeFriction.frictionless().is_frictionless
    assert not ThreadlikeFriction.frictionless().requires_wrap_angle
    assert jnp.array_equal(default.actuation_matrix(q), explicit.actuation_matrix(q))
    assert isinstance(
        default.actuators[0].params.transmission.friction, FrictionlessParams
    )


def test_params_and_law_stay_paired():
    """The params leaf and the runtime law must not drift apart."""
    routing = _routing([0.02])
    q = jnp.array([6.0, 0.0, 0.0])

    robot = _planar(
        actuators=ThreadlikeActuator.tendons(
            routing, friction=ThreadlikeFriction.capstan(coefficient=0.4)
        )
    )
    transmission = robot.actuators[0].transmission
    assert isinstance(transmission.params.friction, CapstanFrictionParams)
    assert transmission.friction.params is transmission.params.friction

    # Replacing a value keeps the installed law.
    replaced = _with_friction(robot, jnp.asarray(0.4))
    assert replaced.actuators[0].transmission.friction.ratio_fn is (
        transmission.friction.ratio_fn
    )
    assert jnp.array_equal(robot.actuation_matrix(q), replaced.actuation_matrix(q))

    # Swapping the params type behind an installed law is refused, as is
    # installing anything that is not a law.
    with pytest.raises(TypeError, match="keep their type"):
        transmission.friction.with_params(
            ExponentialLengthFrictionParams(rate=jnp.asarray(1.0))
        )
    with pytest.raises(TypeError):
        ThreadlikeActuator.tendons(routing, friction="capstan")


@pytest.mark.parametrize("factory", [_planar, _spatial])
def test_length_friction_matches_closed_form(factory):
    """A per-length friction integrates to its analytic mean over a segment."""
    rate, length = 3.0, 0.1
    routing = _routing([0.02])
    q = jnp.zeros((factory().num_coordinates,)).at[0].set(4.0)

    frictionless = factory(actuators=ThreadlikeActuator.tendons(routing))
    attenuated = factory(
        actuators=ThreadlikeActuator.tendons(
            routing, friction=ThreadlikeFriction.exponential_length(rate=rate)
        )
    )
    decay = rate * length
    expected = (1.0 - jnp.exp(-decay)) / decay

    reference = frictionless.actuation_matrix(q)
    row = int(jnp.argmax(jnp.abs(reference[:, 0])))
    assert_allclose(
        attenuated.actuation_matrix(q)[row, 0],
        expected * reference[row, 0],
        rtol=1e-8,
    )


def test_length_friction_runs_on_gvs():
    """GVS hosts a wrap-angle-free law, and still refuses a Capstan law."""
    routing = _routing([0.02])
    q = jnp.zeros((_gvs().num_coordinates,)).at[0].set(3.0)

    frictionless = _gvs(actuators=ThreadlikeActuator.tendons(routing))
    attenuated = _gvs(
        actuators=ThreadlikeActuator.tendons(
            routing, friction=ThreadlikeFriction.exponential_length(rate=4.0)
        )
    )
    reference = jnp.abs(frictionless.actuation_matrix(q))
    weighted = jnp.abs(attenuated.actuation_matrix(q))
    assert jnp.all(weighted <= reference + 1e-15)
    assert jnp.max(reference - weighted) > 1e-6

    with pytest.raises(TypeError, match="wrap-angle model"):
        _gvs(
            actuators=ThreadlikeActuator.tendons(
                routing, friction=ThreadlikeFriction.capstan(coefficient=0.3)
            )
        )


def test_custom_friction_model_needs_no_soromox_change():
    """A third-party law installs through the public contract alone."""
    routing = _routing([0.02])
    q = jnp.array([6.0, 0.0, 0.0])

    frictionless = _planar(actuators=ThreadlikeActuator.tendons(routing))
    custom = _planar(
        actuators=ThreadlikeActuator.tendons(routing, friction=_hyperbolic(0.5))
    )
    reference = frictionless.actuation_matrix(q)
    weighted = custom.actuation_matrix(q)

    assert jnp.all(jnp.abs(weighted) <= jnp.abs(reference) + 1e-15)
    assert jnp.max(jnp.abs(reference - weighted)) > 1e-6
    # The law is not Capstan, so it must not coincide with one.
    capstan = _planar(
        actuators=ThreadlikeActuator.tendons(
            routing, friction=ThreadlikeFriction.capstan(coefficient=0.5)
        )
    ).actuation_matrix(q)
    assert jnp.max(jnp.abs(weighted - capstan)) > 1e-6


def test_model_is_traceable_and_identifiable():
    """Friction parameters stay differentiable leaves under jit, for any model."""
    routing = _routing([0.02, -0.015])
    q = jnp.array([4.0, 0.02, -0.01])
    u = jnp.array([2.0, 1.0])

    def loss_for(build, value):
        # The robot is built once outside the trace; only the friction parameter
        # traced, which is how a law is identified in practice.
        robot = _planar(
            actuators=ThreadlikeActuator.tendons(routing, friction=build(value))
        )

        @jax.jit
        def loss(parameter):
            transmission = robot.actuators[0].params.transmission
            identified = robot.update_actuator_params(
                0,
                transmission=transmission.replace(friction=build(parameter).params),
            )
            return jnp.sum(identified.actuation_force(q, u) ** 2)

        return jax.value_and_grad(loss)(jnp.asarray(value))

    for build, value in (
        (lambda p: ThreadlikeFriction.capstan(coefficient=p), 0.3),
        (lambda p: ThreadlikeFriction.exponential_length(rate=p), 2.0),
        (lambda p: _hyperbolic(p), 0.4),
    ):
        magnitude, gradient = loss_for(build, value)
        assert jnp.isfinite(magnitude) and jnp.isfinite(gradient)
        assert jnp.abs(gradient) > 0.0


def test_context_fields_agree_across_hosts():
    """Host-agnostic context fields must match, so portable laws are portable."""
    routing = _routing([0.02, -0.015])
    planar = _planar(actuators=ThreadlikeActuator.tendons(routing))
    spatial = _spatial(actuators=ThreadlikeActuator.tendons(routing))

    curvature, axial, shear = 5.0, 0.02, -0.01
    planar_strain = jnp.array([curvature, 1.0 + axial, shear])
    spatial_strain = jnp.array([0.0, 0.0, curvature, 1.0 + axial, shear, 0.0])
    abscissa = 0.06
    path_routing = planar.actuators[0].transmission.routing

    def geometry(robot, strain):
        return jax.vmap(
            robot._threadlike_local_geometry,
            in_axes=(None, None, None, None, 0, 0, 0),
            out_axes=0,
        )(
            jnp.asarray(0),
            strain,
            jnp.asarray(abscissa),
            path_routing,
            path_routing.params,
            path_routing.params.start_segment_index_array,
            path_routing.params.end_segment_index_array,
        )

    planar_context = geometry(planar, planar_strain)
    spatial_context = geometry(spatial, spatial_strain)

    for field in (
        "abscissa",
        "path_abscissa",
        "offset",
        "offset_derivative",
        "tangent_norm",
    ):
        assert_allclose(
            getattr(planar_context, field),
            getattr(spatial_context, field),
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"host-agnostic field {field} disagrees",
        )
    assert jnp.array_equal(planar_context.active, spatial_context.active)


@pytest.mark.parametrize(
    ("build", "field"),
    [
        (lambda v: CapstanFrictionParams(coefficient=jnp.asarray(v)), "coefficient"),
        (lambda v: CapstanFrictionParams(eps=jnp.asarray(v)), "eps"),
        (lambda v: ExponentialLengthFrictionParams(rate=jnp.asarray(v)), "rate"),
    ],
)
@pytest.mark.parametrize("value", [-1.0, jnp.nan, jnp.inf])
def test_friction_parameters_reject_negative_and_non_finite_values(build, field, value):
    """A ratio outside (0, 1] would let a transmission generate effort."""
    with pytest.raises(ValueError, match=field):
        build(value)


def test_friction_parameters_reject_a_wrong_per_path_shape():
    """A per-path parameter must carry one entry per routed path."""
    with pytest.raises(ValueError, match="coefficient"):
        ThreadlikeActuator.tendons(
            _routing([0.02, -0.015]),
            friction=ThreadlikeFriction.capstan(coefficient=jnp.zeros((3,))),
        )


def test_a_friction_model_may_not_claim_to_be_frictionless():
    """is_frictionless skips the model, so a friction model must not set it."""
    with pytest.raises(ValueError, match="is_frictionless"):
        ThreadlikeFriction(
            params=CapstanFrictionParams(coefficient=jnp.asarray(0.3)),
            ratio_fn=capstan_transmission_ratio,
            requires_wrap_angle=True,
            is_frictionless=True,
        )


# ---------------------------------------------------------------------------
# Friction accumulates from the path anchor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [_planar, _spatial])
def test_friction_is_measured_from_the_path_anchor(factory):
    """A path anchored mid-robot must not pay for what happens below it."""
    robot = factory(num_segments=3)
    anchored = _anchored_routing(0.02, start=1, end=2)
    path_routing = ThreadlikeActuator.tendons(anchored).transmission.routing
    starts = anchored.params.start_segment_index_array
    anchor, tip = float(robot.L_cum[1]), float(jnp.sum(robot.L))

    # Bend only the first segment, which the path does not span.
    q = jnp.zeros((robot.num_coordinates,)).at[_bend_index(factory)].set(9.0)
    strains = robot.strain(q).reshape((robot.num_segments, -1))
    curvature = (
        robot._threadlike_path_curvature(strains)
        if factory is _planar
        else robot._threadlike_path_curvature(strains, anchored)
    )

    # The wrap angle ignores that turning,
    anchored_angle = robot._threadlike_safe_wrap_angle(
        curvature, tip, starts, jnp.asarray(0.0)
    )
    assert_allclose(anchored_angle, jnp.zeros_like(anchored_angle), atol=1e-12)
    # which the same pose measured from the robot base does not.
    base_angle = robot._threadlike_safe_wrap_angle(
        curvature, tip, jnp.zeros_like(starts), jnp.asarray(0.0)
    )
    assert jnp.all(base_angle > 1e-3)

    # The traversed length is anchor-relative too, so a length law transmits in
    # full at the anchor and decays only beyond it.
    at_anchor = _geometry(robot, path_routing, q, s=anchor, segment_index=1)
    assert_allclose(
        at_anchor.path_abscissa, jnp.zeros_like(at_anchor.path_abscissa), atol=1e-12
    )
    ratio = ThreadlikeFriction.exponential_length(rate=5.0).transmission_ratio(
        at_anchor
    )
    assert_allclose(ratio, jnp.ones_like(ratio), rtol=1e-12)

    travelled = 0.03
    beyond = _geometry(robot, path_routing, q, s=anchor + travelled, segment_index=1)
    assert_allclose(
        beyond.path_abscissa, jnp.full_like(beyond.path_abscissa, travelled)
    )
    # while the backbone coordinate still counts from the robot base.
    assert_allclose(beyond.abscissa, jnp.full_like(beyond.abscissa, anchor + travelled))


@pytest.mark.parametrize("factory", [_planar, _spatial])
def test_capstan_friction_is_unaffected_by_curvature_before_the_anchor(factory):
    """Attenuation of a mid-robot path must not depend on the segments below."""
    anchored = _anchored_routing(0.02, start=1, end=2)
    frictional = factory(
        num_segments=3,
        actuators=ThreadlikeActuator.tendons(
            anchored, friction=ThreadlikeFriction.capstan(coefficient=0.6)
        ),
    )
    reference = factory(num_segments=3, actuators=ThreadlikeActuator.tendons(anchored))
    per_segment, bend = frictional.num_internal_dofs // 3, _bend_index(factory)

    def attenuation(base_curvature):
        # Bend the two segments the path spans, so it is genuinely attenuated,
        # and vary only the curvature of the segment below its anchor.
        q = (
            jnp.zeros((frictional.num_coordinates,))
            .at[bend]
            .set(base_curvature)
            .at[per_segment + bend]
            .set(8.0)
            .at[2 * per_segment + bend]
            .set(-6.0)
        )
        reference_matrix = reference.actuation_matrix(q)
        carried = jnp.abs(reference_matrix) > 1e-9
        return frictional.actuation_matrix(q)[carried] / reference_matrix[carried]

    straight_below = attenuation(0.0)
    bent_below = attenuation(11.0)
    assert straight_below.size > 0
    # The path is attenuated, but by its own turning only.
    assert jnp.all(straight_below < 1.0)
    assert_allclose(straight_below, bent_below, rtol=1e-9, atol=1e-12)
