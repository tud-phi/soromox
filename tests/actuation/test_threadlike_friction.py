# ruff: noqa: E402
"""Capstan guide friction on threadlike transmissions."""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from numpy.testing import assert_allclose
from system_param_builders import (
    pcs_params,
    planar_base_pose,
    planar_pcs_params,
    spatial_base_pose,
)

from soromox.actuation import ThreadlikeActuator, ThreadlikeImpedance, ThreadlikeRouting
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
    wrap_angle_smoothing=0.0,
    num_gauss_points=5,
    gravity=jnp.zeros((2,)),
):
    body = planar_pcs_params(
        base_pose=planar_base_pose(),
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
        PlanarPCSStructure(
            num_gauss_points=num_gauss_points,
            wrap_angle_smoothing=wrap_angle_smoothing,
        ),
        actuators=actuators,
        passive_elements=passive_elements,
    )


def _spatial(*, num_segments=1, actuators=None, wrap_angle_smoothing=0.0):
    body = pcs_params(
        base_pose=spatial_base_pose(),
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
        PCSStructure(num_gauss_points=5, wrap_angle_smoothing=wrap_angle_smoothing),
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


def _impedance(routing, friction_coefficient=0.0):
    return ThreadlikeImpedance(
        routing=routing,
        stiffness=jnp.array([50.0, 30.0]),
        damping=jnp.array([0.3, 0.2]),
        rest_length=jnp.array([0.09, 0.1]),
        friction_coefficient=friction_coefficient,
    )


def _settle(robot, u, iterations=30):
    """Return the static equilibrium reached by Newton iteration from rest."""

    def residual(q):
        return (
            robot.elastic_force(q)
            + robot.gravitational_force(q)
            - robot.actuation_force(q, u)
        )

    q = jnp.zeros((robot.num_dofs,))
    for _ in range(iterations):
        q = q - jnp.linalg.solve(jax.jacobian(residual)(q), residual(q))
    return q


def _with_friction(robot, friction_coefficient):
    transmission = robot.actuators[0].params.transmission.replace(
        friction_coefficient=friction_coefficient
    )
    return robot.update_actuator_params(0, transmission=transmission)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [_planar, _spatial])
def test_zero_friction_is_bit_identical(factory):
    routing = _routing([0.02, -0.015])
    plain = factory(actuators=ThreadlikeActuator.tendons(routing))
    zero = factory(
        actuators=ThreadlikeActuator.tendons(routing, friction_coefficient=0.0)
    )
    q = jnp.linspace(-0.01, 0.04, plain.num_dofs)
    assert jnp.array_equal(plain.actuation_matrix(q), zero.actuation_matrix(q))


def test_capstan_matches_closed_form():
    """Attenuation follows the Euler belt law and compounds along the arc."""
    mu, kappa, length = 0.3, 8.0, 0.1
    routing = _routing([0.02])
    frictionless = _planar(actuators=ThreadlikeActuator.tendons(routing))
    attenuated = _planar(
        actuators=ThreadlikeActuator.tendons(routing, friction_coefficient=mu)
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
        actuators=ThreadlikeActuator.tendons(span, friction_coefficient=0.4),
    )
    q_chain = jnp.zeros((chain_0.num_dofs,)).at[::3].set(jnp.array([6.0, 9.0, 12.0]))
    rows = jnp.arange(num_segments) * 3 + axial_row
    ratio = (
        chain_mu.actuation_matrix(q_chain)[rows, 0]
        / chain_0.actuation_matrix(q_chain)[rows, 0]
    )
    assert jnp.all(ratio[1:] < ratio[:-1])
    assert jnp.all((ratio > 0.0) & (ratio <= 1.0))


def test_torsion_wrap_matches_helix_curvature():
    """An off-axis path under pure torsion wraps at the helix curvature."""
    radius, torsion = 0.02, 5.0
    routing = _routing([radius])
    robot = _spatial(actuators=ThreadlikeActuator.tendons(routing))
    q = jnp.zeros((robot.num_dofs,)).at[0].set(torsion)
    strains = robot.strain(q).reshape((robot.num_segments, 6))

    density = robot._threadlike_wrap_density(
        strains, robot.actuators[0].transmission.routing
    )
    chord = jnp.sqrt(1.0 + (torsion * radius) ** 2)
    assert_allclose(density, jnp.full_like(density, torsion**2 * radius / chord))


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
            routing, friction_coefficient=actuator_friction
        ),
        passive_elements=(_impedance(routing, passive_friction),),
        gravity=jnp.array([0.0, -9.81]),
    )
    q = jnp.array([4.0, 0.02, -0.01])
    qd = jnp.array([0.7, -0.3, 0.5])
    u = jnp.array([3.0, 1.5])

    state_rate = robot.forward_dynamics(0.0, jnp.concatenate([q, qd]), (u,))
    qdd = state_rate[robot.num_dofs :]

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
    robot = _planar(actuators=preset(routing, friction_coefficient=0.5))
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
        actuators=ThreadlikeActuator.tendons(routing, friction_coefficient=0.8)
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
        q = 4.0 * jax.random.normal(keys[0], (robot.num_dofs,))
        qd = jax.random.normal(keys[1], (robot.num_dofs,))
        return (
            qd @ robot.actuation_force(q, control, qd=qd)
            - qd @ robot.damping_matrix(q) @ qd
        )

    rates = jax.vmap(energy_rate)(jnp.arange(200))
    if passive_friction == 0.0:
        assert jnp.all(rates <= 0.0)
    else:
        assert jnp.any(rates > 0.0)


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
        actuators=ThreadlikeActuator.tendons(routing, friction_coefficient=0.7)
    )
    push_rod = _planar(
        actuators=ThreadlikeActuator.push_rods(routing, friction_coefficient=0.7)
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
        actuators=ThreadlikeActuator.tendons(routing, friction_coefficient=0.6)
    )
    gap = (
        jax.jacrev(attenuated.actuator_coordinates)(q)
        - attenuated.actuation_matrix(q).T
    )
    assert jnp.max(jnp.abs(gap)) > 1e-3


def test_friction_coefficient_is_traceable_and_identifiable():
    """The coefficient stays a differentiable leaf under jit."""
    routing = _routing([0.02, -0.015])
    robot = _planar(
        actuators=ThreadlikeActuator.tendons(routing, friction_coefficient=0.2)
    )
    q = jnp.array([4.0, 0.02, -0.01])
    u = jnp.array([2.0, 1.0])

    @jax.jit
    def loss(friction_coefficient):
        identified = _with_friction(robot, friction_coefficient)
        return jnp.sum(identified.actuation_force(q, u) ** 2)

    for start in (0.0, 0.35):
        value, gradient = jax.value_and_grad(loss)(jnp.asarray(start))
        assert jnp.isfinite(value) and jnp.isfinite(gradient)
        assert jnp.abs(gradient) > 0.0

    per_path = jax.grad(loss)(jnp.array([0.1, 0.4]))
    assert per_path.shape == (2,)
    assert jnp.all(jnp.isfinite(per_path)) and jnp.all(per_path != 0.0)


@pytest.mark.parametrize("wrap_angle_smoothing", [0.0, 0.1])
def test_friction_gradients_are_finite_at_singular_states(wrap_angle_smoothing):
    """The |kappa| kink at a straight backbone must not produce NaNs."""
    robot = _planar(
        wrap_angle_smoothing=wrap_angle_smoothing,
        actuators=ThreadlikeActuator.tendons(
            _routing([0.02]), friction_coefficient=0.3
        ),
    )
    straight = jnp.zeros((robot.num_dofs,))

    assert jnp.isfinite(jax.jacrev(robot.actuation_matrix)(straight)).all()
    assert jnp.isfinite(jax.jacrev(robot.actuator_coordinates)(straight)).all()

    def matrix_norm(friction_coefficient):
        identified = _with_friction(robot, friction_coefficient)
        return jnp.sum(identified.actuation_matrix(straight) ** 2)

    sensitivity = jax.grad(matrix_norm)(jnp.asarray(0.3))
    assert jnp.isfinite(sensitivity)
    # Smoothing is what makes the coefficient identifiable at a straight pose.
    if wrap_angle_smoothing == 0.0:
        assert sensitivity == 0.0
    else:
        assert jnp.abs(sensitivity) > 0.0


def test_wrap_angle_smoothing_bias_is_bounded_and_removes_the_kink():
    """Smoothing only ever over-estimates the wrap angle, by at most eps * L."""
    q = jnp.array([3.0, 0.0, 0.0, -2.0, 0.0, 0.0])
    exact = _planar(num_segments=2)
    total_length = float(jnp.sum(exact.L))
    reference = exact._threadlike_wrap_angle(
        exact._threadlike_wrap_density(exact.strain(q).reshape((2, 3))), total_length
    )

    for smoothing in (0.0, 0.05, 0.4):
        smoothed = _planar(num_segments=2, wrap_angle_smoothing=smoothing)
        angle = smoothed._threadlike_wrap_angle(
            smoothed._threadlike_wrap_density(smoothed.strain(q).reshape((2, 3))),
            total_length,
        )
        assert angle >= reference
        assert angle - reference <= smoothing * total_length + 1e-12

    routing = _routing([0.02])
    plain = _planar(
        actuators=ThreadlikeActuator.tendons(routing, friction_coefficient=0.5)
    )
    unsmoothed = _planar(
        wrap_angle_smoothing=0.0,
        actuators=ThreadlikeActuator.tendons(routing, friction_coefficient=0.5),
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
        _gvs(actuators=ThreadlikeActuator.tendons(routing, friction_coefficient=0.3))
