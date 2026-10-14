"""Independent mechanics and parameter-update checks for helicoidal GVS arms."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.systems import (
    GVSParams,
    GVSSegment,
    Helicoidal,
    HelicoidalSectionParams,
    JointSpec,
    LinkSpec,
    StrainBasisSpec,
)

jax.config.update("jax_enable_x64", True)


def segment(length=0.2, active=(1, 1, 1, 1, 1, 1), order=(0,) * 6, gauss=8):
    dof = sum(a * (o + 1) for a, o in zip(active, order))
    return GVSSegment(
        link=LinkSpec.circular(
            length=length,
            radius=0.025,
            density=1.0,
            reference_strain=[0, 0, 0, 1, 0, 0],
            stiffness=jnp.zeros((dof, dof)),
            damping=jnp.zeros((dof, dof)),
        ),
        joint=JointSpec(type="fixed"),
        basis=StrainBasisSpec(
            type="legendre", strain_selector=active, basis_order=order
        ),
        num_gauss_points=gauss,
    )


def law(count=1, c_b=3.0):
    return HelicoidalSectionParams(
        compliance=jnp.tile(jnp.array([[5.0, 2.0, 2.0, 0.01, 0.02, 0.02]]), (count, 1)),
        compliance_gradient=jnp.tile(
            jnp.array([[0.0, c_b, c_b, 0.0, 0.0, 0.0]]), (count, 1)
        ),
        damping_time_constant=jnp.full(count, 0.02),
    )


def build(segments=None, **kwargs):
    segments = [segment()] if segments is None else segments
    count = len(segments)
    return Helicoidal.from_segments(
        segments,
        mass_per_length=jnp.full(count, 0.2),
        rotational_inertia_per_length=jnp.tile(
            jnp.array([[2e-4, 1e-4, 1e-4]]), (count, 1)
        ),
        tip_mass=jnp.full(count, 0.04),
        constitutive_law=law(count),
        **kwargs,
    )


@pytest.mark.parametrize("c_b", [0.0, 3.0, -3.0])
@pytest.mark.parametrize("scaled", [False, True])
def test_analytic_projection(c_b, scaled):
    model = build(scale_rotational_basis_by_length=scaled).with_constitutive_law(
        law(c_b=c_b)
    )
    bending = 0.2 / 2 if c_b == 0 else np.log1p(c_b * 0.2 / 2) / c_b
    diagonal = np.array([0.2 * 0.2, bending, bending, 100 * 0.2, 50 * 0.2, 50 * 0.2])
    if scaled:
        diagonal[:3] /= 0.2**2
    assert_allclose(model.K, np.diag(diagonal), rtol=1e-12, atol=1e-13)
    assert_allclose(model.D_active, 0.02 * model.K, rtol=1e-12, atol=1e-13)


def test_higher_order_projection_and_inactive_padding():
    model = build(
        [
            segment(active=(0, 1, 0, 0, 0, 0), order=(0, 1, 0, 0, 0, 0)),
            segment(length=0.01, active=(0,) * 6),
        ],
        max_dof=4,
    )
    expected = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            expected[i, j] = quad(
                lambda s, i=i, j=j: (2 * s / 0.2 - 1) ** (i + j) / (2 + 3 * (0.2 - s)),
                0,
                0.2,
            )[0]
    assert_allclose(model.K, expected, rtol=1e-12, atol=1e-13)
    assert_allclose(model.params.link.stiffness[1], 0.0)
    assert_allclose(model.params.link.stiffness[0, 2:], 0.0)


def test_platform_and_distributed_mechanics():
    model = build()
    q = jnp.array([0.2, -0.4, 0.3, 0.01, -0.01, 0.02])
    qd = jnp.array([-0.1, 0.2, -0.3, 0.02, 0.01, -0.01])
    unloaded = model.update_link_params(tip_inertia=jnp.zeros((1, 6, 6)))

    def position(coordinates):
        return model.forward_kinematics_tips(coordinates)[0, :3, 3]

    jacobian = jax.jacfwd(position)(q)
    derivative = jax.jvp(jax.jacfwd(position), (q,), (qd,))[1]
    assert_allclose(
        model.inertia_matrix(q) - unloaded.inertia_matrix(q),
        0.04 * jacobian.T @ jacobian,
        atol=1e-11,
    )
    assert_allclose(
        (model.coriolis_matrix(q, qd) - unloaded.coriolis_matrix(q, qd)) @ qd,
        0.04 * jacobian.T @ derivative @ qd,
        atol=1e-11,
    )
    assert_allclose(
        model.gravitational_force(q),
        jax.grad(model._gravitational_energy)(q),
        rtol=1e-9,
        atol=1e-10,
    )
    mass, bias, gravity = model.dynamics_terms(q, qd)
    assert_allclose(mass, model.inertia_matrix(q), rtol=1e-9, atol=1e-11)
    assert_allclose(bias, model.coriolis_matrix(q, qd) @ qd, rtol=1e-9, atol=1e-11)
    assert_allclose(gravity, model.gravitational_force(q), rtol=1e-9, atol=1e-10)

    # Lagrange's kinetic-energy identity checks the velocity bias independently.
    def kinetic(coordinates, velocity):
        return 0.5 * velocity @ model.inertia_matrix(coordinates) @ velocity

    momentum_derivative = jax.jvp(
        lambda coordinates: jax.grad(kinetic, 1)(coordinates, qd), (q,), (qd,)
    )[1]
    assert_allclose(
        bias, momentum_derivative - jax.grad(kinetic)(q, qd), rtol=1e-8, atol=1e-10
    )
    assert np.linalg.eigvalsh(mass).min() > 0
    derivative = model.forward_dynamics(0.0, jnp.concatenate((q, qd)))
    assert_allclose(derivative[:6], qd)
    assert_allclose(
        mass @ derivative[6:],
        -bias - gravity - model.K @ q - model.D_active @ qd,
        rtol=1e-9,
        atol=1e-10,
    )


def test_zero_configuration_batching_and_material_mass_independence():
    model = build()
    q = jnp.zeros(6)
    mass, bias, gravity = model.dynamics_terms(q, q)
    assert np.isfinite(mass).all()
    assert_allclose(bias, 0, atol=1e-13)
    # Axial extension: integral line_mass*s^2 ds plus endpoint_mass*L^2.
    assert_allclose(mass[3, 3], 0.2 * 0.2**3 / 3 + 0.04 * 0.2**2, rtol=1e-12)
    assert_allclose(
        model.update_link_params(
            density=1000 * model.params.link.density
        ).inertia_matrix(q),
        mass,
        atol=1e-12,
    )
    batch = model.dynamics_terms(jnp.stack([q, q]), jnp.stack([q, q]))
    for batched, single in zip(batch, (mass, bias, gravity)):
        assert_allclose(batched, jnp.stack([single, single]), atol=1e-12)


def test_contiguous_tendons_include_transitions_and_massless_gaps():
    route = ThreadlikeRouting.linear(
        intercept=jnp.array([[0.0, 0.015, 0.0]]),
        slope=jnp.zeros((1, 3)),
        start_segment_index=0,
        end_segment_index=(2,),
    )
    model = build(
        [
            segment(active=(0, 1, 0, 1, 0, 0)),
            segment(length=0.01, active=(0,) * 6),
            segment(length=0.008, active=(0, 1, 0, 1, 0, 0)),
        ],
        actuators=ThreadlikeActuator.tendons(route),
    )
    model = model.update_link_params(
        sectional_inertia=model.params.link.sectional_inertia.at[1].set(0),
        tip_inertia=model.params.link.tip_inertia.at[1].set(0),
    )
    q = jnp.zeros(model.num_coordinates)
    assert_allclose(model.actuators[0].path_lengths(model, q), [0.218], atol=1e-12)
    q = jnp.array([0.1, 0.01, -0.2, -0.02])
    tension = jnp.array([1.3])
    assert_allclose(
        model.actuation_matrix(q) @ tension,
        -jax.jacfwd(
            lambda coordinates: model.actuators[0].path_lengths(model, coordinates)
        )(q).T
        @ tension,
        atol=1e-10,
    )
    mass, bias, gravity = model.dynamics_terms(q, jnp.ones_like(q) * 0.01)
    assert (
        np.isfinite(mass).all()
        and np.isfinite(bias).all()
        and np.isfinite(gravity).all()
    )
    assert_allclose(mass, model.inertia_matrix(q), atol=1e-11)


def test_parameter_updates_and_projection_reuse_compilation():
    model = build()
    q = jnp.array([0.2, -0.4, 0.3, 0.01, -0.01, 0.02])
    changed = model.params.replace(
        link=model.params.link.replace(
            length=jnp.array([0.23]),
            sectional_inertia=1.5 * model.params.link.sectional_inertia,
            tip_inertia=1.5 * model.params.link.tip_inertia,
        )
    )
    traces = []

    @eqx.filter_jit
    def evaluate(robot, params, section):
        traces.append(True)
        updated = robot.with_params(params).with_constitutive_law(section)
        return updated.dynamics_terms(q, q * 0.1), updated.K

    evaluate(model, model.params, law())
    actual, stiffness = evaluate(model, changed, law(c_b=4.0))
    assert len(traces) == 1
    fresh = Helicoidal(changed, model.structure).with_constitutive_law(law(c_b=4.0))
    for a, b in zip(actual, fresh.dynamics_terms(q, q * 0.1)):
        assert_allclose(a, b, rtol=1e-10, atol=1e-12)
    assert_allclose(stiffness, fresh.K)
    assert_allclose(model.with_params(changed).K, model.K)
    assert isinstance(model.with_params(changed).params, GVSParams)
    model_traces = []

    @eqx.filter_jit
    def evaluate_model(robot):
        model_traces.append(True)
        return robot.dynamics_terms(q, q * 0.1)

    evaluate_model(model)
    outside_update = model.with_params(changed).with_constitutive_law(law(c_b=4.0))
    for a, b in zip(evaluate_model(outside_update), actual):
        assert_allclose(a, b, rtol=1e-10, atol=1e-12)
    assert len(model_traces) == 1

    derivative = jax.grad(
        lambda c: model.with_constitutive_law(
            law().replace(compliance=law().compliance.at[0, 1:3].set(c))
        ).K[1, 1]
    )(2.0)
    expected = quad(lambda s: -1 / (2 + 3 * (0.2 - s)) ** 2, 0, 0.2)[0]
    assert_allclose(derivative, expected, rtol=1e-12)
    length_derivative = jax.grad(
        lambda length: (
            model.update_link_params(length=jnp.array([length]))
            .with_constitutive_law(law())
            .K[1, 1]
        )
    )(0.2)
    assert_allclose(length_derivative, 1 / (2 + 3 * 0.2), rtol=1e-12)
    mass_derivative = jax.grad(
        lambda m: model.update_link_params(
            tip_inertia=jnp.diag(jnp.array([0.0, 0.0, 0.0, m, m, m]))[None]
        ).inertia_matrix(q)[3, 3]
    )(0.04)
    assert mass_derivative > 0


def test_validation():
    model = build()
    with pytest.raises(ValueError, match="positive throughout"):
        model.with_constitutive_law(law(c_b=-20.0))
    with pytest.raises(ValueError, match="one row per link"):
        model.with_constitutive_law(law(2))
    with pytest.raises(ValueError, match="positive semidefinite"):
        model.update_link_params(tip_inertia=-jnp.eye(6)[None])
    with pytest.raises(ValueError, match="shape"):
        model.update_link_params(sectional_inertia=jnp.zeros((2, 6, 6)))
