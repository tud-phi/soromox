# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose
from system_param_builders import articulated_params, pcs_params, pendulum_params

from soromox.actuation import (
    AffineJointTransmission,
    AffineJointTransmissionParams,
    ArticulatedTendonActuator,
    ArticulatedTendonImpedance,
    ThreadlikeActuator,
    ThreadlikeRouting,
)
from soromox.systems import PCS, ArticulatedSoftRobot, PCSStructure, Pendulum


def _body_params():
    return pendulum_params(
        mass=jnp.array([1.0, 0.8, 0.6]),
        moment_inertia=jnp.array([0.1, 0.08, 0.05]),
        length=jnp.array([0.5, 0.4, 0.3]),
        center_of_mass_length=jnp.array([0.25, 0.2, 0.15]),
        gravity=jnp.array([0.0, -9.81]),
        joint_stiffness=jnp.diag(jnp.array([0.3, 0.2, 0.1])),
        joint_damping=jnp.diag(jnp.array([0.03, 0.02, 0.01])),
        joint_rest_configuration=jnp.array([0.02, -0.01, 0.03]),
    )


def _routing():
    return jnp.array(
        [
            [-0.02, -0.015, 0.0],
            [0.018, 0.012, 0.009],
        ]
    )


def _actuator():
    return ArticulatedTendonActuator.from_routing(
        _routing(),
        reference_configuration=jnp.array([0.1, -0.05, 0.02]),
        coordinate_offset=jnp.array([0.03, 0.04]),
    )


def _passive():
    return ArticulatedTendonImpedance.from_routing(
        _routing(),
        stiffness=jnp.array([60.0, 45.0]),
        damping=jnp.array([1.2, 0.8]),
        reference_configuration=jnp.array([0.1, -0.05, 0.02]),
        coordinate_offset=jnp.array([0.03, 0.04]),
    )


def _spatial_articulated_robot(*, actuators=None):
    params = articulated_params(
        joint_screw=jnp.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
        tip_position=jnp.array([[0.5, 0.0, 0.0]]),
        center_of_mass_position=jnp.array([[0.25, 0.0, 0.0]]),
        mass=jnp.array([1.0]),
        center_of_mass_inertia=jnp.array([jnp.diag(jnp.array([0.01, 0.02, 0.03]))]),
        gravity=jnp.array([0.0, 0.0, -9.81]),
    )
    return ArticulatedSoftRobot(params, actuators=actuators)


def _spatial_pcs(*, actuators=None):
    params = pcs_params(
        length=jnp.array([0.1]),
        radius=jnp.array([0.01]),
        density=jnp.array([1000.0]),
        young_modulus=jnp.array([1.0e5]),
        shear_modulus=jnp.array([5.0e4]),
        damping_matrix=jnp.eye(6),
        gravity=jnp.zeros(3),
    )
    return PCS(params, PCSStructure(num_gauss_points=3), actuators=actuators)


def test_affine_joint_transmission_coordinates_jacobian_and_power():
    params = AffineJointTransmissionParams(
        routing_matrix=_routing(),
        reference_configuration=jnp.array([0.1, -0.05, 0.02]),
        coordinate_offset=jnp.array([0.03, 0.04]),
    )
    transmission = AffineJointTransmission(params)
    robot = Pendulum(_body_params(), actuators=_actuator())
    q = jnp.array([0.25, -0.2, 0.15])
    qd = jnp.array([0.4, -0.1, 0.3])
    effort = jnp.array([12.0, 8.0])

    expected_coordinate = (
        _routing() @ (q - params.reference_configuration) + params.coordinate_offset
    )
    assert_allclose(transmission.coordinates(robot, q), expected_coordinate)
    assert_allclose(
        jax.jacrev(transmission.coordinates, argnums=1)(robot, q),
        transmission.moment_matrix(robot, q).T,
        rtol=1e-12,
        atol=1e-12,
    )
    yad = transmission.velocities(robot, q, qd)
    tau = transmission.moment_matrix(robot, q) @ effort
    assert_allclose(qd @ tau, yad @ effort, rtol=1e-12, atol=1e-12)


def test_pendulum_identity_unactuated_and_composed_construction():
    identity = Pendulum(_body_params())
    unactuated = Pendulum(_body_params(), actuators=())
    composed = Pendulum(
        _body_params(), actuators=_actuator(), passive_elements=(_passive(),)
    )

    assert identity.num_actuators == identity.num_dofs
    assert unactuated.num_actuators == 0
    assert unactuated.actuation_matrix(jnp.zeros(3)).shape == (3, 0)
    assert composed.num_actuators == 2
    assert len(composed.passive_elements) == 1


def test_direct_effort_is_independent_of_optional_qd():
    robot = Pendulum(_body_params(), actuators=_actuator())
    q = jnp.array([0.2, -0.15, 0.1])
    qd = jnp.array([1.1, -0.8, 0.5])
    tension = jnp.array([15.0, 9.0])

    assert_allclose(
        robot.actuator_efforts(q, tension),
        robot.actuator_efforts(q, tension, qd=qd),
        rtol=0.0,
        atol=0.0,
    )
    assert_allclose(
        robot.actuation_force(q, tension),
        robot.actuation_force(q, tension, qd=qd),
        rtol=0.0,
        atol=0.0,
    )


def test_articulated_tendon_matches_legacy_joint_routing_formula():
    routing = _routing()
    robot = Pendulum(_body_params(), actuators=_actuator())
    q = jnp.array([-0.1, 0.2, -0.25])
    qd = jnp.array([0.3, -0.4, 0.2])
    tension = jnp.array([11.0, 7.0])

    assert_allclose(robot.actuation_matrix(q), routing.T, rtol=0.0, atol=0.0)
    assert_allclose(
        robot.actuation_force(q, tension), routing.T @ tension, rtol=0.0, atol=0.0
    )
    assert_allclose(robot.actuator_velocities(q, qd), routing @ qd)


def test_passive_tendon_force_damping_energy_and_body_superposition():
    passive = _passive()
    body = _body_params()
    robot = Pendulum(body, actuators=(), passive_elements=(passive,))
    q = jnp.array([0.2, -0.12, 0.08])
    routing = _routing()
    coordinate = passive.coordinates(robot, q)
    stiffness = passive.params.stiffness
    damping = passive.params.damping

    expected_passive_force = routing.T @ (stiffness * coordinate)
    expected_passive_damping = routing.T @ (damping[:, None] * routing)
    expected_passive_energy = 0.5 * jnp.sum(stiffness * coordinate**2)

    assert_allclose(passive.elastic_force(robot, q), expected_passive_force)
    assert_allclose(passive.damping_matrix(robot, q), expected_passive_damping)
    assert_allclose(passive.elastic_energy(robot, q), expected_passive_energy)
    assert_allclose(
        robot.elastic_force(q),
        body.joint_stiffness @ (q - body.joint_rest_configuration)
        + expected_passive_force,
    )
    assert_allclose(
        robot.damping_matrix(q), body.joint_damping + expected_passive_damping
    )
    assert_allclose(jax.grad(robot.elastic_energy)(q), robot.elastic_force(q))


def test_tendon_topology_validation_and_independent_parameter_updates():
    actuator = _actuator()
    passive = _passive()
    robot = Pendulum(_body_params(), actuators=actuator, passive_elements=(passive,))

    with pytest.raises(ValueError, match="skip joints"):
        Pendulum(
            _body_params(),
            actuators=ArticulatedTendonActuator.from_routing(
                jnp.array([[1.0, 0.0, 1.0]])
            ),
        )
    with pytest.raises(ValueError, match="full row rank"):
        Pendulum(
            _body_params(),
            actuators=ArticulatedTendonActuator.from_routing(
                jnp.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
            ),
        )

    updated_body = robot.update_params(mass=jnp.array([1.1, 0.9, 0.7]))
    assert_allclose(
        updated_body.actuators[0].params.transmission.routing_matrix,
        robot.actuators[0].params.transmission.routing_matrix,
    )
    assert_allclose(
        updated_body.passive_elements[0].params.stiffness,
        robot.passive_elements[0].params.stiffness,
    )

    new_bounds = jnp.array([20.0, 25.0])
    updated_actuator = robot.update_actuator_params(0, upper_bounds=new_bounds)
    assert_allclose(updated_actuator.params.mass, robot.params.mass)
    assert_allclose(updated_actuator.actuators[0].params.upper_bounds, new_bounds)

    new_stiffness = jnp.array([70.0, 55.0])
    updated_passive = robot.update_passive_element_params(0, stiffness=new_stiffness)
    assert_allclose(updated_passive.params.mass, robot.params.mass)
    assert_allclose(updated_passive.passive_elements[0].params.stiffness, new_stiffness)


def test_generic_affine_map_accepts_pcs_coordinates_but_tendon_preset_rejects_pcs():
    robot = _spatial_pcs()
    matrix = jnp.arange(robot.num_dofs, dtype=jnp.float64)[None, :] * 0.01
    transmission = AffineJointTransmission(
        AffineJointTransmissionParams(
            routing_matrix=matrix,
            reference_configuration=jnp.zeros((robot.num_dofs,)),
            coordinate_offset=jnp.array([0.2]),
        )
    )
    q = jnp.linspace(-0.1, 0.2, robot.num_dofs)

    assert_allclose(transmission.coordinates(robot, q), matrix @ q + jnp.array([0.2]))
    assert_allclose(transmission.moment_matrix(robot, q), matrix.T)

    with pytest.raises(TypeError, match="serial articulated host"):
        _spatial_pcs(
            actuators=ArticulatedTendonActuator.from_routing(
                jnp.ones((1, robot.num_dofs))
            )
        )


def test_threadlike_preset_rejects_articulated_host_during_construction():
    routing = ThreadlikeRouting.linear(
        intercept=jnp.array([[0.0, 0.01, 0.0]]),
        slope=jnp.zeros((1, 3)),
        start_segment_index=0,
        end_segment_index=0,
    )
    with pytest.raises(TypeError, match="continuum host"):
        _spatial_articulated_robot(actuators=ThreadlikeActuator.tendons(routing))
