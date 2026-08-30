"""Tests for the low-dimensional rigid-soft pendulum examples."""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples.simulation.gvs.soft_pendulums import (
    AFFINE_CURVATURE_MATRIX,
    SoftInvertedPendulumConfig,
    SoftSwingUpPendulumConfig,
    cart_pole_hanging_configuration,
    cart_pole_upright_configuration,
    furuta_hanging_configuration,
    furuta_upright_configuration,
    make_fixed_pendulum,
    make_r_soft_inverted_pendulum,
    make_soft_cart_pole,
    make_soft_furuta,
    make_soft_pendubot,
    pendubot_hanging_configuration,
    pendubot_upright_configuration,
    r_sip_hanging_configuration,
    r_sip_upright_configuration,
    simulate_open_loop,
    sip_initial_configuration,
    sip_tip_torque_generalized_force,
    split_state,
)


@pytest.fixture(scope="module")
def sip_config() -> SoftInvertedPendulumConfig:
    return SoftInvertedPendulumConfig()


@pytest.fixture(scope="module")
def paper_config() -> SoftSwingUpPendulumConfig:
    return SoftSwingUpPendulumConfig()


@pytest.fixture(scope="module")
def models(sip_config, paper_config):
    return {
        "r_sip": make_r_soft_inverted_pendulum(sip_config),
        "cart_pole": make_soft_cart_pole(sip_config),
        "pendubot": make_soft_pendubot(paper_config),
        "furuta": make_soft_furuta(paper_config),
    }


def test_original_sip_parameters_and_tip_torque_are_retained(sip_config):
    robot = make_fixed_pendulum(sip_config)
    assert robot.num_dofs == 2
    assert robot.num_actuators == 0
    assert robot.structure.scale_rotational_basis_by_length
    assert_allclose(robot.K, sip_config.stiffness)
    assert_allclose(robot.D_active, sip_config.damping)
    assert_allclose(sip_config.stiffness, AFFINE_CURVATURE_MATRIX)
    assert_allclose(sip_initial_configuration(), [np.pi / 4.0, -np.pi / 4.0])
    assert_allclose(sip_tip_torque_generalized_force(2.0), [2.0, 1.0])


def test_coordinate_order_topology_axes_and_length_scaling(models):
    expected = {
        "r_sip": (3, ((1, 2),), (("revolute", "z"),)),
        "cart_pole": (
            4,
            ((1, 0), (1, 2)),
            (("prismatic", "y"), ("revolute", "z")),
        ),
        "pendubot": (
            6,
            ((1, 2), (1, 2)),
            (("revolute", "z"), ("revolute", "z")),
        ),
        "furuta": (
            4,
            ((1, 0), (1, 2)),
            (("revolute", "z"), ("revolute", "y")),
        ),
    }
    for name, robot in models.items():
        num_dofs, segment_dofs, joints = expected[name]
        assert robot.num_dofs == num_dofs
        assert tuple(map(tuple, np.asarray(robot.dofs_per_segment))) == segment_dofs
        assert (
            tuple(
                (segment.joint.type, segment.joint.axis)
                for segment in robot.structure.segments
            )
            == joints
        )
        assert robot.structure.scale_rotational_basis_by_length


def test_joint_damping_matches_the_papers(sip_config, paper_config, models):
    assert_allclose(models["r_sip"].D_active[0, 0], sip_config.r_sip_joint_damping)
    assert_allclose(models["cart_pole"].D_active[1, 1], sip_config.cart_hinge_damping)
    assert_allclose(
        jnp.diag(models["pendubot"].D_active)[jnp.array([0, 3])],
        paper_config.joint_damping,
    )
    assert_allclose(jnp.diag(models["furuta"].D_active)[:2], paper_config.joint_damping)


def test_mass_and_geometry_are_assigned_to_the_intended_links(
    sip_config, paper_config, models
):
    r_sip = models["r_sip"]
    cart = models["cart_pole"]
    pendubot = models["pendubot"]
    furuta = models["furuta"]
    assert_allclose(r_sip.params.link.length, [sip_config.length])
    assert_allclose(r_sip.params.link.density, [sip_config.density])
    assert_allclose(
        cart.params.link.length, [sip_config.cart_body_length, sip_config.length]
    )
    assert_allclose(
        cart.params.link.density, [sip_config.cart_body_density, sip_config.density]
    )
    assert_allclose(
        pendubot.params.link.length,
        [paper_config.soft_length / 2.0, paper_config.soft_length / 2.0],
    )
    assert_allclose(pendubot.params.link.density, paper_config.soft_density)
    assert_allclose(
        pendubot.params.link.cross_section.coefficients[:, 0],
        paper_config.soft_radius,
    )
    assert_allclose(
        furuta.params.link.length,
        [paper_config.furuta_arm_length, paper_config.soft_length],
    )
    assert_allclose(
        furuta.params.link.density,
        [paper_config.furuta_arm_density, paper_config.soft_density],
    )


def test_actuation_matrices_labels_units_and_bounds(models):
    expected = {
        "r_sip": ([1.0, 0.0, 0.0], "base_joint_torque", "N m", -np.inf, np.inf),
        "cart_pole": ([1.0, 0.0, 0.0, 0.0], "cart_force", "N", -200.0, 200.0),
        "pendubot": (
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "first_joint_torque",
            "N m",
            -10.0,
            10.0,
        ),
        "furuta": (
            [1.0, 0.0, 0.0, 0.0],
            "first_joint_torque",
            "N m",
            -80.0,
            80.0,
        ),
    }
    for name, robot in models.items():
        routing, label, units, lower, upper = expected[name]
        q = jnp.zeros((robot.num_dofs,))
        assert_allclose(robot.actuation_matrix(q)[:, 0], routing)
        metadata = robot.actuator_input_metadata[0]
        assert metadata.labels == (label,)
        assert metadata.units == (units,)
        assert_allclose(metadata.lower_bounds, [lower])
        assert_allclose(metadata.upper_bounds, [upper])


def _position(robot, q, s):
    return robot.forward_kinematics(q, jnp.asarray(s))[:3, 3]


def test_upright_and_hanging_helpers_match_forward_kinematics(
    sip_config, paper_config, models
):
    r_sip = models["r_sip"]
    assert_allclose(
        _position(r_sip, r_sip_upright_configuration(), r_sip.length),
        [0.0, 0.0, sip_config.length],
        atol=1e-10,
    )
    assert_allclose(
        _position(r_sip, r_sip_hanging_configuration(), r_sip.length),
        [0.0, 0.0, -sip_config.length],
        atol=1e-10,
    )

    cart = models["cart_pole"]
    pivot_s = sip_config.cart_body_length
    upright_pivot = _position(cart, cart_pole_upright_configuration(), pivot_s)
    upright_tip = _position(cart, cart_pole_upright_configuration(), cart.length)
    hanging_pivot = _position(cart, cart_pole_hanging_configuration(), pivot_s)
    hanging_tip = _position(cart, cart_pole_hanging_configuration(), cart.length)
    assert_allclose(cart_pole_upright_configuration(), jnp.zeros(4))
    assert_allclose(
        upright_tip - upright_pivot, [0.0, 0.0, sip_config.length], atol=1e-10
    )
    assert_allclose(
        hanging_tip - hanging_pivot, [0.0, 0.0, -sip_config.length], atol=1e-10
    )

    pendubot = models["pendubot"]
    midpoint = paper_config.soft_length / 2.0
    assert_allclose(pendubot_hanging_configuration(), jnp.zeros(6))
    assert_allclose(
        _position(pendubot, pendubot_hanging_configuration(), midpoint),
        [0.0, 0.0, -midpoint],
        atol=1e-10,
    )
    assert_allclose(
        _position(pendubot, pendubot_hanging_configuration(), pendubot.length),
        [0.0, 0.0, -paper_config.soft_length],
        atol=1e-10,
    )
    assert_allclose(
        _position(pendubot, pendubot_upright_configuration(), pendubot.length),
        [0.0, 0.0, paper_config.soft_length],
        atol=1e-10,
    )

    furuta = models["furuta"]
    arm_tip = _position(
        furuta, furuta_upright_configuration(), paper_config.furuta_arm_length
    )
    upright_tip = _position(furuta, furuta_upright_configuration(), furuta.length)
    hanging_tip = _position(furuta, furuta_hanging_configuration(), furuta.length)
    assert_allclose(arm_tip, [paper_config.furuta_arm_length, 0.0, 0.0])
    assert_allclose(
        upright_tip - arm_tip, [0.0, 0.0, paper_config.soft_length], atol=1e-10
    )
    assert_allclose(
        hanging_tip - arm_tip, [0.0, 0.0, -paper_config.soft_length], atol=1e-10
    )


@pytest.mark.parametrize(
    ("model_name", "configuration", "bounded_input"),
    [
        ("r_sip", r_sip_hanging_configuration(), 1.0),
        ("cart_pole", cart_pole_hanging_configuration(), 200.0),
        ("pendubot", pendubot_hanging_configuration(), 10.0),
        ("furuta", furuta_hanging_configuration(), 80.0),
    ],
)
@pytest.mark.parametrize("use_bounded_input", [False, True])
def test_short_zero_and_bounded_input_rollouts_are_finite(
    models, model_name, configuration, bounded_input, use_bounded_input
):
    robot = models[model_name]
    trajectory = simulate_open_loop(
        robot,
        initial_configuration=configuration,
        control=bounded_input if use_bounded_input else 0.0,
        t1=0.01,
        solver_dt=1e-3,
        save_dt=0.01,
    )
    assert trajectory.y.shape == (2, 2 * robot.num_dofs)
    assert jnp.all(jnp.isfinite(trajectory.y))
    q_ts, qd_ts = split_state(trajectory.y, robot.num_dofs)
    assert q_ts.shape == qd_ts.shape == (2, robot.num_dofs)
