# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from soromox.systems import ISupport, PneumaticActuatedPlanarPCS


def make_pneumatic_planar_params():
    return {
        "th0": jnp.array(0.0),
        "L": jnp.array([0.1]),
        "r": jnp.array([0.02]),
        "rho": jnp.array([1000.0]),
        "g": jnp.array([0.0, -9.81]),
        "E": jnp.array([2e3]),
        "G": jnp.array([1e3]),
        "D": 1e-3 * jnp.eye(3),
        "r_chamber_in": jnp.array([0.002]),
        "r_chamber_out": jnp.array([0.004]),
        "phi_chamber": jnp.array([jnp.pi / 3.0]),
        "d_chamber": jnp.array([0.01]),
    }


def make_isupport_params():
    return {
        "p0": jnp.zeros((6,)),
        "L": jnp.array([0.1]),
        "r": jnp.array([0.02]),
        "rho": jnp.array([1000.0]),
        "g": jnp.array([0.0, 0.0, -9.81]),
        "E": jnp.array([2e3]),
        "G": jnp.array([1e3]),
        "D": 1e-3 * jnp.eye(6),
        "r_chamber_in": jnp.array([0.002]),
        "r_chamber_out": jnp.array([0.004]),
        "d_chamber": jnp.array([0.01]),
        "varphi_chamber_off": jnp.array([0.0]),
    }


def test_pneumatic_planar_pcs_circular_geometry_and_actuation_matrix():
    params = make_pneumatic_planar_params()
    robot = PneumaticActuatedPlanarPCS(1, params, num_gauss_points=1)

    chamber_area = jnp.pi * (
        params["r_chamber_out"][0] ** 2 - params["r_chamber_in"][0] ** 2
    )
    chamber_second_moment = (
        jnp.pi
        / 4.0
        * (params["r_chamber_out"][0] ** 4 - params["r_chamber_in"][0] ** 4)
        + chamber_area * params["d_chamber"][0] ** 2
    )
    expected_actuation = jnp.array(
        [
            [
                chamber_area * params["d_chamber"][0],
                -chamber_area * params["d_chamber"][0],
            ],
            [chamber_area, chamber_area],
            [0.0, 0.0],
        ]
    )

    assert jnp.allclose(
        robot._local_chamber_cross_sectional_area(jnp.array(0)), chamber_area
    )
    assert jnp.allclose(
        robot._local_chamber_second_moment_of_area(jnp.array(0)),
        chamber_second_moment,
    )
    assert jnp.allclose(robot.actuation_matrix(jnp.zeros((3,))), expected_actuation)


def test_pneumatic_planar_pcs_concentric_geometry_and_updates():
    params = make_pneumatic_planar_params()
    robot = PneumaticActuatedPlanarPCS(
        1,
        params,
        num_gauss_points=1,
        chamber_cross_section_geometry="concentric",
    )

    expected_area = (
        params["phi_chamber"][0]
        / 2.0
        * (params["r_chamber_out"][0] ** 2 - params["r_chamber_in"][0] ** 2)
    )
    expected_second_moment = (
        (params["r_chamber_out"][0] ** 4 - params["r_chamber_in"][0] ** 4)
        * (
            params["phi_chamber"][0]
            - 2.0
            * jnp.sin(params["phi_chamber"][0] / 2.0)
            * jnp.cos(params["phi_chamber"][0] / 2.0)
        )
        / 8.0
    )

    assert jnp.allclose(
        robot._local_chamber_cross_sectional_area(jnp.array(0)), expected_area
    )
    assert jnp.allclose(
        robot._local_chamber_second_moment_of_area(jnp.array(0)),
        expected_second_moment,
    )

    updated = robot.update_params({"d_chamber": jnp.array([0.012])})
    assert jnp.allclose(updated.d_chamber, jnp.array([0.012]))
    assert jnp.allclose(robot.d_chamber, params["d_chamber"])


def test_pneumatic_planar_pcs_validates_chamber_parameters():
    params = make_pneumatic_planar_params()
    params.pop("r_chamber_in")

    with pytest.raises(KeyError, match="r_chamber_in"):
        PneumaticActuatedPlanarPCS(1, params, num_gauss_points=1)

    params = make_pneumatic_planar_params()
    params["r_chamber_out"] = 0.004
    with pytest.raises(TypeError, match="r_chamber_out"):
        PneumaticActuatedPlanarPCS(1, params, num_gauss_points=1)


def test_isupport_geometry_helpers_and_actuation_matrix_shape():
    params = make_isupport_params()
    robot = ISupport(1, params, num_gauss_points=1)

    expected_angles = jnp.array([0.0, 2.0 * jnp.pi / 3.0, 4.0 * jnp.pi / 3.0])
    expected_centroid = jnp.array([0.0, 0.0, params["d_chamber"][0]])
    expected_area = jnp.pi * (
        params["r_chamber_out"][0] ** 2 - params["r_chamber_in"][0] ** 2
    )

    assert jnp.allclose(
        robot._local_actuator_polar_angles(jnp.array(0)), expected_angles
    )
    assert jnp.allclose(
        robot._local_actuator_centroid(jnp.array(0), jnp.array(0.0)),
        expected_centroid,
    )
    assert jnp.allclose(
        robot._local_actuator_cross_sectional_area(jnp.array(0)), expected_area
    )
    assert jnp.allclose(
        robot._local_cross_sectional_area(jnp.array(0)),
        robot.num_chambers_per_segment * expected_area,
    )

    actuation_matrix = robot.actuation_matrix(jnp.zeros((6,)))
    assert actuation_matrix.shape == (6, 3)
    assert not jnp.isnan(actuation_matrix).any()


def test_isupport_update_params_and_validation():
    params = make_isupport_params()
    robot = ISupport(1, params, num_gauss_points=1)

    updated = robot.update_params({"varphi_chamber_off": jnp.array([0.25])})
    assert jnp.allclose(updated.varphi_chamber_off, jnp.array([0.25]))
    assert jnp.allclose(robot.varphi_chamber_off, params["varphi_chamber_off"])

    with pytest.raises(ValueError, match="r_chamber_in"):
        robot.update_params({"r_chamber_in": jnp.array([0.001, 0.002])})
