# ruff: noqa: E402
from dataclasses import fields

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from soromox.systems import (
    ISupport,
    ISupportParams,
    PCSStructure,
    PlanarPCSStructure,
    PneumaticActuatedPlanarPCS,
    PneumaticActuatedPlanarPCSParams,
)


def make_pneumatic_planar_params():
    return PneumaticActuatedPlanarPCSParams(
        base_angle=jnp.array(0.0),
        length=jnp.array([0.1]),
        radius=jnp.array([0.02]),
        density=jnp.array([1000.0]),
        gravity=jnp.array([0.0, -9.81]),
        young_modulus=jnp.array([2e3]),
        shear_modulus=jnp.array([1e3]),
        damping_matrix=1e-3 * jnp.eye(3),
        reference_strain=jnp.array([0.0, 1.0, 0.0]),
        chamber_inner_radius=jnp.array([0.002]),
        chamber_outer_radius=jnp.array([0.004]),
        chamber_angle=jnp.array([jnp.pi / 3.0]),
        chamber_distance=jnp.array([0.01]),
    )


def make_isupport_params(num_segments=1):
    return ISupportParams(
        base_pose=jnp.zeros((6,)),
        length=jnp.full((num_segments,), 0.1),
        radius=jnp.full((num_segments,), 0.02),
        density=jnp.full((num_segments,), 1000.0),
        gravity=jnp.array([0.0, 0.0, -9.81]),
        young_modulus=jnp.full((num_segments,), 2e3),
        shear_modulus=jnp.full((num_segments,), 1e3),
        damping_matrix=1e-3 * jnp.eye(6 * num_segments),
        reference_strain=jnp.tile(
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), num_segments
        ),
        chamber_inner_radius=jnp.full((num_segments,), 0.002),
        chamber_outer_radius=jnp.full((num_segments,), 0.004),
        chamber_distance=jnp.full((num_segments,), 0.01),
        chamber_angle_offset=jnp.zeros((num_segments,)),
    )


def expected_isupport_segment_actuation(params, segment_idx, num_chambers=3):
    chamber_area = jnp.pi * params.chamber_inner_radius[segment_idx] ** 2
    chamber_distance = params.chamber_distance[segment_idx]
    chamber_angles = params.chamber_angle_offset[segment_idx] + jnp.linspace(
        0.0, 2.0 * jnp.pi, num_chambers, endpoint=False
    )
    expected_columns = []
    for angle in chamber_angles:
        expected_columns.append(
            jnp.array(
                [
                    0.0,
                    -chamber_distance * jnp.cos(angle) * chamber_area,
                    chamber_distance * jnp.sin(angle) * chamber_area,
                    chamber_area,
                    0.0,
                    0.0,
                ]
            )
        )
    return jnp.stack(expected_columns, axis=-1)


def test_pneumatic_planar_pcs_circular_geometry_and_actuation_matrix():
    params = make_pneumatic_planar_params()
    robot = PneumaticActuatedPlanarPCS(
        params=params, structure=PlanarPCSStructure(num_gauss_points=1)
    )

    chamber_area = jnp.pi * (
        params.chamber_outer_radius[0] ** 2 - params.chamber_inner_radius[0] ** 2
    )
    chamber_second_moment = (
        jnp.pi
        / 4.0
        * (params.chamber_outer_radius[0] ** 4 - params.chamber_inner_radius[0] ** 4)
        + chamber_area * params.chamber_distance[0] ** 2
    )
    expected_actuation = jnp.array(
        [
            [
                chamber_area * params.chamber_distance[0],
                -chamber_area * params.chamber_distance[0],
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
        params=params,
        structure=PlanarPCSStructure(num_gauss_points=1),
        chamber_cross_section_geometry="concentric",
    )

    expected_area = (
        params.chamber_angle[0]
        / 2.0
        * (params.chamber_outer_radius[0] ** 2 - params.chamber_inner_radius[0] ** 2)
    )
    expected_second_moment = (
        (params.chamber_outer_radius[0] ** 4 - params.chamber_inner_radius[0] ** 4)
        * (
            params.chamber_angle[0]
            - 2.0
            * jnp.sin(params.chamber_angle[0] / 2.0)
            * jnp.cos(params.chamber_angle[0] / 2.0)
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

    updated = robot.update_params(chamber_distance=jnp.array([0.012]))
    assert jnp.allclose(updated.d_chamber, jnp.array([0.012]))
    assert jnp.allclose(robot.d_chamber, params.chamber_distance)


def test_pneumatic_planar_pcs_validates_chamber_parameters():
    params = make_pneumatic_planar_params()
    kwargs = {field.name: getattr(params, field.name) for field in fields(params)}
    kwargs.pop("chamber_inner_radius")
    with pytest.raises(TypeError, match="chamber_inner_radius"):
        PneumaticActuatedPlanarPCSParams(**kwargs)

    params = make_pneumatic_planar_params()
    kwargs = {field.name: getattr(params, field.name) for field in fields(params)}
    kwargs["chamber_outer_radius"] = jnp.array([0.004, 0.005])
    params = PneumaticActuatedPlanarPCSParams(**kwargs)
    with pytest.raises(ValueError, match="chamber_outer_radius"):
        PneumaticActuatedPlanarPCS(
            params=params, structure=PlanarPCSStructure(num_gauss_points=1)
        )


def test_isupport_geometry_helpers_and_actuation_matrix_shape():
    params = make_isupport_params()
    robot = ISupport(params=params, structure=PCSStructure(num_gauss_points=1))

    expected_angles = jnp.array([0.0, 2.0 * jnp.pi / 3.0, 4.0 * jnp.pi / 3.0])
    expected_centroid = jnp.array([0.0, 0.0, params.chamber_distance[0]])
    expected_area = jnp.pi * (
        params.chamber_outer_radius[0] ** 2 - params.chamber_inner_radius[0] ** 2
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


def test_isupport_actuation_matrix_matches_per_segment_chamber_model():
    params = make_isupport_params()
    robot = ISupport(params=params, structure=PCSStructure(num_gauss_points=1))

    actuation_matrix = robot.actuation_matrix(jnp.zeros((robot.num_dofs,)))

    assert actuation_matrix.shape == (6, robot.num_chambers_per_segment)
    assert jnp.allclose(
        actuation_matrix,
        expected_isupport_segment_actuation(params, 0, robot.num_chambers_per_segment),
    )
    assert jnp.all(jnp.linalg.norm(actuation_matrix, axis=0) > 0.0)


def test_isupport_actuation_matrix_assembles_segments_block_diagonal():
    params = make_isupport_params(num_segments=2).replace(
        chamber_inner_radius=jnp.array([0.002, 0.003]),
        chamber_distance=jnp.array([0.01, 0.015]),
        chamber_angle_offset=jnp.array([0.0, 0.2]),
    )
    robot = ISupport(params=params, structure=PCSStructure(num_gauss_points=1))

    actuation_matrix = robot.actuation_matrix(jnp.zeros((robot.num_dofs,)))
    num_chambers = robot.num_chambers_per_segment
    expected_segment_0 = expected_isupport_segment_actuation(params, 0, num_chambers)
    expected_segment_1 = expected_isupport_segment_actuation(params, 1, num_chambers)

    assert robot.num_actuators == 2 * num_chambers
    assert actuation_matrix.shape == (12, robot.num_actuators)
    assert jnp.allclose(actuation_matrix[:6, :num_chambers], expected_segment_0)
    assert jnp.allclose(actuation_matrix[6:, num_chambers:], expected_segment_1)
    assert jnp.allclose(actuation_matrix[:6, num_chambers:], 0.0)
    assert jnp.allclose(actuation_matrix[6:, :num_chambers], 0.0)
    assert jnp.all(jnp.linalg.norm(actuation_matrix, axis=0) > 0.0)


def test_isupport_actuation_matrix_projects_to_active_strains():
    params = make_isupport_params(num_segments=2)
    strain_selector = jnp.array(
        [
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
        ],
        dtype=bool,
    )
    full_robot = ISupport(params=params, structure=PCSStructure(num_gauss_points=1))
    reduced_robot = ISupport(
        params=params,
        structure=PCSStructure(
            num_gauss_points=1,
            strain_selector=strain_selector,
        ),
    )

    full_actuation_matrix = full_robot.actuation_matrix(
        jnp.zeros((full_robot.num_dofs,))
    )
    reduced_actuation_matrix = reduced_robot.actuation_matrix(
        jnp.zeros((reduced_robot.num_dofs,))
    )

    assert reduced_robot.num_actuators == full_robot.num_actuators
    assert reduced_actuation_matrix.shape == (
        reduced_robot.num_dofs,
        reduced_robot.num_actuators,
    )
    assert jnp.allclose(
        reduced_actuation_matrix,
        reduced_robot.B_xi.T @ full_actuation_matrix,
    )


def test_isupport_update_params_and_validation():
    params = make_isupport_params()
    robot = ISupport(params=params, structure=PCSStructure(num_gauss_points=1))

    updated = robot.update_params(chamber_angle_offset=jnp.array([0.25]))
    assert jnp.allclose(updated.varphi_chamber_off, jnp.array([0.25]))
    assert jnp.allclose(robot.varphi_chamber_off, params.chamber_angle_offset)

    with pytest.raises(ValueError, match="chamber_inner_radius"):
        robot.update_params(chamber_inner_radius=jnp.array([0.001, 0.002]))
