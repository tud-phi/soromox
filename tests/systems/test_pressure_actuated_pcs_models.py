# ruff: noqa: E402
from dataclasses import fields

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest
from system_param_builders import planar_base_pose, spatial_base_pose

from soromox.systems import (
    ISupport,
    ISupportParams,
    ISupportStructure,
    PlanarPCSStructure,
    PressureActuatedPlanarPCS,
    PressureActuatedPlanarPCSParams,
)


def make_pressure_planar_params():
    return PressureActuatedPlanarPCSParams(
        base_pose=planar_base_pose(),
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
    angles = 2.0 * jnp.pi * jnp.arange(3) / 3
    return ISupportParams(
        base_pose=spatial_base_pose(),
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
        chamber_azimuth_angles=jnp.tile(angles, (num_segments, 1)),
    )


def expected_isupport_segment_actuation(params, segment_idx, num_chambers=3):
    chamber_area = jnp.pi * params.chamber_inner_radius[segment_idx] ** 2
    chamber_distance = params.chamber_distance[segment_idx]
    chamber_angles = params.chamber_azimuth_angles[segment_idx]
    expected_columns = []
    for angle in chamber_angles:
        expected_columns.append(
            jnp.array(
                [
                    0.0,
                    chamber_distance * jnp.sin(angle) * chamber_area,
                    -chamber_distance * jnp.cos(angle) * chamber_area,
                    chamber_area,
                    0.0,
                    0.0,
                ]
            )
        )
    return jnp.stack(expected_columns, axis=-1)


def test_pressure_planar_pcs_circular_geometry_and_actuation_matrix():
    params = make_pressure_planar_params()
    robot = PressureActuatedPlanarPCS(
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


def test_pressure_planar_pcs_concentric_geometry_and_updates():
    params = make_pressure_planar_params()
    robot = PressureActuatedPlanarPCS(
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


def test_pressure_planar_pcs_validates_chamber_parameters():
    params = make_pressure_planar_params()
    kwargs = {field.name: getattr(params, field.name) for field in fields(params)}
    kwargs.pop("chamber_inner_radius")
    with pytest.raises(TypeError, match="chamber_inner_radius"):
        PressureActuatedPlanarPCSParams(**kwargs)

    params = make_pressure_planar_params()
    kwargs = {field.name: getattr(params, field.name) for field in fields(params)}
    kwargs["chamber_outer_radius"] = jnp.array([0.004, 0.005])
    params = PressureActuatedPlanarPCSParams(**kwargs)
    with pytest.raises(ValueError, match="chamber_outer_radius"):
        PressureActuatedPlanarPCS(
            params=params, structure=PlanarPCSStructure(num_gauss_points=1)
        )


def test_isupport_geometry_helpers_and_actuation_matrix_shape():
    params = make_isupport_params()
    robot = ISupport(params=params, structure=ISupportStructure(num_gauss_points=1))

    expected_angles = jnp.array([0.0, 2.0 * jnp.pi / 3.0, 4.0 * jnp.pi / 3.0])
    expected_centroid = jnp.array([0.0, params.chamber_distance[0], 0.0])
    expected_area = jnp.pi * (
        params.chamber_outer_radius[0] ** 2 - params.chamber_inner_radius[0] ** 2
    )

    assert jnp.allclose(robot.chamber_azimuths(jnp.array(0)), expected_angles)
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


def test_isupport_chamber_azimuth_cardinals_and_symmetry_validation():
    cardinal_angles = jnp.array([[0.0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2]])
    params = make_isupport_params().replace(
        chamber_azimuth_angles=cardinal_angles,
        chamber_distance=jnp.array([0.01]),
    )
    robot = ISupport(params=params, structure=ISupportStructure(num_gauss_points=1))
    assert jnp.allclose(
        robot.local_chamber_offsets(jnp.array(0)),
        jnp.array(
            [
                [0.0, 0.01, 0.0],
                [0.0, 0.0, 0.01],
                [0.0, -0.01, 0.0],
                [0.0, 0.0, -0.01],
            ]
        ),
        atol=1e-12,
    )

    unsorted = params.replace(
        chamber_azimuth_angles=cardinal_angles[:, jnp.array([2, 0, 3, 1])]
    )
    unsorted.validate()
    assert jnp.allclose(
        ISupport(unsorted).chamber_azimuths(0),
        unsorted.chamber_azimuth_angles[0],
    )

    with pytest.raises(ValueError, match="uniformly distributed"):
        params.replace(
            chamber_azimuth_angles=jnp.array([[0.0, 0.5, jnp.pi, 3 * jnp.pi / 2]])
        ).validate()


def test_isupport_defaults_to_canonical_three_chamber_azimuths():
    params = make_isupport_params(num_segments=2).replace(chamber_azimuth_angles=None)
    robot = ISupport(params=params, structure=ISupportStructure(num_gauss_points=1))
    expected = jnp.tile(
        jnp.array([0.0, 2.0 * jnp.pi / 3.0, 4.0 * jnp.pi / 3.0]),
        (2, 1),
    )

    assert robot.num_chambers_per_segment == 3
    assert jnp.allclose(robot.params.chamber_azimuth_angles, expected)
    assert jnp.allclose(robot.pcs_params.chamber_azimuth_angles, expected)


def test_isupport_actuation_matrix_matches_per_segment_chamber_model():
    params = make_isupport_params()
    robot = ISupport(params=params, structure=ISupportStructure(num_gauss_points=1))

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
        chamber_azimuth_angles=jnp.stack(
            [
                2 * jnp.pi * jnp.arange(3) / 3,
                (0.2 + 2 * jnp.pi * jnp.arange(3) / 3)[jnp.array([2, 0, 1])],
            ]
        ),
    )
    robot = ISupport(params=params, structure=ISupportStructure(num_gauss_points=1))

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
    full_robot = ISupport(
        params=params, structure=ISupportStructure(num_gauss_points=1)
    )
    reduced_robot = ISupport(
        params=params,
        structure=ISupportStructure(
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


@pytest.mark.parametrize(
    ("pcs_segment_counts", "rigid_connector_selector"),
    [
        ([2, 2], [False, False, False]),
        ((2, 2), (False, False, False)),
        (jnp.array([2, 2]), jnp.array([False, False, False])),
    ],
)
def test_isupport_structure_accepts_sequence_inputs(
    pcs_segment_counts, rigid_connector_selector
):
    params = make_isupport_params(num_segments=2).replace(
        pcs_segment_lengths=jnp.array([0.04, 0.06, 0.025, 0.075])
    )
    structure = ISupportStructure(
        num_gauss_points=1,
        pcs_segment_counts=pcs_segment_counts,
        rigid_connector_selector=rigid_connector_selector,
    )
    robot = ISupport(params=params, structure=structure)

    assert robot.structure.pcs_segment_counts == (2, 2)
    assert robot.structure.rigid_connector_selector == (False, False, False)
    assert robot.num_pneumatic_segments == 2
    assert robot.num_pcs_segments == 4
    assert jnp.allclose(robot.L, jnp.array([0.04, 0.06, 0.025, 0.075]))
    assert jnp.allclose(
        robot.pcs_segment_to_pneumatic_segment,
        jnp.array([0, 0, 1, 1], dtype=jnp.int32),
    )


def test_isupport_structure_accepts_scalar_topology_inputs():
    params = make_isupport_params(num_segments=2).replace(
        pcs_segment_lengths=jnp.array([0.05, 0.05, 0.05, 0.05])
    )
    robot = ISupport(
        params=params,
        structure=ISupportStructure(
            num_gauss_points=1,
            pcs_segment_counts=2,
            rigid_connector_selector=False,
        ),
    )

    assert robot.structure.pcs_segment_counts == (2, 2)
    assert robot.structure.rigid_connector_selector == (False, False, False)
    assert jnp.allclose(robot.L, jnp.array([0.05, 0.05, 0.05, 0.05]))


def test_isupport_structure_validates_split_sums_and_connector_count():
    params = make_isupport_params(num_segments=2)

    with pytest.raises(ValueError, match="sum"):
        ISupport(
            params=params.replace(pcs_segment_lengths=jnp.array([0.04, 0.05, 0.1])),
            structure=ISupportStructure(
                num_gauss_points=1,
                pcs_segment_counts=(2, 1),
            ),
        )

    with pytest.raises(ValueError, match="pcs_segment_counts"):
        ISupport(
            params=params,
            structure=ISupportStructure(
                num_gauss_points=1,
                pcs_segment_counts=(2, 2, 1),
            ),
        )

    with pytest.raises(ValueError, match="rigid_connector_selector"):
        ISupport(
            params=params,
            structure=ISupportStructure(
                num_gauss_points=1,
                rigid_connector_selector=[False, False],
            ),
        )

    with pytest.raises(ValueError, match="unselected connector"):
        ISupport(
            params=params.replace(rigid_connector_lengths=jnp.array([0.01, 0.0, 0.0])),
            structure=ISupportStructure(num_gauss_points=1),
        )

    with pytest.raises(ValueError, match="Selected rigid connectors"):
        ISupport(
            params=params,
            structure=ISupportStructure(
                num_gauss_points=1,
                rigid_connector_selector=[True, False, False],
            ),
        )


def test_isupport_rigid_connectors_expand_in_documented_sequence():
    params = make_isupport_params(num_segments=2).replace(
        length=jnp.array([0.1, 0.2]),
        pcs_segment_lengths=jnp.array([0.04, 0.06, 0.2]),
        rigid_connector_lengths=jnp.array([0.01, 0.02, 0.03]),
    )
    robot = ISupport(
        params=params,
        structure=ISupportStructure(
            num_gauss_points=1,
            pcs_segment_counts=(2, 1),
            rigid_connector_selector=(True, True, True),
        ),
    )

    assert jnp.allclose(robot.L, jnp.array([0.01, 0.04, 0.06, 0.02, 0.2, 0.03]))
    assert jnp.allclose(
        robot.pcs_segment_to_pneumatic_segment,
        jnp.array([-1, 0, 0, -1, 1, -1], dtype=jnp.int32),
    )
    assert jnp.allclose(
        robot.pcs_segment_is_rigid,
        jnp.array([True, False, False, True, False, True], dtype=bool),
    )
    assert robot.num_dofs == 18

    xi_ref = robot.xi_ref.reshape(robot.num_pcs_segments, 6)
    straight = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert jnp.allclose(xi_ref[0], straight)
    assert jnp.allclose(xi_ref[3], straight)
    assert jnp.allclose(xi_ref[5], straight)


def test_isupport_actuation_groups_pressures_by_pneumatic_segment_after_split():
    params = make_isupport_params(num_segments=2).replace(
        length=jnp.array([0.1, 0.2]),
        chamber_inner_radius=jnp.array([0.002, 0.003]),
        chamber_distance=jnp.array([0.01, 0.015]),
        chamber_azimuth_angles=jnp.stack(
            [2 * jnp.pi * jnp.arange(3) / 3, 0.2 + 2 * jnp.pi * jnp.arange(3) / 3]
        ),
        pcs_segment_lengths=jnp.array([0.04, 0.06, 0.2]),
        rigid_connector_lengths=jnp.array([0.01, 0.02, 0.03]),
    )
    robot = ISupport(
        params=params,
        structure=ISupportStructure(
            num_gauss_points=1,
            pcs_segment_counts=(2, 1),
            rigid_connector_selector=(True, True, True),
        ),
    )

    actuation_matrix = robot.actuation_matrix(jnp.zeros((robot.num_dofs,)))
    num_chambers = robot.num_chambers_per_segment
    expected_segment_0 = expected_isupport_segment_actuation(params, 0, num_chambers)
    expected_segment_1 = expected_isupport_segment_actuation(params, 1, num_chambers)

    assert robot.num_actuators == 2 * num_chambers
    assert actuation_matrix.shape == (18, robot.num_actuators)
    assert jnp.allclose(actuation_matrix[:6, :num_chambers], expected_segment_0)
    assert jnp.allclose(actuation_matrix[6:12, :num_chambers], expected_segment_0)
    assert jnp.allclose(actuation_matrix[12:, num_chambers:], expected_segment_1)
    assert jnp.allclose(actuation_matrix[:12, num_chambers:], 0.0)
    assert jnp.allclose(actuation_matrix[12:, :num_chambers], 0.0)


def test_isupport_update_params_and_validation():
    params = make_isupport_params()
    robot = ISupport(params=params, structure=ISupportStructure(num_gauss_points=1))

    updated_angles = 0.25 + 2 * jnp.pi * jnp.arange(3)[None, :] / 3
    updated = robot.update_params(chamber_azimuth_angles=updated_angles)
    assert jnp.allclose(updated.chamber_azimuth_angles, updated_angles)
    assert jnp.allclose(robot.chamber_azimuth_angles, params.chamber_azimuth_angles)

    with pytest.raises(ValueError, match="chamber_inner_radius"):
        robot.update_params(chamber_inner_radius=jnp.array([0.001, 0.002]))

    split_params = make_isupport_params(num_segments=2).replace(
        length=jnp.array([0.1, 0.2]),
        pcs_segment_lengths=jnp.array([0.04, 0.06, 0.2]),
        rigid_connector_lengths=jnp.array([0.01, 0.02, 0.03]),
    )
    split_robot = ISupport(
        params=split_params,
        structure=ISupportStructure(
            num_gauss_points=1,
            pcs_segment_counts=(2, 1),
            rigid_connector_selector=(True, True, True),
        ),
    )

    updated_split = split_robot.update_params(
        length=jnp.array([0.12, 0.2]),
        pcs_segment_lengths=jnp.array([0.05, 0.07, 0.2]),
        rigid_connector_lengths=jnp.array([0.015, 0.025, 0.035]),
    )

    assert updated_split.structure.pcs_segment_counts == (2, 1)
    assert updated_split.structure.rigid_connector_selector == (True, True, True)
    assert jnp.allclose(
        updated_split.L,
        jnp.array([0.015, 0.05, 0.07, 0.025, 0.2, 0.035]),
    )
