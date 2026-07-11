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


def make_pneumatic_isupport_structure(num_segments=1, **kwargs):
    return ISupportStructure(
        rigid_segment_selector=(False,) * num_segments,
        **kwargs,
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
    robot = ISupport(
        params=params,
        structure=make_pneumatic_isupport_structure(num_gauss_points=1),
    )

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
    robot = ISupport(
        params=params,
        structure=make_pneumatic_isupport_structure(num_gauss_points=1),
    )
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
        ISupport(
            unsorted, structure=make_pneumatic_isupport_structure()
        ).chamber_azimuths(0),
        unsorted.chamber_azimuth_angles[0],
    )

    with pytest.raises(ValueError, match="uniformly distributed"):
        params.replace(
            chamber_azimuth_angles=jnp.array([[0.0, 0.5, jnp.pi, 3 * jnp.pi / 2]])
        ).validate()


def test_isupport_defaults_to_canonical_three_chamber_azimuths():
    params = make_isupport_params(num_segments=2).replace(chamber_azimuth_angles=None)
    robot = ISupport(
        params=params,
        structure=make_pneumatic_isupport_structure(2, num_gauss_points=1),
    )
    expected = jnp.tile(
        jnp.array([0.0, 2.0 * jnp.pi / 3.0, 4.0 * jnp.pi / 3.0]),
        (2, 1),
    )

    assert robot.num_chambers_per_segment == 3
    assert jnp.allclose(robot.params.chamber_azimuth_angles, expected)
    assert jnp.allclose(robot.pcs_params.chamber_azimuth_angles, expected)


def test_isupport_actuation_matrix_matches_per_segment_chamber_model():
    params = make_isupport_params()
    robot = ISupport(
        params=params,
        structure=make_pneumatic_isupport_structure(num_gauss_points=1),
    )

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
    robot = ISupport(
        params=params,
        structure=make_pneumatic_isupport_structure(2, num_gauss_points=1),
    )

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
        params=params,
        structure=make_pneumatic_isupport_structure(2, num_gauss_points=1),
    )
    reduced_robot = ISupport(
        params=params,
        structure=make_pneumatic_isupport_structure(
            2,
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


def make_mixed_isupport_params():
    angles = 2.0 * jnp.pi * jnp.arange(3) / 3
    return make_isupport_params(num_segments=5).replace(
        length=jnp.array([0.01, 0.10, 0.02, 0.20, 0.03]),
        radius=jnp.array([0.011, 0.020, 0.012, 0.025, 0.013]),
        density=jnp.array([1200.0, 1000.0, 1300.0, 1050.0, 1400.0]),
        young_modulus=jnp.array([2e9, 2e3, 2e9, 3e3, 2e9]),
        shear_modulus=jnp.array([8e8, 1e3, 8e8, 1.5e3, 8e8]),
        damping_matrix=1e-3 * jnp.eye(30),
        chamber_inner_radius=jnp.array([0.002, 0.003]),
        chamber_outer_radius=jnp.array([0.004, 0.005]),
        chamber_distance=jnp.array([0.01, 0.015]),
        chamber_azimuth_angles=jnp.stack([angles, 0.2 + angles]),
        pcs_segment_lengths=jnp.array([0.04, 0.06, 0.20]),
    )


@pytest.mark.parametrize("num_segments", [4, 5])
def test_isupport_defaults_to_alternating_physical_segment_types(num_segments):
    params = make_isupport_params(num_segments=num_segments).replace(
        chamber_inner_radius=jnp.array([0.002, 0.003]),
        chamber_outer_radius=jnp.array([0.004, 0.005]),
        chamber_distance=jnp.array([0.01, 0.015]),
        chamber_azimuth_angles=jnp.tile(
            2.0 * jnp.pi * jnp.arange(3)[None, :] / 3, (2, 1)
        ),
    )
    robot = ISupport(params, structure=ISupportStructure(num_gauss_points=1))

    assert robot.structure.rigid_segment_selector == tuple(
        i % 2 == 0 for i in range(num_segments)
    )
    assert robot.num_pneumatic_segments == 2


def test_isupport_expands_arbitrary_physical_layout_and_pneumatic_subdivision():
    params = make_mixed_isupport_params()
    robot = ISupport(
        params,
        structure=ISupportStructure(
            num_gauss_points=1,
            pcs_segment_counts=(2, 1),
            rigid_segment_selector=(True, False, True, False, True),
        ),
    )

    assert jnp.allclose(robot.L, jnp.array([0.01, 0.04, 0.06, 0.02, 0.2, 0.03]))
    assert jnp.allclose(robot.r, jnp.array([0.011, 0.020, 0.020, 0.012, 0.025, 0.013]))
    assert jnp.allclose(
        robot.rho, jnp.array([1200.0, 1000.0, 1000.0, 1300.0, 1050.0, 1400.0])
    )
    assert jnp.array_equal(
        robot.pcs_segment_to_pneumatic_segment, jnp.array([-1, 0, 0, -1, 1, -1])
    )
    assert jnp.array_equal(
        robot.pcs_segment_is_rigid,
        jnp.array([True, False, False, True, False, True]),
    )
    assert robot.num_dofs == 18


def test_isupport_supports_adjacent_and_endpoint_pneumatic_segments():
    params = make_isupport_params(num_segments=4).replace(
        chamber_inner_radius=jnp.array([0.002, 0.003]),
        chamber_outer_radius=jnp.array([0.004, 0.005]),
        chamber_distance=jnp.array([0.01, 0.015]),
        chamber_azimuth_angles=jnp.tile(
            2.0 * jnp.pi * jnp.arange(3)[None, :] / 3, (2, 1)
        ),
    )
    robot = ISupport(
        params,
        structure=ISupportStructure(
            num_gauss_points=1,
            rigid_segment_selector=(False, True, True, False),
        ),
    )

    assert jnp.array_equal(
        robot.pcs_segment_to_pneumatic_segment, jnp.array([0, -1, -1, 1])
    )
    assert robot.num_dofs == 12


def test_isupport_rigid_geometry_uses_filled_cylinders():
    robot = ISupport(
        make_mixed_isupport_params(),
        structure=ISupportStructure(
            num_gauss_points=1,
            pcs_segment_counts=(2, 1),
            rigid_segment_selector=(True, False, True, False, True),
        ),
    )
    radius = robot.r[0]
    assert jnp.allclose(
        robot._local_cross_sectional_area(jnp.array(0)), jnp.pi * radius**2
    )
    assert jnp.allclose(
        robot._local_second_moment_of_area(jnp.array(0)),
        jnp.array(
            [jnp.pi * radius**4 / 2, jnp.pi * radius**4 / 4, jnp.pi * radius**4 / 4]
        ),
    )
    expected_pneumatic_area = 3 * jnp.pi * (0.004**2 - 0.002**2)
    assert jnp.allclose(
        robot._local_cross_sectional_area(jnp.array(1)), expected_pneumatic_area
    )


def test_isupport_preserves_rigid_material_damping_properties():
    params = make_mixed_isupport_params().replace(
        damping_matrix=None,
        material_damping_coefficient=jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
    )
    robot = ISupport(
        params,
        structure=ISupportStructure(
            num_gauss_points=1,
            pcs_segment_counts=(2, 1),
            rigid_segment_selector=(True, False, True, False, True),
        ),
    )

    assert jnp.allclose(
        robot.pcs_params.material_damping_coefficient,
        jnp.array([10.0, 20.0, 20.0, 30.0, 40.0, 50.0]),
    )
    rigid_area = jnp.pi * params.radius[0] ** 2
    expected_axial_damping = params.length[0] * 10.0 * 3.0 * rigid_area
    assert jnp.allclose(robot.D_full[3, 3], expected_axial_damping)
    assert jnp.allclose(robot.D_active, robot.B_xi.T @ robot.D_full @ robot.B_xi)


def test_isupport_actuation_groups_pressures_after_generalized_expansion():
    params = make_mixed_isupport_params()
    robot = ISupport(
        params,
        structure=ISupportStructure(
            num_gauss_points=1,
            pcs_segment_counts=(2, 1),
            rigid_segment_selector=(True, False, True, False, True),
        ),
    )
    actuation_matrix = robot.actuation_matrix(jnp.zeros((robot.num_dofs,)))
    num_chambers = robot.num_chambers_per_segment
    assert actuation_matrix.shape == (18, 2 * num_chambers)
    assert jnp.allclose(
        actuation_matrix[:6, :num_chambers],
        expected_isupport_segment_actuation(params, 0, num_chambers),
    )
    assert jnp.allclose(
        actuation_matrix[6:12, :num_chambers], actuation_matrix[:6, :num_chambers]
    )
    assert jnp.allclose(
        actuation_matrix[12:, num_chambers:],
        expected_isupport_segment_actuation(params, 1, num_chambers),
    )


def test_isupport_layout_validation_and_updates():
    params = make_mixed_isupport_params()
    structure = ISupportStructure(
        num_gauss_points=1,
        pcs_segment_counts=(2, 1),
        rigid_segment_selector=(True, False, True, False, True),
    )
    robot = ISupport(params, structure=structure)

    with pytest.raises(TypeError, match="boolean sequence"):
        ISupportStructure(rigid_segment_selector=False)
    with pytest.raises(ValueError, match="one entry per physical segment"):
        ISupport(params, structure=ISupportStructure(rigid_segment_selector=(False,)))
    with pytest.raises(ValueError, match="at least one pneumatic"):
        ISupport(
            params, structure=ISupportStructure(rigid_segment_selector=(True,) * 5)
        )
    with pytest.raises(ValueError, match="one entry per pneumatic segment"):
        ISupport(
            params,
            structure=ISupportStructure(rigid_segment_selector=(False,) * 5),
        )
    with pytest.raises(ValueError, match="sum"):
        ISupport(
            params.replace(pcs_segment_lengths=jnp.array([0.03, 0.06, 0.20])),
            structure=structure,
        )
    bad_reference = params.reference_strain.at[0].set(0.1)
    with pytest.raises(ValueError, match="Rigid segment reference_strain"):
        ISupport(params.replace(reference_strain=bad_reference), structure=structure)

    updated = robot.update_params(
        length=jnp.array([0.015, 0.12, 0.025, 0.20, 0.035]),
        pcs_segment_lengths=jnp.array([0.05, 0.07, 0.20]),
        radius=jnp.array([0.015, 0.020, 0.016, 0.025, 0.017]),
    )
    assert updated.structure.rigid_segment_selector == structure.rigid_segment_selector
    assert jnp.allclose(updated.L, jnp.array([0.015, 0.05, 0.07, 0.025, 0.20, 0.035]))
    assert jnp.allclose(
        updated.r[jnp.array([0, 3, 5])], jnp.array([0.015, 0.016, 0.017])
    )
