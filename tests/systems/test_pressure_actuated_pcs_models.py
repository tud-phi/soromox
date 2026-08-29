# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import pytest

from soromox.actuation import ThreadlikeActuator
from soromox.rendering import MatplotlibRenderer
from soromox.systems import (
    PCS,
    ContinuumLinkParams,
    CrossSectionParams,
    ISupport,
    ISupportParams,
    ISupportStructure,
    LinkSpec,
)


def make_isupport_params(num_segments=1):
    angles = 2.0 * jnp.pi * jnp.arange(3) / 3
    length = jnp.full((num_segments,), 0.1)
    radius = jnp.full((num_segments,), 0.02)
    young = jnp.full((num_segments,), 2e3)
    shear = jnp.full((num_segments,), 1e3)
    area = jnp.pi * radius**2
    transverse = jnp.pi * radius**4 / 4.0
    polar = 2.0 * transverse
    stiffness = length[:, None, None] * jax.vmap(jnp.diag)(
        jnp.stack(
            [
                shear * polar,
                young * transverse,
                young * transverse,
                young * area,
                shear * area,
                shear * area,
            ],
            axis=1,
        )
    )
    return ISupportParams(
        gravity=jnp.array([0.0, 0.0, -9.81]),
        link=ContinuumLinkParams(
            length=length,
            density=jnp.full((num_segments,), 1000.0),
            reference_strain=jnp.tile(
                jnp.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]),
                (num_segments, 1),
            ),
            cross_section=CrossSectionParams(coefficients=radius[:, None]),
            stiffness=stiffness,
            damping=1e-3 * jnp.tile(jnp.eye(6)[None, :, :], (num_segments, 1, 1)),
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


def expected_isupport_segment_actuation(
    params, pneumatic_segment_idx, segment_length=None, num_chambers=3
):
    if params.chamber_effective_pressure_area is None:
        chamber_area = jnp.pi * (
            params.chamber_outer_radius[pneumatic_segment_idx] ** 2
            - params.chamber_inner_radius[pneumatic_segment_idx] ** 2
        )
    else:
        chamber_area = params.chamber_effective_pressure_area[pneumatic_segment_idx]
    if segment_length is None:
        segment_length = params.link.length[pneumatic_segment_idx]
    chamber_distance = params.chamber_distance[pneumatic_segment_idx]
    chamber_angles = params.chamber_azimuth_angles[pneumatic_segment_idx]
    expected_columns = []
    for angle in chamber_angles:
        expected_columns.append(
            segment_length
            * jnp.array(
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


def updated_isupport_params(params):
    return params.replace(
        gravity=params.gravity.at[2].set(-9.7),
        link=params.link.replace(
            length=1.02 * params.link.length,
            density=1.01 * params.link.density,
            stiffness=1.03 * params.link.stiffness,
            damping=1.04 * params.link.damping,
        ),
        chamber_outer_radius=1.05 * params.chamber_outer_radius,
        chamber_distance=1.06 * params.chamber_distance,
        chamber_azimuth_angles=params.chamber_azimuth_angles + 0.1,
    )


def isupport_runtime_summary(robot):
    actuator = robot.actuators[0]
    return jnp.concatenate(
        [
            robot.fixed_base_pose,
            robot.g,
            robot.L,
            robot.rho,
            robot.r_chamber_out,
            robot.d_chamber,
            robot.K_full.reshape(-1),
            robot.D_full.reshape(-1),
            actuator.params.transmission.routing.intercept.reshape(-1),
            actuator.params.transmission.coordinate_scale,
        ]
    )


def test_isupport_parameter_updates_are_immutable_and_refresh_eager_caches():
    params = make_isupport_params()
    robot = ISupport(
        params=params,
        structure=make_pneumatic_isupport_structure(num_gauss_points=1),
    )
    updated_params = updated_isupport_params(params)
    before = isupport_runtime_summary(robot)

    updated = robot.with_params(updated_params)

    assert jnp.allclose(isupport_runtime_summary(robot), before)
    assert not jnp.allclose(isupport_runtime_summary(updated), before)


def test_isupport_same_structure_parameter_updates_compile_once():
    params = make_isupport_params()
    robot = ISupport(
        params=params,
        structure=make_pneumatic_isupport_structure(num_gauss_points=1),
    )
    updated_params = updated_isupport_params(params)
    trace_count = {"value": 0}

    @eqx.filter_jit
    def compiled_summary(current_params):
        trace_count["value"] += 1
        return isupport_runtime_summary(robot.with_params(current_params))

    baseline = compiled_summary(params)
    actual = compiled_summary(updated_params)

    assert trace_count["value"] == 1
    assert not jnp.allclose(actual, baseline)
    assert jnp.allclose(
        actual, isupport_runtime_summary(robot.with_params(updated_params))
    )


def test_isupport_parameter_gradient_passes_through_compiled_update():
    params = make_isupport_params()
    robot = ISupport(
        params=params,
        structure=make_pneumatic_isupport_structure(num_gauss_points=1),
    )

    @eqx.filter_jit
    def pressure_scale(outer_radius):
        current = params.replace(chamber_outer_radius=outer_radius)
        updated = robot.with_params(current)
        return jnp.sum(updated.actuators[0].params.transmission.coordinate_scale)

    gradient = jax.grad(pressure_scale)(params.chamber_outer_radius)

    assert gradient.shape == params.chamber_outer_radius.shape
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(gradient != 0.0)


def test_isupport_chamber_count_change_requires_reconstruction():
    params = make_isupport_params()
    robot = ISupport(
        params=params,
        structure=make_pneumatic_isupport_structure(num_gauss_points=1),
    )
    four_angles = 2.0 * jnp.pi * jnp.arange(4, dtype=jnp.float64)[None, :] / 4.0

    with pytest.raises(ValueError, match="number of chambers|reconstruction"):
        robot.with_params(params.replace(chamber_azimuth_angles=four_angles))


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


def test_isupport_second_moment_gradient_is_finite_at_centered_chamber():
    """The parallel-axis squared norm must remain differentiable at zero."""

    def second_moment(distance):
        params = make_isupport_params().replace(
            chamber_distance=jnp.asarray([distance])
        )
        robot = ISupport(
            params=params,
            structure=make_pneumatic_isupport_structure(num_gauss_points=1),
        )
        return jnp.sum(
            robot._local_actuator_second_moment_of_area(jnp.array(0), jnp.array(0.0))
        )

    gradient = jax.grad(second_moment)(jnp.array(0.0))

    assert jnp.isfinite(gradient)
    assert gradient == 0.0


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
        expected_isupport_segment_actuation(
            params, 0, num_chambers=robot.num_chambers_per_segment
        ),
    )
    assert jnp.all(jnp.linalg.norm(actuation_matrix, axis=0) > 0.0)


def test_isupport_uses_threadlike_pressure_chambers_and_resolves_area():
    params = make_isupport_params()
    robot = ISupport(
        params=params,
        structure=make_pneumatic_isupport_structure(num_gauss_points=2),
    )

    expected_area = jnp.pi * (
        params.chamber_outer_radius**2 - params.chamber_inner_radius**2
    )
    assert len(robot.actuators) == 1
    assert isinstance(robot.actuators[0], ThreadlikeActuator)
    assert robot.actuators[0].kind == "pneumatic"
    assert robot.actuators[0].transmission.routing.params.start_segment_index == (
        0,
        0,
        0,
    )
    assert robot.actuators[0].transmission.routing.params.end_segment_index == (
        0,
        0,
        0,
    )
    coordinate_scale = robot.actuators[0].params.transmission.coordinate_scale
    assert jnp.allclose(
        coordinate_scale,
        jnp.repeat(expected_area, robot.num_chambers_per_segment),
    )
    assert (
        robot.actuator_visual_layers(
            jnp.zeros((robot.num_dofs,)), jnp.linspace(0.0, robot.length, 5)
        )
        == ()
    )
    assert (
        MatplotlibRenderer(robot, num_points=5).compute_actuator_visual_layers(
            jnp.zeros((robot.num_dofs,))
        )
        == ()
    )


def test_isupport_matches_pr116_threadlike_pressure_mapping():
    params = make_isupport_params().replace(
        chamber_effective_pressure_area=jnp.array([2.5e-5])
    )
    robot = ISupport(
        params=params,
        structure=make_pneumatic_isupport_structure(num_gauss_points=3),
    )
    q = jnp.array([0.2, -0.1, 0.15, 0.03, 0.04, -0.02])
    pressure = jnp.array([1.0e4, 2.0e4, 3.0e4])

    routing = robot.actuators[0].transmission.routing
    raw_path_matrix = robot._threadlike_moment_matrix(q, routing)
    pressure_area = jnp.repeat(
        params.chamber_effective_pressure_area,
        robot.num_chambers_per_segment,
    )
    expected_pressure_matrix = raw_path_matrix * pressure_area[None, :]

    assert jnp.allclose(robot.actuation_matrix(q), expected_pressure_matrix)
    assert jnp.allclose(
        robot.actuator_efforts(q, pressure, qd=jnp.zeros_like(q)), pressure
    )
    assert jnp.allclose(
        robot.actuation_force(q, pressure),
        expected_pressure_matrix @ pressure,
    )
    assert jnp.allclose(
        jax.jacobian(robot.actuator_coordinates)(q),
        robot.actuation_matrix(q).T,
        rtol=1e-7,
        atol=1e-10,
    )
    assert not jnp.allclose(
        robot.actuation_matrix(q),
        robot.actuation_matrix(jnp.zeros_like(q)),
        rtol=1e-7,
        atol=1e-12,
    )


def test_isupport_pressure_area_validation_and_update_semantics():
    params = make_isupport_params()
    structure = make_pneumatic_isupport_structure(num_gauss_points=1)
    robot = ISupport(params=params, structure=structure)

    updated_derived = robot.update_params(
        chamber_outer_radius=jnp.array([0.005]),
        chamber_distance=jnp.array([0.012]),
    )
    assert updated_derived.params.chamber_effective_pressure_area is None
    updated_actuator = updated_derived.actuators[0]
    updated_coordinate_scale = updated_actuator.params.transmission.coordinate_scale
    assert jnp.allclose(
        updated_coordinate_scale,
        jnp.repeat(
            jnp.pi * (0.005**2 - params.chamber_inner_radius**2),
            updated_derived.num_chambers_per_segment,
        ),
    )
    routing_params = updated_derived.actuators[0].transmission.routing.params
    assert jnp.allclose(routing_params.intercept[0, 1], 0.012)

    explicit_area = jnp.array([7.5e-5])
    explicit_robot = ISupport(
        params.replace(chamber_effective_pressure_area=explicit_area),
        structure=structure,
    )
    updated_explicit = explicit_robot.update_params(
        chamber_outer_radius=jnp.array([0.005])
    )
    assert jnp.allclose(
        updated_explicit.actuators[0].params.transmission.coordinate_scale,
        jnp.repeat(explicit_area, updated_explicit.num_chambers_per_segment),
    )

    with pytest.raises(ValueError, match="chamber radii"):
        params.replace(chamber_outer_radius=params.chamber_inner_radius).validate()
    with pytest.raises(ValueError, match="chamber_effective_pressure_area"):
        params.replace(chamber_effective_pressure_area=jnp.array([-1.0])).validate()
    with pytest.raises(ValueError, match="shape"):
        params.replace(chamber_effective_pressure_area=jnp.array([1.0, 2.0])).validate()


def test_isupport_body_update_preserves_actuator_owned_bounds():
    robot = ISupport(
        params=make_isupport_params(),
        structure=make_pneumatic_isupport_structure(num_gauss_points=1),
    )
    lower_bounds = jnp.array([1.0e3, 2.0e3, 3.0e3])
    upper_bounds = jnp.array([1.0e5, 1.1e5, 1.2e5])
    updated_actuator = robot.update_actuator_params(
        0,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    updated_body = updated_actuator.update_link_params(
        density=updated_actuator.params.link.density * 1.01
    )

    assert jnp.allclose(updated_body.actuators[0].params.lower_bounds, lower_bounds)
    assert jnp.allclose(updated_body.actuators[0].params.upper_bounds, upper_bounds)


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
    expected_segment_0 = expected_isupport_segment_actuation(
        params, 0, num_chambers=num_chambers
    )
    expected_segment_1 = expected_isupport_segment_actuation(
        params, 1, num_chambers=num_chambers
    )

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
    lengths = [0.01, 0.10, 0.02, 0.20, 0.03]
    radii = [0.011, 0.020, 0.012, 0.025, 0.013]
    densities = [1200.0, 1000.0, 1300.0, 1050.0, 1400.0]
    young = [2e9, 2e3, 2e9, 3e3, 2e9]
    shear = [8e8, 1e3, 8e8, 1.5e3, 8e8]
    link = PCS.params_from_links(
        [
            LinkSpec.circular(
                length=length,
                radius=radius,
                density=density,
                young_modulus=young_modulus,
                shear_modulus=shear_modulus,
                material_damping_coefficient=1e-3,
                reference_strain=[0, 0, 0, 1, 0, 0],
            )
            for length, radius, density, young_modulus, shear_modulus in zip(
                lengths, radii, densities, young, shear, strict=True
            )
        ]
    ).link
    return make_isupport_params(num_segments=5).replace(
        link=link,
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
    routing_params = robot.actuators[0].transmission.routing.params
    assert routing_params.start_segment_index == (1, 1, 1, 4, 4, 4)
    assert routing_params.end_segment_index == (2, 2, 2, 4, 4, 4)
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
    params = make_mixed_isupport_params()
    damping_coefficients = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    radius = params.link.cross_section.coefficients[:, 0]
    area = jnp.pi * radius**2
    transverse = jnp.pi * radius**4 / 4.0
    polar = 2.0 * transverse
    damping = params.link.length[:, None, None] * jax.vmap(jnp.diag)(
        damping_coefficients[:, None]
        * jnp.stack(
            [polar, 3 * transverse, 3 * transverse, 3 * area, area, area], axis=1
        )
    )
    params = params.replace(
        link=params.link.replace(damping=damping),
    )
    robot = ISupport(
        params,
        structure=ISupportStructure(
            num_gauss_points=1,
            pcs_segment_counts=(2, 1),
            rigid_segment_selector=(True, False, True, False, True),
        ),
    )

    rigid_area = jnp.pi * params.link.cross_section.coefficients[0, 0] ** 2
    expected_axial_damping = params.link.length[0] * 10.0 * 3.0 * rigid_area
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
    metadata = robot.actuator_input_metadata[0]
    assert metadata.kind == "pneumatic"
    assert metadata.units == ("Pa",) * robot.num_actuators
    assert len(metadata.labels) == robot.num_actuators
    num_chambers = robot.num_chambers_per_segment
    assert actuation_matrix.shape == (18, 2 * num_chambers)
    assert jnp.allclose(
        actuation_matrix[:6, :num_chambers],
        expected_isupport_segment_actuation(
            params, 0, segment_length=0.04, num_chambers=num_chambers
        ),
    )
    assert jnp.allclose(
        actuation_matrix[6:12, :num_chambers],
        expected_isupport_segment_actuation(
            params, 0, segment_length=0.06, num_chambers=num_chambers
        ),
    )
    assert jnp.allclose(
        actuation_matrix[12:, num_chambers:],
        expected_isupport_segment_actuation(
            params, 1, segment_length=0.20, num_chambers=num_chambers
        ),
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
    bad_reference = params.link.reference_strain.at[0, 0].set(0.1)
    with pytest.raises(ValueError, match="Rigid segment reference_strain"):
        ISupport(
            params.replace(link=params.link.replace(reference_strain=bad_reference)),
            structure=structure,
        )

    updated_link = params.link.replace(
        length=jnp.array([0.015, 0.12, 0.025, 0.20, 0.035]),
        cross_section=params.link.cross_section.replace(
            coefficients=jnp.array([0.015, 0.020, 0.016, 0.025, 0.017])[:, None]
        ),
    )
    updated = robot.update_params(
        link=updated_link,
        pcs_segment_lengths=jnp.array([0.05, 0.07, 0.20]),
    )
    assert updated.structure.rigid_segment_selector == structure.rigid_segment_selector
    assert jnp.allclose(updated.L, jnp.array([0.015, 0.05, 0.07, 0.025, 0.20, 0.035]))
    assert jnp.allclose(
        updated.r[jnp.array([0, 3, 5])], jnp.array([0.015, 0.016, 0.017])
    )
