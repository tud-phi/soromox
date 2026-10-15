"""Check the native fixture's geometry, routing, and source compliance moments."""

import json

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from examples.simulation.gvs.simulate_helicoidal import (
    FIXTURE,
    build_helix_params,
    build_robot,
)
from soromox.systems.components.inertia import sectional_inertias

jax.config.update("jax_enable_x64", True)


def test_helix_domain_example_projects_reduced_stiffness_and_damping():
    helix = build_helix_params()
    reduced = helix.to_backbone()
    robot = build_robot(parameter_domain="helix")
    assert robot.num_coordinates == 33
    c_base = 1 / float(reduced.stiffness_base[1])
    c_tip = 1 / float(reduced.stiffness_tip[1])
    gj = float(reduced.stiffness_base[0])
    length, tau = float(helix.length), float(helix.damping_time_constant)
    for index in (1, 3, 5):
        assert (
            robot.structure.segments[index].basis.strain_selector
            == reduced.admissible_strains
        )
        stiffness = np.asarray(robot.params.link.stiffness[index])
        damping = np.asarray(robot.params.link.damping[index])
        assert_allclose(stiffness[0, 0], length * gj, rtol=1e-12)
        assert_allclose(
            stiffness[3, 3],
            length * np.log(c_tip / c_base) / (c_tip - c_base),
            rtol=1e-12,
        )
        assert_allclose(
            damping[0, 0],
            tau * np.sqrt(gj) * 2 * length / (np.sqrt(c_base) + np.sqrt(c_tip)),
            rtol=1e-12,
        )
        assert_allclose(damping[3:, 3:], tau * stiffness[3:, 3:], atol=1e-14)
    assert np.linalg.eigvalsh(robot.K).min() > 0
    assert np.linalg.eigvalsh(robot.D_active).min() > 0


def test_cylindrical_fixture_geometry_routing_and_constitutive_reduction():
    robot = build_robot()
    data = json.loads(FIXTURE.read_text())
    assert robot.num_coordinates == 33
    assert robot.num_segments == 6
    assert robot.num_actuators == 9
    q = jnp.zeros(robot.num_coordinates)
    tips = robot.forward_kinematics_tips(q)
    # Source world positions include two connector gaps, but no extra coordinates.
    assert_allclose(
        tips[:, :3, 3],
        np.column_stack(
            (
                np.zeros(6),
                np.zeros(6),
                [0.008, 0.208, 0.2263, 0.4263, 0.4446, 0.6446],
            )
        ),
        atol=1e-12,
    )
    assert_allclose(
        robot.actuators[0].path_lengths(robot, q),
        np.repeat([0.208, 0.4263, 0.6446], 3),
        atol=1e-12,
    )
    for index, source_index, diagnostic in zip(
        [1, 3, 5], [1, 4, 7], data["compliance_diagnostics"], strict=True
    ):
        link = data["links"][source_index]
        c_a, c_b = link["bending_compliance"]
        length = link["length"]
        compliance = np.array(diagnostic["interface_compliance"])
        # Reintegrate compliance moments fitted by the upstream builder.
        assert_allclose(
            c_a * length + c_b * length**2 / 2,
            0.5 * (compliance[0, 0] + compliance[1, 1]),
            rtol=1e-12,
        )
        assert_allclose(
            c_a * length**2 / 2 + c_b * length**3 / 3,
            0.5 * (abs(compliance[0, 4]) + abs(compliance[1, 3])),
            rtol=1e-12,
        )
        assert diagnostic["shear_compliance_residual"] < 0
        assert not diagnostic["shear_strain_active"]
        assert link["shear_stiffness"] is None
        assert robot.structure.segments[index].basis.strain_selector == (
            True,
            True,
            True,
            True,
            False,
            False,
        )
        # The new generalized stiffness integrates EI, not inverse interface compliance.
        bending = np.log1p(c_b * length / c_a) / c_b
        assert_allclose(robot.params.link.stiffness[index, 3, 3], bending, rtol=5e-12)
        assert_allclose(
            robot.params.link.stiffness[index, 0, 0],
            length**2 / compliance[2, 2],
            rtol=1e-12,
        )
        # Integrate tau * sqrt(GJ / C_b(s)) analytically for constant twist.
        # This distinguishes sectional scaling from sqrt of projected stiffness.
        tau = data["damping_time_constant"]
        expected_twist_damping = (
            tau
            * np.sqrt(link["torsional_stiffness"])
            * 2
            * length
            / (np.sqrt(c_a + c_b * length) + np.sqrt(c_a))
        )
        damping = np.asarray(robot.params.link.damping[index])
        assert_allclose(damping[0, 0], expected_twist_damping, rtol=5e-12)
        # Bending and extension keep the fixture's original damping exactly.
        assert_allclose(
            damping[3:, 3:],
            tau * robot.params.link.stiffness[index, 3:, 3:],
            rtol=1e-12,
            atol=1e-16,
        )
        assert np.linalg.eigvalsh(damping[:11, :11]).min() > 0
    for index in (0, 2, 4):
        assert not any(robot.structure.segments[index].basis.strain_selector)
        assert_allclose(robot.params.link.stiffness[index], 0.0)
    assert_allclose(
        robot.params.link.tip_inertia[np.array([1, 3, 5]), 3, 3],
        [0.0, 0.0, 0.01613238885980728],
    )


def test_rigid_cylinders_preserve_total_mass_and_sectional_moments():
    robot = build_robot()
    data = json.loads(FIXTURE.read_text())
    assert_allclose(robot.segment_lengths[np.array([2, 4])], 0.0183, atol=1e-15)
    expected_mass = sum(
        link["length"] * link["mass_per_length"] + link["tip_mass"]
        for link in data["links"]
    )
    sections = np.asarray(
        sectional_inertias(robot.params.link, robot.inner_mass_matrices)
    )
    actual_mass = (
        np.sum(sections[:, :, 3, 3] * np.asarray(robot.inner_integration_weights))
        + np.asarray(robot.params.link.tip_inertia)[:, 3, 3].sum()
    )
    assert_allclose(actual_mass, expected_mass, rtol=1e-13)
    for index, source_index in ((0, 0), (2, 3), (4, 6)):
        source = data["links"][source_index]
        mass = source["length"] * source["mass_per_length"]
        if index:
            mass += data["links"][source_index - 2]["tip_mass"]
        length = float(robot.segment_lengths[index])
        radius = source["radius"]
        # Disk second moments: <x^2>=0, <y^2>=<z^2>=r^2/4.
        expected = (
            mass
            / length
            * np.array([radius**2 / 2, radius**2 / 4, radius**2 / 4, 1, 1, 1])
        )
        assert_allclose(
            sections[index], np.broadcast_to(np.diag(expected), sections[index].shape)
        )
        assert not robot.params.link.sectional_inertia_mask[index]
        assert_allclose(
            robot.params.link.density[index], mass / (length * np.pi * radius**2)
        )
        assert_allclose(robot.params.link.tip_inertia[index], 0.0)


def test_terminal_cylinder_inertia_matches_volume_integration():
    robot = build_robot()
    source = json.loads(FIXTURE.read_text())["links"][-1]
    mass, radius, length = (
        source["tip_mass"],
        source["radius"],
        source["tip_body_length"],
    )
    assert_allclose(length, 0.0103)
    nodes, weights = np.polynomial.legendre.leggauss(3)
    integrated = np.zeros((6, 6))
    # Four disk points at r/sqrt(2) reproduce its area second moments exactly;
    # axial Gauss quadrature covers the cylinder extending from the backbone tip.
    for node, weight in zip(nodes, weights):
        for theta in np.arange(4) * np.pi / 2:
            x = length * (node + 1) / 2
            y = radius / np.sqrt(2) * np.cos(theta)
            z = radius / np.sqrt(2) * np.sin(theta)
            velocity_map = np.column_stack(
                (np.array([[0, z, -y], [-z, 0, x], [y, -x, 0]]), np.eye(3))
            )
            integrated += mass * weight / 8 * velocity_map.T @ velocity_map
    assert_allclose(robot.params.link.tip_inertia[-1], integrated, atol=1e-16)
