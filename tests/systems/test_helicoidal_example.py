"""Check the native fixture's geometry, routing, and source compliance moments."""

import json

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from examples.simulation.gvs.simulate_helicoidal import FIXTURE, build_robot

jax.config.update("jax_enable_x64", True)


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
    assert_allclose(robot.params.link.sectional_inertia[np.array([2, 4])], 0.0)
    for index in (0, 2, 4):
        assert not any(robot.structure.segments[index].basis.strain_selector)
        assert_allclose(robot.params.link.stiffness[index], 0.0)
    assert_allclose(
        robot.params.link.tip_inertia[np.array([1, 3, 5]), 3, 3],
        [0.04226477771961455, 0.04226477771961455, 0.01613238885980728],
    )


def test_merged_rigid_interfaces_preserve_mass_and_ring_inertia():
    robot = build_robot()
    data = json.loads(FIXTURE.read_text())
    assert_allclose(robot.segment_lengths[np.array([2, 4])], 0.0183, atol=1e-15)
    expected_mass = sum(
        link["length"] * link["mass_per_length"] + link["tip_mass"]
        for link in data["links"]
    )
    actual_mass = (
        np.asarray(robot.params.link.sectional_inertia)[:, 3, 3]
        @ np.asarray(robot.segment_lengths)
        + np.asarray(robot.params.link.tip_inertia)[:, 3, 3].sum()
    )
    assert_allclose(actual_mass, expected_mass, rtol=1e-13)
    nodes, weights = np.polynomial.legendre.leggauss(3)
    for index, source_index in ((2, 3), (4, 6)):
        source = data["links"][source_index]
        length = source["length"]
        sectional = np.diag(
            source["rotational_inertia_per_length"] + [source["mass_per_length"]] * 3
        )
        integrated = np.zeros((6, 6))
        for node, weight in zip(nodes, weights):
            # A rigid body's point velocity is v_tip + omega cross r.
            r_x = length * (node - 1) / 2
            velocity_map = np.eye(6)
            velocity_map[4, 2] = r_x
            velocity_map[5, 1] = -r_x
            integrated += (
                length * weight / 2 * velocity_map.T @ sectional @ velocity_map
            )
        assert_allclose(
            robot.params.link.tip_inertia[index], integrated, rtol=1e-13, atol=1e-16
        )
