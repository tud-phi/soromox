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
    assert robot.num_coordinates == 36
    assert robot.num_segments == 8
    assert robot.num_actuators == 9
    q = jnp.zeros(robot.num_coordinates)
    tips = robot.forward_kinematics_tips(q)
    # Source world positions include two connector gaps, but no extra coordinates.
    assert_allclose(
        tips[:, :3, 3],
        np.column_stack(
            (
                np.zeros(8),
                np.zeros(8),
                [0.008, 0.208, 0.2183, 0.2263, 0.4263, 0.4366, 0.4446, 0.6446],
            )
        ),
        atol=1e-12,
    )
    assert_allclose(
        robot.actuators[0].path_lengths(robot, q),
        np.repeat([0.208, 0.4263, 0.6446], 3),
        atol=1e-12,
    )
    for index, diagnostic in zip([1, 4, 7], data["compliance_diagnostics"]):
        link = data["links"][index]
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
        assert diagnostic["shear_fallback_used"]
        assert link["shear_stiffness"] == 1e6
        # The new generalized stiffness integrates EI, not inverse interface compliance.
        bending = np.log1p(c_b * length / c_a) / c_b
        assert_allclose(robot.params.link.stiffness[index, 1, 1], bending, rtol=5e-12)
        assert_allclose(
            robot.params.link.stiffness[index, 0, 0],
            length**2 / compliance[2, 2],
            rtol=1e-12,
        )
    assert_allclose(robot.params.mass_per_length[np.array([2, 5])], 0.0)
    assert_allclose(
        robot.params.tip_mass[np.array([1, 4, 7])],
        [0.04226477771961455, 0.04226477771961455, 0.01613238885980728],
    )
