"""N6 module specification, both mountings, tendon geometry and the chord string load."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples.simulation.gvs.n6_instron_model import (
    HORIZONTAL_BASE_POSE,
    N6ModuleSpec,
    RingMonitor,
    TendonBendingConfig,
    TendonExitGeometry,
    TendonLoad,
    TendonPath,
    build_horizontal_n6_module,
    build_upright_n6_module,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def spec():
    return N6ModuleSpec()


@pytest.fixture(scope="module")
def upright(spec):
    _, helix, robot = build_upright_n6_module(spec)
    return helix, robot


@pytest.fixture(scope="module")
def config():
    return TendonBendingConfig()


@pytest.fixture(scope="module")
def path(spec, config):
    _, _, robot = build_horizontal_n6_module(spec, config)
    return TendonPath(robot, spec, config)


def prescribed_configurations(num_q: int, count: int = 3) -> list[np.ndarray]:
    """Bent, twisted and stretched test configurations (not solved)."""
    rng = np.random.default_rng(2024)
    scale = np.r_[np.full(num_q - 2, 2.0), np.full(2, 0.05)]
    return [rng.normal(size=num_q) * scale for _ in range(count)]


# --------------------------------------------------------------------------- #
# N6 specification
# --------------------------------------------------------------------------- #
def test_n6_spec_matches_requested_geometry_and_material(spec):
    assert spec.total_strands == 12 and spec.strands_per_family == 6
    assert spec.turns_per_strand == 1.0
    assert spec.strand_width == pytest.approx(0.012)
    assert spec.strand_thickness == pytest.approx(2.926792e-3)
    assert spec.inner_radius == pytest.approx(0.018)
    assert spec.outer_radius == pytest.approx(0.030)
    assert spec.centerline_radius == pytest.approx(0.024)
    assert spec.length == pytest.approx(0.108)
    assert spec.phase == 0.0 and spec.family_phase == 0.0
    assert spec.young_modulus == pytest.approx(2.48e6)
    assert spec.end_ring_heights == (0.0075, 0.0075)
    assert spec.end_ring_height == pytest.approx(0.015)
    assert spec.specimen_height == pytest.approx(0.123)
    assert spec.shear_modulus == pytest.approx(2.48e6 / (2 * 1.485))
    assert spec.gravity == 0.0


def test_n6_spec_rejects_inconsistent_inputs_and_roundtrips_json(spec):
    with pytest.raises(ValueError):
        N6ModuleSpec(inner_radius=0.018, outer_radius=0.031)
    with pytest.raises(ValueError):
        N6ModuleSpec(poisson_ratio=0.6)
    with pytest.raises(ValueError):
        N6ModuleSpec(end_ring_heights=(0.015,))
    assert N6ModuleSpec.from_json(spec.to_json()) == spec


def test_helix_params_carry_the_solid_rectangular_section(upright, spec):
    helix, _ = upright
    width, height = spec.strand_width, spec.strand_thickness
    assert float(helix.strand_area) == pytest.approx(width * height)
    assert_allclose(
        np.asarray(helix.strand_second_moments),
        [width * height**3 / 12, height * width**3 / 12],
    )
    # The retained section drives the renderer: 12 mm spans the radial axis.
    assert float(helix.strand_section.radial_extent) == pytest.approx(width / 2)
    assert float(helix.radius_base) == pytest.approx(0.024)
    assert float(helix.num_turns) == 1.0
    assert helix.helices_per_direction == 6
    assert float(helix.young_modulus) == pytest.approx(2.48e6)
    assert float(helix.shear_modulus) == pytest.approx(2.48e6 / (2 * 1.485))
    assert helix.to_backbone().admissible_strains[:4] == (True, True, True, True)


# --------------------------------------------------------------------------- #
# Mountings
# --------------------------------------------------------------------------- #
def test_upright_module_is_base_fixed_and_gravity_free(upright, spec):
    _, robot = upright
    q0 = jnp.zeros(robot.num_coordinates)
    tips = np.asarray(robot.forward_kinematics_tips(q0))
    assert_allclose(tips[-1, :3, 3], [0.0, 0.0, spec.length], atol=1e-12)
    # The rigid end rings never enter the deformable backbone length.
    assert float(robot.segment_length[0]) == pytest.approx(0.108)
    assert_allclose(np.asarray(robot.params.gravity), 0.0)
    assert robot.structure.segments[0].joint.type == "fixed"


def test_horizontal_module_is_clamped_with_material_z_up(path):
    robot = path.robot
    assert robot.num_segments == 1
    assert robot.structure.segments[0].joint.type == "fixed"
    assert_allclose(np.asarray(robot.fixed_base_pose), HORIZONTAL_BASE_POSE)
    q0 = path.reference_configuration()
    tip = np.asarray(path.tip_pose(q0))
    assert_allclose(tip[:3, :3], np.eye(3), atol=1e-12)
    assert_allclose(tip[:3, 3], [0.108, 0.0, 0.0], atol=1e-12)
    # The clamped hole sits at the base frame whatever the configuration.
    hole = np.asarray(path.clamped_hole_opening())
    for q in prescribed_configurations(robot.num_coordinates):
        pose = np.asarray(robot.forward_kinematics(jnp.asarray(q), jnp.asarray(0.0)))
        assert_allclose(pose, np.asarray(path.base_pose()), atol=1e-12)
        assert_allclose(pose[:3, 3] + pose[:3, :3] @ path.config.hole_offset_local(), hole, atol=1e-12)
    # The free end carries no constraint: the tip Jacobian has full rank.
    assert np.linalg.matrix_rank(np.asarray(robot.jacobian_tips(q0)[-1])) == 6


def test_gravity_setting_builds_cross_axis_self_weight(spec):
    _, _, robot = build_horizontal_n6_module(spec, TendonBendingConfig(gravity=9.80665))
    assert_allclose(np.asarray(robot.params.gravity), [0.0, 0.0, -9.80665])
    q0 = jnp.zeros(robot.num_coordinates)
    assert float(jnp.linalg.norm(robot.gravitational_force(q0))) > 0.0
    _, _, no_gravity = build_horizontal_n6_module(spec, TendonBendingConfig())
    assert float(jnp.linalg.norm(no_gravity.gravitational_force(q0))) == 0.0


# --------------------------------------------------------------------------- #
# Tendon configuration and string path
# --------------------------------------------------------------------------- #
def test_tendon_offset_follows_from_ring_and_hole_geometry(config):
    assert config.ring_outer_radius == pytest.approx(0.040)
    assert config.hole_diameter == pytest.approx(3.2e-3)
    assert config.ring_edge_gap == pytest.approx(3.4e-3)
    assert config.tendon_offset == pytest.approx(0.040 - 0.0034 - 0.0016)
    assert config.tendon_azimuth == pytest.approx(math.pi / 2)
    assert_allclose(config.hole_offset_local(), [0.0, 0.0, 0.035], atol=1e-15)
    assert_allclose(TendonBendingConfig(tendon_azimuth=0.0).hole_offset_local(), [0.0, 0.035, 0.0], atol=1e-15)
    with pytest.raises(ValueError, match="tendon_offset must equal"):
        TendonBendingConfig(tendon_offset=0.040)
    with pytest.raises(ValueError):
        TendonExitGeometry(direction=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        TendonExitGeometry(travel_ratio=0.0)


def test_config_json_round_trip(config):
    assert TendonBendingConfig.from_json(config.to_json()) == config
    custom = TendonBendingConfig(
        exit=TendonExitGeometry(direction=(0.0, 0.0, 2.0), run_length=0.2, travel_ratio=0.9)
    )
    assert custom.exit.direction == (0.0, 0.0, 1.0)
    assert TendonBendingConfig.from_json(custom.to_json()) == custom


def test_tendon_runs_from_free_ring_through_clamped_ring_to_the_instron(path):
    q0 = path.reference_configuration()
    clamped = np.asarray(path.clamped_hole_opening())
    free = np.asarray(path.free_hole_opening(q0))
    assert_allclose(clamped, [0.0, 0.0, 0.035], atol=1e-12)
    assert_allclose(free, [0.108, 0.0, 0.035], atol=1e-12)  # holes aligned, chord parallel to the axis
    nodes = np.asarray(path.nodes(q0))
    assert nodes.shape == (6, 3)
    anchor, p1, p2, p3, p4, grip = nodes
    assert_allclose(anchor, [0.108 + 0.0075, 0.0, 0.035], atol=1e-12)  # knot on the outer face
    assert p1[0] == pytest.approx(0.108) and p2[0] == pytest.approx(0.0)
    assert p3[0] == pytest.approx(-0.0075)
    assert_allclose(p4, p3, atol=1e-12)  # redirect at the hole edge by default
    assert_allclose(grip - p4, [0.0, 0.0, path.config.exit.run_length], atol=1e-12)
    assert np.all(np.diff(nodes[:4, 0]) < 0)  # x decreases from the anchor to the exit


def test_reference_length_and_shortening_are_consistent(path):
    q0 = path.reference_configuration()
    assert float(path.internal_chord(q0)) == pytest.approx(0.108)
    assert path.constant_length() == pytest.approx(0.0075 + 0.0075 + 0.120)
    assert path.reference_length() == pytest.approx(0.108 + path.constant_length())
    assert float(path.shortening(q0)) == pytest.approx(0.0, abs=1e-15)
    lattice_anchor = TendonPath(path.robot, path.spec, TendonBendingConfig(free_anchor_face="lattice"))
    assert lattice_anchor.constant_length() == pytest.approx(0.0075 + 0.120)
    # Bending about the material y axis lifts the free end and shortens the top-side string.
    e = np.zeros(path.num_q)
    e[3] = -0.5
    pose = path.free_end_pose(jnp.asarray(e))
    assert pose["displacement"][2] > 0.0 and pose["rotation_angle"] > 0.0
    assert float(path.shortening(jnp.asarray(e))) > 0.0


# --------------------------------------------------------------------------- #
# Chord string load and ring monitor
# --------------------------------------------------------------------------- #
def test_string_wrench_follows_the_chord_with_the_hole_moment_arm(path):
    load = TendonLoad(path)
    q0 = path.reference_configuration()
    w = np.asarray(load.wrench(q0, jnp.asarray(2.0)))
    # Force along -x (toward the clamped guide hole), moment about -y from the
    # +z hole offset: (0, 0, r) x (-T, 0, 0) = (0, -r T, 0).
    assert_allclose(w[3:], [-2.0, 0.0, 0.0], atol=1e-12)
    assert_allclose(w[:3], [0.0, -2.0 * path.config.tendon_offset, 0.0], atol=1e-12)
    assert_allclose(np.asarray(load.moment_arm(q0)), [0.0, 0.0, path.config.tendon_offset], atol=1e-12)
    for q in prescribed_configurations(path.num_q):
        q = jnp.asarray(q)
        direction = np.asarray(load.direction(q))
        chord = np.asarray(path.clamped_hole_opening() - path.free_hole_opening(q))
        assert_allclose(direction, chord / np.linalg.norm(chord), atol=1e-12)
        assert_allclose(np.asarray(load.wrench(q, jnp.asarray(1.0)))[3:], direction, atol=1e-12)


def test_ring_monitor_reports_chord_and_collision(path):
    monitor = RingMonitor(path)
    reference = monitor.evaluate(path.reference_configuration())
    assert reference["internal_chord"] == pytest.approx(0.108)
    assert reference["ring_collision"] is False
