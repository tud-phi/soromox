"""Regression tests for the fast pneumatic-arm simulation example."""

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

spec = importlib.util.spec_from_file_location(
    "simulate_fast_pneumatic_arm",
    Path(__file__).resolve().parents[2]
    / "examples/simulation/gvs/simulate_fast_pneumatic_arm.py",
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_robot_matches_reported_geometry_and_actuator_layout():
    robot = example.make_robot()
    assert_allclose(robot.length, 0.45)
    assert_allclose(example.MAXIMUM_DIAMETER, 0.0625)
    assert robot.num_internal_dofs == 2
    assert robot.num_actuators == 4
    assert robot.structure.scale_rotational_basis_by_length
    assert robot.actuator_input_metadata[0].labels == (
        "local +y",
        "local +z",
        "local -y",
        "local -z",
    )
    assert robot.actuator_input_metadata[0].units == ("N",) * 4
    assert_allclose(
        robot.forward_kinematics(jnp.zeros(2), robot.length)[:3, 3],
        [0.0, 0.0, -0.45],
        atol=1e-12,
    )

    phases = jnp.linspace(0.0, 2.0 * jnp.pi, 17)[:-1]
    bend_magnitude = 1.0
    tip_positions = np.asarray(
        [
            robot.forward_kinematics(
                bend_magnitude * jnp.asarray([jnp.cos(phase), jnp.sin(phase)]),
                robot.length,
            )[:3, 3]
            for phase in phases
        ]
    )
    base_transform = np.asarray(robot.base_transform)
    base_frame_tips = (tip_positions - base_transform[:3, 3]) @ base_transform[:3, :3]
    assert_allclose(np.ptp(base_frame_tips[:, 0]), 0.0, atol=1e-12)
    assert_allclose(
        np.linalg.norm(base_frame_tips[:, 1:3], axis=1),
        np.linalg.norm(base_frame_tips[0, 1:3]),
        atol=1e-12,
    )


def test_muscle_command_is_antagonistic_nonnegative_and_phase_continuous():
    tensions = example.muscle_tensions(
        jnp.asarray(0.0), amplitude=4.0, amplitude_taper=0.0
    )
    assert_allclose(tensions, [7.5, 11.5, 7.5, 3.5])
    assert_allclose(tensions[0] + tensions[2], 2 * example.DEFAULT_PRELOAD)
    assert_allclose(tensions[1] + tensions[3], 2 * example.DEFAULT_PRELOAD)

    quarter_period = example.muscle_tensions(
        jnp.asarray(0.25),
        start_frequency=1.0,
        end_frequency=1.0,
        amplitude=4.0,
        amplitude_taper=0.0,
    )
    assert_allclose(quarter_period, [11.5, 7.5, 3.5, 7.5], atol=1e-12)

    epsilon = 1e-8
    before = example.muscle_tensions(
        jnp.asarray(-epsilon), amplitude=4.0, amplitude_taper=0.0
    )
    after = example.muscle_tensions(
        jnp.asarray(epsilon), amplitude=4.0, amplitude_taper=0.0
    )
    assert_allclose(before, after, atol=1e-6)
    assert np.all(np.asarray(before) >= 0.0)
    assert np.all(np.asarray(after) >= 0.0)


def test_default_muscle_command_accelerates_across_the_sweep():
    def phase_at(time: float) -> float:
        tensions = np.asarray(example.muscle_tensions(jnp.asarray(time)))
        return float(np.arctan2(tensions[0] - tensions[2], tensions[1] - tensions[3]))

    dt = 1e-3
    early_phase_step = np.angle(np.exp(1j * (phase_at(dt) - phase_at(0.0))))
    late_phase_step = np.angle(
        np.exp(
            1j
            * (
                phase_at(example.DEFAULT_DURATION)
                - phase_at(example.DEFAULT_DURATION - dt)
            )
        )
    )
    assert late_phase_step > 1.6 * early_phase_step


def test_short_simulation_and_tracker_trajectories_are_finite():
    robot = example.make_robot()
    trajectory = example.simulate_motion(
        robot,
        duration=0.02,
        frame_rate=100.0,
        solver_dt=1e-3,
        warmup_cycles=0.0,
    )
    q_ts = example.configurations(robot, trajectory)
    tracker_positions = example.points_along_arm(robot, q_ts, example.TRACKER_POSITIONS)
    assert trajectory.y.shape == (3, robot.state_size)
    assert trajectory.u.shape == (3, robot.num_actuators)
    assert q_ts.shape == (3, robot.num_coordinates)
    assert tracker_positions.shape == (4, 3, 3)
    assert np.isfinite(np.asarray(trajectory.y)).all()
    assert np.isfinite(np.asarray(tracker_positions)).all()


def test_renderer_config_uses_square_hanging_studio_and_frontal_camera():
    config = example.make_renderer_config("neutral")
    assert config.scene.backdrop.enabled
    assert config.scene.ground.normal == (0.0, 0.0, -1.0)
    assert config.scene.ground.height_reference == "base_mounting_face"
    assert config.camera.position[1] < 0.0
    assert config.camera.up == (0.0, 0.0, 1.0)
    assert config.output.width == example.DEFAULT_RENDER_SIZE
    assert config.output.height == example.DEFAULT_RENDER_SIZE
    assert_allclose(
        config.colors.actuators.kind_radii["muscle"],
        example.MUSCLE_RENDER_RADIUS,
    )
