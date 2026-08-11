import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

MODULE_DIR = (
    Path(__file__).resolve().parents[3] / "paper_results" / "secVf_parallel_rl" / "code"
)
sys.path.insert(0, str(MODULE_DIR))

import render_rl_video  # noqa: E402


def test_show_trajectory_defaults_to_enabled():
    args = render_rl_video.parse_args([])

    assert args.show_trajectory is True
    assert args.max_envs is None
    assert args.output == (render_rl_video.OUTPUT_DIR / "rl_rollout_trained_1_env.mp4")
    assert args.gif_output == (
        render_rl_video.OUTPUT_DIR / "rl_rollout_trained_1_env.gif"
    )


def test_default_outputs_derive_from_data_filename_in_case_outputs():
    args = render_rl_video.parse_args(["--data", "/tmp/custom_parallel_rollout.npz"])

    assert args.output == render_rl_video.OUTPUT_DIR / "custom_parallel_rollout.mp4"
    assert args.gif_output == render_rl_video.OUTPUT_DIR / "custom_parallel_rollout.gif"


def test_show_trajectory_can_be_disabled():
    args = render_rl_video.parse_args(["--no-show-trajectory"])

    assert args.show_trajectory is False


def test_load_rollout_supports_first_class_unbatched_single_arm(tmp_path):
    path = tmp_path / "single.npz"
    np.savez_compressed(
        path,
        ts=np.arange(3, dtype=np.float64),
        q_ts=np.zeros((3, 6), dtype=np.float64),
        ball_ts=np.ones((3, 3), dtype=np.float64),
    )

    rollout = render_rl_video.load_rollout(path)

    assert rollout.q_ts.shape == (1, 3, 6)
    assert rollout.ball_ts.shape == (1, 3, 3)
    assert rollout.arm_length == render_rl_video.DEFAULT_ARM_LENGTH
    assert rollout.arm_radius == render_rl_video.DEFAULT_ARM_RADIUS


def test_load_rollout_selects_first_environments_and_reads_geometry(tmp_path):
    path = tmp_path / "parallel.npz"
    q_ts = np.arange(4 * 3 * 6, dtype=np.float64).reshape(4, 3, 6)
    ball_ts = np.arange(4 * 3 * 3, dtype=np.float64).reshape(4, 3, 3)
    np.savez_compressed(
        path,
        ts=np.arange(4, dtype=np.float64),
        q_ts=q_ts,
        ball_ts=ball_ts,
        arm_length=0.31,
        arm_radius=0.021,
        policy="random",
    )

    rollout = render_rl_video.load_rollout(path, max_envs=2)

    assert rollout.q_ts.shape == (2, 4, 6)
    assert rollout.ball_ts.shape == (2, 4, 3)
    assert_allclose(rollout.q_ts, np.transpose(q_ts[:, :2], (1, 0, 2)))
    assert rollout.arm_length == pytest.approx(0.31)
    assert rollout.arm_radius == pytest.approx(0.021)
    assert rollout.policy == "random"


@pytest.mark.parametrize(
    ("q_shape", "ball_shape", "match"),
    [
        ((2, 6), (3, 3), "time dimension"),
        ((3, 2, 6), (3, 3, 3), "time/environment dimensions"),
        ((3, 2, 6), (3, 2, 2), "positions must have size 3"),
    ],
)
def test_load_rollout_rejects_mismatched_shapes(tmp_path, q_shape, ball_shape, match):
    path = tmp_path / "invalid.npz"
    np.savez_compressed(
        path,
        ts=np.arange(3, dtype=np.float64),
        q_ts=np.zeros(q_shape, dtype=np.float64),
        ball_ts=np.zeros(ball_shape, dtype=np.float64),
    )

    with pytest.raises(ValueError, match=match):
        render_rl_video.load_rollout(path)


def test_grid_dimensions_and_offsets_are_centered():
    rows, cols = render_rl_video.resolve_grid_dimensions(4, None, None)
    offsets = render_rl_video.make_grid_offsets(4, rows, cols, 0.2)

    assert (rows, cols) == (2, 2)
    assert_allclose(
        offsets,
        np.array(
            [
                [-0.1, -0.1, 0.0],
                [0.1, -0.1, 0.0],
                [-0.1, 0.1, 0.0],
                [0.1, 0.1, 0.0],
            ]
        ),
    )


def test_grid_dimensions_require_complete_covering_override():
    with pytest.raises(ValueError, match="provided together"):
        render_rl_video.resolve_grid_dimensions(4, 2, None)
    with pytest.raises(ValueError, match="must cover"):
        render_rl_video.resolve_grid_dimensions(5, 2, 2)


def test_camera_uses_fixed_paper_view_for_one_arm_and_auto_fit_for_grid():
    single = render_rl_video.make_render_camera_config(
        num_envs=1,
        arm_length=0.25,
        fov=60.0,
        up=(0.0, 0.0, 1.0),
        distance_factor=None,
        position_offset=None,
    )
    grid = render_rl_video.make_render_camera_config(
        num_envs=4,
        arm_length=0.25,
        fov=60.0,
        up=(0.0, 0.0, 1.0),
        distance_factor=None,
        position_offset=None,
    )

    assert single.position is not None
    assert single.look_at is not None
    assert grid.position is None
    assert grid.look_at is None
    assert grid.distance_factor == render_rl_video.GRID_CAMERA_DISTANCE_FACTOR
    assert grid.position_offset == render_rl_video.GRID_CAMERA_POSITION_OFFSET
    robot = render_rl_video.build_rl_robot()
    camera_pos, look_at = grid.compute_auto_position(
        np.array([1.0, 2.0, 3.0]),
        0.5,
        reference_transform=np.asarray(robot.base_transform),
    )
    assert_allclose(look_at, (1.0, 2.0, 3.0))
    assert_allclose(camera_pos, (3.2, -0.2, 5.475))
    world_offset = camera_pos - look_at
    assert world_offset[0] == pytest.approx(-world_offset[1])


def test_manual_auto_camera_override_applies_to_single_arm():
    camera = render_rl_video.make_render_camera_config(
        num_envs=1,
        arm_length=0.25,
        fov=55.0,
        up=(0.0, 0.0, 1.0),
        distance_factor=4.0,
        position_offset=(1.0, 0.0, 0.5),
    )

    assert camera.position is None
    assert camera.look_at is None
    assert camera.distance_factor == 4.0
    assert camera.position_offset == (1.0, 0.0, 0.5)


def test_shared_renderer_vectorizes_tendon_geometry_over_envs_and_time():
    robot = render_rl_video.build_rl_robot()
    renderer = render_rl_video.HeadlessRLVideoRenderer(
        robot,
        width=64,
        height=64,
        num_points=5,
        sphere_resolution=3,
    )
    q_ts = np.zeros((2, 3, 6), dtype=np.float64)
    q_ts[1, 2, :3] = (0.1, -0.2, 0.05)
    offsets = np.array([[0.0, 0.0, 0.0], [0.3, -0.2, 0.0]])
    actuator_inputs = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

    (trajectory_layer,) = renderer.compute_actuator_visual_layers_trajectory(
        q_ts,
        offsets,
        actuator_inputs=actuator_inputs,
    )
    (reference_layer,) = renderer.compute_actuator_visual_layers(
        q_ts[1, 2], actuator_inputs=actuator_inputs[1, 2]
    )

    assert trajectory_layer.points.shape == (2, 3, 4, 5, 3)
    assert_allclose(
        trajectory_layer.points[1, 2],
        reference_layer.points + offsets[1, None, None, :],
    )
    assert_allclose(trajectory_layer.scalar_fields["input"], actuator_inputs)

    (single_layer,) = renderer.compute_actuator_visual_layers_trajectory(
        q_ts[:1],
        offsets[:1],
        actuator_inputs=actuator_inputs[0],
    )
    assert_allclose(single_layer.scalar_fields["input"], actuator_inputs[:1])


def test_output_paths_and_preflight_cover_mp4_and_gif(tmp_path):
    output, gif_output = render_rl_video.resolve_output_paths(
        tmp_path / "nested" / "rollout.mp4",
        None,
    )
    render_rl_video.preflight_outputs((output, gif_output), force=False)

    assert output.name == "rollout.mp4"
    assert gif_output.name == "rollout.gif"
    assert output.parent.is_dir()

    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="--force"):
        render_rl_video.preflight_outputs((output, gif_output), force=False)


def test_gif_encoder_constructs_palette_optimized_ffmpeg_command(
    monkeypatch,
    tmp_path,
):
    calls = []
    monkeypatch.setattr(
        render_rl_video.subprocess,
        "run",
        lambda command, check: calls.append((command, check)),
    )

    render_rl_video.encode_gif_from_mp4(
        mp4_path=tmp_path / "rollout.mp4",
        gif_path=tmp_path / "rollout.gif",
        fps=8,
        width=640,
    )

    command, check = calls[0]
    assert check is True
    assert command[:4] == ["ffmpeg", "-y", "-i", str(tmp_path / "rollout.mp4")]
    assert "fps=8" in command[5]
    assert "scale=640:-1" in command[5]
    assert command[-1] == str(tmp_path / "rollout.gif")
