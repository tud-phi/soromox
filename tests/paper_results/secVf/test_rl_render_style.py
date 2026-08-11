import sys
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

MODULE_DIR = (
    Path(__file__).resolve().parents[3] / "paper_results" / "secVf_parallel_rl" / "code"
)
sys.path.insert(0, str(MODULE_DIR))

import render_rl_video  # noqa: E402
import rl_render_style  # noqa: E402
import run_rl_policy  # noqa: E402
from paper_style import PAPER_COLORS  # noqa: E402


def test_run_policy_defaults_to_shared_paper_render_style():
    args = render_rl_video.parse_args([])

    assert args.width == rl_render_style.RENDER_WIDTH
    assert args.height == rl_render_style.RENDER_HEIGHT
    assert args.num_points == rl_render_style.BACKBONE_NUM_POINTS
    assert args.camera_fov == 60.0
    assert args.camera_distance_factor is None
    assert args.camera_position_offset is None


def test_run_policy_defaults_to_all_64_parallel_environments():
    args = run_rl_policy.parse_args([])

    assert args.num_envs == 64


def test_shared_camera_matches_section_vf_viewpoint():
    camera = rl_render_style.make_rl_camera_config(0.25)

    assert camera.fov == 60.0
    assert_allclose(camera.position, (-1.51471109, -0.77178385, 0.5), atol=1e-8)
    assert_allclose(camera.look_at, (0.0, 0.0, 0.0875), atol=1e-12)


def test_shared_trained_policy_and_target_colors_match_paper_palette():
    color_config = rl_render_style.make_rl_color_config("post_opt_1")

    assert rl_render_style.PAPER_COLORS is PAPER_COLORS
    assert_allclose(
        color_config.backbone.segment_colors,
        np.array([[241.0, 85.0, 46.0]]) / 255.0,
        atol=1e-12,
    )
    assert_allclose(
        rl_render_style.TARGET_COLOR,
        np.array([14.0, 173.0, 105.0]) / 255.0,
        atol=1e-12,
    )


def test_target_trail_style_supports_batched_trajectories():
    ball_positions = np.arange(2 * 5 * 3, dtype=np.float64).reshape(2, 5, 3)

    positions, radii, colors = rl_render_style.make_target_trail_spheres(ball_positions)

    assert positions.shape == (6, 3)
    assert radii.shape == (6,)
    assert colors.shape == (6, 3)
    assert_allclose(radii, rl_render_style.TARGET_TRAIL_RADIUS)
    assert_allclose(
        colors,
        np.tile(np.asarray(rl_render_style.TARGET_TRAIL_COLOR)[None, :], (6, 1)),
    )
