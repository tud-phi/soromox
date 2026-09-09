import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

MODULE_DIR = (
    Path(__file__).resolve().parents[3]
    / "paper_results"
    / "secVe_safety_constrained_control"
    / "code"
)
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import render_pcs_with_cf_cbf_clf_video as render_video  # noqa: E402


def test_explicit_trajectory_controls_style_and_recording_terminates(
    monkeypatch, tmp_path
):
    data_path = tmp_path / "rollout_without_cbf.npz"
    data_path.touch()
    output_path = tmp_path / "video.mp4"
    args = Namespace(
        controller="with-cbf",
        data_dir=tmp_path,
        trajectory=data_path,
        output=output_path,
        loop=False,
    )
    result = {
        "controller_key": np.asarray("without-cbf"),
        "label": np.asarray("without CBF"),
        "ts": np.array([0.0, 0.1]),
        "q_ts": np.zeros((2, 1)),
        "obs_centers": np.zeros((1, 3)),
        "obs_radii": np.array([0.1]),
        "target_center": np.ones(3),
    }
    captured = {}

    class DummyRobot:
        L_cum = np.array([1.0])
        fixed_base_pose = np.array([0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 0.0])

        def with_fixed_base_pose(self, pose):
            captured["display_base_pose"] = pose
            return self

    class DummyRenderer:
        def __init__(self, *args, **kwargs):
            captured["color_config"] = kwargs["config"].colors

        def render_sequence(self, **kwargs):
            captured["render_kwargs"] = kwargs

    monkeypatch.setattr(render_video, "parse_args", lambda: args)
    monkeypatch.setattr(render_video, "load_trajectory", lambda path: result)
    monkeypatch.setattr(
        render_video,
        "build_simulation_setup",
        lambda: SimpleNamespace(robot=DummyRobot()),
    )
    monkeypatch.setattr(render_video, "Open3DRenderer", DummyRenderer)

    render_video.main()

    expected_gradient = render_video.cmap_pre_opt_1(np.array([0.75, 1.0]))[:, :3]
    np.testing.assert_allclose(
        captured["color_config"].backbone.segment_colors,
        expected_gradient,
    )
    assert captured["render_kwargs"]["record_path"] == str(output_path)
    assert captured["render_kwargs"]["loop"] is False
    assert captured["render_kwargs"]["close_when_recording_done"] is True
    assert captured["render_kwargs"]["camera_config"].up == (0.0, 0.0, -1.0)
    # Robot and markers must receive the same rigid rotation, preserving distances.
    Rotation = render_video.Rotation
    original = Rotation.from_quat(DummyRobot.fixed_base_pose[:4])
    displayed = Rotation.from_quat(captured["display_base_pose"][:4])
    world_rotation = displayed * original.inv()
    positions = captured["render_kwargs"]["static_spheres_positions"]
    np.testing.assert_allclose(
        positions[-1], world_rotation.apply(result["target_center"])
    )
    np.testing.assert_allclose(np.linalg.norm(positions[-1] - positions[0]), np.sqrt(3))
