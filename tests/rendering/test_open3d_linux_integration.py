"""Opt-in native Open3D rendering checks for Ubuntu CI and workstations."""

import json
import os
import platform
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from jax import numpy as jnp


def _is_ubuntu_linux() -> bool:
    """Return whether this process is running on an Ubuntu Linux host."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        return platform.freedesktop_os_release().get("ID") == "ubuntu"
    except OSError:
        return False


if not _is_ubuntu_linux():
    pytest.skip("native Open3D integration is Ubuntu-only", allow_module_level=True)
if os.environ.get("SOROMOX_RUN_RENDERING_INTEGRATION") != "1":
    pytest.skip(
        "set SOROMOX_RUN_RENDERING_INTEGRATION=1 to run native rendering integration",
        allow_module_level=True,
    )

pytest.importorskip("open3d")

from soromox.rendering.open3d_render_config import Open3DRenderConfig  # noqa: E402
from soromox.rendering.open3d_renderer import Open3DRenderer  # noqa: E402
from soromox.systems.components import CrossSectionGeometry  # noqa: E402

pytestmark = pytest.mark.rendering_integration


class _IntegrationRobot:
    """Small spatial robot that exercises swept-mesh updates without downloads."""

    is_planar = False
    floating_base = False
    length = jnp.array(1.0)
    segment_length = jnp.array([1.0])
    fixed_base_pose = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    base_transform = jnp.eye(4)

    def forward_kinematics_abscissa_batched(self, q, s_points):
        """Return a gently curved, configuration-dependent centerline."""
        angles = q[0] * s_points
        cosines = jnp.cos(angles)
        sines = jnp.sin(angles)
        rotations = jnp.stack(
            (
                jnp.stack((cosines, -sines, jnp.zeros_like(angles)), axis=-1),
                jnp.stack((sines, cosines, jnp.zeros_like(angles)), axis=-1),
                jnp.stack(
                    (
                        jnp.zeros_like(angles),
                        jnp.zeros_like(angles),
                        jnp.ones_like(angles),
                    ),
                    axis=-1,
                ),
            ),
            axis=-2,
        )
        positions = jnp.stack((s_points, q[1] * s_points**2, q[2] * s_points), axis=-1)
        transforms = jnp.broadcast_to(jnp.eye(4), (s_points.size, 4, 4))
        transforms = transforms.at[:, :3, :3].set(rotations)
        return transforms.at[:, :3, 3].set(positions)

    def cross_section_geometry(self, q, s):
        """Use a constant circular cross-section."""
        del q, s
        return CrossSectionGeometry.CIRCULAR, jnp.array([0.05])


def _renderer(*, xvfb_compatible: bool = False) -> Open3DRenderer:
    """Create a small renderer that keeps native integration checks fast."""
    return Open3DRenderer(
        _IntegrationRobot(),
        width=96,
        height=72,
        num_points=24,
        cross_section_resolution=12,
        render_config=(
            Open3DRenderConfig(shadows=False, ambient_occlusion=False)
            if xvfb_compatible
            else None
        ),
    )


def _run_xvfb_test_in_fresh_process(test_name: str) -> bool:
    """Isolate Open3D's modern and legacy GUI teardown from one another."""
    child_test = os.environ.get("SOROMOX_OPEN3D_XVFB_CHILD")
    if child_test == test_name:
        return False
    env = os.environ.copy()
    env["SOROMOX_OPEN3D_XVFB_CHILD"] = test_name
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            f"{Path(__file__).resolve()}::{test_name}",
        ],
        check=True,
        env=env,
    )
    return True


def test_open3d_surfaceless_frame_and_mp4_export(tmp_path):
    """Render real RGB pixels and encode a short video without a display."""
    assert os.environ.get("EGL_PLATFORM") == "surfaceless"
    assert not os.environ.get("DISPLAY")
    renderer = _renderer()
    frame = renderer.render_frame(np.array([[0.25, 0.08, 0.04]]))
    assert frame.shape == (72, 96, 3)
    assert frame.dtype == np.uint8
    assert np.ptp(frame) > 0

    timestamps = np.array([0.0, 0.04, 0.08])
    trajectory = np.array(
        [
            [
                [0.25, 0.08, 0.04],
                [0.35, 0.12, 0.06],
                [0.20, 0.05, 0.03],
            ]
        ]
    )
    video = tmp_path / "open3d.mp4"
    renderer.render_sequence(timestamps, trajectory, record_path=str(video))
    assert video.stat().st_size > 0

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["codec_name"] == "h264"
    assert (stream["width"], stream["height"]) == (96, 72)
    assert int(stream["nb_frames"]) == 3


def test_open3d_show_opens_and_closes_real_xvfb_window(monkeypatch):
    """Exercise the public modern interactive viewer with a real X display."""
    assert os.environ.get("DISPLAY")
    assert os.environ.get("EGL_PLATFORM") != "surfaceless"
    if _run_xvfb_test_in_fresh_process(
        "test_open3d_show_opens_and_closes_real_xvfb_window"
    ):
        return
    # Xvfb exposes only Filament feature level 1 on GitHub's software GLX
    # stack; VSM and SSAO require higher feature levels than the window test.
    renderer = _renderer(xvfb_compatible=True)
    create_window = renderer._create_modern_window

    def create_and_schedule_close(*args, **kwargs):
        app, window, widget = create_window(*args, **kwargs)

        def close_window():
            time.sleep(0.25)
            app.post_to_main_thread(window, window.close)

        threading.Thread(target=close_window, daemon=True).start()
        return app, window, widget

    monkeypatch.setattr(renderer, "_create_modern_window", create_and_schedule_close)
    renderer.show(np.array([[0.25, 0.08, 0.04]]))


def test_open3d_interactive_sequence_advances_in_real_xvfb_window(monkeypatch):
    """Exercise public interactive playback and multiple real mesh updates."""
    assert os.environ.get("DISPLAY")
    assert os.environ.get("EGL_PLATFORM") != "surfaceless"
    if _run_xvfb_test_in_fresh_process(
        "test_open3d_interactive_sequence_advances_in_real_xvfb_window"
    ):
        return
    renderer = _renderer()
    create_visualizer = renderer._create_visualizer
    seen_frames: list[int] = []

    class _ClosingVisualizer:
        def __init__(self, visualizer):
            self._visualizer = visualizer
            self._poll_count = 0

        def __getattr__(self, name):
            return getattr(self._visualizer, name)

        def poll_events(self):
            self._poll_count += 1
            return self._poll_count < 80 and self._visualizer.poll_events()

    def create_closing_visualizer(window_name):
        visualizer, control = create_visualizer(window_name)
        return _ClosingVisualizer(visualizer), control

    update_scene = renderer._update_scene

    def record_update(*args, **kwargs):
        seen_frames.append(kwargs["frame_idx"])
        return update_scene(*args, **kwargs)

    monkeypatch.setattr(renderer, "_create_visualizer", create_closing_visualizer)
    monkeypatch.setattr(renderer, "_update_scene", record_update)
    timestamps = np.array([0.0, 0.01, 0.02])
    trajectory = np.array(
        [
            [
                [0.25, 0.08, 0.04],
                [0.35, 0.12, 0.06],
                [0.20, 0.05, 0.03],
            ]
        ]
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Open3D viewer assumes cross_section_geometry.*",
            category=RuntimeWarning,
        )
        renderer.render_sequence(
            timestamps,
            trajectory,
            playback_speed=10.0,
            autoplay=True,
        )
    assert {0, 1, 2}.issubset(seen_frames)
