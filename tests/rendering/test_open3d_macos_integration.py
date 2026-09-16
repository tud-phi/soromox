"""Opt-in native Metal export and window checks on macOS."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = [
    pytest.mark.rendering_integration,
    pytest.mark.skipif(
        sys.platform != "darwin"
        or os.environ.get("SOROMOX_RUN_RENDERING_INTEGRATION") != "1",
        reason="requires macOS and SOROMOX_RUN_RENDERING_INTEGRATION=1",
    ),
]
ROOT = Path(__file__).resolve().parents[2]


def test_open3d_metal_png_and_mp4_export(tmp_path):
    """Export the real tentacle gallery and verify RGB pixels and video frames."""
    video = tmp_path / "open3d.mp4"
    subprocess.run(
        [
            sys.executable,
            "examples/rendering/preset_gallery.py",
            "--backend",
            "open3d",
            "--preset",
            "neutral",
            "--count",
            "1",
            "--width",
            "320",
            "--height",
            "240",
            "--output-dir",
            str(tmp_path),
            "--video-output",
            str(video),
            "--write-manifest",
        ],
        cwd=ROOT,
        check=True,
        timeout=180,
    )
    with Image.open(tmp_path / "open3d_neutral.png") as image:
        pixels = np.asarray(image)
        assert pixels.shape == (240, 320, 3)
        assert pixels.dtype == np.uint8
        assert np.std(pixels) > 1
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
    assert (stream["width"], stream["height"]) == (320, 240)
    assert int(stream["nb_frames"]) == 60


def test_open3d_metal_show_opens_and_closes_window():
    """Open the public modern viewer and close it through the native event loop."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys
                import threading
                import time
                sys.path.insert(0, 'examples/rendering')
                from tentacle_scene import make_comparison, make_tentacle
                from soromox.rendering import Open3DRenderer

                config, q, offsets = make_comparison(
                    'neutral', count=1, width=320, height=240)
                renderer = Open3DRenderer(make_tentacle(), config=config)
                create_window = renderer._create_modern_window
                closed = threading.Event()

                def create_and_schedule_close(*args, **kwargs):
                    app, window, widget = create_window(*args, **kwargs)
                    def close_on_main_thread():
                        window.close()
                        closed.set()
                    def schedule_close():
                        time.sleep(1.0)
                        app.post_to_main_thread(window, close_on_main_thread)
                    threading.Thread(target=schedule_close, daemon=True).start()
                    return app, window, widget

                renderer._create_modern_window = create_and_schedule_close
                renderer.show(q, base_offsets=offsets, render_actuators=False)
                assert closed.is_set(), 'Viewer exited before scheduled close'
                print('Native Metal window opened and closed successfully')
                """
            ),
        ],
        cwd=ROOT,
        check=True,
        timeout=90,
    )
