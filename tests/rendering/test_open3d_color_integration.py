"""Opt-in native checks for Open3D's color-grading implementation."""

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.rendering_integration


@pytest.mark.skipif(
    os.environ.get("SOROMOX_RUN_RENDERING_INTEGRATION") != "1",
    reason="set SOROMOX_RUN_RENDERING_INTEGRATION=1 for native graphics tests",
)
def test_neutral_grading_preserves_greys_and_honors_tone_mapper_selection():
    """Check neutral highlights and distinct linear/ACES results in a fresh context.

    The background material passes through color grading, so these pixels isolate
    the output transform from lights, mesh normals and environment-map colors.
    A subprocess keeps native graphics teardown separate from other GUI tests.
    """
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import numpy as np
                import open3d as o3d

                render = o3d.visualization.rendering.OffscreenRenderer(96, 72)
                scene = render.scene
                scene.show_skybox(False)
                scene.view.set_post_processing(True)
                grading = o3d.visualization.rendering.ColorGrading
                modes = grading.ToneMapping
                assert hasattr(modes, 'PBR_NEUTRAL'), 'Update the patched Open3D build'
                results = {}
                for mode in (modes.FILMIC, modes.PBR_NEUTRAL, modes.LINEAR, modes.ACES):
                    scene.view.set_color_grading(grading(grading.Quality.ULTRA, mode))
                    values = []
                    for level in (0.18, 0.58, 1.0):
                        scene.set_background([level, level, level, 1.0])
                        pixels = np.asarray(render.render_to_image())
                        values.append(pixels[10:-10, 10:-10, :3].mean(axis=(0, 1)))
                    results[mode] = np.array(values)
                assert np.max(np.ptp(results[modes.PBR_NEUTRAL], axis=1)) < 2.0
                neutral = results[modes.FILMIC]
                assert np.max(np.ptp(neutral, axis=1)) < 2.0, neutral
                assert np.all(np.diff(neutral.mean(axis=1)) > 0), neutral
                assert np.mean(neutral[-1]) > 225, neutral
                # Previously every enum silently selected the same legacy mapper.
                difference = np.abs(results[modes.LINEAR] - results[modes.ACES])
                assert np.max(difference) > 10, difference
                # Technical scenes use linear mapping to retain a white background.
                white = results[modes.LINEAR][-1]
                assert np.min(white) >= 252, white
                assert np.ptp(white) < 1, white
                print('Neutral grey RGB:', neutral.tolist())
                """
            ),
        ],
        check=True,
        timeout=60,
    )
