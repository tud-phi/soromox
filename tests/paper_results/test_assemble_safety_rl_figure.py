import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "paper_results"))
import assemble_safety_rl_figure as assembly  # noqa: E402


def test_shared_crop_covers_both_extreme_poses_and_mount_margin():
    frames = [np.full((200, 150, 3), 150, dtype=np.uint8) for _ in range(2)]
    frames[0][30:100, 20:40] = [240, 50, 30]
    frames[1][90:160, 100:130] = [20, 150, 40]
    left, top, right, bottom = assembly.shared_crop(frames)
    assert left <= 20 and top < 30
    assert right >= 130 and bottom > 160
    assert 0 <= left < right <= 150
    assert 0 <= top < bottom <= 200


def test_shared_crop_rejects_missing_geometry_or_mismatched_resolution():
    blank = np.zeros((20, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="No colored"):
        assembly.shared_crop([blank])
    with pytest.raises(ValueError, match="matching frame"):
        assembly.shared_crop([blank, blank[:10]])


def test_existing_outputs_are_preserved_without_force(tmp_path):
    output = tmp_path / "figure"
    output.with_suffix(".pdf").write_bytes(b"existing figure")
    with pytest.raises(SystemExit):
        assembly.main(["--output-base", str(output)])
    assert output.with_suffix(".pdf").read_bytes() == b"existing figure"


def test_snapshot_timestamps_match_axis_labels_and_clear_row_titles():
    with assembly.plt.rc_context({"axes.labelsize": 8}):
        fig, _ = assembly.build_figure(
            assembly.SAFETY, assembly.RL, (0, 0.75, 2, 7.9), (0, 5, 10, 15)
        )
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            snapshot_axes = [ax for ax in fig.axes if ax.images]
            for ax in snapshot_axes:
                label = ax.texts[0]
                assert label.get_fontsize() == assembly.plt.rcParams["axes.labelsize"]
                box = label.get_window_extent(renderer)
                assert ax.get_window_extent(renderer).y0 - box.y1 >= 5 * fig.dpi / 72
                assert all(
                    not box.overlaps(title.get_window_extent(renderer))
                    for title in fig.texts
                )
        finally:
            assembly.plt.close(fig)


def test_exports_preserve_transparent_canvas_and_axes(tmp_path, monkeypatch):
    import xml.etree.ElementTree as ET

    from PIL import Image

    def build(*args):
        fig, ax = assembly.plt.subplots()
        ax.plot([0, 1], [0, 1])
        return fig, []

    monkeypatch.setattr(assembly, "build_figure", build)
    output = tmp_path / "transparent"
    assembly.main(
        [
            "--output-base",
            str(output),
            "--safety-dir",
            str(tmp_path),
            "--rl-dir",
            str(tmp_path),
        ]
    )
    pixels = np.array(Image.open(output.with_suffix(".png")))
    assert pixels.shape[2] == 4
    assert pixels[0, 0, 3] == 0
    assert (
        np.count_nonzero(pixels[:, :, 3] == 0) > pixels.shape[0] * pixels.shape[1] * 0.9
    )
    svg = ET.parse(output.with_suffix(".svg"))
    for patch_id in ("patch_1", "patch_2"):
        patch = svg.find(f".//{{*}}g[@id='{patch_id}']/{{*}}path")
        assert "fill: none" in patch.attrib["style"]
