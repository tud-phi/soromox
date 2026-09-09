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
