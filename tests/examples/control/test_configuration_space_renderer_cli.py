import sys
from pathlib import Path

import pytest

MODULE_DIR = (
    Path(__file__).resolve().parents[3] / "examples" / "control" / "configuration_space"
)
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import render_configuration_space_comparison  # noqa: E402


@pytest.mark.parametrize("incompatible_flag", ["--no-autoplay", "--loop"])
def test_video_export_rejects_nonterminating_playback(
    monkeypatch,
    incompatible_flag,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_configuration_space_comparison.py",
            "comparison.npz",
            "--video-output",
            incompatible_flag,
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        render_configuration_space_comparison.parse_args()

    assert exc_info.value.code == 2
