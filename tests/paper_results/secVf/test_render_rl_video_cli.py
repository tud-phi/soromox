import sys
from pathlib import Path

MODULE_DIR = (
    Path(__file__).resolve().parents[3] / "paper_results" / "secVf_parallel_rl" / "code"
)
sys.path.insert(0, str(MODULE_DIR))

import render_rl_video  # noqa: E402


def test_show_trajectory_defaults_to_enabled():
    args = render_rl_video.parse_args([])

    assert args.show_trajectory is True


def test_show_trajectory_can_be_disabled():
    args = render_rl_video.parse_args(["--no-show-trajectory"])

    assert args.show_trajectory is False
