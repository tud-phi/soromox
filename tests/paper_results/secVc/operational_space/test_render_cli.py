import sys
from pathlib import Path

import pytest

MODULE_DIR = (
    Path(__file__).resolve().parents[4]
    / "paper_results"
    / "secVc_model_based_control"
    / "operational_space_impedance_control"
    / "code"
)
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import plot_control_pcs_with_impedance as plot_cli  # noqa: E402
import render_control_pcs_with_impedance as render_cli  # noqa: E402


def test_render_cli_uses_canonical_four_snapshot_times_and_outputs():
    args = render_cli.parse_args([])

    assert tuple(args.snapshot_times) == (4.5, 5.5, 6.5, 7.5)
    assert args.video_output == render_cli.DEFAULT_VIDEO_OUTPUT
    assert args.snapshot_dir == render_cli.DEFAULT_SNAPSHOT_DIR
    assert args.strip_output == render_cli.DEFAULT_STRIP_OUTPUT


def test_render_cli_requires_exactly_four_snapshot_times():
    with pytest.raises(SystemExit):
        render_cli.parse_args(["--snapshot-times", "1", "2", "3"])


def test_snapshot_outputs_are_stable_and_unique(tmp_path):
    paths = render_cli.snapshot_output_paths(
        tmp_path,
        (4.5, 5.5, 6.5, 7.5),
    )

    assert [path.name for path in paths] == [
        "snapshot_t04p50s.png",
        "snapshot_t05p50s.png",
        "snapshot_t06p50s.png",
        "snapshot_t07p50s.png",
    ]


def test_preflight_refuses_existing_outputs_without_force(tmp_path):
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="Pass --force"):
        render_cli.preflight_outputs([output], force=False)

    render_cli.preflight_outputs([output], force=True)
    assert output.read_bytes() == b"existing"


@pytest.mark.parametrize(
    "removed_option",
    ["--render", "--video-output", "--robot-opacity", "--viser-port"],
)
def test_plot_cli_rejects_removed_rendering_options(removed_option):
    with pytest.raises(SystemExit):
        plot_cli.parse_args([removed_option])
