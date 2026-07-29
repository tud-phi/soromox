import subprocess
import sys
from pathlib import Path


def test_output_file_ends_with_newline(tmp_path):
    changelog = tmp_path / "changelog.md"
    changelog.write_text(
        "# Changelog\n\n"
        "## [0.2.0] - 2026-07-29\n\n"
        "### Fixed\n\n"
        "- Preserve the GitHub output delimiter.\n"
    )
    output = tmp_path / "release-notes.md"
    script = Path(__file__).parents[1] / "extract_changelog.py"

    subprocess.run(
        [
            sys.executable,
            script,
            "0.2.0",
            "--changelog",
            changelog,
            "--output",
            output,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.read_text() == (
        "### Fixed\n\n- Preserve the GitHub output delimiter.\n"
    )
