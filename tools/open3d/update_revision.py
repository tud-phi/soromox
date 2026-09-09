"""Compare or update the Open3D main revision used by the source adapter."""

import argparse
import subprocess
from pathlib import Path

REPOSITORY = "https://github.com/isl-org/Open3D.git"
REVISION_FILE = Path(__file__).resolve().parent / "OPEN3D_REVISION"


def _main_revision() -> str:
    """Resolve the current immutable commit at upstream's main branch."""
    output = subprocess.check_output(
        ["git", "ls-remote", REPOSITORY, "refs/heads/main"], text=True
    )
    fields = output.split()
    if len(fields) != 2 or fields[1] != "refs/heads/main":
        raise RuntimeError(f"Unexpected git ls-remote output: {output!r}")
    return fields[0]


def main() -> int:
    """Compare revisions by default, or write the latest main revision."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update",
        action="store_true",
        help="replace OPEN3D_REVISION with the current upstream main commit",
    )
    args = parser.parse_args()

    current = REVISION_FILE.read_text().strip()
    latest = _main_revision()
    if args.update:
        REVISION_FILE.write_text(f"{latest}\n")
        print(f"Open3D revision: {current} -> {latest}")
        return 0
    if current != latest:
        print(
            "Open3D main has advanced: "
            f"pinned={current} latest={latest}. Run {Path(__file__).name} --update."
        )
        return 1
    print(f"Open3D revision is current: {current}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
