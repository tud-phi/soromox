#!/usr/bin/env python3
"""Synchronize SoRoMoX release metadata and optionally create a release tag.

The script updates the tracked sources of release metadata:

- ``pyproject.toml``
- ``CITATION.cff``
- ``docs/citation.md``
- ``docs/development/changelog.md``
- ``uv.lock``

Generated ``*.egg-info`` metadata is deliberately excluded. Build tools
regenerate it from ``pyproject.toml``.
"""

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

RELEASE_FILES = (
    Path("pyproject.toml"),
    Path("CITATION.cff"),
    Path("docs/citation.md"),
    Path("docs/development/changelog.md"),
    Path("uv.lock"),
)


def parse_version(version_string: str) -> tuple[int, int, int]:
    """Parse a semantic version into major, minor, and patch components."""
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:-.*)?", version_string)
    if not match:
        raise ValueError(f"Invalid version format: {version_string}")
    return tuple(int(component) for component in match.groups())


def format_version(major: int, minor: int, patch: int) -> str:
    """Format semantic-version components."""
    return f"{major}.{minor}.{patch}"


def get_current_version(file_path: Path) -> str:
    """Read the project version from ``pyproject.toml``."""
    match = re.search(
        r'^version = "([^"]+)"', file_path.read_text(), flags=re.MULTILINE
    )
    if not match:
        raise ValueError(f"Could not find version in {file_path}")
    return match.group(1)


def increment_version(current_version: str, bump_type: str) -> str:
    """Increment a semantic version."""
    major, minor, patch = parse_version(current_version)
    if bump_type == "major":
        return format_version(major + 1, 0, 0)
    if bump_type == "minor":
        return format_version(major, minor + 1, 0)
    if bump_type == "patch":
        return format_version(major, minor, patch + 1)
    raise ValueError(f"Invalid bump type: {bump_type}")


def replace_required(
    content: str, pattern: str, replacement: str, *, description: str
) -> str:
    """Replace one required metadata value."""
    updated, count = re.subn(pattern, replacement, content, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Could not update {description}")
    return updated


def update_pyproject_toml(file_path: Path, new_version: str) -> None:
    """Update the canonical package version."""
    content = replace_required(
        file_path.read_text(),
        r'^version = "[^"]+"',
        f'version = "{new_version}"',
        description=f"version in {file_path}",
    )
    file_path.write_text(content)


def update_citation_cff(file_path: Path, new_version: str, release_date: date) -> None:
    """Update the CFF software version, release date, and artifact URL."""
    content = file_path.read_text()
    content = replace_required(
        content,
        r'^version: "[^"]+"',
        f'version: "{new_version}"',
        description=f"version in {file_path}",
    )
    content = replace_required(
        content,
        r'^date-released: "[^"]+"',
        f'date-released: "{release_date.isoformat()}"',
        description=f"release date in {file_path}",
    )
    content = replace_required(
        content,
        r'^repository-artifact: "[^"]+"',
        f'repository-artifact: "https://pypi.org/project/soromox/{new_version}/"',
        description=f"artifact URL in {file_path}",
    )
    file_path.write_text(content)


def update_software_bibtex_citation(
    file_path: Path, new_version: str, release_date: date
) -> None:
    """Update the software BibTeX block embedded in Markdown."""
    content = file_path.read_text()
    block_match = re.search(
        r"@software\{soromox_v\d+_\d+_\d+,.*?\n\s*\}",
        content,
        flags=re.DOTALL,
    )
    if not block_match:
        raise ValueError(f"Could not find software citation in {file_path}")

    block = block_match.group(0)
    citation_version = new_version.replace(".", "_")
    block = re.sub(
        r"@software\{soromox_v\d+_\d+_\d+,",
        f"@software{{soromox_v{citation_version},",
        block,
        count=1,
    )
    block = replace_required(
        block,
        r"^(\s*year = )\{\d{4}\},$",
        rf"\g<1>{{{release_date.year}}},",
        description=f"citation year in {file_path}",
    )
    block = replace_required(
        block,
        r"^(\s*version = )\{[^}]+\},$",
        rf"\g<1>{{{new_version}}},",
        description=f"citation version in {file_path}",
    )
    block = replace_required(
        block,
        r"^(\s*url = )\{https://github\.com/tud-phi/soromox/releases/tag/v[^}]+\},$",
        rf"\g<1>{{https://github.com/tud-phi/soromox/releases/tag/v{new_version}}},",
        description=f"citation release URL in {file_path}",
    )
    file_path.write_text(
        content[: block_match.start()] + block + content[block_match.end() :]
    )


def update_changelog(file_path: Path, new_version: str, release_date: date) -> None:
    """Move unreleased notes into a new entry while retaining release history."""
    content = file_path.read_text()
    if re.search(rf"^## \[{re.escape(new_version)}\]", content, flags=re.MULTILINE):
        raise ValueError(f"Changelog already contains version {new_version}")

    match = re.search(
        r"^## \[Unreleased\]\s*\n(?P<body>.*?)(?=^## \[)",
        content,
        flags=re.DOTALL | re.MULTILINE,
    )
    if not match:
        raise ValueError(
            f"Could not find a bounded [Unreleased] section in {file_path}"
        )

    notes = match.group("body").strip()
    if not re.search(r"(?m)^- ", notes):
        notes = f"### Changed\n\n- Release version {new_version}."

    replacement = (
        "## [Unreleased]\n\n"
        "### Added\n\n"
        "### Changed\n\n"
        "### Fixed\n\n"
        f"## [{new_version}] - {release_date.isoformat()}\n\n"
        f"{notes}\n\n"
    )
    file_path.write_text(
        content[: match.start()] + replacement + content[match.end() :]
    )


def update_uv_lock(file_path: Path, new_version: str) -> None:
    """Update only the local SoRoMoX package entry in ``uv.lock``."""
    content = file_path.read_text()
    content, count = re.subn(
        r'(\[\[package\]\]\nname = "soromox"\nversion = ")[^"]+(")',
        rf"\g<1>{new_version}\g<2>",
        content,
        count=1,
    )
    if count != 1:
        raise ValueError(f"Could not find the local soromox entry in {file_path}")
    file_path.write_text(content)


def stage_commit_tag_and_push(
    root_dir: Path,
    new_version: str,
    *,
    create_tag: bool,
    push: bool,
) -> None:
    """Commit only managed release files, then optionally tag and push."""
    subprocess.run(
        ["git", "add", "--", *(str(path) for path in RELEASE_FILES)],
        check=True,
        cwd=root_dir,
    )
    subprocess.run(
        ["git", "commit", "-m", f"Release SoRoMoX {new_version}"],
        check=True,
        cwd=root_dir,
    )

    tag_name = f"v{new_version}"
    if create_tag:
        subprocess.run(
            ["git", "tag", "-a", tag_name, "-m", f"SoRoMoX {new_version}"],
            check=True,
            cwd=root_dir,
        )
    if push:
        subprocess.run(["git", "push"], check=True, cwd=root_dir)
        if create_tag:
            subprocess.run(
                ["git", "push", "origin", tag_name], check=True, cwd=root_dir
            )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    version_group = parser.add_mutually_exclusive_group(required=True)
    version_group.add_argument("version", nargs="?", help="New version, e.g. 0.2.0")
    version_group.add_argument("--major", action="store_true")
    version_group.add_argument("--minor", action="store_true")
    version_group.add_argument("--patch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", "-y", action="store_true")
    parser.add_argument("--create-tag", action="store_true")
    parser.add_argument("--push", action="store_true")
    return parser


def main() -> None:
    """Run the version update."""
    args = build_parser().parse_args()
    root_dir = Path(__file__).resolve().parent
    paths = {path: root_dir / path for path in RELEASE_FILES}

    try:
        current_version = get_current_version(paths[Path("pyproject.toml")])
        if args.version:
            parse_version(args.version)
            new_version = args.version
        else:
            bump_type = "major" if args.major else "minor" if args.minor else "patch"
            new_version = increment_version(current_version, bump_type)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    release_date = date.today()
    print(f"Current version: {current_version}")
    print(f"New version: {new_version}")
    print(f"Release date: {release_date.isoformat()}")

    if current_version == new_version:
        print("Error: new version matches the current version", file=sys.stderr)
        raise SystemExit(1)

    if args.dry_run:
        print("Dry run; would update:")
        for relative_path in RELEASE_FILES:
            print(f"  - {relative_path}")
        return

    if not args.yes:
        response = input(
            f"Update version from {current_version} to {new_version}? [y/N]: "
        )
        if response.lower() not in {"y", "yes"}:
            print("Aborted.")
            return

    try:
        update_pyproject_toml(paths[Path("pyproject.toml")], new_version)
        update_citation_cff(paths[Path("CITATION.cff")], new_version, release_date)
        update_software_bibtex_citation(
            paths[Path("docs/citation.md")], new_version, release_date
        )
        update_changelog(
            paths[Path("docs/development/changelog.md")],
            new_version,
            release_date,
        )
        update_uv_lock(paths[Path("uv.lock")], new_version)

        if args.create_tag or args.push:
            stage_commit_tag_and_push(
                root_dir,
                new_version,
                create_tag=args.create_tag,
                push=args.push,
            )
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"Release update failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(f"Updated release metadata to {new_version}.")
    if not (args.create_tag and args.push):
        print("Review and commit the scoped changes before creating the release tag.")


if __name__ == "__main__":
    main()
