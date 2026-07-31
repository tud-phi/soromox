#!/usr/bin/env python3
"""
Changelog parser for extracting release notes from CHANGELOG.md.

This script extracts the changelog content for a specific version
to be used in GitHub releases.
"""

import argparse
import re
import sys
from pathlib import Path


def extract_version_changelog(changelog_path: Path, version: str) -> str | None:
    """Extract changelog content for a specific version."""
    if not changelog_path.exists():
        return None

    content = changelog_path.read_text()

    # Pattern to match version headers like ## [0.1.0] - 2025-XX-XX
    version_pattern = rf"^## \[{re.escape(version)}\].*$"
    next_version_pattern = r"^## \[.*\].*$"

    lines = content.split("\n")
    in_version_section = False
    changelog_lines = []

    for line in lines:
        if re.match(version_pattern, line):
            in_version_section = True
            continue
        elif in_version_section and re.match(next_version_pattern, line):
            # Found the next version section, stop collecting
            break
        elif in_version_section:
            changelog_lines.append(line)

    if not changelog_lines:
        return None

    # Clean up the changelog content
    changelog_content = "\n".join(changelog_lines).strip()

    # Remove empty lines at the beginning and end
    while changelog_content.startswith("\n"):
        changelog_content = changelog_content[1:]
    while changelog_content.endswith("\n\n"):
        changelog_content = changelog_content[:-1]

    return changelog_content if changelog_content else None


def main():
    parser = argparse.ArgumentParser(
        description="Extract changelog content for a specific version"
    )
    parser.add_argument(
        "version", help="Version to extract changelog for (e.g., 0.1.0)"
    )
    parser.add_argument(
        "--changelog",
        default="docs/development/changelog.md",
        help="Path to changelog file (default: docs/development/changelog.md)",
    )
    parser.add_argument(
        "--output", help="Output file for changelog content (default: stdout)"
    )

    args = parser.parse_args()

    changelog_path = Path(args.changelog)
    changelog_content = extract_version_changelog(changelog_path, args.version)

    if changelog_content is None:
        print(
            f"❌ Could not find changelog for version {args.version}", file=sys.stderr
        )
        sys.exit(1)

    if args.output:
        Path(args.output).write_text(f"{changelog_content}\n")
        print(f"✓ Changelog for v{args.version} written to {args.output}")
    else:
        print(changelog_content)


if __name__ == "__main__":
    main()
