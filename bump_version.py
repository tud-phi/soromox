#!/usr/bin/env python3
"""
Version bump script for SoRoMoX project.

This script updates version information across all relevant files:
- pyproject.toml
- CITATION.cff
- docs/development/changelog.md
- src/soromox.egg-info/PKG-INFO (if it exists)

Usage:
    python bump_version.py <new_version>
    python bump_version.py --patch  # increment patch version (0.1.0 -> 0.1.1)
    python bump_version.py --minor  # increment minor version (0.1.0 -> 0.2.0)
    python bump_version.py --major  # increment major version (0.1.0 -> 1.0.0)

Examples:
    python bump_version.py 0.2.0
    python bump_version.py --minor
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def parse_version(version_string: str) -> Tuple[int, int, int]:
    """Parse a semantic version string into major, minor, patch components."""
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)(?:-.*)?$', version_string)
    if not match:
        raise ValueError(f"Invalid version format: {version_string}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def format_version(major: int, minor: int, patch: int) -> str:
    """Format version components into a semantic version string."""
    return f"{major}.{minor}.{patch}"


def get_current_version(file_path: Path) -> str:
    """Extract current version from pyproject.toml."""
    content = file_path.read_text()
    match = re.search(r'^version = "([^"]+)"', content, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find version in {file_path}")
    return match.group(1)


def update_pyproject_toml(file_path: Path, new_version: str) -> None:
    """Update version in pyproject.toml."""
    content = file_path.read_text()
    updated_content = re.sub(
        r'^version = "[^"]+"',
        f'version = "{new_version}"',
        content,
        flags=re.MULTILINE
    )
    file_path.write_text(updated_content)
    print(f"✓ Updated {file_path}")


def update_citation_cff(file_path: Path, new_version: str) -> None:
    """Update version in CITATION.cff."""
    if not file_path.exists():
        print(f"⚠ {file_path} does not exist, skipping")
        return
    
    content = file_path.read_text()
    updated_content = re.sub(
        r'^version: "[^"]+"',
        f'version: "{new_version}"',
        content,
        flags=re.MULTILINE
    )
    file_path.write_text(updated_content)
    print(f"✓ Updated {file_path}")


def update_changelog(file_path: Path, new_version: str, old_version: str) -> None:
    """Update changelog with new version entry."""
    if not file_path.exists():
        print(f"⚠ {file_path} does not exist, skipping")
        return
    
    content = file_path.read_text()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Replace the unreleased version with the new version and date
    if f"## [{old_version}]" in content:
        updated_content = content.replace(
            f"## [{old_version}] - TBD",
            f"## [{new_version}] - {today}"
        ).replace(
            f"## [{old_version}] - 2025-XX-XX",
            f"## [{new_version}] - {today}"
        )
    else:
        # Add new version entry at the top of the changelog
        changelog_entry = f"""
## [{new_version}] - {today}

### Added
- Version bump to {new_version}

"""
        # Find the position after the header to insert new entry
        lines = content.split('\n')
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('## [') and 'Unreleased' not in line:
                insert_pos = i
                break
        
        if insert_pos > 0:
            lines.insert(insert_pos, changelog_entry.strip())
            updated_content = '\n'.join(lines)
        else:
            updated_content = content + changelog_entry
    
    file_path.write_text(updated_content)
    print(f"✓ Updated {file_path}")


def update_pkg_info(file_path: Path, new_version: str) -> None:
    """Update version in PKG-INFO if it exists."""
    if not file_path.exists():
        print(f"⚠ {file_path} does not exist, skipping")
        return
    
    content = file_path.read_text()
    updated_content = re.sub(
        r'^Version: [^\n]+',
        f'Version: {new_version}',
        content,
        flags=re.MULTILINE
    )
    file_path.write_text(updated_content)
    print(f"✓ Updated {file_path}")


def increment_version(current_version: str, bump_type: str) -> str:
    """Increment version based on bump type."""
    major, minor, patch = parse_version(current_version)
    
    if bump_type == "major":
        return format_version(major + 1, 0, 0)
    elif bump_type == "minor":
        return format_version(major, minor + 1, 0)
    elif bump_type == "patch":
        return format_version(major, minor, patch + 1)
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Bump version across all SoRoMoX project files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else ""
    )
    
    version_group = parser.add_mutually_exclusive_group(required=True)
    version_group.add_argument("version", nargs="?", help="New version number (e.g., 0.2.0)")
    version_group.add_argument("--major", action="store_true", help="Increment major version")
    version_group.add_argument("--minor", action="store_true", help="Increment minor version")
    version_group.add_argument("--patch", action="store_true", help="Increment patch version")
    
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--create-tag", action="store_true", help="Create and push git tag after version bump")
    parser.add_argument("--push", action="store_true", help="Push changes and tags to origin after version bump")
    
    args = parser.parse_args()
    
    # Define file paths
    root_dir = Path(__file__).parent
    pyproject_path = root_dir / "pyproject.toml"
    citation_path = root_dir / "CITATION.cff"
    changelog_path = root_dir / "docs" / "development" / "changelog.md"
    pkg_info_path = root_dir / "src" / "soromox.egg-info" / "PKG-INFO"
    
    # Get current version
    try:
        current_version = get_current_version(pyproject_path)
        print(f"Current version: {current_version}")
    except Exception as e:
        print(f"Error reading current version: {e}")
        sys.exit(1)
    
    # Determine new version
    if args.version:
        new_version = args.version
        try:
            parse_version(new_version)  # Validate format
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        bump_type = "major" if args.major else "minor" if args.minor else "patch"
        try:
            new_version = increment_version(current_version, bump_type)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    print(f"New version: {new_version}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No files will be modified")
        print(f"Would update version from {current_version} to {new_version} in:")
        print(f"  - {pyproject_path}")
        print(f"  - {citation_path}")
        print(f"  - {changelog_path}")
        print(f"  - {pkg_info_path}")
        return
    
    # Confirm before making changes
    if current_version == new_version:
        print("⚠ New version is the same as current version. Proceeding anyway...")
    
    if not args.yes:
        response = input(f"\nUpdate version from {current_version} to {new_version}? [y/N]: ")
        if response.lower() not in ('y', 'yes'):
            print("Aborted.")
            sys.exit(0)
    else:
        print(f"\n✓ Auto-confirmed version update from {current_version} to {new_version}")
    
    # Update files
    print(f"\n🚀 Updating version to {new_version}...")
    
    try:
        update_pyproject_toml(pyproject_path, new_version)
        update_citation_cff(citation_path, new_version)
        update_changelog(changelog_path, new_version, current_version)
        update_pkg_info(pkg_info_path, new_version)
        
        print(f"\n✅ Successfully updated version to {new_version}")
        
        # Git operations
        if args.create_tag or args.push:
            print(f"\n� Performing git operations...")
            try:
                # Stage and commit changes
                subprocess.run(["git", "add", "."], check=True, cwd=root_dir)
                subprocess.run(
                    ["git", "commit", "-m", f"Bump version to {new_version}"], 
                    check=True, cwd=root_dir
                )
                print(f"✓ Committed version bump")
                
                if args.create_tag:
                    # Create git tag
                    subprocess.run(
                        ["git", "tag", f"v{new_version}"], 
                        check=True, cwd=root_dir
                    )
                    print(f"✓ Created git tag v{new_version}")
                
                if args.push:
                    # Push changes and tags
                    subprocess.run(["git", "push"], check=True, cwd=root_dir)
                    if args.create_tag:
                        subprocess.run(["git", "push", "--tags"], check=True, cwd=root_dir)
                    print(f"✓ Pushed changes and tags to origin")
                    
            except subprocess.CalledProcessError as e:
                print(f"❌ Git operation failed: {e}")
                print("You may need to perform git operations manually.")
                sys.exit(1)
        
        print(f"\n�📝 Next steps:")
        if not (args.create_tag and args.push):
            print("1. Review the changes with: git diff")
            if not args.create_tag:
                print("2. Commit the changes: git add . && git commit -m 'Bump version to {}'".format(new_version))
                print("3. Create a git tag: git tag v{}".format(new_version))
            if not args.push:
                print("4. Push changes: git push && git push --tags")
        print("5. GitHub Actions will automatically create a release when the tag is pushed")
        print("6. The release will include changelog content extracted automatically")
        
    except Exception as e:
        print(f"❌ Error updating files: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
