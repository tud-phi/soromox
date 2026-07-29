# Version Bump and Release System

This directory contains scripts for automated version bumping and GitHub release creation for the SoRoMoX project.

## Files Included

- **`bump_version.py`** - Main version bump script
- **`extract_changelog.py`** - Changelog parser for release notes
- **`.github/workflows/release.yml`** - GitHub Actions workflow for automated releases
- **Makefile targets** - Convenient commands for version management

## Files Updated

The version bump script updates version information in:

- `pyproject.toml` - Main project version
- `CITATION.cff` - Citation version and release date
- `README.md` and `docs/index.md` - BibTeX version, year, and citation key
- `docs/development/changelog.md` - Dated release entry
- `uv.lock` - Local package version

Generated `*.egg-info` files are intentionally excluded because build tools
regenerate them from `pyproject.toml`.

## Usage

### 🚀 Direct Release Process

These commands bump the version, commit the managed metadata, create an
annotated tag, and push it. Use them only when the release commit is already on
the protected default branch:

```bash
# Create a patch release (0.1.0 -> 0.1.1)
make release-patch

# Create a minor release (0.1.0 -> 0.2.0)
make release-minor

# Create a major release (0.1.0 -> 1.0.0)
make release-major

# Create a specific version release
make release VERSION=0.2.0
```

### 📝 Version Bump Only

If you just want to update version numbers without creating a release:

```bash
# Increment patch version
make bump-patch

# Increment minor version
make bump-minor

# Increment major version
make bump-major

# Set specific version
make bump-version VERSION=0.2.0
```

### 🔧 Advanced Script Usage

```bash
# Preview changes without making them
python bump_version.py --patch --dry-run

# Bump version with git operations
python bump_version.py --minor --yes --create-tag --push

# Skip confirmation prompt
python bump_version.py --patch --yes

# Create tag but don't push automatically
python bump_version.py --patch --yes --create-tag
```

## 🤖 Automated GitHub Release Process

When you push a version tag (e.g., `v0.1.1`), the following happens automatically:

1. **GitHub Actions triggers** on the new tag
2. **Version verification** - The tag must match the version in `pyproject.toml`
3. **Build validation** - The wheel and source distribution are built once and
   checked with Twine
4. **Changelog extraction** - Release notes are extracted from `docs/development/changelog.md`
5. **GitHub Release created** with:
   - Release title: "SoRoMoX v0.1.1"
   - Description from changelog
   - Installation instructions
   - Built distribution files (`.whl` and `.tar.gz`)
6. **PyPI publication** - The same validated files are published through the
   protected `pypi` environment and PyPI trusted publishing

## 📋 What Each Script Does

### bump_version.py
1. Reads current version from `pyproject.toml`
2. Calculates new version based on increment type or uses specified version
3. Updates all version-related files
4. Updates changelog with new entry and current date
5. Optionally commits only the managed release files and creates an annotated
   git tag
6. Optionally pushes to origin to trigger release

### extract_changelog.py
1. Parses `docs/development/changelog.md`
2. Extracts changelog content for a specific version
3. Used by GitHub Actions to create release descriptions

### GitHub Actions Workflow
1. Triggers on version tags (`v*`)
2. Extracts version from tag
3. Verifies the tag and builds the distributions once
4. Checks the distributions and extracts the changelog
5. Creates the GitHub release with the validated assets
6. Publishes the same assets to PyPI

## 🎯 Recommended Workflow

1. **Develop your features** and add notes below `[Unreleased]` in
   `docs/development/changelog.md`.
2. **Create a release branch** from the current default branch.
3. **Bump the metadata without tagging**, for example with
   `make bump-version VERSION=0.2.0`.
4. **Validate and merge a release PR** after the test and documentation
   workflows pass.
5. **Create an annotated version tag** on the verified default-branch commit
   and push only that tag.
6. **Monitor the release workflow** and approve the protected `pypi`
   environment when prompted.
7. **Verify the GitHub release, PyPI metadata and attestations, and a fresh
   installation from PyPI.**

## ✅ Features

- ✅ **Semantic versioning** support (major.minor.patch)
- ✅ **Automated changelog updates** with dates
- ✅ **Git tag creation and pushing**
- ✅ **GitHub release automation** with changelog content
- ✅ **PyPI publication** with distribution files
- ✅ **Dry run mode** to preview changes
- ✅ **Conda environment** support (`jsrm` environment)
- ✅ **Makefile integration** for easy usage
- ✅ **Release asset upload** (wheels and source distributions)

## 🛠️ Requirements

- Python 3.10+
- `jsrm` conda environment
- Git repository with proper remotes configured
- GitHub repository with Actions enabled
- PyPI account with a trusted publisher configured for:
  - Project: `soromox`
  - Owner: `tud-phi`
  - Repository: `soromox`
  - Workflow: `release.yml`
  - Environment: `pypi`
- Changelog following [Keep a Changelog](https://keepachangelog.com/) format

## 🔒 Publishing Security

The `publish-to-pypi` job uses a protected GitHub environment named `pypi`
and requests an OpenID Connect token with `id-token: write`.

No long-lived `PYPI_API_TOKEN` secret is required.

The release helper stages an explicit allowlist of metadata files. It never
runs `git add .`, so unrelated worktree files cannot enter a release commit
silently.
