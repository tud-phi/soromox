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
- `CITATION.cff` - GitHub citation file
- `docs/development/changelog.md` - Changelog with new version entry
- `src/soromox.egg-info/PKG-INFO` - Package info (if exists)

## Usage

### 🚀 Full Release Process (Recommended)

These commands will bump the version, commit changes, create a git tag, and push to trigger the automated GitHub release:

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
2. **Changelog extraction** - The release notes are automatically extracted from `docs/development/changelog.md`
3. **GitHub Release created** with:
   - Release title: "SoRoMoX v0.1.1"
   - Description from changelog
   - Installation instructions
   - Built distribution files (`.whl` and `.tar.gz`)
4. **PyPI publication** - Package is automatically published to PyPI

## 📋 What Each Script Does

### bump_version.py
1. Reads current version from `pyproject.toml`
2. Calculates new version based on increment type or uses specified version
3. Updates all version-related files
4. Updates changelog with new entry and current date
5. Optionally commits changes and creates git tags
6. Optionally pushes to origin to trigger release

### extract_changelog.py
1. Parses `docs/development/changelog.md`
2. Extracts changelog content for a specific version
3. Used by GitHub Actions to create release descriptions

### GitHub Actions Workflow
1. Triggers on version tags (`v*`)
2. Extracts version from tag
3. Gets changelog content for the version
4. Creates GitHub release with description and assets
5. Publishes to PyPI

## 🎯 Recommended Workflow

1. **Develop your features** and update the changelog in `docs/development/changelog.md`
2. **Create a release** using one of the release commands:
   - `make release-patch` for bug fixes
   - `make release-minor` for new features
   - `make release-major` for breaking changes
3. **Monitor the release** - GitHub Actions will automatically create the release and publish to PyPI
4. **Verify the release** - Check GitHub releases and PyPI to ensure everything worked

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
- PyPI account with trusted publishing configured (optional)
- Changelog following [Keep a Changelog](https://keepachangelog.com/) format

## 🔒 GitHub Secrets

For PyPI publication, you may need to set up:
- `PYPI_API_TOKEN` - PyPI API token (if not using trusted publishing)

The workflow is configured to use GitHub's trusted publishing feature, which is more secure than API tokens.
