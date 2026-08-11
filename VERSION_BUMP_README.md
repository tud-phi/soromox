# Version Management and Releases

This is the authoritative maintainer guide for version bumps and releases.
End-user and contributor setup belongs in `README.md` and `CONTRIBUTING.md`.

## Canonical release workflow

Releases are prepared in a branch, reviewed in a pull request, merged, and then
triggered by tagging the verified merge commit. Do not create the tag from an
unmerged release branch.

1. Start a release branch from the current default branch and ensure the
   worktree is clean.
2. Add release notes below `[Unreleased]` in
   `docs/development/changelog.md`.
3. Preview the metadata update:

   ```bash
   python bump_version.py 0.2.0 --dry-run
   ```

4. Apply a version-only update without tagging or pushing:

   ```bash
   make bump-version VERSION=0.2.0
   ```

   For semantic increments, use `make bump-patch`, `make bump-minor`, or
   `make bump-major`.

5. Review the managed-file diff and run the release checks:

   ```bash
   uv run --extra test pytest
   uv run --extra docs zensical build --clean
   uv run --extra dev check-manifest
   python -m build
   python -m twine check dist/*
   ```

6. Open and merge the release pull request after the required checks pass.
7. Update the local default branch, verify that its version matches the
   intended tag, and tag the exact merged commit:

   ```bash
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push origin v0.2.0
   ```

8. Monitor the GitHub Actions release workflow, approve the protected `pypi`
   environment if prompted, and verify the GitHub release and PyPI package.

## What the version bump manages

`bump_version.py` updates only release metadata:

- `pyproject.toml`: package version
- `CITATION.cff`: software version, release date, and PyPI artifact URL
- `docs/citation.md`: software BibTeX version, year, key, and release URL
- `docs/development/changelog.md`: dated release entry
- `uv.lock`: local package version

Generated `*.egg-info` metadata is not tracked or edited; builds regenerate it
from `pyproject.toml`. The helper stages an explicit allowlist and never runs
`git add .`, so unrelated worktree changes are not silently included.

## Dry runs and script options

Preview patch, minor, or explicit updates without modifying files:

```bash
python bump_version.py --patch --dry-run
python bump_version.py --minor --dry-run
python bump_version.py 0.2.0 --dry-run
```

The lower-level script can create a tag or push, but those flags are not part
of the canonical branch-and-PR workflow:

```bash
python bump_version.py --patch --yes
python bump_version.py --patch --yes --create-tag
python bump_version.py --patch --yes --create-tag --push
```

## Automated tag release

Pushing a `v*` tag triggers `.github/workflows/release.yml`. The workflow:

1. verifies that the tag matches `pyproject.toml`;
2. builds the wheel and source distribution once;
3. validates both distributions with Twine;
4. extracts the matching notes from `docs/development/changelog.md`;
5. creates a GitHub release and attaches the validated artifacts; and
6. publishes those same artifacts to PyPI using trusted publishing.

The `publish-to-pypi` job uses a protected GitHub environment named `pypi` and
an OpenID Connect token (`id-token: write`). No long-lived PyPI token is
required.

## Recovery-only direct publication

The `make release-*` targets combine a version bump, commit, annotated tag, and
push. They bypass the normal review boundary and are therefore reserved for a
maintainer-approved recovery situation:

```bash
make release-patch
make release-minor
make release-major
make release VERSION=0.2.0
```

Before using one, confirm that you are on the protected default branch at the
intended release commit, the worktree is clean, all release checks have passed,
and you have explicit permission to push the commit and tag. Prefer repairing
and rerunning the tag workflow over manually uploading distributions to PyPI.

## Maintainer prerequisites

- Python 3.11 or newer with the development, documentation, build, and Twine
  tools installed
- a clean Git checkout with the expected `origin` remote
- GitHub Actions enabled for `tud-phi/soromox`
- a protected `pypi` environment and PyPI trusted publisher configured for
  `.github/workflows/release.yml`
- a changelog following [Keep a Changelog](https://keepachangelog.com/)
