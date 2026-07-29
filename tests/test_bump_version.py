from datetime import date

import pytest

import bump_version


def test_parser_requires_exactly_one_version_selection():
    parser = bump_version.build_parser()

    assert parser.parse_args(["0.3.0"]).version == "0.3.0"
    assert parser.parse_args(["--patch"]).patch is True
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["0.3.0", "--patch"])


def test_update_changelog_preserves_history(tmp_path):
    changelog = tmp_path / "changelog.md"
    changelog.write_text(
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "### Added\n\n"
        "- New feature.\n\n"
        "## [0.1.0] - 2025-09-01\n\n"
        "### Added\n\n"
        "- Development milestone.\n"
    )

    bump_version.update_changelog(changelog, "0.2.0", date(2026, 7, 29))

    content = changelog.read_text()
    assert "## [Unreleased]" in content
    assert "## [0.2.0] - 2026-07-29" in content
    assert "## [0.1.0] - 2025-09-01" in content
    assert content.index("## [0.2.0]") < content.index("## [0.1.0]")


def test_update_changelog_rejects_duplicate_version(tmp_path):
    changelog = tmp_path / "changelog.md"
    changelog.write_text(
        "# Changelog\n\n## [Unreleased]\n\n### Added\n\n## [0.2.0] - 2026-07-29\n"
    )

    with pytest.raises(ValueError, match="already contains"):
        bump_version.update_changelog(changelog, "0.2.0", date(2026, 7, 29))


def test_update_uv_lock_changes_only_local_package(tmp_path):
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text(
        '[[package]]\nname = "dependency"\nversion = "0.1.0"\n\n'
        '[[package]]\nname = "soromox"\nversion = "0.1.0"\n'
    )

    bump_version.update_uv_lock(lockfile, "0.2.0")

    content = lockfile.read_text()
    assert 'name = "dependency"\nversion = "0.1.0"' in content
    assert 'name = "soromox"\nversion = "0.2.0"' in content


def test_update_citations_synchronizes_version_date_and_year(tmp_path):
    cff = tmp_path / "CITATION.cff"
    cff.write_text(
        'version: "0.1.0"\n'
        'date-released: "2025-09-01"\n'
    )
    markdown = tmp_path / "README.md"
    markdown.write_text(
        "```bibtex\n"
        "@software{soromox2025,\n"
        "  year = {2025},\n"
        "  version = {0.1.0},\n"
        "}\n"
        "```\n"
    )

    release_date = date(2026, 7, 29)
    bump_version.update_citation_cff(cff, "0.2.0", release_date)
    bump_version.update_bibtex_citation(markdown, "0.2.0", release_date)

    assert 'version: "0.2.0"' in cff.read_text()
    assert 'date-released: "2026-07-29"' in cff.read_text()
    assert "@software{soromox2026," in markdown.read_text()
    assert "year = {2026}," in markdown.read_text()
    assert "version = {0.2.0}," in markdown.read_text()
