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
        'repository-artifact: "https://pypi.org/project/soromox/0.1.0/"\n'
        "preferred-citation:\n"
        '  title: "Paper metadata must remain unchanged"\n'
        "  year: 2025\n"
    )
    markdown = tmp_path / "citation.md"
    markdown.write_text(
        "@misc{paper2025,\n"
        "  title = {Paper metadata must remain unchanged},\n"
        "  year = {2025},\n"
        "}\n\n"
        "```bibtex\n"
        "@software{soromox_v0_1_0,\n"
        "  year = {2025},\n"
        "  version = {0.1.0},\n"
        "  url = {https://github.com/tud-phi/soromox/releases/tag/v0.1.0},\n"
        "}\n"
        "```\n"
    )

    release_date = date(2026, 7, 29)
    bump_version.update_citation_cff(cff, "0.2.0", release_date)
    bump_version.update_software_bibtex_citation(markdown, "0.2.0", release_date)

    cff_content = cff.read_text()
    assert 'version: "0.2.0"' in cff_content
    assert 'date-released: "2026-07-29"' in cff_content
    assert (
        'repository-artifact: "https://pypi.org/project/soromox/0.2.0/"' in cff_content
    )
    assert 'title: "Paper metadata must remain unchanged"' in cff_content
    assert "  year: 2025" in cff_content

    markdown_content = markdown.read_text()
    assert "@software{soromox_v0_2_0," in markdown_content
    assert "year = {2026}," in markdown_content
    assert "version = {0.2.0}," in markdown_content
    assert (
        "url = {https://github.com/tud-phi/soromox/releases/tag/v0.2.0},"
        in markdown_content
    )
    assert "@misc{paper2025," in markdown_content
    assert "title = {Paper metadata must remain unchanged}," in markdown_content
