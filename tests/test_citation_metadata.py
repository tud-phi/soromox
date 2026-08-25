import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARXIV_ID = "2608.06650"
ARXIV_DOI = f"10.48550/arXiv.{ARXIV_ID}"
PAPER_TITLE = "SoRoMoX: Fast, Differentiable, and Parallelizable Soft Robot Models"


def _extract_bibtex(text: str, entry_type: str, key: str) -> str:
    match = re.search(
        rf"@{re.escape(entry_type)}\{{{re.escape(key)},.*?^\}}$",
        text,
        flags=re.DOTALL | re.MULTILINE,
    )
    assert match is not None
    return match.group(0)


def test_preprint_is_primary_citation_with_distinct_authorship():
    cff = (ROOT / "CITATION.cff").read_text()
    root_metadata, preferred = cff.split("preferred-citation:\n", maxsplit=1)

    assert root_metadata.count("family-names:") == 9
    assert preferred.count("family-names:") == 12
    assert PAPER_TITLE in preferred
    assert f'doi: "{ARXIV_DOI}"' in preferred
    assert f'url: "https://arxiv.org/abs/{ARXIV_ID}"' in preferred
    assert "status: preprint" in preferred
    for paper_only_author in ("Tarnini", "Mathew", "Renda"):
        assert paper_only_author not in root_metadata
        assert f'family-names: "{paper_only_author}"' in preferred


def test_readme_and_docs_share_primary_bibtex_and_badge():
    readme = (ROOT / "README.md").read_text()
    citation_docs = (ROOT / "docs" / "citation.md").read_text()

    readme_bibtex = _extract_bibtex(readme, "misc", "stolzle2026soromox")
    docs_bibtex = _extract_bibtex(citation_docs, "misc", "stolzle2026soromox")
    assert readme_bibtex == docs_bibtex
    assert ARXIV_ID in readme_bibtex
    assert ARXIV_DOI in readme_bibtex
    assert (
        f"[![arXiv](https://img.shields.io/badge/arXiv-{ARXIV_ID}-b31b1b.svg)]"
        f"(https://arxiv.org/abs/{ARXIV_ID})" in readme
    )


def test_software_release_and_project_urls_are_version_specific():
    cff = (ROOT / "CITATION.cff").read_text()
    citation_docs = (ROOT / "docs" / "citation.md").read_text()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    version = project["version"]
    citation_key = f"soromox_v{version.replace('.', '_')}"

    assert f'repository-artifact: "https://pypi.org/project/soromox/{version}/"' in cff
    software_bibtex = _extract_bibtex(citation_docs, "software", citation_key)
    assert f"version = {{{version}}}," in software_bibtex
    assert (
        f"url = {{https://github.com/tud-phi/soromox/releases/tag/v{version}}},"
        in software_bibtex
    )
    assert project["urls"]["Paper"] == f"https://doi.org/{ARXIV_DOI}"
    assert project["urls"]["Citation"] == "https://tud-phi.github.io/soromox/citation/"


def test_controller_citation_identifies_relevant_thesis_chapter():
    citation_docs = (ROOT / "docs" / "citation.md").read_text()

    assert "described in Chapter 2" in citation_docs


def test_hsa_citation_uses_final_publication_metadata():
    paths = (
        ROOT / "README.md",
        ROOT / "docs" / "citation.md",
        ROOT / "docs" / "api" / "systems" / "pcs" / "planar-hsa.md",
        ROOT / "src" / "soromox" / "systems" / "pcs" / "pcs.py",
    )
    contents = "\n".join(path.read_text() for path in paths)

    assert "stolzle2023experimental" not in contents
    assert "stolzle2024experimental" in contents
    assert "10.1007/978-3-031-63596-0_14" in contents
