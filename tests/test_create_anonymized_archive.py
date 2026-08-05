from __future__ import annotations

import binascii
import importlib.util
import io
import struct
import subprocess
import sys
import tomllib
import zipfile
import zlib
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "tools" / "create_anonymized_archive.py"
SPEC = importlib.util.spec_from_file_location("create_anonymized_archive", SCRIPT_PATH)
assert SPEC and SPEC.loader
archive_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = archive_module
SPEC.loader.exec_module(archive_module)


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True)


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _png_with_text_metadata() -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        crc = binascii.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    scanline = zlib.compress(b"\x00\x00\x00\x00\x00")
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)),
            chunk(b"caBX", b"c2pa OpenAI API"),
            chunk(b"tEXt", b"Software\x00Matplotlib v3.10.9"),
            chunk(b"IDAT", scanline),
            chunk(b"IEND", b""),
        )
    )


def _pdf_with_metadata() -> bytes:
    return b"".join(
        (
            b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n",
            b"1 0 obj\n<< /Type /Catalog >>\nendobj\n",
            b"2 0 obj\n<< /Creator (Alice Example) /CreationDate (D:20260101000000Z) >>\nendobj\n",
            b"trailer\n<< /Size 3 /Root 1 0 R /Info 2 0 R >>\n",
            b"startxref\n0\n%%EOF\n",
        )
    )


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q")
    _write(
        repo / "pyproject.toml",
        """\
[project]
name = "review-project"
version = "1.0.0"
authors = [
  {name = "Alice Example"},
  {name = "Bob Reviewer"},
]
maintainers = [
  {name = "Alice Example"},
]
[project.urls]
Source = "https://github.com/identifying-group/review-project"
""",
    )
    _write(
        repo / "CITATION.cff",
        """\
cff-version: 1.2.0
title: Review Project
authors:
  - family-names: "Example"
    given-names: "Alice"
  - family-names: "Reviewer"
    given-names: "Bob"
repository-code: "https://github.com/identifying-group/review-project"
url: "https://identifying.example/review-project"
""",
    )
    _write(
        repo / "docs" / "authors.md",
        """\
# Authors & Maintainers

## Authors

### Alice Example

Identifying Robotics Lab, Example University<br>
Somewhere

**Contact:** <alice@example.edu>

**Links:**
- [Profile](https://example.edu/people/alice)

## Acknowledgments

This work was funded by Grant 123.
""",
    )
    _write(
        repo / "README.md",
        """\
# Review Project

![Project logo](results/figure.png)

> **Note**: Review Project is the successor to the [JSRM package](https://example.invalid/jsrm).

Alice Example maintains this package at https://github.com/identifying-group/review-project.

## Citation

### Software Citation

```bibtex
@software{package, author={Example, Alice}}
```

### Related Reference

Example, A. (2024). A legitimate scholarly reference.

### Planar HSA Model Citation

Please cite the revealing conference paper.

### Model-Based Controllers Citation

Please cite the revealing PhD thesis.

## Acknowledgments

Our previous work was funded by Grant 123.
""",
    )
    _write(
        repo / "docs" / "installation.md",
        """\
# Installation

!!! warning "Migration from JSRM"

    This package succeeds an identifying predecessor.
    This whole warning must be removed.

Install the anonymous package normally.
""",
    )
    _write(
        repo / "docs" / "api" / "control" / "index.md",
        """\
# Control

## References and Citation

Please cite the revealing PhD thesis.

## API

Public controller documentation.
""",
    )
    _write(repo / "src" / "module.py", "VALUE = 1\n")
    _write(repo / "results" / "figure.png", b"\x89PNG\r\n\x1a\ncreator=Alice Example")
    _write(repo / "paper_results" / "plot.png", _png_with_text_metadata())
    _write(repo / "paper_results" / "plot.pdf", _pdf_with_metadata())
    _write(
        repo / "docs" / "assets" / "logo" / "favicon" / "favicon-32x32.png",
        _png_with_text_metadata(),
    )
    _write(
        repo / "docs" / "assets" / "logo" / "soromox_logo.png",
        _png_with_text_metadata(),
    )
    _git(repo, "add", ".")
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    _write(repo / "local-secret.txt", "must never be archived\n")
    _write(repo / "src" / "module.py", "VALUE = 'uncommitted change'\n")
    return repo


def test_archive_uses_committed_files_and_anonymizes(
    source_repo: Path, tmp_path: Path
) -> None:
    output = tmp_path / "anonymous.zip"
    result = archive_module.create_archive(source_repo, output)

    assert result.output == output
    assert result.report == tmp_path / archive_module.REPORT_PATH
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        root = archive_module.ARCHIVE_ROOT
        assert f"{root}/local-secret.txt" not in names
        assert f"{root}/results/figure.png" not in names
        assert archive.read(f"{root}/src/module.py") == b"VALUE = 1\n"

        readme = archive.read(f"{root}/README.md").decode()
        assert "Alice Example maintains" not in readme
        assert "identifying-group" not in readme
        assert "Software Citation" not in readme
        assert "Planar HSA Model Citation" not in readme
        assert "Model-Based Controllers Citation" not in readme
        assert "Acknowledgments" not in readme
        assert "Example, A. (2024)" in readme
        assert "JSRM" not in readme

        installation = archive.read(f"{root}/docs/installation.md").decode()
        assert "JSRM" not in installation
        assert "whole warning" not in installation
        assert "Install the anonymous package normally" in installation

        control_docs = archive.read(f"{root}/docs/api/control/index.md").decode()
        assert "revealing PhD thesis" not in control_docs
        assert "Public controller documentation" in control_docs

        project = tomllib.loads(archive.read(f"{root}/pyproject.toml").decode())
        assert project["project"]["authors"] == [{"name": "Anonymous"}]
        assert project["project"]["maintainers"] == [{"name": "Anonymous"}]

        author_page = archive.read(f"{root}/docs/authors.md").decode()
        assert "identities are withheld" in author_page
        assert "Alice" not in author_page
        assert "Example University" not in author_page

        cff = archive.read(f"{root}/CITATION.cff").decode()
        assert 'name: "Anonymous"' in cff
        assert "repository-code" not in cff
        assert "Alice" not in cff

        assert "figure.png" not in readme

        paper_png = archive.read(f"{root}/paper_results/plot.png")
        assert b"Matplotlib" not in paper_png
        assert f"{root}/paper_results/plot.pdf" in names
        retained_favicon = archive.read(
            f"{root}/docs/assets/logo/favicon/favicon-32x32.png"
        )
        assert b"Matplotlib" not in retained_favicon
        retained_logo = archive.read(f"{root}/docs/assets/logo/soromox_logo.png")
        assert b"Matplotlib" not in retained_logo
        assert b"c2pa" not in retained_logo

        report = archive.read(f"{root}/{archive_module.REPORT_PATH}").decode()
        assert "results/figure.png" in report
        assert "paper_results/plot.pdf (PDF document metadata replaced)" in report
        assert "Git history are absent" in report
        assert result.report.read_text(encoding="utf-8") == report

        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.comment == b""
            assert info.extra == b""


def test_refuses_to_overwrite_existing_archive(
    source_repo: Path, tmp_path: Path
) -> None:
    output = tmp_path / "anonymous.zip"
    output.write_bytes(b"existing")

    with pytest.raises(archive_module.AnonymizationError, match="already exists"):
        archive_module.create_archive(source_repo, output)

    output.unlink()
    report = tmp_path / archive_module.REPORT_PATH
    report.write_text("existing report", encoding="utf-8")
    with pytest.raises(archive_module.AnonymizationError, match="already exists"):
        archive_module.create_archive(source_repo, output)


def test_cli_prints_archive_and_report_paths(
    source_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "cli-output.zip"
    report = tmp_path / "cli-report.txt"
    monkeypatch.chdir(source_repo)

    assert archive_module.main(["--output", str(output), "--report", str(report)]) == 0

    stdout = capsys.readouterr().out
    assert f"Created archive: {output}" in stdout
    assert f"Anonymization report: {report}" in stdout


def test_name_matching_does_not_confuse_ordinary_words() -> None:
    assert archive_module._normal_form("wrong shape") == "wrong shape"
    assert archive_module._normal_form(r"St{\"o}lzle") == "stolzle"


def test_pdf_metadata_is_replaced_and_literal_identity_is_removed() -> None:
    original = _pdf_with_metadata()
    sanitized = archive_module._sanitize_pdf_metadata(original)

    assert b"Alice Example" not in sanitized
    assert b"D:20260101000000Z" not in sanitized
    assert sanitized.rstrip().endswith(b"%%EOF")
    assert sanitized.rfind(b"/Info 3 0 R") > sanitized.rfind(b"/Info 2 0 R")


def test_nested_zip_metadata_is_normalized() -> None:
    source = io.BytesIO()
    with zipfile.ZipFile(source, "w") as archive:
        info = zipfile.ZipInfo("system_info.txt", (2026, 8, 5, 12, 0, 0))
        archive.writestr(info, "- OS: identifying workstation\n")

    rules = archive_module.IdentityRules(
        full_names=("Alice Example",),
        family_names=("Example",),
        name_variants=("Alice Example",),
        affiliations=(),
        identifying_urls=(),
        identifying_emails=(),
        identifying_hosts=(),
        repository_owners=(),
        project_name="review-project",
    )
    sanitized = archive_module._canonicalize_nested_zip(
        source.getvalue(), xlsx=False, rules=rules
    )
    with zipfile.ZipFile(io.BytesIO(sanitized)) as archive:
        info = archive.getinfo("system_info.txt")
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert archive.read(info).startswith(b"System information removed")


@pytest.mark.parametrize("path", sorted(archive_module.INTERNAL_ANONYMIZER_PATHS))
def test_internal_anonymizer_files_are_excluded(path: str) -> None:
    assert "internal anonymization utility" in archive_module._is_risky(path)


@pytest.mark.parametrize(
    "path",
    (
        "paper_results/figure.pdf",
        "paper_results/video.mp4",
        "docs/assets/logo/favicon/favicon.ico",
        "docs/assets/logo/soromox_logo.png",
        "docs/assets/logo/soromox_logo_s.png",
    ),
)
def test_explicitly_retained_artifacts_are_not_excluded(path: str) -> None:
    assert archive_module._is_risky(path) is None


def test_unrelated_logo_remains_excluded() -> None:
    assert archive_module._is_risky("docs/assets/logo/lab_logo.png") is not None
