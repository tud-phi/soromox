from __future__ import annotations

import importlib.util
import subprocess
import sys
import tomllib
import zipfile
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

Alice Example maintains this package at https://github.com/identifying-group/review-project.

## Citation

### Software Citation

```bibtex
@software{package, author={Example, Alice}}
```

### Related Reference

Example, A. (2024). A legitimate scholarly reference.

## Acknowledgments

Our previous work was funded by Grant 123.
""",
    )
    _write(repo / "src" / "module.py", "VALUE = 1\n")
    _write(repo / "results" / "figure.png", b"\x89PNG\r\n\x1a\ncreator=Alice Example")
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
        assert "Acknowledgments" not in readme
        assert "Example, A. (2024)" in readme

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

        report = archive.read(f"{root}/{archive_module.REPORT_PATH}").decode()
        assert "results/figure.png" in report
        assert "Git history are absent" in report

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


def test_name_matching_does_not_confuse_ordinary_words() -> None:
    assert archive_module._normal_form("wrong shape") == "wrong shape"
    assert archive_module._normal_form(r"St{\"o}lzle") == "stolzle"
