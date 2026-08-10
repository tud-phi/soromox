#!/usr/bin/env python3
"""Create a fail-closed, double-anonymous ZIP from a committed Git tree.

The archive is built from ``git archive`` rather than the working tree. This
means that untracked, ignored, and uncommitted local files are never included.
Potentially identifying text is rewritten or removed, visual and opaque
document formats are omitted by default, and every ZIP entry receives neutral,
deterministic metadata.

The anonymizer discovers project identities from the committed package and
citation metadata; it intentionally does not embed a repository-specific list
of people or institutions in this script. Bibliographic names are retained in
recognizable reference contexts, but ownership language such as "our previous
work" is not allowed.
"""

from __future__ import annotations

import argparse
import binascii
import io
import os
import re
import struct
import subprocess
import tarfile
import tempfile
import tomllib
import unicodedata
import urllib.parse
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ARCHIVE_ROOT = "anonymous-supplementary-material"
REPORT_PATH = "ANONYMIZATION_REPORT.txt"
NEUTRAL_URL = "https://anonymous.invalid/resource"
NEUTRAL_EMAIL = "contact@anonymous.invalid"
NEUTRAL_TEXT = "[redacted for double-anonymous review]"

# These formats can carry author/creator metadata, lab logos, or recognizable
# faces that cannot be reviewed reliably by a dependency-free script. Source
# code that generates them remains in the archive.
RISKY_SUFFIXES = frozenset(
    {
        ".7z",
        ".avi",
        ".bmp",
        ".bz2",
        ".doc",
        ".docx",
        ".eps",
        ".gif",
        ".gz",
        ".heic",
        ".ico",
        ".jpeg",
        ".jpg",
        ".key",
        ".mat",
        ".mkv",
        ".mov",
        ".mp4",
        ".mpeg",
        ".pdf",
        ".png",
        ".ppt",
        ".pptx",
        ".ps",
        ".rar",
        ".svg",
        ".tar",
        ".tif",
        ".tiff",
        ".webm",
        ".webp",
        ".xls",
        ".xlsx",
        ".xz",
        ".zip",
    }
)
RISKY_PATH_PARTS = frozenset(
    {"headshot", "headshots", "logo", "logos", "portrait", "portraits"}
)
INTERNAL_ANONYMIZER_PATHS = frozenset(
    {
        "tests/test_create_anonymized_archive.py",
        "tools/create_anonymized_archive.py",
    }
)

DISCOVERY_SUFFIXES = frozenset(
    {
        "",
        ".bib",
        ".cff",
        ".html",
        ".md",
        ".py",
        ".tex",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)

URL_RE = re.compile(r"(?:https?://|mailto:)[^\s<>'\"\])}]+", re.IGNORECASE)
EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
MARKDOWN_HEADING_RE = re.compile(r"^(\s*)(#{1,6})\s+(.+?)\s*$")
REFERENCE_TITLE_RE = re.compile(
    r"\b(?:bibliography|citation|citations|references|related work)\b", re.IGNORECASE
)
REMOVED_SECTION_RE = re.compile(
    r"\b(?:acknowledg(?:e)?ments?|author contributions?|funding|financial support|"
    r"sponsorship|competing interests?|conflicts? of interest)\b",
    re.IGNORECASE,
)
ACKNOWLEDGMENT_RE = re.compile(
    r"\b(?:acknowledg(?:e|ed|es|ing|ment|ments)?|funded by|funding from|"
    r"financial support|grant (?:agreement|number|no\.?))\b",
    re.IGNORECASE,
)
OWNERSHIP_RE = re.compile(
    r"\b(?:in\s+)?our\s+(?:previous|prior|earlier)\s+work\b", re.IGNORECASE
)
PREDECESSOR_PACKAGE_RE = re.compile(r"\bjsrm\b", re.IGNORECASE)
LOCAL_PATH_RES = (
    re.compile(r"/Users/[^/\s]+"),
    re.compile(r"/home/[^/\s]+"),
    re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.IGNORECASE),
)
SENSITIVE_ASSET_REFERENCE_RE = re.compile(
    r"\b(?:headshots?|logos?|portraits?)\b", re.IGNORECASE
)
IDENTIFYING_CITATION_SECTION_RE = re.compile(
    r"^(?:citation|software citation|planar hsa model citation|"
    r"model-based controllers citation|references and citation)$",
    re.IGNORECASE,
)
RETAINED_VISUAL_PREFIXES = (
    "docs/assets/logo/favicon/",
    "paper_results/",
)
RETAINED_VISUAL_PATHS = frozenset(
    {
        "docs/assets/logo/soromox_logo.png",
        "docs/assets/logo/soromox_logo_s.png",
    }
)
PNG_METADATA_CHUNKS = frozenset({b"caBX", b"eXIf", b"iTXt", b"tEXt", b"tIME", b"zTXt"})
PDF_INFO_VALUE_RE = re.compile(
    rb"/(Author|Creator|Producer|CreationDate|ModDate|Title|Subject|Keywords)"
    rb"\s*(\((?:\\.|[^\\)])*\)|<[0-9A-Fa-f\s]*>)",
    re.DOTALL,
)


class AnonymizationError(RuntimeError):
    """Raised when a safe archive cannot be produced."""


@dataclass(frozen=True)
class IdentityRules:
    """Identity markers discovered from the committed source tree."""

    full_names: tuple[str, ...]
    family_names: tuple[str, ...]
    name_variants: tuple[str, ...]
    affiliations: tuple[str, ...]
    identifying_urls: tuple[str, ...]
    identifying_emails: tuple[str, ...]
    identifying_hosts: tuple[str, ...]
    repository_owners: tuple[str, ...]
    project_name: str


@dataclass(frozen=True)
class ArchiveResult:
    """Summary returned after successfully creating an archive."""

    output: Path
    report: Path
    included_count: int
    sanitized_paths: tuple[str, ...]
    metadata_sanitized_paths: tuple[str, ...]
    excluded_paths: tuple[str, ...]


def _run_git(
    repo: Path,
    arguments: Sequence[str],
    *,
    stdout: int | io.BufferedWriter = subprocess.PIPE,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            stdout=stdout,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise AnonymizationError("Git is required but was not found on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise AnonymizationError(detail or "Git command failed.") from exc


def find_repository(start: Path) -> Path:
    """Return the enclosing Git repository root."""

    result = _run_git(start, ["rev-parse", "--show-toplevel"])
    return Path(result.stdout.decode("utf-8").strip()).resolve()


def _normal_form(value: str) -> str:
    """Normalize Unicode and common TeX accents for conservative matching."""

    value = re.sub(r"\\[\"'`^~=.uvHckbr]\s*\{?([A-Za-z])\}?", r"\1", value)
    value = value.replace("{", "").replace("}", "")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        character for character in value if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _phrase_present(text: str, phrase: str) -> bool:
    normalized_text = f" {_normal_form(text)} "
    normalized_phrase = _normal_form(phrase)
    return bool(normalized_phrase and f" {normalized_phrase} " in normalized_text)


def _safe_decode(data: bytes) -> str | None:
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _read_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise AnonymizationError(f"Could not read tracked file: {member.name}")
    return extracted.read()


def _write_git_tar(repo: Path, ref: str, destination: Path) -> None:
    _run_git(repo, ["rev-parse", "--verify", f"{ref}^{{commit}}"])
    with destination.open("wb") as output:
        _run_git(repo, ["archive", "--format=tar", ref], stdout=output)


def _discovery_texts(tar_path: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    with tarfile.open(tar_path, "r:") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            path = PurePosixPath(member.name)
            if path.suffix.casefold() not in DISCOVERY_SUFFIXES:
                continue
            # Citation/identity discovery never needs giant generated text blobs.
            if member.size > 5 * 1024 * 1024:
                continue
            text = _safe_decode(_read_member(archive, member))
            if text is not None:
                texts[member.name] = text
    return texts


def _cff_people(cff_text: str) -> list[tuple[str, str]]:
    people: list[tuple[str, str]] = []
    in_authors = False
    family = ""
    given = ""
    for line in cff_text.splitlines():
        if line.rstrip() == "authors:":
            in_authors = True
            continue
        if in_authors and line and not line[0].isspace():
            break
        family_match = re.match(r'^\s*-\s*family-names:\s*["\'](.+?)["\']\s*$', line)
        given_match = re.match(r'^\s*given-names:\s*["\'](.+?)["\']\s*$', line)
        if family_match:
            if family and given:
                people.append((given, family))
            family = family_match.group(1)
            given = ""
        elif given_match:
            given = given_match.group(1)
    if family and given:
        people.append((given, family))
    return people


def _author_page_affiliations(author_page: str) -> set[str]:
    affiliations: set[str] = set()
    in_authors_section = False
    in_profile_header = False
    for raw_line in author_page.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("## ") and not stripped.startswith("### "):
            in_authors_section = stripped.casefold() == "## authors"
            in_profile_header = False
            continue
        if in_authors_section and stripped.startswith("### "):
            in_profile_header = True
            continue
        if not in_profile_header:
            continue
        if stripped.startswith("**Contact:") or stripped.startswith("**Links:"):
            in_profile_header = False
            continue
        if not stripped or stripped == "---":
            continue
        cleaned = re.sub(r"<br\s*/?>", "", stripped, flags=re.IGNORECASE).strip()
        if not cleaned:
            continue
        affiliations.add(cleaned)
        for part in re.split(r"\s*,\s*", cleaned):
            part = part.strip()
            has_institution_word = re.search(
                r"\b(?:college|department|institute|lab(?:oratory)?|polytechnique|"
                r"robotics|school|university)\b",
                part,
                re.IGNORECASE,
            )
            has_acronym = re.search(r"\b[A-Z]{2,}\b", part)
            if len(part) >= 4 and (has_institution_word or has_acronym):
                affiliations.add(part)
    return affiliations


def _url_host(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    return (parsed.hostname or "").casefold().removeprefix("www.")


def discover_identity_rules(texts: dict[str, str]) -> IdentityRules:
    """Discover names, affiliations, and links from project metadata."""

    pyproject_text = texts.get("pyproject.toml")
    if not pyproject_text:
        raise AnonymizationError("The committed tree does not contain pyproject.toml.")
    try:
        project = tomllib.loads(pyproject_text).get("project", {})
    except tomllib.TOMLDecodeError as exc:
        raise AnonymizationError(
            f"Cannot parse committed pyproject.toml: {exc}"
        ) from exc

    project_name = str(project.get("name", "")).strip()
    people = [
        str(person.get("name", "")).strip()
        for group in (project.get("authors", []), project.get("maintainers", []))
        for person in group
        if isinstance(person, dict) and person.get("name")
    ]

    cff_text = texts.get("CITATION.cff", "")
    cff_people = _cff_people(cff_text)
    people.extend(f"{given} {family}" for given, family in cff_people)
    full_names = {
        person for person in people if person and _normal_form(person) != "anonymous"
    }
    if not full_names:
        raise AnonymizationError(
            "No author or maintainer identities were discovered; refusing to claim anonymity."
        )

    family_names = {family for _, family in cff_people if family}
    if not family_names:
        family_names.update(name.rsplit(" ", 1)[-1] for name in full_names)

    name_variants = set(full_names)
    for given, family in cff_people:
        name_variants.add(f"{family}, {given}")
        name_variants.add(f"{given} {family}")
    name_variants.update(
        "".join(
            character
            for character in unicodedata.normalize("NFKD", name)
            if not unicodedata.combining(character)
        )
        for name in tuple(name_variants)
    )

    author_page = texts.get("docs/authors.md", "")
    identifying_emails = set(EMAIL_RE.findall(author_page))
    identifying_urls = {
        match.group(0).rstrip(".,;:") for match in URL_RE.finditer(author_page)
    }

    project_urls = project.get("urls", {})
    if isinstance(project_urls, dict):
        identifying_urls.update(str(value) for value in project_urls.values())
    for key in ("repository-code", "url"):
        match = re.search(
            rf'^\s*{re.escape(key)}:\s*["\']?([^"\'\s]+)', cff_text, re.MULTILINE
        )
        if match:
            identifying_urls.add(match.group(1))

    affiliations = _author_page_affiliations(author_page)
    bib_affiliation_re = re.compile(
        r'^\s*(?:affiliation|institution|school)\s*=\s*[\{"\'](.+?)[\}"\'],?\s*$',
        re.IGNORECASE | re.MULTILINE,
    )
    for text in texts.values():
        affiliations.update(
            match.group(1).strip() for match in bib_affiliation_re.finditer(text)
        )

    identifying_hosts = {
        domain.casefold()
        for email in identifying_emails
        for domain in [email.rsplit("@", 1)[-1]]
        if domain
    }
    identifying_hosts.update(
        host for value in identifying_urls for host in [_url_host(value)] if host
    )

    repository_owners: set[str] = set()
    for value in identifying_urls:
        parsed = urllib.parse.urlparse(value)
        host = (parsed.hostname or "").casefold().removeprefix("www.")
        path_parts = [part for part in parsed.path.split("/") if part]
        if host == "github.com" and len(path_parts) >= 2:
            repository_owners.add(path_parts[0])

    # github.com hosts both identifying repositories and unrelated references,
    # so only owner/path matching is used for it rather than host-wide removal.
    identifying_hosts.discard("github.com")
    identifying_hosts.discard("www.github.com")

    return IdentityRules(
        full_names=tuple(sorted(full_names, key=str.casefold)),
        family_names=tuple(sorted(family_names, key=str.casefold)),
        name_variants=tuple(sorted(name_variants, key=len, reverse=True)),
        affiliations=tuple(sorted(affiliations, key=len, reverse=True)),
        identifying_urls=tuple(sorted(identifying_urls, key=len, reverse=True)),
        identifying_emails=tuple(sorted(identifying_emails, key=len, reverse=True)),
        identifying_hosts=tuple(sorted(identifying_hosts, key=len, reverse=True)),
        repository_owners=tuple(sorted(repository_owners, key=len, reverse=True)),
        project_name=project_name,
    )


def _is_risky(path: str) -> str | None:
    if path in INTERNAL_ANONYMIZER_PATHS:
        return (
            "internal anonymization utility containing detection rules or test fixtures"
        )
    if path in RETAINED_VISUAL_PATHS or path.startswith(RETAINED_VISUAL_PREFIXES):
        return None
    pure_path = PurePosixPath(path)
    suffix = pure_path.suffix.casefold()
    if suffix in RISKY_SUFFIXES:
        return f"potentially identifying visual or metadata-bearing {suffix or 'file'}"
    lowered_parts = {part.casefold() for part in pure_path.parts}
    risky_parts = lowered_parts & RISKY_PATH_PARTS
    if risky_parts:
        return f"potentially identifying visual asset ({sorted(risky_parts)[0]})"
    return None


def _remove_markdown_sections(text: str, title_pattern: re.Pattern[str]) -> str:
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    removed_level: int | None = None
    for line in lines:
        heading = MARKDOWN_HEADING_RE.match(line.rstrip("\r\n"))
        if removed_level is not None:
            if heading and len(heading.group(2)) <= removed_level:
                removed_level = None
            else:
                continue
        if heading and title_pattern.search(heading.group(3)):
            removed_level = len(heading.group(2))
            continue
        output.append(line)
    return "".join(output)


def _remove_markdown_admonitions_containing(text: str, marker: re.Pattern[str]) -> str:
    """Remove a MkDocs admonition whose title or body contains ``marker``."""

    lines = text.splitlines(keepends=True)
    output: list[str] = []
    index = 0
    while index < len(lines):
        if not re.match(r"^\s*!!!\s+", lines[index]):
            output.append(lines[index])
            index += 1
            continue
        end = index + 1
        while end < len(lines):
            candidate = lines[end]
            if candidate.strip() == "" or candidate.startswith(("    ", "\t")):
                end += 1
                continue
            break
        block = "".join(lines[index:end])
        if not marker.search(block):
            output.append(block)
        index = end
    return "".join(output)


def _sanitize_pyproject(text: str) -> str:
    for field in ("authors", "maintainers"):
        text, count = re.subn(
            rf"(?ms)^{field}[ \t]*=[ \t]*\[.*?^\][ \t]*(?:#[^\r\n]*)?$",
            f'{field} = [{{name = "Anonymous"}}]',
            text,
            count=1,
        )
        if count != 1:
            raise AnonymizationError(
                f"Could not safely replace project.{field} metadata."
            )
    return text


def _sanitize_cff(text: str) -> str:
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    skipping_authors = False
    skipping_preferred_citation = False
    found_authors = False
    for line in lines:
        if line.rstrip("\r\n") == "preferred-citation:":
            skipping_preferred_citation = True
            continue
        if skipping_preferred_citation:
            if line and not line[0].isspace():
                skipping_preferred_citation = False
            else:
                continue
        if line.rstrip("\r\n") == "authors:":
            output.extend(("authors:\n", '  - name: "Anonymous"\n'))
            skipping_authors = True
            found_authors = True
            continue
        if skipping_authors:
            if line and not line[0].isspace():
                skipping_authors = False
            else:
                continue
        if re.match(r"^(?:repository-artifact|repository-code|url):", line):
            continue
        output.append(line)
    if not found_authors:
        raise AnonymizationError("Could not safely replace CITATION.cff authors.")
    return "".join(output)


def _reference_flags(lines: Sequence[str], path: str) -> list[bool]:
    if PurePosixPath(path).suffix.casefold() in {".bib", ".ris"}:
        return [True] * len(lines)

    flags = [False] * len(lines)
    reference_heading_level: int | None = None
    reference_fence = False
    docstring_references = False
    for index, line in enumerate(lines):
        heading = MARKDOWN_HEADING_RE.match(line)
        if heading:
            level = len(heading.group(2))
            if reference_heading_level is not None and level <= reference_heading_level:
                reference_heading_level = None
            if REFERENCE_TITLE_RE.search(heading.group(3)):
                reference_heading_level = level

        fence = re.match(r"^\s*(```+|~~~+)\s*([^\s]*)", line)
        if fence:
            language = fence.group(2).casefold()
            if reference_fence:
                flags[index] = True
                reference_fence = False
                continue
            if language in {"bib", "bibtex", "citation", "ris"}:
                reference_fence = True

        if re.match(r"^\s*(?:references|bibliography)\s*:\s*$", line, re.IGNORECASE):
            docstring_references = True
        if docstring_references and index > 0 and re.search(r"(?:'''|\"\"\")", line):
            flags[index] = True
            docstring_references = False
            continue

        citation_line = bool(
            re.search(
                r"\b(?:19|20)\d{2}[a-z]?\b|\bet\s+al\.\b|https?://doi\.org/", line
            )
        )
        flags[index] = bool(
            flags[index]
            or reference_heading_level is not None
            or reference_fence
            or docstring_references
            or citation_line
        )
    return flags


def _url_is_identifying(url: str, rules: IdentityRules) -> bool:
    candidate = url.rstrip(".,;:")
    if candidate.casefold().startswith("mailto:"):
        return True
    parsed = urllib.parse.urlparse(candidate)
    host = (parsed.hostname or "").casefold().removeprefix("www.")
    lowered = urllib.parse.unquote(candidate).casefold()

    if rules.project_name and rules.project_name.casefold() in lowered:
        return True
    if any(lowered.startswith(value.casefold()) for value in rules.identifying_urls):
        return True
    if any(
        host == known or host.endswith(f".{known}") for known in rules.identifying_hosts
    ):
        return True
    if host == "github.com":
        path_parts = [part.casefold() for part in parsed.path.split("/") if part]
        if path_parts and path_parts[0] in {
            owner.casefold() for owner in rules.repository_owners
        }:
            return True
    return False


def _replace_identifying_urls(text: str, rules: IdentityRules) -> str:
    def replacement(match: re.Match[str]) -> str:
        value = match.group(0)
        return NEUTRAL_URL if _url_is_identifying(value, rules) else value

    return URL_RE.sub(replacement, text)


def _replace_names(text: str, rules: IdentityRules) -> str:
    for name in rules.name_variants:
        if not name:
            continue
        text = re.sub(re.escape(name), "Anonymous", text, flags=re.IGNORECASE)
    return text


def _replace_affiliations(text: str, rules: IdentityRules) -> str:
    for affiliation in rules.affiliations:
        if len(_normal_form(affiliation)) < 4:
            continue
        text = re.sub(re.escape(affiliation), NEUTRAL_TEXT, text, flags=re.IGNORECASE)
    return text


def sanitize_text(path: str, text: str, rules: IdentityRules) -> str:
    """Return anonymized UTF-8 text while preserving scholarly references."""

    if path == "docs/citation.md":
        return (
            "# Citation\n\nCitation guidance is withheld for double-anonymous review.\n"
        )
    if path == "docs/authors.md":
        return (
            "# Authors & Maintainers\n\n"
            "Author and maintainer identities are withheld for double-anonymous review.\n"
        )
    if path == "pyproject.toml":
        text = _sanitize_pyproject(text)
    elif path == "CITATION.cff":
        text = _sanitize_cff(text)
    elif path == "mkdocs.yml":
        text = re.sub(
            r"^site_author:.*$", "site_author: Anonymous", text, flags=re.MULTILINE
        )

    if PurePosixPath(path).suffix.casefold() in {".md", ".markdown"}:
        text = _remove_markdown_admonitions_containing(text, PREDECESSOR_PACKAGE_RE)
        text = _remove_markdown_sections(text, REMOVED_SECTION_RE)
        text = _remove_markdown_sections(text, IDENTIFYING_CITATION_SECTION_RE)
        # Standalone prose and legacy changelog bullets can also disclose the
        # predecessor relationship. The full line is removed to avoid retaining
        # an identifying package name or URL fragment.
        text = "".join(
            line
            for line in text.splitlines(keepends=True)
            if not PREDECESSOR_PACKAGE_RE.search(line)
        )

    if PurePosixPath(path).suffix.casefold() == ".svg":
        text = re.sub(
            r"\s*<metadata\b[^>]*>.*?</metadata>\s*",
            "\n",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

    text = OWNERSHIP_RE.sub("previous work", text)
    text = re.sub(r"\bour software\b", "the software", text, flags=re.IGNORECASE)
    text = re.sub(r"\bour package\b", "the package", text, flags=re.IGNORECASE)

    if PurePosixPath(path).suffix.casefold() in {
        ".html",
        ".markdown",
        ".md",
        ".yaml",
        ".yml",
    }:
        text = "".join(
            line
            for line in text.splitlines(keepends=True)
            if not (
                SENSITIVE_ASSET_REFERENCE_RE.search(line)
                and not any(
                    allowed in line.casefold()
                    for allowed in ("favicon", "soromox_logo")
                )
            )
        )

    text = EMAIL_RE.sub(NEUTRAL_EMAIL, text)
    text = _replace_identifying_urls(text, rules)
    text = _replace_affiliations(text, rules)

    for owner in rules.repository_owners:
        text = re.sub(
            rf"(?<![\w-]){re.escape(owner)}(?=/|\b)",
            "anonymous",
            text,
            flags=re.IGNORECASE,
        )
    for local_path_re in LOCAL_PATH_RES:
        text = local_path_re.sub("/anonymous/user", text)

    lines = text.splitlines(keepends=True)
    flags = _reference_flags([line.rstrip("\r\n") for line in lines], path)
    for index, line in enumerate(lines):
        if not flags[index]:
            lines[index] = _replace_names(line, rules)
    return "".join(lines)


def _text_violations(path: str, text: str, rules: IdentityRules) -> list[str]:
    violations: list[str] = []
    lines = text.splitlines()
    reference_flags = _reference_flags(lines, path)
    for number, line in enumerate(lines, start=1):
        if PREDECESSOR_PACKAGE_RE.search(line):
            violations.append(f"{path}:{number}: predecessor-package relationship")
        if OWNERSHIP_RE.search(line):
            violations.append(f"{path}:{number}: ownership language")
        if ACKNOWLEDGMENT_RE.search(line):
            violations.append(f"{path}:{number}: acknowledgment or funding language")
        if EMAIL_RE.search(line) and NEUTRAL_EMAIL not in line:
            violations.append(f"{path}:{number}: email address")
        if any(pattern.search(line) for pattern in LOCAL_PATH_RES):
            violations.append(f"{path}:{number}: local user path")
        for url_match in URL_RE.finditer(line):
            if _url_is_identifying(url_match.group(0), rules):
                violations.append(f"{path}:{number}: identifying link")
                break
        if any(
            _phrase_present(line, affiliation) for affiliation in rules.affiliations
        ):
            violations.append(f"{path}:{number}: affiliation or lab name")
        if not reference_flags[number - 1] and any(
            _phrase_present(line, family) for family in rules.family_names
        ):
            violations.append(f"{path}:{number}: author or maintainer name")
    return violations


def _binary_markers(rules: IdentityRules) -> tuple[bytes, ...]:
    values = {
        *rules.full_names,
        *rules.name_variants,
        *rules.affiliations,
        *rules.identifying_urls,
        *rules.identifying_emails,
    }
    values.update(rules.repository_owners)
    markers: set[bytes] = set()
    for value in values:
        for variant in (value, _normal_form(value)):
            if len(variant) >= 4:
                markers.add(variant.casefold().encode("utf-8"))
    return tuple(sorted(markers, key=len, reverse=True))


def _binary_violations(path: str, data: bytes, rules: IdentityRules) -> list[str]:
    lowered = data.lower()
    violations = [
        f"{path}: embedded identity marker"
        for marker in _binary_markers(rules)
        if marker.lower() in lowered
    ]
    if any(
        pattern.search(data.decode("latin-1", errors="ignore"))
        for pattern in LOCAL_PATH_RES
    ):
        violations.append(f"{path}: embedded local user path")

    # Inspect ZIP-based scientific and office containers instead of trusting
    # their outer compressed bytes alone.
    if PurePosixPath(path).suffix.casefold() in {".npz", ".xlsx", ".zip"}:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as nested:
                for info in nested.infolist():
                    if any(
                        marker in info.filename.casefold().encode()
                        for marker in _binary_markers(rules)
                    ):
                        violations.append(f"{path}: identifying nested member name")
                    payload = nested.read(info)
                    lowered_payload = payload.lower()
                    if any(
                        marker.lower() in lowered_payload
                        for marker in _binary_markers(rules)
                    ):
                        violations.append(f"{path}: identity marker in nested member")
                    nested_text = payload.decode("latin-1", errors="ignore")
                    if any(pattern.search(nested_text) for pattern in LOCAL_PATH_RES):
                        violations.append(f"{path}: local path in nested member")
        except (OSError, zipfile.BadZipFile) as exc:
            violations.append(f"{path}: unreadable ZIP-based container ({exc})")
    return sorted(set(violations))


def _sanitize_pdf_metadata(data: bytes) -> bytes:
    """Replace active PDF document metadata without changing rendered pages."""

    if not data.startswith(b"%PDF-"):
        raise AnonymizationError("Retained .pdf file does not have a PDF signature.")
    if re.search(rb"/Encrypt\b", data):
        raise AnonymizationError("Encrypted PDFs cannot be safely anonymized.")
    if re.search(rb"/Metadata\s+\d+\s+\d+\s+R", data):
        raise AnonymizationError(
            "PDF contains an XMP metadata stream that cannot be safely scrubbed."
        )

    root_matches = list(re.finditer(rb"/Root\s+(\d+)\s+(\d+)\s+R", data))
    size_matches = list(re.finditer(rb"/Size\s+(\d+)", data))
    startxref_matches = list(re.finditer(rb"startxref\s+(\d+)\s+%%EOF", data))
    if not (root_matches and size_matches and startxref_matches):
        raise AnonymizationError("Could not locate the PDF trailer safely.")

    # Blank literal metadata values in-place so simple, superseded Info objects
    # do not retain forensic copies. Keeping byte lengths unchanged preserves all
    # existing cross-reference offsets.
    def blank_value(match: re.Match[bytes]) -> bytes:
        replacement = b"/" + match.group(1) + b"()"
        return replacement.ljust(len(match.group(0)), b" ")

    scrubbed = PDF_INFO_VALUE_RE.sub(blank_value, data)
    root_object, root_generation = root_matches[-1].groups()
    old_startxref = int(startxref_matches[-1].group(1))
    new_object = int(size_matches[-1].group(1))

    prefix = b"\n"
    object_offset = len(scrubbed) + len(prefix)
    info_object = f"{new_object} 0 obj\n<< /Producer (Anonymous) >>\nendobj\n".encode()
    xref_offset = object_offset + len(info_object)
    incremental_update = (
        prefix
        + info_object
        + f"xref\n{new_object} 1\n{object_offset:010d} 00000 n \n".encode()
        + b"trailer\n"
        + (
            f"<< /Size {new_object + 1} /Root {root_object.decode()} "
            f"{root_generation.decode()} R /Info {new_object} 0 R "
            f"/Prev {old_startxref} >>\n"
        ).encode()
        + f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return scrubbed + incremental_update


def _sanitize_png_metadata(data: bytes) -> bytes:
    """Drop textual/EXIF/time/C2PA chunks while preserving pixel-bearing chunks."""

    signature = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(signature):
        raise AnonymizationError("Retained .png file does not have a PNG signature.")
    output = bytearray(signature)
    offset = len(signature)
    saw_header = False
    saw_end = False
    while offset < len(data):
        if offset + 12 > len(data):
            raise AnonymizationError("Truncated PNG chunk.")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        end = offset + 12 + length
        if end > len(data):
            raise AnonymizationError("Truncated PNG chunk payload.")
        chunk_type = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length : end])[0]
        actual_crc = binascii.crc32(chunk_type + payload) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise AnonymizationError("PNG chunk CRC verification failed.")
        if not saw_header and chunk_type != b"IHDR":
            raise AnonymizationError("PNG does not start with an IHDR chunk.")
        saw_header = True
        if chunk_type not in PNG_METADATA_CHUNKS:
            output.extend(data[offset:end])
        if chunk_type == b"IEND":
            saw_end = True
            if end != len(data):
                raise AnonymizationError("PNG contains trailing data after IEND.")
        offset = end
    if not (saw_header and saw_end):
        raise AnonymizationError("Incomplete PNG file.")
    return bytes(output)


def _sanitize_mat_metadata(data: bytes) -> bytes:
    """Neutralize the descriptive MATLAB v5 header without touching array data."""

    if len(data) < 128 or not data.startswith(b"MATLAB 5.0 MAT-file"):
        raise AnonymizationError("Only MATLAB v5 files can be safely anonymized.")
    header = b"MATLAB 5.0 MAT-file, Platform: anonymous, anonymized for peer review"
    return header.ljust(116, b" ") + data[116:]


def _safe_nested_name(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise AnonymizationError(f"Unsafe nested ZIP member name: {name}")


def _canonicalize_nested_zip(data: bytes, *, xlsx: bool, rules: IdentityRules) -> bytes:
    """Normalize nested ZIP metadata and scrub known identifying members."""

    source_buffer = io.BytesIO(data)
    output_buffer = io.BytesIO()
    try:
        source = zipfile.ZipFile(source_buffer)
    except (OSError, zipfile.BadZipFile) as exc:
        raise AnonymizationError(f"Unreadable retained ZIP container: {exc}") from exc

    with (
        source,
        zipfile.ZipFile(
            output_buffer,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as destination,
    ):
        destination.comment = b""
        for original_info in source.infolist():
            _safe_nested_name(original_info.filename)
            if original_info.is_dir():
                continue
            payload = source.read(original_info)
            if xlsx and original_info.filename == "docProps/core.xml":
                payload = re.sub(
                    rb"<dc:creator\b[^>]*>.*?</dc:creator>",
                    b"<dc:creator>Anonymous</dc:creator>",
                    payload,
                    flags=re.DOTALL,
                )
                payload = re.sub(
                    rb"<cp:lastModifiedBy\b[^>]*>.*?</cp:lastModifiedBy>",
                    b"<cp:lastModifiedBy>Anonymous</cp:lastModifiedBy>",
                    payload,
                    flags=re.DOTALL,
                )
                for date_field in (b"created", b"modified"):
                    payload = re.sub(
                        rb"<dcterms:"
                        + date_field
                        + rb"\b[^>]*>.*?</dcterms:"
                        + date_field
                        + rb">",
                        b"<dcterms:"
                        + date_field
                        + b' xsi:type="dcterms:W3CDTF">1980-01-01T00:00:00Z</dcterms:'
                        + date_field
                        + b">",
                        payload,
                        flags=re.DOTALL,
                    )
            elif (
                not xlsx
                and PurePosixPath(original_info.filename).name == "system_info.txt"
            ):
                payload = b"System information removed for double-anonymous review.\n"
            elif not xlsx:
                nested_text = _safe_decode(payload)
                if nested_text is not None:
                    payload = sanitize_text(
                        original_info.filename, nested_text, rules
                    ).encode("utf-8")
            executable = bool((original_info.external_attr >> 16) & 0o111)
            destination.writestr(
                _zip_info(original_info.filename, executable=executable), payload
            )
    return output_buffer.getvalue()


def _sanitize_binary_metadata(
    path: str, data: bytes, rules: IdentityRules
) -> tuple[bytes, str | None]:
    suffix = PurePosixPath(path).suffix.casefold()
    if suffix == ".pdf":
        return _sanitize_pdf_metadata(data), "PDF document metadata replaced"
    if suffix == ".png":
        sanitized = _sanitize_png_metadata(data)
        return (
            sanitized,
            "PNG textual, EXIF, timestamp, and content-credential chunks removed",
        )
    if suffix == ".mat":
        return _sanitize_mat_metadata(
            data
        ), "MATLAB platform and timestamp header replaced"
    if suffix == ".xlsx":
        return (
            _canonicalize_nested_zip(data, xlsx=True, rules=rules),
            "XLSX creator/timestamp and ZIP-entry metadata normalized",
        )
    if suffix == ".zip":
        return (
            _canonicalize_nested_zip(data, xlsx=False, rules=rules),
            "nested ZIP metadata and system information normalized",
        )
    return data, None


def _zip_info(path: str, executable: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    mode = 0o755 if executable else 0o644
    info.external_attr = (0o100000 | mode) << 16
    info.comment = b""
    info.extra = b""
    return info


def _report(
    excluded: Sequence[tuple[str, str]],
    sanitized: Sequence[str],
    metadata_sanitized: Sequence[tuple[str, str]],
) -> str:
    lines = [
        "DOUBLE-ANONYMOUS ARCHIVE REPORT",
        "=================================",
        "",
        "This ZIP was generated only from files committed to a Git tree.",
        "Untracked, ignored, modified-but-uncommitted files and Git history are absent.",
        "Archive entry timestamps and platform metadata were normalized.",
        "Author/maintainer metadata, affiliations, acknowledgments/funding, lab names,",
        "and identifying links were removed. Bibliographic names may remain in references.",
        "Predecessor-package references and identifying citation requests were removed.",
        "",
        "The SoRoMoX package logo, favicon, and files under paper_results/ were retained",
        "by explicit policy. Metadata-bearing retained formats were scrubbed; all",
        "retained visuals still require manual review for faces and identifying logos.",
        "",
        f"Sanitized text files: {len(sanitized)}",
        f"Metadata-sanitized retained files: {len(metadata_sanitized)}",
        f"Excluded potentially identifying files: {len(excluded)}",
    ]
    if metadata_sanitized:
        lines.extend(("", "Sanitized retained-file metadata:"))
        lines.extend(
            f"- {path} ({description})" for path, description in metadata_sanitized
        )
    if sanitized:
        lines.extend(("", "Sanitized text files:"))
        lines.extend(f"- {path}" for path in sanitized)
    if excluded:
        lines.extend(("", "Excluded files:"))
        lines.extend(f"- {path} ({reason})" for path, reason in excluded)
    lines.append("")
    return "\n".join(lines)


def create_archive(
    repo: Path,
    output: Path,
    *,
    report_output: Path | None = None,
    ref: str = "HEAD",
    force: bool = False,
) -> ArchiveResult:
    """Create and verify a double-anonymous archive from ``ref``."""

    repo = repo.resolve()
    output = output.resolve()
    report_output = (
        report_output.resolve() if report_output else output.parent / REPORT_PATH
    )
    if output == report_output:
        raise AnonymizationError("The ZIP and report output paths must be different.")
    existing_outputs = [path for path in (output, report_output) if path.exists()]
    if existing_outputs and not force:
        existing = ", ".join(str(path) for path in existing_outputs)
        raise AnonymizationError(
            f"Output already exists: {existing} (use --force to replace it)"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="anonymous-archive-"
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        tar_path = temporary / "source.tar"
        zip_path = temporary / "archive.zip"
        report_path = temporary / REPORT_PATH
        _write_git_tar(repo, ref, tar_path)
        rules = discover_identity_rules(_discovery_texts(tar_path))

        excluded: list[tuple[str, str]] = []
        sanitized: list[str] = []
        metadata_sanitized: list[tuple[str, str]] = []
        violations: list[str] = []
        included_count = 0

        with (
            tarfile.open(tar_path, "r:") as source,
            zipfile.ZipFile(
                zip_path,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
                strict_timestamps=True,
            ) as destination,
        ):
            destination.comment = b""
            for member in source.getmembers():
                if member.isdir():
                    continue
                if not member.isfile():
                    raise AnonymizationError(
                        f"Unsupported tracked entry type for {member.name}; refusing unsafe output."
                    )

                reason = _is_risky(member.name)
                if reason:
                    excluded.append((member.name, reason))
                    continue

                original = _read_member(source, member)
                text = _safe_decode(original)
                if text is None:
                    result, metadata_change = _sanitize_binary_metadata(
                        member.name, original, rules
                    )
                    violations.extend(_binary_violations(member.name, result, rules))
                    if metadata_change and result != original:
                        metadata_sanitized.append((member.name, metadata_change))
                else:
                    had_svg_metadata = bool(
                        PurePosixPath(member.name).suffix.casefold() == ".svg"
                        and re.search(r"<metadata\b", text, re.IGNORECASE)
                    )
                    anonymized = sanitize_text(member.name, text, rules)
                    violations.extend(_text_violations(member.name, anonymized, rules))
                    result = anonymized.encode("utf-8")
                    if result != original:
                        sanitized.append(member.name)
                    if had_svg_metadata:
                        metadata_sanitized.append(
                            (member.name, "SVG metadata element removed")
                        )

                archive_path = f"{ARCHIVE_ROOT}/{member.name}"
                destination.writestr(
                    _zip_info(archive_path, executable=bool(member.mode & 0o111)),
                    result,
                )
                included_count += 1

            if violations:
                details = "\n".join(
                    f"- {item}" for item in sorted(set(violations))[:100]
                )
                extra = (
                    ""
                    if len(set(violations)) <= 100
                    else "\n- (additional violations omitted)"
                )
                raise AnonymizationError(
                    "Anonymization verification failed; no archive was created:\n"
                    f"{details}{extra}"
                )

            report_text = _report(
                excluded,
                sorted(set(sanitized)),
                sorted(set(metadata_sanitized)),
            )
            destination.writestr(
                _zip_info(f"{ARCHIVE_ROOT}/{REPORT_PATH}"), report_text.encode("utf-8")
            )

        report_path.write_text(report_text, encoding="utf-8")
        os.replace(zip_path, output)
        os.replace(report_path, report_output)

    return ArchiveResult(
        output=output,
        report=report_output,
        included_count=included_count,
        sanitized_paths=tuple(sorted(set(sanitized))),
        metadata_sanitized_paths=tuple(
            path for path, _ in sorted(set(metadata_sanitized))
        ),
        excluded_paths=tuple(path for path, _ in sorted(excluded)),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a verified double-anonymous ZIP from committed, tracked repository files."
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output ZIP (default: <repo>/anonymous-supplementary-material.zip)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="standalone report (default: ANONYMIZATION_REPORT.txt beside the ZIP)",
    )
    parser.add_argument(
        "--ref",
        default="HEAD",
        help="committed Git tree to archive (default: HEAD)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing ZIP and report outputs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        repo = find_repository(Path.cwd())
        output = arguments.output or repo / "anonymous-supplementary-material.zip"
        result = create_archive(
            repo,
            output,
            report_output=arguments.report,
            ref=arguments.ref,
            force=arguments.force,
        )
    except AnonymizationError as exc:
        parser.exit(1, f"error: {exc}\n")

    print(f"Created archive: {result.output}")
    print(f"Anonymization report: {result.report}")
    print(f"Included committed files: {result.included_count}")
    print(f"Sanitized text files: {len(result.sanitized_paths)}")
    print(f"Metadata-sanitized retained files: {len(result.metadata_sanitized_paths)}")
    print(f"Excluded risky files: {len(result.excluded_paths)}")
    status = _run_git(repo, ["status", "--porcelain"]).stdout
    if status:
        print(
            "Note: working-tree changes were ignored; only the committed ref was archived."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
