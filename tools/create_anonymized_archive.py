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
import io
import os
import re
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
LOCAL_PATH_RES = (
    re.compile(r"/Users/[^/\s]+"),
    re.compile(r"/home/[^/\s]+"),
    re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.IGNORECASE),
)
SENSITIVE_ASSET_REFERENCE_RE = re.compile(
    r"\b(?:headshots?|logos?|portraits?)\b", re.IGNORECASE
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
    included_count: int
    sanitized_paths: tuple[str, ...]
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
    found_authors = False
    for line in lines:
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
        if re.match(r"^(?:repository-code|url):", line):
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
        text = _remove_markdown_sections(text, REMOVED_SECTION_RE)
        text = _remove_markdown_sections(
            text, re.compile(r"\bsoftware citation\b", re.IGNORECASE)
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
            if not SENSITIVE_ASSET_REFERENCE_RE.search(line)
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

    # NPZ is intentionally allowed because it is a common scientific-data
    # container. Inspect its member names and payloads instead of trusting the
    # outer compressed bytes alone.
    if PurePosixPath(path).suffix.casefold() == ".npz":
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
        except (OSError, zipfile.BadZipFile) as exc:
            violations.append(f"{path}: unreadable NPZ container ({exc})")
    return sorted(set(violations))


def _zip_info(path: str, executable: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    mode = 0o755 if executable else 0o644
    info.external_attr = (0o100000 | mode) << 16
    info.comment = b""
    info.extra = b""
    return info


def _report(excluded: Sequence[tuple[str, str]], sanitized: Sequence[str]) -> str:
    lines = [
        "DOUBLE-ANONYMOUS ARCHIVE REPORT",
        "=================================",
        "",
        "This ZIP was generated only from files committed to a Git tree.",
        "Untracked, ignored, modified-but-uncommitted files and Git history are absent.",
        "Archive entry timestamps and platform metadata were normalized.",
        "Author/maintainer metadata, affiliations, acknowledgments/funding, lab names,",
        "and identifying links were removed. Bibliographic names may remain in references.",
        "",
        "Rendered images, documents, videos, nested archives, and other formats that can",
        "carry identifying metadata, logos, or faces were omitted by fail-safe policy.",
        "Regenerate necessary figures from the included source only after manual review.",
        "",
        f"Sanitized text files: {len(sanitized)}",
        f"Excluded potentially identifying files: {len(excluded)}",
    ]
    if excluded:
        lines.extend(("", "Excluded files:"))
        lines.extend(f"- {path} ({reason})" for path, reason in excluded)
    lines.append("")
    return "\n".join(lines)


def create_archive(
    repo: Path,
    output: Path,
    *,
    ref: str = "HEAD",
    force: bool = False,
) -> ArchiveResult:
    """Create and verify a double-anonymous archive from ``ref``."""

    repo = repo.resolve()
    output = output.resolve()
    if output.exists() and not force:
        raise AnonymizationError(
            f"Output already exists: {output} (use --force to replace it)"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="anonymous-archive-"
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        tar_path = temporary / "source.tar"
        zip_path = temporary / "archive.zip"
        _write_git_tar(repo, ref, tar_path)
        rules = discover_identity_rules(_discovery_texts(tar_path))

        excluded: list[tuple[str, str]] = []
        sanitized: list[str] = []
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
                    violations.extend(_binary_violations(member.name, original, rules))
                    result = original
                else:
                    anonymized = sanitize_text(member.name, text, rules)
                    violations.extend(_text_violations(member.name, anonymized, rules))
                    result = anonymized.encode("utf-8")
                    if result != original:
                        sanitized.append(member.name)

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

            report_text = _report(excluded, sorted(set(sanitized)))
            destination.writestr(
                _zip_info(f"{ARCHIVE_ROOT}/{REPORT_PATH}"), report_text.encode("utf-8")
            )

        os.replace(zip_path, output)

    return ArchiveResult(
        output=output,
        included_count=included_count,
        sanitized_paths=tuple(sorted(set(sanitized))),
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
        "--ref",
        default="HEAD",
        help="committed Git tree to archive (default: HEAD)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output ZIP",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        repo = find_repository(Path.cwd())
        output = arguments.output or repo / "anonymous-supplementary-material.zip"
        result = create_archive(repo, output, ref=arguments.ref, force=arguments.force)
    except AnonymizationError as exc:
        parser.exit(1, f"error: {exc}\n")

    print(f"Created {result.output}")
    print(f"Included committed files: {result.included_count}")
    print(f"Sanitized text files: {len(result.sanitized_paths)}")
    print(f"Excluded risky files: {len(result.excluded_paths)}")
    status = _run_git(repo, ["status", "--porcelain"]).stdout
    if status:
        print(
            "Note: working-tree changes were ignored; only the committed ref was archived."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
