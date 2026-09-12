"""
extractor.py — Unstructured ingestion for candidate materials.

Handles PDF resumes, Markdown project write-ups / READMEs, and raw text
(transcripts, pasted bullet points). Produces a flat list of
`CandidateSnippet` objects that downstream stages (matcher.py, agent_prompts)
treat as atomic, independently-scorable units of evidence.

Design notes:
- We deliberately snippet at a paragraph/bullet granularity rather than
  whole-document, because the matcher needs to score individual
  accomplishments against individual JD requirements, and the XYZ rewriter
  needs one accomplishment at a time to avoid conflating unrelated facts.
- PDF parsing tries pymupdf (fitz) first because it's faster and handles
  multi-column resume layouts better than pypdf; pypdf is the fallback if
  pymupdf isn't available in a given environment.
- Everything here is pure CPU-bound parsing — no network calls, no model
  loads — so it's safe to unit test without the LLM or embedding model
  present at all (see tests/fixtures/).
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from loguru import logger

from backend.schemas import CandidateSnippet, SourceType

try:
    import fitz  # pymupdf

    _HAS_PYMUPDF = True
except ImportError:  # pragma: no cover - environment dependent
    _HAS_PYMUPDF = False

try:
    from pypdf import PdfReader

    _HAS_PYPDF = True
except ImportError:  # pragma: no cover
    _HAS_PYPDF = False


class ExtractionError(RuntimeError):
    """Raised when a source file can't be parsed by any available backend."""


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def _extract_pdf_text(path: Path) -> str:
    if _HAS_PYMUPDF:
        try:
            doc = fitz.open(path)
            pages = [page.get_text("text") for page in doc]
            doc.close()
            return "\n".join(pages)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"pymupdf failed on {path.name}: {exc}, trying pypdf")

    if _HAS_PYPDF:
        try:
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:  # noqa: BLE001
            raise ExtractionError(f"pypdf failed on {path.name}: {exc}") from exc

    raise ExtractionError(
        "No PDF backend available. Install pymupdf or pypdf (see requirements.txt)."
    )


# ---------------------------------------------------------------------------
# Segmentation — turning raw text blobs into atomic snippets
# ---------------------------------------------------------------------------

_BULLET_PREFIX_RE = re.compile(r"^\s*[-*•▪◦‣]\s+")
_MD_HEADER_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)
_WHITESPACE_RUN_RE = re.compile(r"[ \t]+")


def _clean_line(line: str) -> str:
    line = _BULLET_PREFIX_RE.sub("", line)
    line = _WHITESPACE_RUN_RE.sub(" ", line).strip()
    return line


def _segment_into_snippets(text: str, min_len: int = 25) -> list[str]:
    """
    Split raw text into candidate snippets at bullet/line boundaries first,
    falling back to sentence-ish splitting for prose paragraphs (e.g.
    transcript narrative, README prose sections).

    min_len filters out fragments too short to be a meaningful standalone
    accomplishment (e.g. stray headers, "Skills:" labels).
    """
    lines = [_clean_line(l) for l in text.splitlines()]
    lines = [l for l in lines if l]

    snippets: list[str] = []
    buffer: list[str] = []

    def flush():
        if buffer:
            candidate = " ".join(buffer).strip()
            if len(candidate) >= min_len:
                snippets.append(candidate)
            buffer.clear()

    for line in lines:
        # Treat markdown headers as hard segment boundaries.
        if _MD_HEADER_RE.match(line):
            flush()
            continue
        # A line that looks like its own bullet becomes its own snippet
        # immediately rather than being merged with neighbors.
        if len(line) >= min_len:
            flush()
            snippets.append(line)
        else:
            buffer.append(line)
    flush()

    return snippets


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_from_file(path: str | Path, source_type: SourceType | None = None) -> list[CandidateSnippet]:
    """
    Extract candidate snippets from a single file on disk.

    If `source_type` isn't provided, it's inferred from the file extension:
    .pdf -> PDF_RESUME, .md -> MARKDOWN, everything else -> RAW_TEXT.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if source_type is None:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            source_type = SourceType.PDF_RESUME
        elif suffix == ".md":
            source_type = SourceType.MARKDOWN
        else:
            source_type = SourceType.RAW_TEXT

    if source_type == SourceType.PDF_RESUME:
        raw_text = _extract_pdf_text(path)
    else:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")

    return extract_from_text(raw_text, source_type=source_type, source_name=path.name)


def extract_from_text(
    raw_text: str,
    source_type: SourceType,
    source_name: str,
) -> list[CandidateSnippet]:
    """Extract candidate snippets from an in-memory text blob."""
    if not raw_text or not raw_text.strip():
        logger.warning(f"{source_name}: empty content, no snippets produced")
        return []

    fragments = _segment_into_snippets(raw_text)
    snippets = [
        CandidateSnippet(
            snippet_id=f"snip_{uuid.uuid4().hex[:10]}",
            source_type=source_type,
            source_name=source_name,
            raw_text=fragment,
        )
        for fragment in fragments
    ]
    logger.info(f"{source_name}: extracted {len(snippets)} snippets")
    return snippets


def extract_job_description(raw_text: str, title: str, company: str | None = None) -> "JobDescriptionDraft":
    """
    Lightweight JD requirement segmentation. Returns raw requirement strings;
    matcher.py embeds them, agent_prompts.py's Phase 1 call further refines
    which ones are truly "required" vs "nice to have" using the LLM, since
    that distinction needs semantic judgment this regex-based pass can't make.
    """
    requirements = _segment_into_snippets(raw_text, min_len=15)
    return JobDescriptionDraft(title=title, company=company, raw_text=raw_text, requirements=requirements)


from dataclasses import dataclass, field  # noqa: E402  (kept local to avoid top-level clutter)


@dataclass
class JobDescriptionDraft:
    title: str
    company: str | None
    raw_text: str
    requirements: list[str] = field(default_factory=list)
