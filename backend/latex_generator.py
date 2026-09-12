"""
latex_generator.py — Renders a ResumeDocument into a compiled PDF via a
Jinja2-templated LaTeX file, compiled fully offline with `tectonic`
(preferred) or `pdflatex` (fallback).

Two things this module is careful about, because both are easy to get
subtly wrong and both fail silently/ugly if mishandled:

1. **LaTeX special-character escaping.** Free-text resume content (bullets,
   summaries, project names) will routinely contain `%`, `&`, `$`, `_`, `#`,
   and other TeX-meaningful characters. We escape everything BEFORE it
   reaches the Jinja2 template via a custom filter, rather than trusting
   the LLM output or the template author to have done it — this is a
   correctness boundary, not a style choice.

2. **Jinja2 delimiter collision with LaTeX.** LaTeX uses `{`, `}`, `%`, `#`
   heavily — most importantly, `\newcommand{\foo}[1]{... #1 ...}` collides
   directly with Jinja2's default `{# #}` comment syntax. We therefore
   configure a custom Jinja2 environment (below) with LaTeX-safe delimiters:
   `\VAR{ ... }` for variables, `\BLOCK{ ... }` for control flow, and
   `\#{ ... }` for comments. `templates/resume_template.tex.jinja` is
   written against these custom delimiters, not Jinja2's defaults.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from loguru import logger

from backend.schemas import ResumeDocument

TEMPLATE_DIR = Path("templates")
TEMPLATE_NAME = "resume_template.tex.jinja"

# Order matters: backslash must be escaped first, or we'd double-escape the
# backslashes introduced by the other substitutions.
_TEX_SPECIAL_CHARS: list[tuple[str, str]] = [
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
]

_LATEX_LINK_FIELDS = {"linkedin", "github", "website"}


class LatexCompilationError(RuntimeError):
    pass


def _escape_latex(value: str) -> str:
    """Escape LaTeX special characters in free-text content."""
    if value is None:
        return ""
    text = str(value)
    for char, replacement in _TEX_SPECIAL_CHARS:
        text = text.replace(char, replacement)
    return text


def _escape_recursive(obj, _key: str | None = None):
    """
    Recursively escape every string field in a nested dict/list structure,
    EXCEPT fields we know are raw URLs destined for \\href{...} (those need
    to stay unescaped for the URL itself, though the displayed anchor text
    in the template is a separate literal, e.g. "LinkedIn", "GitHub").
    """
    if isinstance(obj, str):
        if _key in _LATEX_LINK_FIELDS or _key == "email":
            return obj  # raw URL/email used inside \href{...}, not displayed literally
        return _escape_latex(obj)
    if isinstance(obj, list):
        return [_escape_recursive(item, _key) for item in obj]
    if isinstance(obj, dict):
        return {k: _escape_recursive(v, k) for k, v in obj.items()}
    return obj


def resume_document_to_context(resume: ResumeDocument) -> dict:
    """Convert a validated ResumeDocument into an escaped Jinja2 context dict."""
    raw = resume.model_dump()
    return _escape_recursive(raw)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _make_latex_jinja_env(template_dir: Path) -> Environment:
    """
    LaTeX-safe Jinja2 environment. Default `{{ }}` / `{% %}` / `{# #}`
    delimiters collide with real LaTeX syntax (e.g. `{#1}` inside a
    \\newcommand definition looks like a Jinja comment opener), so we
    remap to delimiters that never appear in ordinary LaTeX source.
    """
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        undefined=StrictUndefined,  # fail loudly on a missing field rather
        # than silently rendering "None" into the .tex source
        trim_blocks=True,
        lstrip_blocks=True,
        block_start_string=r"\BLOCK{",
        block_end_string="}",
        variable_start_string=r"\VAR{",
        variable_end_string="}",
        comment_start_string=r"\#{",
        comment_end_string="}",
    )


def render_tex(resume: ResumeDocument, template_dir: Path = TEMPLATE_DIR) -> str:
    env = _make_latex_jinja_env(template_dir)
    template = env.get_template(TEMPLATE_NAME)
    context = resume_document_to_context(resume)
    return template.render(**context)


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def _find_compiler() -> tuple[str, list[str]]:
    """
    Returns (binary_name, base_args) for whichever offline LaTeX compiler is
    available. Prefers tectonic (self-contained, deterministic, no TeX Live
    tree required) and falls back to pdflatex if that's what's staged.
    """
    if shutil.which("tectonic"):
        return "tectonic", ["--keep-logs", "--synctex=none"]
    if shutil.which("pdflatex"):
        return "pdflatex", ["-interaction=nonstopmode", "-halt-on-error"]
    raise LatexCompilationError(
        "No LaTeX compiler found on PATH. Stage `tectonic` (preferred) or a "
        "portable TeX Live with `pdflatex` per MANIFEST_USB.md."
    )


def compile_pdf(tex_source: str, output_path: Path, timeout_s: int = 60) -> Path:
    """
    Compiles a rendered .tex string to a PDF at `output_path`, entirely
    offline. Runs in a temp directory so intermediate LaTeX build artifacts
    (.aux, .log, etc.) never pollute the real output directory.
    """
    compiler, base_args = _find_compiler()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_tex = Path(tmpdir) / "resume.tex"
        tmp_tex.write_text(tex_source, encoding="utf-8")

        cmd = [compiler, *base_args, str(tmp_tex.name)]
        logger.info(f"Compiling with: {' '.join(cmd)} (cwd={tmpdir})")

        try:
            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise LatexCompilationError(
                f"{compiler} timed out after {timeout_s}s — check for a "
                "malformed .tex (unbalanced braces are the usual culprit)."
            ) from exc

        tmp_pdf = Path(tmpdir) / "resume.pdf"
        if result.returncode != 0 or not tmp_pdf.exists():
            log_tail = (result.stdout[-2000:] + result.stderr[-2000:])
            raise LatexCompilationError(
                f"{compiler} failed (exit {result.returncode}). Tail of output:\n{log_tail}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp_pdf, output_path)
        logger.info(f"Compiled resume PDF -> {output_path}")
        return output_path


def generate_resume_pdf(
    resume: ResumeDocument,
    output_path: str | Path,
    template_dir: Path = TEMPLATE_DIR,
) -> Path:
    """End-to-end: ResumeDocument -> rendered .tex -> compiled PDF."""
    tex_source = render_tex(resume, template_dir=template_dir)
    return compile_pdf(tex_source, Path(output_path))


# ---------------------------------------------------------------------------
# ATS sanity check
# ---------------------------------------------------------------------------

def verify_ats_text_extractable(pdf_path: Path, expect_substring: str) -> bool:
    """
    Quick sanity check that text extracted back out of the compiled PDF
    (as an ATS parser would do) actually contains recognizable content,
    rather than e.g. glyph-mapped garbage from a font/encoding issue.
    Uses `pdftotext` (poppler-utils) if available, else pymupdf.
    """
    text = ""
    if shutil.which("pdftotext"):
        result = subprocess.run(
            ["pdftotext", str(pdf_path), "-"], capture_output=True, text=True
        )
        text = result.stdout
    else:
        import fitz  # pymupdf

        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text("text") for page in doc)
        doc.close()

    found = expect_substring.lower() in text.lower()
    if not found:
        logger.warning(
            f"ATS extractability check failed: '{expect_substring}' not found "
            f"in extracted PDF text — check font embedding / glyph mapping."
        )
    return found
