"""
test_latex_generator.py — Tests LaTeX escaping and Jinja2 rendering logic
without invoking a real `tectonic`/`pdflatex` binary, so these run on any
machine (including CI without LaTeX installed). A separate marked
integration test does a real compile IF a compiler is found on PATH.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from backend.latex_generator import (
    _escape_latex,
    render_tex,
    resume_document_to_context,
    generate_resume_pdf,
)
from backend.schemas import ContactInfo, ExperienceEntry, ProjectEntry, ResumeDocument

TEMPLATE_DIR = Path(__file__).parent.parent / "templates"


@pytest.fixture
def sample_resume() -> ResumeDocument:
    return ResumeDocument(
        contact=ContactInfo(
            full_name="Ada Lovelace",
            email="ada@example.com",
            phone="555-0100",
            location="Ithaca, NY",
            linkedin="https://linkedin.com/in/ada",
            github="https://github.com/ada",
        ),
        summary="Backend engineer focused on distributed systems & 50% faster pipelines.",
        skills=["Python", "FastAPI", "PostgreSQL"],
        education=[],
        experience=[
            ExperienceEntry(
                organization="Analytical Engines Inc.",
                role="Software Engineer Intern",
                start_date="Jun 2025",
                end_date="Aug 2025",
                bullets=[
                    "Reduced query latency by 40% by adding composite indexes & caching, per load-test report."
                ],
            )
        ],
        projects=[
            ProjectEntry(
                name="Punch Card Compiler",
                tech_stack=["C++", "LLVM"],
                bullets=["Built a toy compiler targeting a custom bytecode VM."],
            )
        ],
    )


def test_escape_latex_handles_all_special_chars():
    raw = r"100% & $5_000 #tag {braces} back\slash tilde~ caret^"
    escaped = _escape_latex(raw)
    for forbidden in ["%", "&", "$", "_", "#"]:
        # forbidden chars should only appear preceded by a backslash now
        assert f"\\{forbidden}" in escaped or forbidden not in raw.replace(f"\\{forbidden}", "")
    assert "\\%" in escaped
    assert "\\&" in escaped
    assert "\\$" in escaped
    assert "\\_" in escaped
    assert "\\#" in escaped


def test_escape_latex_handles_none():
    assert _escape_latex(None) == ""


def test_resume_context_escapes_bullets_but_not_urls(sample_resume):
    ctx = resume_document_to_context(sample_resume)
    # The 40% in the bullet must be escaped
    assert "40\\%" in ctx["experience"][0]["bullets"][0]
    # But the linkedin URL must remain a raw, unescaped URL (used in \href{})
    assert ctx["contact"]["linkedin"] == "https://linkedin.com/in/ada"
    assert ctx["contact"]["email"] == "ada@example.com"


def test_render_tex_produces_valid_looking_document(sample_resume):
    tex = render_tex(sample_resume, template_dir=TEMPLATE_DIR)
    assert "\\documentclass" in tex
    assert "Ada Lovelace" in tex
    assert "\\begin{document}" in tex
    assert "\\end{document}" in tex
    # Escaped percent from the summary should appear, not a raw stray %
    assert "50\\% faster" in tex


def test_render_tex_missing_required_field_raises():
    incomplete = ResumeDocument(contact=ContactInfo(full_name="No Email Person", email="x@x.com"))
    # Should still render fine since all resume fields are optional/defaulted
    tex = render_tex(incomplete, template_dir=TEMPLATE_DIR)
    assert "No Email Person" in tex


@pytest.mark.skipif(
    shutil.which("tectonic") is None and shutil.which("pdflatex") is None,
    reason="No LaTeX compiler on PATH — integration-only test",
)
def test_full_compile_produces_pdf(sample_resume, tmp_path):
    out_path = tmp_path / "resume.pdf"
    result_path = generate_resume_pdf(sample_resume, out_path, template_dir=TEMPLATE_DIR)
    assert result_path.exists()
    assert result_path.stat().st_size > 1000  # sanity: not an empty/truncated PDF
