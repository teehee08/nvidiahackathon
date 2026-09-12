"""
Shared data contracts for the TalentForge AI pipeline.

Every LLM phase in agent_prompts.py is instructed to emit JSON that
validates against one of these models. Keeping the schema in one place
means the LaTeX template, the orchestrator, and the prompts can never
silently drift out of sync with each other.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
    PDF_RESUME = "pdf_resume"
    MARKDOWN = "markdown"
    README = "readme"
    TRANSCRIPT = "transcript"
    RAW_TEXT = "raw_text"


class CandidateSnippet(BaseModel):
    """One atomic, independently-matchable unit of candidate experience."""

    snippet_id: str
    source_type: SourceType
    source_name: str = Field(..., description="Filename or origin label")
    raw_text: str
    # Populated by matcher.py, not by the extractor
    embedding: Optional[list[float]] = None


class JobDescription(BaseModel):
    jd_id: str
    title: str
    company: Optional[str] = None
    raw_text: str
    requirements: list[str] = Field(
        default_factory=list,
        description="Extracted/segmented requirement bullets from the JD",
    )


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

class MatchScore(BaseModel):
    snippet_id: str
    requirement: str
    similarity: float = Field(..., ge=-1.0, le=1.0)


class MatchResult(BaseModel):
    jd_id: str
    ranked_snippets: list[MatchScore]
    covered_requirements: list[str]
    uncovered_requirements: list[str]


# ---------------------------------------------------------------------------
# Phase 1: Skill Gap & Alignment Analysis
# ---------------------------------------------------------------------------

class SkillAlignment(BaseModel):
    skill: str
    evidence_snippet_ids: list[str] = Field(
        default_factory=list,
        description="Snippet IDs that substantiate this skill. Empty list "
                     "means the model found NO grounding evidence.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)


class Phase1Output(BaseModel):
    jd_id: str
    aligned_skills: list[SkillAlignment]
    missing_skills: list[str] = Field(
        description="Required by the JD, not evidenced anywhere in source material"
    )
    notes: str = Field(
        default="",
        description="Terse rationale, no marketing language, no invented facts",
    )

    @field_validator("aligned_skills")
    @classmethod
    def no_ungrounded_claims(cls, v: list[SkillAlignment]) -> list[SkillAlignment]:
        for skill in v:
            if not skill.evidence_snippet_ids:
                raise ValueError(
                    f"Skill '{skill.skill}' has no evidence_snippet_ids — "
                    "an aligned skill must cite at least one source snippet."
                )
        return v


# ---------------------------------------------------------------------------
# Phase 2: Action-Verb & Impact Rewriter (XYZ formula)
# ---------------------------------------------------------------------------

class XYZBullet(BaseModel):
    """
    Google XYZ formula: "Accomplished [X] as measured by [Y], by doing [Z]"
    """

    source_snippet_id: str
    x_accomplishment: str
    y_metric: str = Field(
        description="Quantified or qualitative measurement. If the source "
                     "material has no number, this must say so honestly "
                     "(e.g. 'no quantified metric available in source') "
                     "rather than inventing one."
    )
    z_method: str
    rendered_bullet: str = Field(
        description="Final single-sentence bullet combining X/Y/Z"
    )
    metric_is_estimated: bool = Field(
        default=False,
        description="True if y_metric was not explicitly present in the "
                     "source snippet and had to be qualitatively described "
                     "instead of invented as a number.",
    )

    @field_validator("rendered_bullet")
    @classmethod
    def must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("rendered_bullet cannot be empty")
        return v


class Phase2Output(BaseModel):
    jd_id: str
    bullets: list[XYZBullet]


# ---------------------------------------------------------------------------
# Phase 3: Resume JSON matching the LaTeX schema
# ---------------------------------------------------------------------------

class ContactInfo(BaseModel):
    full_name: str
    email: str
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None


class EducationEntry(BaseModel):
    institution: str
    degree: str
    graduation_date: str
    gpa: Optional[str] = None
    relevant_coursework: list[str] = Field(default_factory=list)


class ExperienceEntry(BaseModel):
    organization: str
    role: str
    location: Optional[str] = None
    start_date: str
    end_date: str = "Present"
    bullets: list[str] = Field(default_factory=list)


class ProjectEntry(BaseModel):
    name: str
    tech_stack: list[str] = Field(default_factory=list)
    bullets: list[str] = Field(default_factory=list)
    link: Optional[str] = None


class ResumeDocument(BaseModel):
    """The exact shape latex_generator.py expects to render the template."""

    contact: ContactInfo
    summary: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
