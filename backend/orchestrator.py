"""
orchestrator.py — Ties extractor -> matcher -> (Phase 1/2/3 LLM calls) ->
latex_generator into a single callable pipeline.

Also implements the post-hoc grounding check mentioned in
agent_prompts.py: every snippet_id the LLM cites as evidence is verified to
actually exist in the indexed snippet pool before we trust it. This is a
cheap, deterministic backstop against hallucinated citations that no amount
of prompt engineering can fully guarantee against on a small local model.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from backend.agent_prompts import (
    LLMConfig,
    LocalLLMClient,
    PHASE1_SYSTEM_PROMPT,
    PHASE2_SYSTEM_PROMPT,
    PHASE3_SYSTEM_PROMPT,
    build_phase1_prompt,
    build_phase2_prompt,
    build_phase3_prompt,
    parse_phase1_response,
    parse_phase2_response,
    parse_phase3_response,
)
from backend.extractor import extract_from_file, extract_job_description
from backend.latex_generator import generate_resume_pdf
from backend.matcher import MatcherConfig, SemanticMatcher
from backend.schemas import (
    CandidateSnippet,
    ContactInfo,
    Phase1Output,
    Phase2Output,
    ResumeDocument,
    SourceType,
)


class GroundingError(RuntimeError):
    """Raised when the LLM cites a snippet_id that doesn't exist — a hard
    signal of hallucination that must not silently pass through to the
    final resume."""


@dataclass
class PipelineResult:
    jd_id: str
    phase1: Phase1Output
    phase2: Phase2Output
    resume: ResumeDocument
    pdf_path: Path


class TalentForgePipeline:
    def __init__(
        self,
        matcher: SemanticMatcher | None = None,
        llm: LocalLLMClient | None = None,
    ):
        self.matcher = matcher or SemanticMatcher(MatcherConfig())
        self.llm = llm or LocalLLMClient(LLMConfig())

    # -- Step 0: ingestion ---------------------------------------------

    def ingest_candidate_files(self, file_paths: list[str]) -> list[CandidateSnippet]:
        snippets: list[CandidateSnippet] = []
        for fp in file_paths:
            snippets.extend(extract_from_file(fp))
        if not snippets:
            raise ValueError("No candidate snippets extracted from provided files")
        self.matcher.build_index(snippets)
        return snippets

    # -- Step 1: matching -------------------------------------------------

    def match_jd(self, jd_id: str, jd_title: str, jd_raw_text: str, company: str | None = None):
        jd_draft = extract_job_description(jd_raw_text, title=jd_title, company=company)
        match_result = self.matcher.match_all(jd_id, jd_draft.requirements)
        return jd_draft, match_result

    # -- Step 2: Phase 1 (per requirement) --------------------------------

    def run_phase1(self, jd_id: str, requirements: list[str]) -> Phase1Output:
        aligned_skills = []
        missing_skills: list[str] = []

        for req in requirements:
            top_matches = self.matcher.match_requirement(req)
            evidence = [
                (m.snippet_id, self.matcher.snippet_by_id(m.snippet_id).raw_text)
                for m in top_matches
                if self.matcher.snippet_by_id(m.snippet_id) is not None
            ]
            if not evidence:
                missing_skills.append(req)
                continue

            prompt = build_phase1_prompt(jd_id, req, evidence)
            raw = self.llm.generate_json(PHASE1_SYSTEM_PROMPT, prompt)
            result = parse_phase1_response(raw)

            self._verify_grounding(result)

            aligned_skills.extend(result.aligned_skills)
            missing_skills.extend(result.missing_skills)

        return Phase1Output(
            jd_id=jd_id,
            aligned_skills=aligned_skills,
            missing_skills=sorted(set(missing_skills)),
            notes="Aggregated across per-requirement Phase 1 calls.",
        )

    def _verify_grounding(self, phase1: Phase1Output) -> None:
        """Hard-fail if the LLM cited a snippet_id that doesn't actually exist."""
        for skill in phase1.aligned_skills:
            for sid in skill.evidence_snippet_ids:
                if self.matcher.snippet_by_id(sid) is None:
                    raise GroundingError(
                        f"LLM cited nonexistent snippet_id '{sid}' for skill "
                        f"'{skill.skill}' — rejecting this claim to avoid a "
                        f"hallucinated credential reaching the resume."
                    )

    # -- Step 3: Phase 2 (rewrite top snippets into XYZ bullets) ----------

    def run_phase2(self, jd_id: str, phase1: Phase1Output) -> Phase2Output:
        # Only rewrite snippets that survived Phase 1 grounding — i.e. were
        # actually cited as evidence for a real, aligned skill.
        snippet_ids = sorted({sid for s in phase1.aligned_skills for sid in s.evidence_snippet_ids})

        bullets = []
        for sid in snippet_ids:
            snippet = self.matcher.snippet_by_id(sid)
            if snippet is None:
                continue
            prompt = build_phase2_prompt(jd_id, sid, snippet.raw_text)
            raw = self.llm.generate_json(PHASE2_SYSTEM_PROMPT, prompt)
            result = parse_phase2_response(raw)
            bullets.extend(result.bullets)

        return Phase2Output(jd_id=jd_id, bullets=bullets)

    # -- Step 4: Phase 3 (assemble final resume JSON) ---------------------

    def run_phase3(
        self,
        contact: ContactInfo,
        phase2: Phase2Output,
        education: list[dict],
        skills: list[str],
        summary: str | None = None,
    ) -> ResumeDocument:
        assembly_context = {
            "contact": contact.model_dump(),
            "summary": summary,
            "skills": skills,
            "education": education,
            "approved_bullets": [b.model_dump() for b in phase2.bullets],
        }
        prompt = build_phase3_prompt(assembly_context)
        raw = self.llm.generate_json(PHASE3_SYSTEM_PROMPT, prompt)
        return parse_phase3_response(raw)

    # -- Step 5: render PDF ------------------------------------------------

    def run_latex(self, resume: ResumeDocument, output_dir: str | Path = "output") -> Path:
        out_path = Path(output_dir) / f"resume_{uuid.uuid4().hex[:8]}.pdf"
        return generate_resume_pdf(resume, out_path)

    # -- Full pipeline ------------------------------------------------------

    def run_full_pipeline(
        self,
        candidate_files: list[str],
        jd_title: str,
        jd_raw_text: str,
        contact: ContactInfo,
        education: list[dict],
        company: str | None = None,
        summary: str | None = None,
    ) -> PipelineResult:
        jd_id = f"jd_{uuid.uuid4().hex[:8]}"

        logger.info("Step 0: ingesting candidate files")
        self.ingest_candidate_files(candidate_files)

        logger.info("Step 1: matching JD requirements")
        jd_draft, match_result = self.match_jd(jd_id, jd_title, jd_raw_text, company)

        logger.info("Step 2: Phase 1 skill gap analysis")
        phase1 = self.run_phase1(jd_id, jd_draft.requirements)

        logger.info("Step 3: Phase 2 XYZ bullet rewriting")
        phase2 = self.run_phase2(jd_id, phase1)

        logger.info("Step 4: Phase 3 resume assembly")
        # Skills ordered by how many aligned_skills entries reference them,
        # descending — a simple relevance proxy without another LLM call.
        skills = [s.skill for s in sorted(phase1.aligned_skills, key=lambda s: -s.confidence)]
        resume = self.run_phase3(contact, phase2, education, skills, summary)

        logger.info("Step 5: rendering LaTeX -> PDF")
        pdf_path = self.run_latex(resume)

        return PipelineResult(jd_id=jd_id, phase1=phase1, phase2=phase2, resume=resume, pdf_path=pdf_path)
