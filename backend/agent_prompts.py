"""
agent_prompts.py — Structured-output prompts for the three-phase LLM
pipeline, plus a thin client wrapper around a local Ollama server
(swap-in for llama-cpp-python's OpenAI-compatible server if preferred —
see LocalLLMClient docstring).

Every prompt below:
  1. States the JSON schema explicitly and demands ONLY JSON in the response
     (no markdown fences, no preamble) — small local models drift toward
     chatty preambles unless told not to, repeatedly, in the system prompt.
  2. Explicitly forbids inventing facts not present in the provided
     evidence. This is the primary anti-hallucination guardrail: it's a
     prompt-level constraint reinforced by schema validation in schemas.py
     (e.g. Phase1Output requires evidence_snippet_ids for every claimed
     skill) and, for extra safety, a post-hoc grounding check in
     orchestrator.py that verifies every cited snippet_id actually exists.
  3. Is deliberately narrow in scope per call — one JD requirement's worth
     of context at a time for Phase 1/2 — so a 7B-14B local model has a
     realistic chance of staying on-schema. Wide multi-requirement prompts
     are where small models start dropping fields or truncating JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from loguru import logger

from backend.schemas import Phase1Output, Phase2Output, ResumeDocument

# ---------------------------------------------------------------------------
# Local LLM client
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    model_name: str = "talentforge-qwen"  # Ollama model tag, see scripts/setup_usb.sh
    host: str = "http://localhost:11434"
    temperature: float = 0.1  # low temp: schema adherence > creativity here
    max_retries: int = 2
    timeout_s: int = 120


class LocalLLMClient:
    """
    Thin wrapper around the Ollama Python client. To swap to
    llama-cpp-python's server mode instead, point `host` at its
    OpenAI-compatible endpoint and swap the `_call_ollama` body for an
    equivalent `requests.post(f"{host}/v1/chat/completions", ...)` call —
    the JSON-mode contract (`format="json"`) is the only Ollama-specific bit.
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        import ollama

        client = ollama.Client(host=self.config.host)
        response = client.chat(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format="json",  # Ollama's constrained JSON-mode grammar
            options={"temperature": self.config.temperature},
        )
        return response["message"]["content"]

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 2):
            try:
                raw = self._call_ollama(system_prompt, user_prompt)
                return json.loads(raw)
            except (json.JSONDecodeError, KeyError, Exception) as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(f"LLM call attempt {attempt} failed: {exc}")
        raise RuntimeError(
            f"LLM failed to produce valid JSON after {self.config.max_retries + 1} attempts: {last_error}"
        )


# ---------------------------------------------------------------------------
# Phase 1: Skill Gap & Alignment Analysis
# ---------------------------------------------------------------------------

PHASE1_SYSTEM_PROMPT = """You are a precise technical recruiting analyst.
You will be given a job requirement and a numbered list of candidate
evidence snippets (real excerpts from the candidate's resume, project
write-ups, or transcripts).

Your task: decide which skills/requirements are genuinely evidenced by the
snippets, and which are not evidenced at all.

STRICT RULES:
- Respond with ONLY a single JSON object. No markdown code fences. No prose
  before or after the JSON.
- Every skill you list as "aligned" MUST cite the snippet_id(s) that actually
  support it. If you cannot cite a real snippet_id from the provided list,
  do not claim the skill is aligned.
- Do NOT invent skills, technologies, or accomplishments that are not
  written in the snippets. If the evidence is weak or absent, say so in
  missing_skills instead of stretching a citation to fit.
- confidence is a float 0.0-1.0 reflecting how directly the snippet
  demonstrates the skill (a passing mention is not the same as sustained
  ownership of it).

Output JSON schema:
{
  "jd_id": string,
  "aligned_skills": [
    {"skill": string, "evidence_snippet_ids": [string, ...], "confidence": float}
  ],
  "missing_skills": [string, ...],
  "notes": string
}
"""


def build_phase1_prompt(
    jd_id: str,
    requirement_text: str,
    evidence_snippets: list[tuple[str, str]],  # (snippet_id, raw_text)
) -> str:
    evidence_block = "\n".join(f"- [{sid}]: {text}" for sid, text in evidence_snippets)
    return f"""jd_id: {jd_id}

Job requirement to evaluate:
"{requirement_text}"

Candidate evidence snippets (only these may be cited):
{evidence_block}

Return the JSON object described in the system prompt, evaluating ONLY this
requirement against ONLY these snippets."""


def parse_phase1_response(raw_json: dict) -> Phase1Output:
    return Phase1Output.model_validate(raw_json)


# ---------------------------------------------------------------------------
# Phase 2: Action-Verb & Impact Rewriter (XYZ formula enforcement)
# ---------------------------------------------------------------------------

PHASE2_SYSTEM_PROMPT = """You are an expert resume writer who strictly
follows Google's XYZ formula for resume bullets:

  "Accomplished [X] as measured by [Y], by doing [Z]"

You will be given ONE raw candidate snippet (a real excerpt describing
something they did). Rewrite it as a single polished XYZ-formula bullet.

STRICT RULES:
- Respond with ONLY a single JSON object. No markdown fences, no prose.
- Use ONLY facts present in the source snippet. Do not add technologies,
  team sizes, percentages, or outcomes that are not stated or directly
  implied by the source text.
- If the source snippet contains NO quantifiable metric, do not invent one
  (e.g. do not fabricate "increased efficiency by 30%" from nothing). Set
  metric_is_estimated=true and write y_metric as an honest qualitative
  measure instead (e.g. "enabling faster onboarding for new contributors,
  per project README" ), never a fabricated number.
- Lead with a strong past-tense action verb (Architected, Reduced,
  Automated, Led, Optimized — chosen because it truly matches what the
  snippet describes, not generically).
- rendered_bullet must be one sentence, resume-ready, no first-person
  pronouns, no trailing period is fine either way but be consistent.

Output JSON schema:
{
  "jd_id": string,
  "bullets": [
    {
      "source_snippet_id": string,
      "x_accomplishment": string,
      "y_metric": string,
      "z_method": string,
      "rendered_bullet": string,
      "metric_is_estimated": boolean
    }
  ]
}
"""


def build_phase2_prompt(jd_id: str, snippet_id: str, snippet_text: str) -> str:
    return f"""jd_id: {jd_id}

Source snippet [{snippet_id}]:
"{snippet_text}"

Rewrite this single snippet as one XYZ-formula bullet, following every rule
in the system prompt. Return a "bullets" array with exactly one entry."""


def parse_phase2_response(raw_json: dict) -> Phase2Output:
    return Phase2Output.model_validate(raw_json)


# ---------------------------------------------------------------------------
# Phase 3: Resume JSON generation matching the LaTeX schema
# ---------------------------------------------------------------------------

PHASE3_SYSTEM_PROMPT = """You are assembling a final resume as structured
JSON. You will be given: contact info, a set of already-approved XYZ-format
bullets grouped by source (experience vs project), and education info.

STRICT RULES:
- Respond with ONLY a single JSON object matching the schema below. No
  markdown fences, no prose.
- Do NOT invent employers, dates, degrees, or bullets that were not
  provided to you. Your job here is ASSEMBLY and light copy-editing
  (fixing tense/grammar consistency across bullets), not content creation.
- Preserve every provided bullet's factual content exactly; you may only
  adjust phrasing for grammatical consistency, never add new claims.
- skills: dedupe and order by relevance to the job description provided.

Output JSON schema:
{
  "contact": {"full_name": string, "email": string, "phone": string|null,
              "location": string|null, "linkedin": string|null,
              "github": string|null, "website": string|null},
  "summary": string|null,
  "skills": [string, ...],
  "education": [{"institution": string, "degree": string,
                  "graduation_date": string, "gpa": string|null,
                  "relevant_coursework": [string, ...]}],
  "experience": [{"organization": string, "role": string,
                   "location": string|null, "start_date": string,
                   "end_date": string, "bullets": [string, ...]}],
  "projects": [{"name": string, "tech_stack": [string, ...],
                 "bullets": [string, ...], "link": string|null}]
}
"""


def build_phase3_prompt(assembly_context: dict) -> str:
    return (
        "Assemble the final resume JSON from this pre-approved content:\n\n"
        + json.dumps(assembly_context, indent=2)
        + "\n\nReturn only the JSON object described in the system prompt."
    )


def parse_phase3_response(raw_json: dict) -> ResumeDocument:
    return ResumeDocument.model_validate(raw_json)
