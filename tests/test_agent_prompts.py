"""
test_agent_prompts.py — Tests JSON schema parsing/validation for all three
phases, and the anti-hallucination guardrail in Phase1Output, WITHOUT
requiring a real Ollama server or model — LLM calls are mocked so these run
instantly while iterating on prompt/schema design.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from backend.agent_prompts import (
    LocalLLMClient,
    build_phase1_prompt,
    build_phase2_prompt,
    parse_phase1_response,
    parse_phase2_response,
    parse_phase3_response,
)


def test_phase1_valid_response_parses():
    raw = {
        "jd_id": "jd_1",
        "aligned_skills": [
            {"skill": "REST API development", "evidence_snippet_ids": ["snip_1"], "confidence": 0.9}
        ],
        "missing_skills": ["Kubernetes"],
        "notes": "Strong FastAPI evidence found.",
    }
    result = parse_phase1_response(raw)
    assert result.aligned_skills[0].skill == "REST API development"
    assert "Kubernetes" in result.missing_skills


def test_phase1_rejects_ungrounded_skill_claim():
    """A skill claimed as aligned with NO cited evidence must fail validation
    — this is the core anti-hallucination guardrail at the schema level."""
    raw = {
        "jd_id": "jd_1",
        "aligned_skills": [
            {"skill": "Kubernetes orchestration", "evidence_snippet_ids": [], "confidence": 0.8}
        ],
        "missing_skills": [],
        "notes": "",
    }
    with pytest.raises(ValidationError):
        parse_phase1_response(raw)


def test_phase2_valid_bullet_parses():
    raw = {
        "jd_id": "jd_1",
        "bullets": [
            {
                "source_snippet_id": "snip_1",
                "x_accomplishment": "Reduced manual stock-check time",
                "y_metric": "from 3 hours/week to 20 minutes/week",
                "z_method": "building a real-time inventory tracking system in Python and PostgreSQL",
                "rendered_bullet": "Reduced manual stock-check time from 3 hours/week to 20 "
                "minutes/week by building a real-time inventory tracking system in Python and PostgreSQL",
                "metric_is_estimated": False,
            }
        ],
    }
    result = parse_phase2_response(raw)
    assert len(result.bullets) == 1
    assert result.bullets[0].metric_is_estimated is False


def test_phase2_rejects_empty_rendered_bullet():
    raw = {
        "jd_id": "jd_1",
        "bullets": [
            {
                "source_snippet_id": "snip_1",
                "x_accomplishment": "Did something",
                "y_metric": "no quantified metric available in source",
                "z_method": "doing the thing",
                "rendered_bullet": "   ",
                "metric_is_estimated": True,
            }
        ],
    }
    with pytest.raises(ValidationError):
        parse_phase2_response(raw)


def test_phase3_valid_resume_parses():
    raw = {
        "contact": {"full_name": "Ada Lovelace", "email": "ada@example.com", "phone": None,
                     "location": None, "linkedin": None, "github": None, "website": None},
        "summary": None,
        "skills": ["Python"],
        "education": [],
        "experience": [],
        "projects": [],
    }
    result = parse_phase3_response(raw)
    assert result.contact.full_name == "Ada Lovelace"


def test_build_phase1_prompt_includes_all_evidence():
    prompt = build_phase1_prompt("jd_1", "REST APIs", [("snip_1", "Built a FastAPI service")])
    assert "snip_1" in prompt
    assert "Built a FastAPI service" in prompt
    assert "REST APIs" in prompt


def test_llm_client_retries_on_bad_json():
    client = LocalLLMClient()
    with patch.object(client, "_call_ollama", side_effect=["not json", '{"ok": true}']):
        result = client.generate_json("system", "user")
    assert result == {"ok": True}


def test_llm_client_raises_after_exhausting_retries():
    client = LocalLLMClient()
    client.config.max_retries = 1
    with patch.object(client, "_call_ollama", return_value="still not json"):
        with pytest.raises(RuntimeError):
            client.generate_json("system", "user")
