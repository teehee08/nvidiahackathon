"""
test_matcher.py — Unit tests for SemanticMatcher.

Mocks SentenceTransformer entirely so these tests run instantly, offline,
without needing the actual bge-small-en-v1.5 weights present — useful for
iterating on matcher.py logic while the real model is still loading/copying
from the USB drive.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.extractor import extract_from_text
from backend.matcher import MatcherConfig, SemanticMatcher
from backend.schemas import SourceType


FIXTURES = Path(__file__).parent / "fixtures"


def _fake_encode(texts, **kwargs):
    """
    Deterministic fake embeddings: hash each string into a fixed-size
    pseudo-random vector, then L2-normalize, so cosine similarity behaves
    sensibly (identical text -> similarity 1.0) without a real model.
    """
    single = isinstance(texts, str)
    text_list = [texts] if single else texts
    vectors = []
    for t in text_list:
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        v = rng.normal(size=32).astype("float32")
        v = v / np.linalg.norm(v)
        vectors.append(v)
    arr = np.array(vectors, dtype="float32")
    return arr[0] if single else arr


@pytest.fixture
def sample_snippets():
    text = (FIXTURES / "sample_resume.txt").read_text()
    return extract_from_text(text, source_type=SourceType.MARKDOWN, source_name="sample_resume.txt")


@pytest.fixture
def mock_matcher():
    matcher = SemanticMatcher(MatcherConfig(model_dir=Path("fake/does/not/matter")))
    fake_model = MagicMock()
    fake_model.encode.side_effect = _fake_encode
    matcher._model = fake_model  # bypass _load_model()'s filesystem check
    return matcher


def test_extractor_produces_multiple_snippets(sample_snippets):
    assert len(sample_snippets) >= 3
    assert all(s.raw_text for s in sample_snippets)


def test_build_index_populates_embeddings(mock_matcher, sample_snippets):
    mock_matcher.build_index(sample_snippets)
    assert len(mock_matcher._indexed_snippets) == len(sample_snippets)
    assert all(s.embedding is not None for s in mock_matcher._indexed_snippets)


def test_match_requirement_returns_ranked_scores(mock_matcher, sample_snippets):
    mock_matcher.build_index(sample_snippets)
    scores = mock_matcher.match_requirement("Experience building REST APIs with FastAPI")
    assert len(scores) > 0
    # Scores should be sorted descending by similarity
    sims = [s.similarity for s in scores]
    assert sims == sorted(sims, reverse=True)


def test_match_all_splits_covered_and_uncovered(mock_matcher, sample_snippets):
    mock_matcher.build_index(sample_snippets)
    mock_matcher.config.coverage_threshold = -1.0  # force everything "covered"
    result = mock_matcher.match_all("jd_test", ["REST APIs", "Docker containers"])
    assert set(result.covered_requirements) == {"REST APIs", "Docker containers"}
    assert result.uncovered_requirements == []


def test_snippet_by_id_lookup(mock_matcher, sample_snippets):
    mock_matcher.build_index(sample_snippets)
    target = sample_snippets[0]
    found = mock_matcher.snippet_by_id(target.snippet_id)
    assert found is not None
    assert found.raw_text == target.raw_text


def test_build_index_rejects_empty_list(mock_matcher):
    with pytest.raises(ValueError):
        mock_matcher.build_index([])
