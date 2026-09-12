"""
matcher.py — Local embedding-based semantic matching between JD requirements
and candidate snippets.

Uses sentence-transformers (BAAI/bge-small-en-v1.5 by default) loaded fully
offline from a local model directory, with an in-memory FAISS index for
similarity search. No network calls at runtime — `local_files_only=True` is
enforced so a misconfigured environment fails loudly instead of silently
trying (and stalling on) a HuggingFace Hub lookup over weak venue WiFi.

This module is intentionally LLM-agnostic: it does pure vector math. The
LLM (agent_prompts.py) is only invoked afterward, on the *already
narrowed-down* top-K snippets per requirement, which keeps prompts short
and keeps the pipeline fast on local hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from backend.schemas import CandidateSnippet, MatchResult, MatchScore

try:
    import faiss

    _HAS_FAISS = True
except ImportError:  # pragma: no cover
    _HAS_FAISS = False

DEFAULT_MODEL_DIR = Path("models/embeddings/bge-small-en-v1.5")

# bge models are trained with an asymmetric instruction prefix for queries.
# Applying it correctly meaningfully improves retrieval quality; skipping it
# is a common silent-degradation bug.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class MatcherConfig:
    model_dir: Path = DEFAULT_MODEL_DIR
    top_k_per_requirement: int = 5
    coverage_threshold: float = 0.45  # cosine sim below this = "not covered"
    device: str = "cpu"  # GB10 CPU embedding is plenty fast for bge-small


class SemanticMatcher:
    def __init__(self, config: MatcherConfig | None = None):
        self.config = config or MatcherConfig()
        self._model = None  # lazy-loaded
        self._index = None
        self._indexed_snippets: list[CandidateSnippet] = []

    # -- model loading ------------------------------------------------------

    def _load_model(self):
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        if not self.config.model_dir.exists():
            raise FileNotFoundError(
                f"Embedding model not found at {self.config.model_dir}. "
                "Copy it from the USB drive to models/embeddings/ first "
                "(see MANIFEST_USB.md)."
            )

        logger.info(f"Loading embedding model from {self.config.model_dir} (offline)")
        self._model = SentenceTransformer(
            str(self.config.model_dir),
            device=self.config.device,
            local_files_only=True,
        )
        return self._model

    # -- indexing -------------------------------------------------------

    def build_index(self, snippets: list[CandidateSnippet]) -> None:
        """Embed all candidate snippets and build a searchable index."""
        if not snippets:
            raise ValueError("Cannot build an index from zero snippets")

        model = self._load_model()
        texts = [s.raw_text for s in snippets]
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,  # so inner product == cosine similarity
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")

        for snippet, emb in zip(snippets, embeddings):
            snippet.embedding = emb.tolist()

        self._indexed_snippets = snippets

        if _HAS_FAISS:
            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine
            self._index.add(embeddings)
        else:
            logger.warning("faiss not available, falling back to numpy brute-force search")
            self._index = embeddings  # plain ndarray fallback

        logger.info(f"Indexed {len(snippets)} snippets ({'faiss' if _HAS_FAISS else 'numpy'} backend)")

    # -- search -----------------------------------------------------------

    def _search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self._index is None:
            raise RuntimeError("Call build_index() before searching")

        if _HAS_FAISS:
            scores, idxs = self._index.search(query_embedding.reshape(1, -1), top_k)
            return list(zip(idxs[0].tolist(), scores[0].tolist()))

        # numpy fallback: brute-force cosine via dot product on normalized vectors
        sims = self._index @ query_embedding
        top_idx = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in top_idx]

    def match_requirement(self, requirement_text: str, top_k: int | None = None) -> list[MatchScore]:
        model = self._load_model()
        top_k = top_k or self.config.top_k_per_requirement

        query_emb = model.encode(
            _BGE_QUERY_PREFIX + requirement_text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        results = self._search(query_emb, top_k)
        return [
            MatchScore(
                snippet_id=self._indexed_snippets[idx].snippet_id,
                requirement=requirement_text,
                similarity=round(score, 4),
            )
            for idx, score in results
            if idx < len(self._indexed_snippets)
        ]

    def match_all(self, jd_id: str, requirements: list[str]) -> MatchResult:
        """Score every JD requirement against the indexed snippet pool."""
        all_scores: list[MatchScore] = []
        covered, uncovered = [], []

        for req in requirements:
            scores = self.match_requirement(req)
            all_scores.extend(scores)

            best = max((s.similarity for s in scores), default=-1.0)
            if best >= self.config.coverage_threshold:
                covered.append(req)
            else:
                uncovered.append(req)

        return MatchResult(
            jd_id=jd_id,
            ranked_snippets=all_scores,
            covered_requirements=covered,
            uncovered_requirements=uncovered,
        )

    def snippet_by_id(self, snippet_id: str) -> CandidateSnippet | None:
        return next((s for s in self._indexed_snippets if s.snippet_id == snippet_id), None)
