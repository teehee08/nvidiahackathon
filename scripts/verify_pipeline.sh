#!/usr/bin/env bash
# verify_pipeline.sh — End-to-end offline smoke test. Turn off WiFi before
# running this if you want a true "will this survive venue conditions" test.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "== 1. Unit tests (no model/network dependency) =="
pytest tests/ -v

echo "== 2. Ollama reachable? =="
curl -s http://localhost:11434/api/tags | head -c 300 && echo || {
    echo "Ollama not reachable on :11434 — run 'ollama serve' first."; exit 1;
}

echo "== 3. Embedding model loads from local dir? =="
python3 - <<'PYEOF'
from backend.matcher import SemanticMatcher, MatcherConfig
m = SemanticMatcher(MatcherConfig())
m._load_model()
print("Embedding model loaded OK.")
PYEOF

echo "== 4. LaTeX compiler present? =="
if command -v tectonic &>/dev/null; then
    echo "tectonic found: $(tectonic --version)"
elif command -v pdflatex &>/dev/null; then
    echo "pdflatex found: $(pdflatex --version | head -n1)"
else
    echo "No LaTeX compiler found on PATH." && exit 1
fi

echo "== 5. Full pipeline dry run on fixture data =="
python3 - <<'PYEOF'
from backend.orchestrator import TalentForgePipeline
from backend.schemas import ContactInfo

pipeline = TalentForgePipeline()
result = pipeline.run_full_pipeline(
    candidate_files=["tests/fixtures/sample_resume.txt"],
    jd_title="Backend Software Engineer Intern",
    jd_raw_text=open("tests/fixtures/sample_jd.txt").read(),
    contact=ContactInfo(full_name="Test Candidate", email="test@example.com"),
    education=[{"institution": "Cornell University", "degree": "B.S. CS",
                "graduation_date": "May 2027", "gpa": None, "relevant_coursework": []}],
)
print(f"Pipeline succeeded. PDF at: {result.pdf_path}")
PYEOF

echo "== All checks passed. You're ready for eHub. =="
