# TalentForge AI

Offline, multi-agent resume tailoring & skills-matching pipeline. Built for
100% local inference on a Dell Pro Max with GB10 (NVIDIA Grace Blackwell,
DGX OS, aarch64). No cloud API calls anywhere in the critical path.

## Architecture

```
 ┌─────────────┐      ┌──────────────┐      ┌────────────────────┐
 │  Streamlit   │ HTTP │   FastAPI     │      │  Local LLM (Ollama  │
 │  Frontend    │◄────►│   Backend     │◄────►│  / llama-cpp-python) │
 └─────────────┘      │  orchestrator │      └────────────────────┘
                       │      │        │
                       │      ▼        │      ┌────────────────────┐
                       │  extractor.py │      │  Embedding model    │
                       │  matcher.py   │◄────►│  (bge-small-en-v1.5) │
                       │  agent_prompts│      └────────────────────┘
                       │  latex_gen.py │
                       └──────┬────────┘
                              ▼
                     ┌─────────────────┐
                     │ tectonic (local) │──► resume.pdf
                     └─────────────────┘
```

Pipeline stages (see `backend/orchestrator.py`):

1. **Ingest** — `extractor.py` pulls raw text out of PDFs, Markdown, READMEs,
   transcripts.
2. **Match** — `matcher.py` embeds JD requirements and candidate snippets,
   scores fit with cosine similarity, surfaces gaps.
3. **Analyze (Phase 1)** — LLM call with a rigid JSON schema
   (`agent_prompts.py::PHASE1_SCHEMA`) identifies skill alignment and gaps,
   grounded only in retrieved snippets (no hallucinated skills).
4. **Rewrite (Phase 2)** — LLM call rewrites top-ranked bullets into strict
   Google XYZ format, constrained to facts present in the source snippet.
5. **Structure (Phase 3)** — LLM call emits a resume JSON object matching the
   LaTeX schema exactly.
6. **Render** — `latex_generator.py` fills a Jinja2-templated `.tex` file and
   shells out to `tectonic` (or `pdflatex`) to compile a PDF, fully offline.

## Repo layout

```
talentforge-ai/
├── backend/
│   ├── schemas.py            # Pydantic models shared across all phases
│   ├── extractor.py          # PDF / Markdown / text ingestion
│   ├── matcher.py            # Embedding-based JD <-> snippet retrieval
│   ├── agent_prompts.py      # Rigid JSON-schema prompts + LLM client wrapper
│   ├── latex_generator.py    # Jinja2 -> .tex -> tectonic -> PDF
│   ├── orchestrator.py       # Wires the phases together, FastAPI-callable
│   └── api.py                # FastAPI app exposing the pipeline
├── frontend/
│   └── app_streamlit.py      # Upload JD + materials, review, download PDF
├── models/                   # USB mount point — see MANIFEST_USB.md
│   └── README.md
├── templates/
│   └── resume_template.tex.jinja
├── tests/
│   ├── fixtures/
│   │   ├── sample_resume.txt
│   │   └── sample_jd.txt
│   ├── test_matcher.py
│   ├── test_latex_generator.py
│   └── test_agent_prompts.py
├── scripts/
│   ├── setup_usb.sh          # Copy USB payload into place, register Ollama model
│   └── verify_pipeline.sh    # End-to-end offline smoke test
├── requirements.txt
├── MANIFEST_USB.md
└── PREP_CHECKLIST.md
```

## Running it

```bash
# 1. Stand up the local model server (Ollama assumed here)
ollama serve &
ollama create talentforge-qwen -f ./models/llm/qwen2.5-14b/Modelfile

# 2. Backend
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload

# 3. Frontend
streamlit run frontend/app_streamlit.py
```

See `PREP_CHECKLIST.md` for the exact pre-hackathon verification sequence,
and `MANIFEST_USB.md` for everything to stage on the drive tonight.
