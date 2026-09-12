"""
app_streamlit.py — Interactive local UI for TalentForge AI.

Calls the pipeline directly (in-process) rather than through the FastAPI
layer, so the demo works even if you didn't start uvicorn separately —
useful when you're juggling a live hackathon demo and want one fewer
process to babysit. Swap `run_full_pipeline` for an `httpx` call to the
FastAPI backend if you want the two decoupled.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.orchestrator import GroundingError, TalentForgePipeline  # noqa: E402
from backend.schemas import ContactInfo  # noqa: E402

st.set_page_config(page_title="TalentForge AI", page_icon="🛠️", layout="wide")
st.title("🛠️ TalentForge AI — Offline Resume Tailoring Agent")
st.caption("100% local inference · Dell Pro Max with GB10 · no cloud calls")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = TalentForgePipeline()

with st.sidebar:
    st.header("1. Candidate materials")
    uploaded_files = st.file_uploader(
        "Upload resume PDF, project READMEs (.md), or transcripts (.txt)",
        accept_multiple_files=True,
        type=["pdf", "md", "txt"],
    )

    st.header("2. Contact info")
    full_name = st.text_input("Full name")
    email = st.text_input("Email")
    phone = st.text_input("Phone (optional)")
    location = st.text_input("Location (optional)")
    linkedin = st.text_input("LinkedIn URL (optional)")
    github = st.text_input("GitHub URL (optional)")

    st.header("3. Education")
    institution = st.text_input("Institution", value="Cornell University")
    degree = st.text_input("Degree", value="B.S. Computer Science")
    grad_date = st.text_input("Graduation date", value="May 2027")
    gpa = st.text_input("GPA (optional)")

col1, col2 = st.columns(2)
with col1:
    jd_title = st.text_input("Target role title", placeholder="e.g. Machine Learning Engineer")
    company = st.text_input("Company (optional)")
with col2:
    summary = st.text_area("Professional summary (optional)", height=80)

jd_raw_text = st.text_area("Paste the full job description", height=220)

generate = st.button("🚀 Generate tailored resume", type="primary")

if generate:
    if not uploaded_files:
        st.error("Upload at least one candidate material file.")
    elif not jd_raw_text.strip():
        st.error("Paste a job description.")
    elif not (full_name and email):
        st.error("Full name and email are required.")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_paths = []
            for f in uploaded_files:
                dest = Path(tmpdir) / f.name
                dest.write_bytes(f.getbuffer())
                saved_paths.append(str(dest))

            contact = ContactInfo(
                full_name=full_name,
                email=email,
                phone=phone or None,
                location=location or None,
                linkedin=linkedin or None,
                github=github or None,
            )
            education = [
                {
                    "institution": institution,
                    "degree": degree,
                    "graduation_date": grad_date,
                    "gpa": gpa or None,
                    "relevant_coursework": [],
                }
            ]

            progress = st.status("Running pipeline...", expanded=True)
            try:
                progress.write("Ingesting candidate materials & building embedding index...")
                progress.write("Matching JD requirements against evidence...")
                progress.write("Phase 1: skill gap & alignment analysis (local LLM)...")
                progress.write("Phase 2: rewriting bullets into XYZ format (local LLM)...")
                progress.write("Phase 3: assembling resume JSON (local LLM)...")
                progress.write("Compiling LaTeX -> PDF offline...")

                result = st.session_state.pipeline.run_full_pipeline(
                    candidate_files=saved_paths,
                    jd_title=jd_title,
                    jd_raw_text=jd_raw_text,
                    contact=contact,
                    education=education,
                    company=company or None,
                    summary=summary or None,
                )
                progress.update(label="Done!", state="complete")

                st.success("Resume generated.")
                with open(result.pdf_path, "rb") as fh:
                    st.download_button(
                        "⬇️ Download tailored resume PDF",
                        data=fh.read(),
                        file_name="tailored_resume.pdf",
                        mime="application/pdf",
                    )

                with st.expander("Skill gap analysis (Phase 1 output)"):
                    st.json(json.loads(result.phase1.model_dump_json()))
                with st.expander("Missing skills to address before applying"):
                    st.write(result.phase1.missing_skills or "None detected.")
                with st.expander("Raw resume JSON (Phase 3 output)"):
                    st.json(json.loads(result.resume.model_dump_json()))

            except GroundingError as exc:
                progress.update(label="Grounding check failed", state="error")
                st.error(f"Rejected a hallucinated claim before it reached the resume: {exc}")
            except Exception as exc:  # noqa: BLE001
                progress.update(label="Pipeline error", state="error")
                st.exception(exc)
