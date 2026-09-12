"""Offline, chat-oriented Streamlit experience for TalentForge AI."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.orchestrator import GroundingError, TalentForgePipeline  # noqa: E402
from backend.schemas import ContactInfo  # noqa: E402
from frontend.theme import (  # noqa: E402
    coverage_pct,
    html,
    inject_global_css,
    render_dossier_header,
    render_dossier_message,
    render_hero,
    render_pdf_download,
)


st.set_page_config(page_title="TalentForge AI", page_icon="A", layout="centered", initial_sidebar_state="collapsed")
inject_global_css()

if "screen" not in st.session_state:
    st.session_state.screen = "welcome"
if "stage" not in st.session_state:
    st.session_state.stage = "form"
if "pipeline" not in st.session_state:
    st.session_state.pipeline = TalentForgePipeline()
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "CHAT"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def render_intake_form() -> None:
    html('<div class="tf-chat-card"><p>Hey, I\'m TalentForge AI. Upload your resume, coursework, projects, or notes and paste a target job description. I\'ll turn the strongest evidence into a focused application package.</p></div>')
    with st.form("resume_intake"):
        uploaded_files = st.file_uploader("Add your source materials", accept_multiple_files=True, type=["pdf", "md", "txt"])
        jd_title = st.text_input("Target role", placeholder="e.g. Associate Product Manager")
        company = st.text_input("Company", placeholder="e.g. NVIDIA")
        jd_raw_text = st.text_area("Target job description", height=170, placeholder="Paste the full job description here...")
        with st.expander("Candidate details"):
            full_name = st.text_input("Full name")
            email = st.text_input("Email")
            phone = st.text_input("Phone (optional)")
            location = st.text_input("Location (optional)")
            linkedin = st.text_input("LinkedIn URL (optional)")
            github = st.text_input("GitHub URL (optional)")
            institution = st.text_input("Institution", value="Cornell University")
            degree = st.text_input("Degree", value="B.S. Computer Science")
            grad_date = st.text_input("Graduation date", value="May 2027")
            gpa = st.text_input("GPA (optional)")
            summary = st.text_area("Professional summary (optional)", height=80)
        generate = st.form_submit_button("✣ Synthesize my dossier", type="primary", use_container_width=True)

    if not generate:
        return
    if not uploaded_files:
        st.error("Add at least one resume, project, or notes file.")
        return
    if not jd_raw_text.strip() or not jd_title.strip():
        st.error("Add a target role and paste its job description.")
        return
    if not full_name.strip() or not email.strip():
        st.error("Full name and email are required for the PDF.")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_paths = []
        uploaded_metadata = []
        for uploaded in uploaded_files:
            destination = Path(tmpdir) / uploaded.name
            destination.write_bytes(uploaded.getbuffer())
            saved_paths.append(str(destination))
            uploaded_metadata.append({"name": uploaded.name, "size": uploaded.size, "type": uploaded.type})

        contact = ContactInfo(full_name=full_name, email=email, phone=phone or None, location=location or None, linkedin=linkedin or None, github=github or None)
        education = [{"institution": institution, "degree": degree, "graduation_date": grad_date, "gpa": gpa or None, "relevant_coursework": []}]
        progress = st.status("TalentForge AI is synthesizing your dossier...", expanded=True)
        try:
            progress.write("Ingesting source evidence and matching the target role...")
            pipeline = st.session_state.pipeline
            # Capture the grounded raw-document baseline before the full run.
            snippets = pipeline.ingest_candidate_files(saved_paths)
            _, baseline_match = pipeline.match_jd(f"baseline_{jd_title}", jd_title, jd_raw_text, company or None)
            result = pipeline.run_full_pipeline(candidate_files=saved_paths, jd_title=jd_title, jd_raw_text=jd_raw_text, contact=contact, education=education, company=company or None, summary=summary or None)
            progress.update(label="Dossier synthesized", state="complete")
            st.session_state.result = result
            st.session_state.match_result = baseline_match
            st.session_state.snippets = snippets
            st.session_state.uploaded_metadata = uploaded_metadata
            st.session_state.jd_title = jd_title
            st.session_state.jd_raw_text = jd_raw_text
            st.session_state.stage = "dossier"
            st.session_state.chat_history = [
                {"role": "user", "content": jd_raw_text, "files": uploaded_metadata},
                {"role": "assistant", "content": "dossier"},
            ]
            st.rerun()
        except GroundingError as exc:
            progress.update(label="Grounding check failed", state="error")
            st.error(f"TalentForge rejected an unsupported claim: {exc}")
        except Exception as exc:  # noqa: BLE001
            progress.update(label="Pipeline error", state="error")
            st.exception(exc)


def render_user_message(message: dict) -> None:
    st.markdown(message["content"])
    for file_info in message.get("files", []):
        size = f"{file_info['size'] / 1024:.1f} KB" if file_info.get("size") else "size unavailable"
        html(f'<span class="tf-chip">▧ {file_info["name"]} <small>{size}</small></span>')


def render_ats_tab(match_result) -> None:
    score = coverage_pct(match_result)
    html(f'<div class="tf-section-card"><div class="tf-section-label">ATS fit calibration</div><div class="tf-metric-value">{score}%</div><div class="tf-metric-bar"><span style="width:{score}%"></span></div></div>')
    left, right = st.columns(2)
    with left:
        st.markdown("**Covered requirements**")
        for requirement in match_result.covered_requirements:
            st.markdown(f"- {requirement}")
    with right:
        st.markdown("**Uncovered requirements**")
        for requirement in match_result.uncovered_requirements:
            st.markdown(f"- {requirement}")


def render_vault_tab() -> None:
    snippets_by_file = {}
    for snippet in st.session_state.get("snippets", []):
        snippets_by_file[snippet.source_name] = snippets_by_file.get(snippet.source_name, 0) + 1
    for file_info in st.session_state.get("uploaded_metadata", []):
        name = file_info["name"]
        html(f'<div class="tf-file-row"><span>▧ {name}<br><span class="tf-muted">{file_info.get("type") or "file"}</span></span><strong>{snippets_by_file.get(name, 0)} snippets</strong></div>')


def render_document_tab(result) -> None:
    html(f'<div class="tf-section-card"><div class="tf-section-label">Generated document</div><strong>{result.pdf_path.name}</strong><p class="tf-muted">PDF size is {result.pdf_path.stat().st_size / 1024:.1f} KB.</p></div>')
    # TODO: Add an inline PDF viewer only if a compatible component is staged in the offline USB wheels cache.
    render_pdf_download(result)


def render_dossier() -> None:
    result = st.session_state.result
    match_result = st.session_state.match_result
    pipeline = st.session_state.pipeline
    render_dossier_header(st.session_state.jd_title, coverage_pct(match_result))

    if st.session_state.active_tab == "CHAT":
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    render_user_message(message)
                else:
                    render_dossier_message(result, match_result, pipeline)
        with st.chat_message("assistant"):
            missing = result.phase1.missing_skills[0] if result.phase1.missing_skills else "technical depth"
            st.markdown(f"Would you like to refine the emphasis on **{missing}** or another section?")
        html('<div class="tf-section-label">Suggested prompts</div>')
        prompt_columns = st.columns(3)
        for column, prompt in zip(prompt_columns, ("Emphasize execution", "Strengthen technical depth", "Highlight user impact")):
            with column:
                if st.button(prompt, key=f"prompt_{prompt}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    st.rerun()
        left, right = st.columns(2)
        with left:
            render_pdf_download(result)
        with right:
            if st.button("Fine-tune Sections", use_container_width=True):
                st.session_state.focus_chat = True
                st.rerun()
        user_message = st.chat_input("Ask TalentForge to refine or rewrite...")
        if user_message:
            # TODO: Feed follow-up text into a future Phase 2/3 rerun.
            st.session_state.chat_history.append({"role": "user", "content": user_message})
            st.rerun()
    elif st.session_state.active_tab == "DOCUMENT":
        render_document_tab(result)
    elif st.session_state.active_tab == "ATS SCORE":
        render_ats_tab(match_result)
    else:
        render_vault_tab()

    html('<div class="tf-tabbar">')
    tab_columns = st.columns(4)
    for column, tab in zip(tab_columns, ("CHAT", "DOCUMENT", "ATS SCORE", "VAULT")):
        with column:
            if st.button(tab, key=f"tab_{tab}", use_container_width=True):
                st.session_state.active_tab = tab
                st.rerun()
    html('</div>')
    if st.button("← Back", key="back_to_landing"):
        st.session_state.screen = "welcome"
        st.session_state.stage = "form"
        st.rerun()


if st.session_state.screen == "welcome":
    render_hero()
    if st.button("Build My Resume  →", type="primary", use_container_width=True):
        st.session_state.screen = "workspace"
        st.session_state.stage = "form"
        st.rerun()
    if st.button("Existing user?  Sign In", use_container_width=True):
        st.session_state.screen = "workspace"
        st.session_state.stage = "form"
        st.rerun()
    html('<div class="tf-legal">By continuing you agree to Amelia\'s Terms of Service &amp; Privacy Policy.</div>')
elif st.session_state.stage == "dossier" and "result" in st.session_state:
    render_dossier()
else:
    render_intake_form()
