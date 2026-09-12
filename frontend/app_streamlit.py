"""Offline, chat-oriented Streamlit experience for TalentForge AI."""

from __future__ import annotations

import base64
import sys
import tempfile
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.orchestrator import GroundingError, TalentForgePipeline  # noqa: E402
from backend.extractor import extract_from_file  # noqa: E402
from backend.matcher import MatcherConfig, SemanticMatcher  # noqa: E402
import backend.matcher as matcher_module  # noqa: E402
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
    st.session_state.screen = "login"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "pipeline" not in st.session_state:
    # Use the matcher's tested NumPy fallback when a platform's FAISS build is incomplete.
    if matcher_module._HAS_FAISS and not hasattr(matcher_module.faiss, "IndexFlatIP"):
        matcher_module._HAS_FAISS = False
    embedding_dir = Path("models/embeddings/bge-small-en-v1.5")
    staged_embedding_dir = Path("usb/models/embeddings/bge-small-en-v1.5")
    if not embedding_dir.exists() and staged_embedding_dir.exists():
        embedding_dir = staged_embedding_dir
    st.session_state.pipeline = TalentForgePipeline(
        matcher=SemanticMatcher(MatcherConfig(model_dir=embedding_dir))
    )
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "CHAT"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def clear_sensitive_state() -> None:
    for key in ("result", "match_result", "snippets", "uploaded_metadata", "jd_title", "jd_raw_text", "chat_history"):
        st.session_state.pop(key, None)
    st.session_state.logged_in = False
    st.session_state.screen = "login"


def render_login() -> None:
    html('<div class="tf-hero"><div class="eyebrow">• PRIVATE WORKSPACE</div><div class="tf-hero-mark">A</div><h1><span>TalentForge.</span></h1><h2>Your local resume strategist</h2><p>Enter a password to open your private, offline resume workspace.</p></div>')
    with st.form("login_form"):
        password = st.text_input("Workspace password", type="password", placeholder="Enter any password to continue")
        submitted = st.form_submit_button("Unlock workspace  →", type="primary", use_container_width=True)
    if submitted:
        if not password.strip():
            st.error("Enter a password to continue.")
            return
        st.session_state.logged_in = True
        st.session_state.screen = "home"
        st.rerun()


def render_home() -> None:
    render_hero()
    if st.button("Build My Resume  →", type="primary", use_container_width=True):
        st.session_state.screen = "intake"
        st.rerun()
    html('<div class="tf-legal">Your files stay in this local workspace during processing.</div>')
    if st.button("Log out", key="home_logout", use_container_width=True):
        clear_sensitive_state()
        st.rerun()


class _JobPageParser(HTMLParser):
    """Collect page title and readable text without adding a scraping dependency."""

    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        cleaned = " ".join(data.split())
        if not cleaned or self._ignored_depth:
            return
        if self._in_title:
            self.title_parts.append(cleaned)
        else:
            self.text_parts.append(cleaned)


def fetch_job_page(url: str) -> tuple[str, str]:
    """Fetch a public job page and return (derived title, readable text)."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a complete http:// or https:// job posting URL.")
    request = Request(url, headers={"User-Agent": "TalentForge AI local resume assistant"})
    with urlopen(request, timeout=15) as response:  # noqa: S310 - user-selected public URL
        payload = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
    parser = _JobPageParser()
    parser.feed(payload.decode(charset, errors="ignore"))
    page_text = "\n".join(parser.text_parts)
    page_title = " ".join(parser.title_parts).strip()
    if len(page_text) < 80:
        raise ValueError("That page did not expose enough readable job-description text. Try uploading the JD or pasting it instead.")
    return derive_role_title(page_title or urlparse(url).path), page_text


def derive_role_title(source_name: str) -> str:
    """Turn a filename or page title into a readable role label."""
    title = re.sub(r"\.(pdf|md|txt|html?)$", "", source_name, flags=re.IGNORECASE)
    title = re.sub(r"[_-]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip(" /|")
    return title or "Target role"


def ollama_is_available() -> bool:
    """Return whether the local Ollama HTTP service is reachable."""
    try:
        with urlopen("http://localhost:11434/api/tags", timeout=2):  # noqa: S310 - fixed local URL
            return True
    except (OSError, URLError):
        return False


def render_intake_form() -> None:
    html('<div class="tf-brandbar"><div><div class="tf-brand"><span class="tf-logo-mark">A</span><span>TalentForge AI</span><span class="tf-pill-badge">PRO</span></div><div class="tf-brand-subtitle">Resume Strategist</div></div><div class="tf-icon-row">● LOCAL</div></div>')
    if st.button("Log out", key="intake_logout", use_container_width=True):
        clear_sensitive_state()
        st.rerun()
    html('<div class="tf-section-card"><div class="tf-section-label">Build your dossier</div><h2>Bring the evidence. We\'ll shape the story.</h2><p class="tf-muted">Start with your resume and the job you want. Supporting files are welcome when they add useful context.</p></div>')
    with st.form("resume_intake"):
        st.markdown("### 1. Upload your resume")
        st.caption("Upload your current resume first. PDF, Markdown, and text files are supported.")
        uploaded_files = st.file_uploader("Resume and source materials", accept_multiple_files=True, type=["pdf", "md", "txt"])
        st.markdown("### 2. Add the job description")
        st.caption("Upload the JD document or paste a public job-posting URL. TalentForge will identify the role title for you.")
        jd_file = st.file_uploader("Job description document", type=["pdf", "md", "txt"], accept_multiple_files=False)
        jd_url = st.text_input("Public job website URL (optional)", placeholder="https://company.com/jobs/associate-product-manager")
        st.caption("If the website blocks automated reading, upload the JD document or use the paste fallback below.")
        jd_pasted_text = st.text_area("Paste the job description (optional fallback)", height=150, placeholder="Paste the full job description here if you cannot upload it or the page cannot be read...")
        company = st.text_input("Company (optional)", placeholder="e.g. NVIDIA")
        with st.expander("3. Candidate details for the PDF"):
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
        generate = st.form_submit_button("✣ Generate my tailored resume", type="primary", use_container_width=True)

    if not generate:
        return
    if not uploaded_files:
        st.error("Upload at least one resume, project, or supporting-material file.")
        return
    if not jd_file and not jd_url.strip() and not jd_pasted_text.strip():
        st.error("Upload a job description, enter a public job URL, or paste the JD text.")
        return
    if not full_name.strip() or not email.strip():
        st.error("Full name and email are required for the PDF.")
        return

    st.session_state.screen = "generating"
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_paths = []
        uploaded_metadata = []
        for uploaded in uploaded_files:
            destination = Path(tmpdir) / uploaded.name
            destination.write_bytes(uploaded.getbuffer())
            saved_paths.append(str(destination))
            uploaded_metadata.append({"name": uploaded.name, "size": uploaded.size, "type": uploaded.type})

        try:
            if jd_file:
                jd_path = Path(tmpdir) / jd_file.name
                jd_path.write_bytes(jd_file.getbuffer())
                jd_snippets = extract_from_file(jd_path)
                jd_raw_text = "\n".join(snippet.raw_text for snippet in jd_snippets)
                jd_title = derive_role_title(jd_file.name)
                jd_source = jd_file.name
            elif jd_url.strip():
                jd_title, jd_raw_text = fetch_job_page(jd_url.strip())
                jd_source = jd_url.strip()
            else:
                jd_raw_text = jd_pasted_text.strip()
                jd_title = "Pasted job description"
                jd_source = "pasted job description"
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the job description source: {exc}")
            return
        if not jd_raw_text.strip():
            st.error("The selected job description source did not contain readable text.")
            return

        contact = ContactInfo(full_name=full_name, email=email, phone=phone or None, location=location or None, linkedin=linkedin or None, github=github or None)
        education = [{"institution": institution, "degree": degree, "graduation_date": grad_date, "gpa": gpa or None, "relevant_coursework": []}]
        progress = st.status("TalentForge AI is generating your resume...", expanded=True)
        try:
            if not ollama_is_available():
                raise RuntimeError(
                    "The local Ollama service is not running. Start `ollama serve`, "
                    "then register the `talentforge-qwen` model before generating."
                )
            pipeline = st.session_state.pipeline
            progress.write("Reading uploaded documents...")
            snippets = pipeline.ingest_candidate_files(saved_paths)
            progress.write("Matching candidate evidence to the job description...")
            _, baseline_match = pipeline.match_jd(f"baseline_{jd_title}", jd_title, jd_raw_text, company or None)
            progress.write("Running grounded skill analysis, bullet rewriting, and resume assembly...")
            result = pipeline.run_full_pipeline(candidate_files=saved_paths, jd_title=jd_title, jd_raw_text=jd_raw_text, contact=contact, education=education, company=company or None, summary=summary or None)
            progress.write("Identified aligned and missing skills.")
            progress.write("Rewrote grounded resume bullets.")
            progress.write("Assembled the resume document.")
            progress.write("Rendering the PDF...")
            progress.update(label="Resume ready", state="complete")
            st.session_state.result = result
            st.session_state.match_result = baseline_match
            st.session_state.snippets = snippets
            st.session_state.uploaded_metadata = uploaded_metadata
            st.session_state.jd_title = jd_title
            st.session_state.jd_raw_text = jd_raw_text
            st.session_state.chat_history = [
                {"role": "user", "content": f"Job description source: {jd_source}\n\n{jd_raw_text}", "files": uploaded_metadata},
                {"role": "assistant", "kind": "dossier", "content": "dossier"},
            ]
            st.session_state.screen = "dossier"
            st.rerun()
        except GroundingError as exc:
            progress.update(label="Grounding check failed", state="error")
            st.error(f"TalentForge rejected an unsupported claim: {exc}")
            st.session_state.screen = "intake"
        except Exception as exc:  # noqa: BLE001
            progress.update(label="Resume generation failed", state="error")
            st.error(f"TalentForge could not finish generating the resume: {exc}")
            st.exception(exc)
            st.session_state.screen = "intake"


def render_user_message(message: dict) -> None:
    st.markdown(message["content"])
    for file_info in message.get("files", []):
        size = f"{file_info['size'] / 1024:.1f} KB" if file_info.get("size") else "size unavailable"
        html(f'<span class="tf-chip">▧ {file_info["name"]} <small>{size}</small></span>')


def render_activity_message(result, match_result, uploaded_metadata) -> None:
    aligned = sorted(result.phase1.aligned_skills, key=lambda item: item.confidence, reverse=True)
    skill_text = ", ".join(item.skill for item in aligned[:3]) or "no directly aligned skills"
    estimated_count = sum(bullet.metric_is_estimated for bullet in result.phase2.bullets)
    grounded_count = len(result.phase2.bullets) - estimated_count
    st.markdown(
        f"I read **{len(uploaded_metadata)} uploaded file(s)**, matched **{len(match_result.covered_requirements)} covered requirement(s)**, and found **{len(match_result.uncovered_requirements)} uncovered requirement(s)**. "
        f"The strongest aligned skills were **{skill_text}**. I rewrote **{len(result.phase2.bullets)} bullet(s)**: **{grounded_count}** used directly grounded metrics and **{estimated_count}** used non-quantified descriptions."
    )


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
    pdf_size = result.pdf_path.stat().st_size / 1024
    html(f'<div class="tf-section-card"><div class="tf-section-label">Generated document</div><strong>{result.pdf_path.name}</strong><p class="tf-muted">PDF size: {pdf_size:.1f} KB</p></div>')
    render_pdf_preview(result)


def render_pdf_preview(result) -> None:
    """Render a browser-native PDF preview without adding a dependency."""
    try:
        pdf_data = result.pdf_path.read_bytes()
        encoded_pdf = base64.b64encode(pdf_data).decode("ascii")
        st.components.v1.html(f'<iframe src="data:application/pdf;base64,{encoded_pdf}" width="100%" height="720" style="border:1px solid #dfe4f2;border-radius:14px"></iframe>', height=740)
    except OSError:
        st.warning("The generated PDF preview is unavailable, but you can still download the file below.")
    # TODO: A dedicated PDF viewer component could improve preview controls if it is staged in the offline USB wheels cache.
    render_pdf_download(result)


def render_dossier() -> None:
    result = st.session_state.get("result")
    match_result = st.session_state.get("match_result")
    if result is None or match_result is None:
        st.session_state.screen = "intake"
        st.rerun()
        return
    pipeline = st.session_state.pipeline
    render_dossier_header(st.session_state.get("jd_title", "Target role"), coverage_pct(match_result))
    if st.button("Log out", key="dossier_top_logout", use_container_width=True):
        clear_sensitive_state()
        st.rerun()

    if st.session_state.active_tab == "CHAT":
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    render_user_message(message)
                elif message.get("kind") == "dossier":
                    render_dossier_message(result, match_result, pipeline)
                    render_activity_message(result, match_result, st.session_state.get("uploaded_metadata", []))
                else:
                    st.markdown(message["content"])
        with st.chat_message("assistant"):
            missing = result.phase1.missing_skills[0] if result.phase1.missing_skills else "another section"
            st.markdown(f"Your resume is ready. Would you like to refine the emphasis on **{missing}** or share another direction?")
        html('<div class="tf-section-label">Suggested prompts</div>')
        prompt_columns = st.columns(4)
        for column, prompt in zip(prompt_columns, ("More technical", "Emphasize leadership", "Shorten summary", "Product impact")):
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
        html('<div class="tf-section-label">Generated PDF preview</div>')
        render_pdf_preview(result)
        user_message = st.chat_input("Ask TalentForge to refine or rewrite...")
        if user_message:
            # TODO: Send this instruction through a future Phase 2/3 rerun; this version stores it without claiming the PDF changed.
            st.session_state.chat_history.append({"role": "user", "content": user_message})
            st.session_state.chat_history.append({"role": "assistant", "kind": "text", "content": "I recorded that refinement request. The current PDF remains unchanged until a rerun is connected."})
            st.rerun()
    elif st.session_state.active_tab == "DOCUMENT":
        render_document_tab(result)
    elif st.session_state.active_tab == "ATS SCORE":
        render_ats_tab(match_result)
    else:
        render_vault_tab()

    tab_columns = st.columns(4)
    for column, tab in zip(tab_columns, ("CHAT", "DOCUMENT", "ATS SCORE", "VAULT")):
        with column:
            if st.button(tab, key=f"tab_{tab}", use_container_width=True):
                st.session_state.active_tab = tab
                st.rerun()
    if st.button("← Back to home", key="back_to_home"):
        st.session_state.screen = "home"
        st.rerun()


if not st.session_state.logged_in:
    st.session_state.screen = "login"

if st.session_state.screen == "login":
    render_login()
elif st.session_state.screen == "home":
    render_home()
elif st.session_state.screen == "intake":
    render_intake_form()
elif st.session_state.screen == "generating":
    html('<div class="tf-section-card"><div class="tf-section-label">Generating</div><h2>TalentForge is shaping your resume...</h2><p class="tf-muted">Your local pipeline is reading evidence, matching requirements, and rendering the final document.</p></div>')
    if st.button("Log out", key="generating_logout", use_container_width=True):
        clear_sensitive_state()
        st.rerun()
elif st.session_state.screen == "dossier":
    render_dossier()
else:
    st.session_state.screen = "home"
    st.rerun()
