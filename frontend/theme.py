"""Shared visual helpers for the Streamlit TalentForge experience."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from backend.orchestrator import PipelineResult, TalentForgePipeline
    from backend.schemas import MatchResult

PRIMARY = "#354ba4"
PRIMARY_DARK = "#20265d"
ACCENT = "#13b98a"
TEXT_DARK = "#20265d"
TEXT_MUTED = "#71809f"


def inject_global_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root { --ink:#20265d; --blue:#354ba4; --lavender:#eef0ff; --line:#dfe4f2; --muted:#71809f; --mint:#13b98a; }
        .stApp { background:#f7f8fc; color:var(--ink); }
        .block-container { max-width:720px; padding:1.1rem 1rem 6.5rem; }
        html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
        h1, h2, h3 { font-family:'Space Grotesk',sans-serif; color:var(--ink); letter-spacing:0; }
        div.stButton > button { border-radius:16px; min-height:2.8rem; font-weight:700; border:0; box-shadow:0 12px 25px #29356e1c; }
        div.stButton > button[kind="primary"] { background:#354ba4; color:#fff; }
        .tf-logo-mark { display:grid; place-items:center; width:34px; height:34px; border-radius:11px; background:#eef0ff; color:#354ba4; font:700 21px 'Space Grotesk'; }
        .tf-pill-badge { color:#7958ed; background:#eee5ff; border-radius:20px; padding:.18rem .48rem; font-size:.64rem; font-weight:700; white-space:nowrap; }
        .tf-brandbar { display:flex; align-items:center; justify-content:space-between; padding:.3rem 0 1rem; position:sticky; top:0; z-index:5; background:#f7f8fc; }
        .tf-brand { display:flex; align-items:center; gap:.55rem; font:700 1rem 'Space Grotesk'; }
        .tf-brand-subtitle { color:var(--muted); font-size:.7rem; font-weight:500; margin-left:2.75rem; margin-top:-.4rem; }
        .tf-icon-row { display:flex; gap:.25rem; color:#596992; font-size:1.05rem; }
        .tf-role-card, .tf-section-card { border:1px solid var(--line); border-radius:18px; background:#fff; padding:1rem; margin:.8rem 0; box-shadow:0 8px 22px #26366e0b; }
        .tf-role-card { display:flex; justify-content:space-between; align-items:center; gap:1rem; }
        .tf-role-label, .tf-section-label { color:#4c5fac; font-size:.68rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }
        .tf-role-title { font:700 1.05rem 'Space Grotesk'; margin-top:.25rem; }
        .tf-match { color:#079d73; background:#e3faf2; border-radius:16px; padding:.35rem .55rem; font-size:.72rem; font-weight:700; white-space:nowrap; }
        .tf-chat-card { border-radius:18px; padding:1rem 1.1rem; margin:.8rem 0; background:#354ba4; color:#fff; }
        .tf-chat-card.assistant { background:#20265d; }
        .tf-chat-card p { margin:.35rem 0 0; line-height:1.55; font-size:.88rem; }
        .tf-chip, .tf-feature-pill { display:inline-block; border:1px solid var(--line); border-radius:18px; padding:.28rem .55rem; color:#5264bd; font-size:.68rem; background:#fff; margin:.3rem .15rem 0 0; }
        .tf-chip { color:#596992; background:#f5f7ff; }
        .tf-chip small { color:var(--muted); }
        .tf-status { display:inline-block; color:#078b6b; background:#e2f8f0; border-radius:12px; padding:.25rem .45rem; font-size:.62rem; font-weight:700; }
        .tf-metric { background:#f5f7ff; border-radius:14px; padding:.85rem; color:var(--ink); margin-top:.8rem; }
        .tf-metric-head { display:flex; justify-content:space-between; align-items:center; font-size:.72rem; font-weight:700; color:#5c6d91; }
        .tf-metric-value { color:#4b3ed1; font:700 1.65rem 'Space Grotesk'; }
        .tf-metric-bar { height:8px; background:#dfe4f3; border-radius:8px; margin:.55rem 0; overflow:hidden; }
        .tf-metric-bar span { display:block; height:100%; background:#354ba4; border-radius:8px; }
        .tf-tag { display:inline-block; padding:.3rem .5rem; border-radius:8px; background:#eef0ff; color:#5264bd; font-size:.68rem; margin:.25rem .15rem 0 0; }
        .tf-tag.mint { color:#078b6b; background:#e2f8f0; }
        .tf-bullet-card { border:1px solid var(--line); border-radius:14px; padding:.75rem; margin:.55rem 0; background:#fff; }
        .tf-raw { color:#8b94aa; text-decoration:line-through; font-size:.78rem; background:#f5f6fa; padding:.5rem; border-radius:8px; }
        .tf-synthesis { color:var(--ink); font-size:.82rem; margin-top:.5rem; line-height:1.45; }
        .tf-tabbar { display:flex; justify-content:space-around; gap:.3rem; background:#fff; border-top:1px solid var(--line); padding:.55rem .2rem .7rem; margin-top:1rem; }
        .tf-tabbar .tf-tab { color:#7884a1; font-size:.65rem; text-align:center; }
        .tf-tabbar .active { color:#6054e8; font-weight:700; }
        .tf-file-row { display:flex; justify-content:space-between; gap:.7rem; padding:.65rem 0; border-bottom:1px solid #edf0f7; font-size:.8rem; }
        .tf-muted { color:var(--muted); font-size:.78rem; }
        .tf-hero { text-align:center; padding:2.3rem 1rem 1.7rem; }
        .tf-hero .eyebrow { display:inline-block; border:1px solid var(--line); border-radius:24px; padding:.43rem .95rem; color:#34489b; font-size:.72rem; letter-spacing:.06em; font-weight:600; }
        .tf-hero-mark { width:140px; height:140px; margin:2.25rem auto 1.8rem; border-radius:29px; display:grid; place-items:center; background:#dfe4fa; box-shadow:0 15px 28px #4254a11c; font:700 6rem 'Space Grotesk'; color:#354ba4; }
        .tf-hero h1 { font-size:2rem; margin:0; } .tf-hero h1 span { color:#354ba4; }
        .tf-hero h2 { color:#4054a8; font-size:1.15rem; font-weight:600; margin:.35rem 0 .8rem; }
        .tf-hero p { max-width:420px; margin:0 auto; color:#30344b; line-height:1.55; font-size:.92rem; }
        .tf-pill-row { display:flex; flex-wrap:wrap; justify-content:center; gap:.4rem; margin:1.45rem auto 2.25rem; }
        .tf-pill-row .tf-feature-pill { padding:.3rem .7rem; }
        .tf-legal { color:#9298aa; text-align:center; font-size:.68rem; margin-top:.8rem; }
        [data-testid="stFileUploader"] section { border-radius:14px; border-color:#ccd4ee; background:#fff; }
        [data-testid="stExpander"] { border-color:var(--line); border-radius:14px; background:#fff; }
        @media (max-width:520px) { .block-container { padding-left:.85rem; padding-right:.85rem; } .tf-hero { padding-top:1.1rem; } .tf-hero-mark { margin-top:1.6rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def render_hero() -> None:
    html('<div class="tf-hero"><div class="eyebrow">• AMELIA INTELLIGENCE</div><div class="tf-hero-mark">A</div><h1><span>Amelia.</span> <small class="tf-pill-badge">AI CO-PILOT</small></h1><h2>Your Strategic Career &amp; Resume Co-pilot</h2><p>Turn scattered coursework, notes, and metrics into tailored, recruiter-ready resumes built for dream roles.</p><div class="tf-pill-row"><span class="tf-feature-pill">✣ ATS Calibration</span><span class="tf-feature-pill"><strong>▮</strong> Role Specific Narratives</span><span class="tf-feature-pill">ϟ Instant PDF Generation</span></div></div>')


def render_dossier_header(jd_title: str, match_pct: int) -> None:
    html(f'<div class="tf-brandbar"><div><div class="tf-brand"><span class="tf-logo-mark">A</span><span>TalentForge AI</span><span class="tf-pill-badge">PRO</span></div><div class="tf-brand-subtitle">Resume Strategist</div></div><div class="tf-icon-row"><span>↗</span><span>⋯</span><span>◉</span></div></div><div class="tf-role-card"><div><div class="tf-role-label">💼 Target role</div><div class="tf-role-title">{escape(jd_title)}</div></div><div class="tf-match">{match_pct}% match</div></div>')


def coverage_pct(match_result: MatchResult) -> int:
    total = len(match_result.covered_requirements) + len(match_result.uncovered_requirements)
    return round(100 * len(match_result.covered_requirements) / total) if total else 0


def dossier_data(result: PipelineResult, match_result: MatchResult, pipeline: TalentForgePipeline) -> dict:
    aligned = sorted(result.phase1.aligned_skills, key=lambda item: item.confidence, reverse=True)
    transformations = []
    for bullet in result.phase2.bullets[:2]:
        snippet = pipeline.matcher.snippet_by_id(bullet.source_snippet_id)
        transformations.append({"raw": snippet.raw_text if snippet else "Source snippet unavailable", "bullet": bullet})
    subtitle = result.resume.summary or (result.resume.education[0].degree if result.resume.education else "Tailored candidate dossier")
    baseline_pct = coverage_pct(match_result)
    final_pct = baseline_pct
    return {"pct": final_pct, "baseline_pct": baseline_pct, "boost": final_pct - baseline_pct, "aligned": aligned, "keywords": [item.skill for item in aligned[:6]], "transformations": transformations, "subtitle": subtitle}


def render_dossier_message(result: PipelineResult, match_result: MatchResult, pipeline: TalentForgePipeline) -> None:
    data = dossier_data(result, match_result, pipeline)
    html(f'<div class="tf-chat-card assistant"><div><strong>TalentForge AI</strong> <span class="tf-status">READY FOR INPUT</span></div><p>Your evidence is grounded against the target role. Here is the strategist readout.</p><div class="tf-metric"><div class="tf-metric-head"><span>ATS FIT CALIBRATION</span><span class="tf-metric-value">{data["pct"]}%</span></div><div class="tf-metric-bar"><span style="width:{data["pct"]}%"></span></div><small>Baseline from raw documents: {data["baseline_pct"]}% &nbsp; <span class="tf-tag mint">{data["boost"]:+d}% Boost</span></small></div></div>')
    # TODO: The backend returns no post-assembly MatchResult, so final coverage intentionally reuses the grounded pre-assembly match.
    left, right = st.columns(2)
    with left:
        skills = "".join(f'<div>• {escape(item.skill)} <span class="tf-muted">{item.confidence:.0%}</span></div>' for item in data["aligned"][:3]) or '<div class="tf-muted">No aligned skills returned.</div>'
        html(f'<div class="tf-section-card"><div class="tf-section-label">Top pillars</div>{skills}</div>')
    with right:
        html('<div class="tf-section-card"><div class="tf-section-label">Noise filtered</div><p class="tf-muted">Unsupported claims and low-signal evidence stay out of the tailored document.</p><span class="tf-tag mint">ATS SAFE</span></div>')
    html(f'<div class="tf-section-card"><div class="tf-section-label">Candidate</div><strong>{escape(result.resume.contact.full_name)}</strong><div class="tf-muted">{escape(data["subtitle"])}</div><span class="tf-tag mint">ALIGNED</span></div>')
    html('<div class="tf-section-card"><div class="tf-section-label">Impact transformation</div>')
    if data["transformations"]:
        for item in data["transformations"]:
            bullet = item["bullet"]
            rendered = escape(bullet.rendered_bullet)
            metric = escape(bullet.y_metric)
            if metric and metric in rendered:
                rendered = rendered.replace(metric, f"<strong>{metric}</strong>", 1)
            badge = '<span class="tf-tag mint">Quantified</span>' if not bullet.metric_is_estimated else ''
            html(f'<div class="tf-bullet-card"><div class="tf-section-label">Raw input</div><div class="tf-raw">{escape(item["raw"])}</div><div class="tf-section-label" style="margin-top:.5rem">Synthesis {badge}</div><div class="tf-synthesis">{rendered}</div></div>')
    else:
        html('<div class="tf-muted">No grounded rewrite bullets returned.</div>')
    html('</div>')
    keywords = "".join(f'<span class="tf-tag">{escape(keyword)}</span>' for keyword in data["keywords"]) or '<span class="tf-muted">No aligned keywords returned.</span>'
    html(f'<div class="tf-section-card"><div class="tf-section-label">High-weight keywords injected</div>{keywords}</div>')


def render_pdf_download(result: PipelineResult, label: str = "Inspect Full 1-Page Resume (PDF)") -> None:
    with open(result.pdf_path, "rb") as pdf_file:
        st.download_button(label, data=pdf_file.read(), file_name="talentforge_tailored_resume.pdf", mime="application/pdf", use_container_width=True)
