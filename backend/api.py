"""
api.py — FastAPI surface over TalentForgePipeline.

Kept deliberately thin: request validation + calling the orchestrator +
returning results or a file download. All real logic lives in
orchestrator.py / matcher.py / agent_prompts.py / latex_generator.py so it
stays independently testable without spinning up a server.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from loguru import logger

from backend.orchestrator import GroundingError, TalentForgePipeline
from backend.schemas import ContactInfo

app = FastAPI(title="TalentForge AI", version="0.1.0")
pipeline = TalentForgePipeline()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-resume")
async def generate_resume(
    jd_title: str = Form(...),
    jd_raw_text: str = Form(...),
    company: str | None = Form(None),
    full_name: str = Form(...),
    email: str = Form(...),
    phone: str | None = Form(None),
    location: str | None = Form(None),
    linkedin: str | None = Form(None),
    github: str | None = Form(None),
    education_json: str = Form(..., description="JSON list of education entries"),
    summary: str | None = Form(None),
    files: list[UploadFile] = File(...),
):
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_paths = []
        for f in files:
            dest = Path(tmpdir) / f.filename
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            saved_paths.append(str(dest))

        try:
            education = json.loads(education_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"education_json is not valid JSON: {exc}") from exc

        contact = ContactInfo(
            full_name=full_name,
            email=email,
            phone=phone,
            location=location,
            linkedin=linkedin,
            github=github,
        )

        try:
            result = pipeline.run_full_pipeline(
                candidate_files=saved_paths,
                jd_title=jd_title,
                jd_raw_text=jd_raw_text,
                contact=contact,
                education=education,
                company=company,
                summary=summary,
            )
        except GroundingError as exc:
            logger.error(f"Grounding check failed: {exc}")
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline failed")
            raise HTTPException(500, f"Pipeline error: {exc}") from exc

        return FileResponse(
            path=result.pdf_path,
            media_type="application/pdf",
            filename=result.pdf_path.name,
        )
