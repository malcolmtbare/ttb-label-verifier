"""FastAPI backend.

Endpoints:
  GET  /                -> the UI
  GET  /api/models      -> models that are actually configured (for the picker)
  POST /api/verify      -> one image + expected application JSON -> VerificationReport
  POST /api/batch       -> many images (Janet's batch requirement), bounded concurrency

The heavy lifting (extraction, matching) lives in the modules above; this file is
just transport + orchestration.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles

load_dotenv()

from . import images  # noqa: E402
from . import ocr  # noqa: E402
from .adapters.base import available_adapters, get_adapter  # noqa: E402
from .schema import ApplicationData, LabelExtraction, VerificationReport  # noqa: E402
from .verifier import verify  # noqa: E402

app = FastAPI(title="TTB Label Verifier")
STATIC = Path(__file__).parent.parent / "static"

# How many images to process at once in a batch. Keeps a 200-label dump from
# serializing while respecting the per-label speed expectation.
BATCH_CONCURRENCY = 5

_ALLOWED = {"image/jpeg", "image/png", "image/webp"}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/batch")
def batch_page():
    return FileResponse(STATIC / "batch.html")


@app.get("/api/models")
def models():
    return [
        {"key": a.info.key, "label": a.info.label,
         "boundary": a.info.boundary, "boundary_note": a.info.boundary_note}
        for a in available_adapters()
    ]


def _parse_expected(expected_json: str) -> ApplicationData:
    try:
        return ApplicationData.model_validate(json.loads(expected_json or "{}"))
    except Exception as e:
        raise HTTPException(400, f"Could not read the application data: {e}")


async def _run_one(model: str, data: bytes, media_type: str, expected: ApplicationData) -> VerificationReport:
    adapter = get_adapter(model)
    t0 = time.perf_counter()
    # The SDK calls are blocking; run them off the event loop.
    extraction = await asyncio.to_thread(adapter.extract, data, media_type)
    latency = int((time.perf_counter() - t0) * 1000)
    return verify(extraction, expected, adapter.info.label, latency)


@app.post("/api/verify", response_model=VerificationReport)
async def verify_one(
    model: str = Form(...),
    expected: str = Form("{}"),
    file: UploadFile = File(...),
):
    data = await file.read()
    data, media = images.normalize(data, file.content_type, file.filename)
    if media not in _ALLOWED:
        raise HTTPException(400, "Please upload a JPEG, PNG, WebP, or HEIC image.")
    try:
        return await _run_one(model, data, media, _parse_expected(expected))
    except KeyError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"The model could not process this image: {e}")


@app.post("/api/convert")
async def convert_image(file: UploadFile = File(...)):
    """Convert an upload (notably iPhone HEIC) to a browser-displayable JPEG. The
    single-view UI routes HEIC through here so the preview and the locate/straighten
    overlay work in any browser; other formats pass straight through."""
    data = await file.read()
    data, media = images.normalize(data, file.content_type, file.filename)
    if media not in _ALLOWED:
        raise HTTPException(400, "Please upload a JPEG, PNG, WebP, or HEIC image.")
    return Response(content=data, media_type=media)


_MANIFEST_FIELDS = ("brand_name", "class_type", "alcohol_content",
                    "net_contents", "name_address", "country_of_origin")


def _parse_manifest(raw: bytes) -> dict[str, ApplicationData]:
    """Parse a CSV mapping filename -> application fields. Header must include
    'filename' plus any of the application field columns. Robust to commas inside
    quoted address fields (handled by csv module). In production this data comes
    from the COLA application records; the CSV is the standalone stand-in."""
    text = raw.decode("utf-8-sig", errors="replace")
    out: dict[str, ApplicationData] = {}
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        fn = (row.get("filename") or "").strip()
        if not fn:
            continue
        data = {k: ((row.get(k) or "").strip() or None) for k in _MANIFEST_FIELDS}
        out[fn] = ApplicationData(**data)
    return out


@app.post("/api/batch")
async def verify_batch(
    model: str = Form(...),
    files: list[UploadFile] = File(...),
    manifest: UploadFile | None = File(None),  # optional CSV: filename -> application fields
):
    """Process a drop of labels concurrently. With a manifest, each label is
    compared to its application row; without one, it's warning-only (the warning
    is mandatory on every label, so the batch still surfaces missing/incorrect
    warnings)."""
    mani: dict[str, ApplicationData] = {}
    if manifest is not None:
        try:
            mani = _parse_manifest(await manifest.read())
        except Exception as e:
            raise HTTPException(400, f"Could not read the manifest CSV: {e}")

    sem = asyncio.Semaphore(BATCH_CONCURRENCY)

    async def worker(f: UploadFile):
        async with sem:
            data = await f.read()
            data, media = images.normalize(data, f.content_type, f.filename)
            if media not in _ALLOWED:
                return {"filename": f.filename, "error": "Unsupported image type."}
            exp = mani.get(f.filename) or mani.get(os.path.basename(f.filename or "")) or ApplicationData()
            try:
                report = await _run_one(model, data, media, exp)
                return {"filename": f.filename, "report": report.model_dump()}
            except Exception as e:
                return {"filename": f.filename, "error": str(e)}

    results = await asyncio.gather(*(worker(f) for f in files))
    summary = {"pass": 0, "fail": 0, "review": 0, "error": 0}
    for r in results:
        if "error" in r:
            summary["error"] += 1
        else:
            summary[r["report"]["overall"]] = summary.get(r["report"]["overall"], 0) + 1
    return JSONResponse({"count": len(results), "matched_manifest": bool(mani),
                         "summary": summary, "results": results})


@app.post("/api/ocr")
async def ocr_locate(
    file: UploadFile = File(...),
    detected: str = Form("{}"),
):
    """Locate the detected field values on the image using a dedicated OCR pass.
    Called asynchronously by the UI after extraction. Returns normalized boxes per
    field, or configured=false if Azure AI Vision isn't set up (UI then hides the
    locate affordance). Never blocks or alters the extraction result."""
    try:
        fields = json.loads(detected) or {}
    except Exception:
        fields = {}
    # Locate concise fields and, specially, the government warning (which may be
    # rotated on a can — the OCR module returns its angle so the UI can straighten it).
    wanted = {k: str(v) for k, v in fields.items() if v}
    data = await file.read()
    data, media = images.normalize(data, file.content_type, file.filename)
    if media not in _ALLOWED:
        raise HTTPException(400, "Unsupported image type.")
    try:
        return JSONResponse(ocr.locate(data, media, wanted))
    except Exception as e:
        # OCR is a non-critical enhancement — fail soft so the page keeps working.
        return JSONResponse({"configured": True, "boxes": {}, "regions": {}, "error": str(e)})


@app.post("/api/compare", response_model=VerificationReport)
async def compare(
    detected: str = Form(...),
    expected: str = Form("{}"),
    model_used: str = Form("(edited)"),
):
    """Re-run matching only — no model call. Lets the UI recompute the verdict
    instantly as the agent corrects detected fields or fills in application data."""
    try:
        ext = LabelExtraction.model_validate_json(detected)
    except Exception as e:
        raise HTTPException(400, f"Could not read the label data: {e}")
    return verify(ext, _parse_expected(expected), model_used, 0)


# Serve any other static assets (favicon, etc.) if added later.
app.mount("/static", StaticFiles(directory=STATIC), name="static")