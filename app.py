"""FastAPI dashboard around a pretrained TorchXRayVision chest X-ray classifier."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import requests
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core.report import (
    FALLBACK_EXPLANATION,
    build_predictions,
    describe_for_prompt,
    overall_status,
    parse_report,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("xray-analysis")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"  # outside static/: uploads are never served back
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "15"))

app = FastAPI(title="Chest X-ray Analysis (research prototype)")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

_engine = None


def get_engine():
    """Load the model on first use so the app (and its tests) can start without PyTorch."""
    global _engine
    if _engine is None:
        from core.xray_engine import XRayInferenceEngine

        engine = XRayInferenceEngine()
        engine.load_model()
        _engine = engine
    return _engine


def generate_human_report(predictions) -> dict[str, str]:
    """Ask a local Ollama model for a plain-language explanation; fixed wording on failure."""
    prompt = f"""
    You are explaining the output of an experimental chest X-ray classifier.
    It is not a medical device and its scores are not a diagnosis. Do not diagnose.

    Rewrite this for a non-specialist:

    Model scores: {describe_for_prompt(predictions)}

    Format:
    Summary:
    Meaning:
    Recommendation:

    Keep it short and always recommend review by a clinician.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT_S,
        )
        if response.status_code == 200:
            return parse_report(response.json().get("response", ""))
    except (requests.RequestException, ValueError) as exc:
        logger.info("Ollama unavailable (%s); using fixed explanation.", exc)
    return dict(FALLBACK_EXPLANATION)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _engine is not None}


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    # Only the extension is taken from the client filename, and only if whitelisted.
    ext = Path(image.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return _error(400, "Unsupported file type. Use JPG, PNG or BMP.")

    payload = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        return _error(413, "File too large.")
    if not payload:
        return _error(400, "Empty file.")

    file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    file_path.write_bytes(payload)

    try:
        try:
            results = get_engine().predict(file_path, top_k=5)
        except FileNotFoundError:
            return _error(400, "File is not a readable image.")

        top = results["top_predictions"]
        predictions = build_predictions(top)

        return {
            "predictions": predictions,
            "status": overall_status(top),
            "explanation": generate_human_report(top),
            "confidence_scores": [p["confidence"] for p in predictions],
            "labels": [p["disease"] for p in predictions],
        }
    except Exception:
        logger.exception("Analysis failed")
        return _error(500, "Internal analysis error.")
    finally:
        # X-ray images are sensitive: never keep an upload after the request.
        file_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    # Binds to localhost by default; set HOST=0.0.0.0 to expose on the network.
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8001")))
