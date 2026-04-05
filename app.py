import shutil
import uuid
import requests
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from core.xray_engine import XRayInferenceEngine

# -----------------------------------
# INIT APP
# -----------------------------------
app = FastAPI()

UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"

# -----------------------------------
# LOAD MODEL ONCE
# -----------------------------------
engine = XRayInferenceEngine()
engine.load_model()

# -----------------------------------
# STATIC + TEMPLATE SETUP
# -----------------------------------
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------------------
# OLLAMA HUMAN REPORT
# -----------------------------------
def generate_human_report(predictions):
    try:
        significant = [p for p in predictions if p[1] > 0.1][:3]

        if not significant:
            diseases_str = "No major abnormalities detected"
        else:
            diseases_str = ", ".join(
                [f"{d} ({p*100:.1f}%)" for d, p in significant]
            )

        prompt = f"""
        You are a medical assistant AI.

        Convert this into simple patient-friendly explanation:

        Predictions: {diseases_str}

        Format:
        Summary:
        Meaning:
        Recommendation:

        Keep it short and clear.
        """

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=15
        )

        if response.status_code == 200:
            text = response.json().get("response", "")

            result = {
                "summary": "N/A",
                "meaning": "N/A",
                "recommendation": "N/A"
            }

            for line in text.split("\n"):
                line = line.strip()

                if line.lower().startswith("summary"):
                    result["summary"] = line.split(":", 1)[1].strip()
                elif line.lower().startswith("meaning"):
                    result["meaning"] = line.split(":", 1)[1].strip()
                elif line.lower().startswith("recommendation"):
                    result["recommendation"] = line.split(":", 1)[1].strip()

            return result

    except Exception as e:
        print("[OLLAMA ERROR]", e)

    return {
        "summary": "AI detected patterns in the X-ray.",
        "meaning": "Some areas may require medical attention.",
        "recommendation": "Consult a healthcare professional."
    }

# -----------------------------------
# ROUTES
# -----------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    try:
        # SAVE IMAGE
        ext = Path(image.filename).suffix
        filename = f"{uuid.uuid4()}{ext}"
        file_path = UPLOAD_DIR / filename

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # RUN MODEL
        results = engine.predict(file_path, top_k=5)

        predictions = []
        labels = []
        scores = []

        for disease, prob in results["top_predictions"]:
            risk = (
                "HIGH" if prob > 0.7 else
                "MODERATE" if prob > 0.4 else
                "LOW"
            )

            predictions.append({
                "disease": disease,
                "confidence": round(prob * 100, 2),
                "risk": risk
            })

            labels.append(disease)
            scores.append(round(prob * 100, 2))

        # STATUS
        top_prob = results["top_predictions"][0][1]

        status = (
            "CRITICAL" if top_prob > 0.7 else
            "NEEDS ATTENTION" if top_prob > 0.4 else
            "NORMAL"
        )

        # OLLAMA
        explanation = generate_human_report(results["top_predictions"])

        return {
            "image_url": f"/static/uploads/{filename}",
            "predictions": predictions,
            "status": status,
            "explanation": explanation,
            "confidence_scores": scores,
            "labels": labels
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# -----------------------------------
# RUN
# -----------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)