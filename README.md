# Chest X-ray Analysis (research prototype)

[![CI](https://github.com/projectbin07-stack/xray-analysis-aiml/actions/workflows/ci.yml/badge.svg)](https://github.com/projectbin07-stack/xray-analysis-aiml/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A FastAPI dashboard around the pretrained
[TorchXRayVision](https://github.com/mlmed/torchxrayvision) DenseNet121
chest X-ray classifier. It shows the model's per-pathology scores, bands them
for display, and can ask a local LLM (Ollama) to rephrase the scores in plain
language.

> **Not a medical device.** This is experimental research and learning
> software. The model was not trained or validated by this project, its scores
> are uncalibrated, and nothing here may be used for diagnosis or treatment
> decisions.

> **Correction in v0.1.1.** Versions up to and including v0.1.0 applied a second
> sigmoid to the model's output. Every score they displayed was squeezed into
> 50-73 %: low scores were inflated to at least 50 %, high scores were capped at
> 73 %, and every image was labelled at least "NEEDS ATTENTION". Scores, bands and
> statuses from those versions are wrong and should not be relied on. See
> [CHANGELOG.md](CHANGELOG.md).

![Dashboard](assets/images/dashboard.png)

## What this project is (and is not)

- It **is** an application layer: upload handling, inference wrapper, score
  banding, an optional LLM explanation step, a web dashboard and a CLI.
- It is **not** a new model. Classification is done entirely by the
  TorchXRayVision weights `densenet121-res224-all`, used unchanged. No training,
  fine-tuning or evaluation was performed in this repository, so no accuracy
  figures are reported here. See the TorchXRayVision paper and repository for
  the model's own evaluation.

## Features

- Multi-label pathology scores for a chest X-ray (top 5 shown)
- Display banding per score: LOW (< 50%), MODERATE (50-70%), HIGH (> 70%), and
  an overall status from the highest score (`NO ELEVATED SCORE`,
  `ELEVATED SCORE`, `HIGH SCORE`). The model normalises its scores so that 50%
  is its own operating-point threshold for each pathology (see
  [How it works](#how-it-works)); the other cut-offs are presentation choices,
  not clinically derived thresholds. The labels describe scores, not patients.
- Optional plain-language explanation from a local Ollama model, with fixed
  non-diagnostic wording when Ollama is not running. Neither gives medical
  advice.
- Web dashboard (vanilla JS + Chart.js) and a command-line tool
- Uploads are validated, analysed and deleted; nothing is stored or served back

## Quick start

Requires Python 3.10+. The first run downloads the model weights
(TorchXRayVision caches them in your user directory).

```bash
git clone https://github.com/projectbin07-stack/xray-analysis-aiml.git
cd xray-analysis-aiml
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:8001` and upload a JPG or PNG chest X-ray. The server
binds to localhost only; set `HOST=0.0.0.0` to expose it on your network.

Command line, without the web server:

```bash
python -m core.xray_engine --image path/to/xray.png --top-k 5
```

### Optional: LLM explanation

Install [Ollama](https://ollama.com), run `ollama pull llama3`, and keep it
running. Without it the dashboard still works and shows a fixed explanation.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `8001` | Port |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Local LLM endpoint |
| `OLLAMA_MODEL` | `llama3` | Model used for the explanation |
| `OLLAMA_TIMEOUT_S` | `15` | Timeout before the fixed explanation is used |
| `LOG_LEVEL` | `INFO` | Python logging level |

## API

`POST /analyze` with multipart field `image` (jpg, png or bmp, up to 15 MB).

Real response from v0.1.1 (CPU, torch 2.15, torchxrayvision 1.5.7, Ollama not
running) for `00000001_000.png`, the public NIH ChestX-ray14 sample image in
TorchXRayVision's
[`tests/`](https://github.com/mlmed/torchxrayvision/tree/master/tests) folder.
It is shown only to document the output format; it says nothing about how well
the model works.

```json
{
  "predictions": [
    {"disease": "Cardiomegaly", "confidence": 66.01, "risk": "MODERATE"},
    {"disease": "Fibrosis", "confidence": 53.29, "risk": "MODERATE"},
    {"disease": "Infiltration", "confidence": 51.54, "risk": "MODERATE"},
    {"disease": "Pleural_Thickening", "confidence": 51.03, "risk": "MODERATE"},
    {"disease": "Nodule", "confidence": 50.92, "risk": "MODERATE"}
  ],
  "status": "ELEVATED SCORE",
  "explanation": {
    "summary": "The model produced pathology scores for this image.",
    "meaning": "These are uncalibrated outputs of a pretrained research model, not findings.",
    "recommendation": "No recommendation is given: this is research software, not a medical device, and its scores are not a diagnosis."
  },
  "confidence_scores": [66.01, 53.29, 51.54, 51.03, 50.92],
  "labels": ["Cardiomegaly", "Fibrosis", "Infiltration", "Pleural_Thickening", "Nodule"]
}
```

For the same image the 18 scores span 1.2-66.0 %. v0.1.0 reported 50.3-65.9 %
for them, with Fibrosis at 63.0 % instead of 53.3 %. (`confidence` is the model
score in percent and `risk` its display band; both names are kept for API
compatibility.)

Errors are returned as `{"error": "..."}` with a 4xx/5xx status.
`GET /health` returns `{"status": "ok", "model_loaded": true|false}`.

## How it works

```
upload ──> validate (type, size) ──> temp file ──> XRayInferenceEngine
                                                      │  grayscale, normalise,
                                                      │  centre-crop, 224x224
                                                      ▼
                                              DenseNet121 (TorchXRayVision)
                                                      │  scores in [0, 1]: sigmoid + op_norm
                                                      │  inside the model (no extra sigmoid)
                                                      ▼
                              core/report.py: banding + status ──> JSON ──> dashboard
                                                      │
                                                      └─> optional Ollama rephrasing
```

| Path | Role |
|---|---|
| `app.py` | FastAPI routes, upload validation, Ollama call |
| `core/xray_engine.py` | Model loading, preprocessing, inference, CLI |

TorchXRayVision's pretrained `DenseNet.forward` returns
`op_norm(sigmoid(logits), op_threshs)`: the sigmoid is already applied, and each
score is rescaled piecewise-linearly so that 0.5 is that pathology's operating
point (below it, the raw probability is under the threshold). The engine uses
these scores as they are.
| `core/report.py` | Score banding and LLM-output parsing (no PyTorch dependency) |
| `static/`, `templates/` | Dashboard |
| `tests/` | pytest suite |

## Development

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
pytest
```

CI runs these on Python 3.10 and 3.13 without PyTorch, with the engine replaced
by a fake. A second CI job (`engine`) installs CPU PyTorch and TorchXRayVision
and runs `tests/test_engine.py`: the inference wrapper against a model that
follows TorchXRayVision's forward contract, against TorchXRayVision's own
`DenseNet` class with random weights, and against the real pretrained weights
on the public sample image above. To run those locally:

```bash
pip install -r requirements.txt pytest httpx
XRAY_SAMPLE_IMAGE=path/to/00000001_000.png pytest tests/test_engine.py
```

Command-line output for the same sample image (real run, v0.1.1):

```text
======================================================================
TECHNICAL MODEL OUTPUT
======================================================================
Cardiomegaly                    66.01%   [MODERATE]
Fibrosis                        53.29%   [MODERATE]
Infiltration                    51.54%   [MODERATE]
Pleural_Thickening              51.03%   [MODERATE]
Nodule                          50.92%   [MODERATE]
======================================================================

======================================================================
MODEL OUTPUT IN PLAIN LANGUAGE (research prototype)
======================================================================

Pathology scores at or above 50%:

- Cardiomegaly: score 66.0%
  Label for an enlarged heart silhouette.
- Fibrosis: score 53.3%
- Infiltration: score 51.5%
- Pleural_Thickening: score 51.0%
- Nodule: score 50.9%

IMPORTANT:
Research and educational software, not a medical device.
These scores are model outputs, not a diagnosis, and must not be used
for diagnosis or treatment decisions.

======================================================================
```

## Known limitations

- Scores are TorchXRayVision's operating-point-normalised outputs of a
  pretrained multi-dataset model. They are not calibrated probabilities, the
  operating points come from the upstream model, and the 70% band is arbitrary.
- No check that the uploaded image is actually a frontal chest X-ray; any image
  produces scores.
- DICOM is not supported (JPG/PNG/BMP only) and images are centre-cropped to a
  square, which can cut off anatomy.
- No localisation or saliency maps: the "high contrast" toggle is a display
  filter, not model output.
- LLM explanations can be wrong or overconfident; the prompt forbids diagnosis
  but output is not otherwise checked.

## Roadmap

- Grad-CAM overlays for the top predictions
- Input sanity check (reject non-X-ray images)
- DICOM input

## License

MIT for the code in this repository - see [LICENSE](LICENSE). TorchXRayVision
and its model weights are distributed under their own terms (Apache-2.0 for the
library; check the upstream repository for dataset-related conditions).

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). Never
post identifiable patient images in issues or pull requests.

## Author

Navaneeth KG
