import pytest
import requests
from fastapi.testclient import TestClient

import app as app_module

IMAGE_BYTES = b"stand-in bytes; the fake engine never decodes them"


class FakeEngine:
    """Stands in for the PyTorch model; records what it was asked to read."""

    def __init__(self, scores):
        self.scores = scores
        self.seen = []

    def predict(self, image_path, top_k=5):
        assert image_path.exists(), "the upload must be on disk while the model reads it"
        self.seen.append(image_path)
        ranked = sorted(self.scores, key=lambda x: x[1], reverse=True)
        return {"all_predictions": ranked, "top_predictions": ranked[:top_k]}


@pytest.fixture
def engine(monkeypatch):
    fake = FakeEngine([("Effusion", 0.82), ("Pneumonia", 0.45), ("Mass", 0.03)])
    monkeypatch.setattr(app_module, "get_engine", lambda: fake)

    def no_ollama(*args, **kwargs):
        raise requests.ConnectionError("offline test")

    monkeypatch.setattr(app_module.requests, "post", no_ollama)
    return fake


@pytest.fixture
def client(engine):
    return TestClient(app_module.app)


def test_home_page_has_no_performance_or_certification_claims(client):
    r = client.get("/")
    assert r.status_code == 200
    page = r.text.lower()
    assert "not a medical device" in page
    assert "certified" not in page
    assert "confusion matrix" not in page


def test_health_reports_model_not_loaded_before_first_use(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": False}


def test_analyze_shapes_response_from_model_scores(client):
    r = client.post("/analyze", files={"image": ("scan.png", IMAGE_BYTES, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "CRITICAL"
    assert body["labels"] == ["Effusion", "Pneumonia", "Mass"]
    assert body["confidence_scores"] == [82.0, 45.0, 3.0]
    assert [p["risk"] for p in body["predictions"]] == ["HIGH", "MODERATE", "LOW"]
    assert body["explanation"] == app_module.FALLBACK_EXPLANATION
    assert "image_url" not in body


def test_upload_is_deleted_and_never_stored_under_static(client, engine):
    r = client.post("/analyze", files={"image": ("../../evil.png", IMAGE_BYTES, "image/png")})
    assert r.status_code == 200
    stored = engine.seen[0]
    assert stored.parent == app_module.UPLOAD_DIR
    assert "evil" not in stored.name
    assert not stored.exists()
    assert "static" not in stored.parts


@pytest.mark.parametrize("name", ["scan.dcm", "run.py", "noext"])
def test_rejects_unsupported_types(client, name):
    r = client.post("/analyze", files={"image": (name, b"data", "application/octet-stream")})
    assert r.status_code == 400
    assert "error" in r.json()


def test_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 10)
    r = client.post("/analyze", files={"image": ("scan.png", IMAGE_BYTES, "image/png")})
    assert r.status_code == 413


def test_unreadable_image_is_a_client_error(client, engine, monkeypatch):
    def unreadable(image_path, top_k=5):
        raise FileNotFoundError("Image not found or unreadable.")

    monkeypatch.setattr(engine, "predict", unreadable)
    r = client.post("/analyze", files={"image": ("scan.png", b"junk", "image/png")})
    assert r.status_code == 400


def test_model_failure_does_not_leak_details(client, engine, monkeypatch):
    def boom(image_path, top_k=5):
        raise RuntimeError("secret internal path C:/models/x")

    monkeypatch.setattr(engine, "predict", boom)
    r = client.post("/analyze", files={"image": ("scan.png", IMAGE_BYTES, "image/png")})
    assert r.status_code == 500
    assert "secret" not in r.text


def test_ollama_text_is_parsed_when_available(client, monkeypatch):
    class Resp:
        status_code = 200

        def json(self):
            return {"response": "Summary: s\nMeaning: m\nRecommendation: r"}

    monkeypatch.setattr(app_module.requests, "post", lambda *a, **k: Resp())
    r = client.post("/analyze", files={"image": ("scan.png", IMAGE_BYTES, "image/png")})
    assert r.json()["explanation"] == {"summary": "s", "meaning": "m", "recommendation": "r"}
