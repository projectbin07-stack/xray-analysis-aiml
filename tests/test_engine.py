"""Tests for core/xray_engine.py against torchxrayvision's real output contract.

They need PyTorch, OpenCV and torchxrayvision, so the lightweight CI job skips them and
the ``engine`` CI job runs them.

torchxrayvision's pretrained ``DenseNet(weights=...)`` returns, from ``forward``,
``op_norm(sigmoid(logits), op_threshs)``: scores already in [0, 1], with 0.5 at each
pathology's operating-point threshold. v0.1.0 applied ``torch.sigmoid`` to that again,
which pushed every score into 50-73 %. These tests pin the contract.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")
xrv = pytest.importorskip("torchxrayvision")

from core.report import build_predictions, overall_status  # noqa: E402
from core.xray_engine import XRayInferenceEngine  # noqa: E402

# The 18 labels of densenet121-res224-all, in model order.
PATHOLOGIES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema", "Emphysema",
    "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening", "Cardiomegaly", "Nodule",
    "Mass", "Hernia", "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum",
]  # fmt: skip
# Operating points shipped with densenet121-res224-all.
OP_THRESHS = [
    0.07422872, 0.038290843, 0.09814756, 0.0098118475, 0.023601074, 0.0022490358,
    0.010060724, 0.103246614, 0.056810737, 0.026791653, 0.050318155, 0.023985857,
    0.01939503, 0.042889766, 0.053369623, 0.035975814, 0.20204692, 0.05015312,
]  # fmt: skip


class XrvContractModel(torch.nn.Module):
    """Fake model whose ``forward`` is torchxrayvision's: op_norm(sigmoid(logits))."""

    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.pathologies = PATHOLOGIES
        self.logits = torch.tensor([logits], dtype=torch.float32)
        self.op_threshs = torch.tensor(OP_THRESHS, dtype=torch.float32)

    def forward(self, x):
        out = torch.sigmoid(self.logits.expand(x.shape[0], -1))
        return xrv.models.op_norm(out, self.op_threshs)


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "scan.png"
    rng = np.random.default_rng(0)
    cv2.imwrite(str(path), rng.integers(0, 256, size=(260, 300), dtype=np.uint8))
    return path


def engine_with(model) -> XRayInferenceEngine:
    engine = XRayInferenceEngine(device=torch.device("cpu"))
    engine.model = model.eval()
    engine.pathologies = model.pathologies
    return engine


def scores(engine: XRayInferenceEngine, image: Path) -> dict[str, float]:
    result = engine.predict(image, top_k=5)
    return {name: float(p) for name, p in result["all_predictions"]}


def test_near_zero_model_output_is_reported_near_zero(image: Path) -> None:
    # sigmoid(-12) = 6e-6, far below every operating point (the lowest is 0.0022).
    engine = engine_with(XrvContractModel([-12.0] * 18))
    got = scores(engine, image)
    assert max(got.values()) < 0.01  # v0.1.0 reported 50.0-50.4 % here
    top = engine.predict(image)["top_predictions"]
    assert overall_status(top) == "NO ELEVATED SCORE"
    assert all(p["risk"] == "LOW" for p in build_predictions(top))


def test_scores_are_the_model_output_unchanged(image: Path) -> None:
    logits = [-12.0] * 18
    logits[PATHOLOGIES.index("Effusion")] = 4.0
    model = XrvContractModel(logits)
    with torch.no_grad():
        expected = model(torch.zeros(1, 1, 224, 224))[0].numpy()
    got = scores(engine_with(model), image)
    assert [got[name] for name in PATHOLOGIES] == pytest.approx(expected.tolist(), abs=1e-6)
    assert got["Effusion"] == pytest.approx(1 - (1 - 1 / (1 + np.exp(-4.0))) / ((1 - OP_THRESHS[7]) * 2))
    assert got["Effusion"] > 0.98  # v0.1.0 could never exceed sigmoid(1) = 73.1 %
    top = engine_with(model).predict(image)["top_predictions"]
    assert top[0][0] == "Effusion"
    assert overall_status(top) == "HIGH SCORE"


def test_score_of_one_half_means_raw_probability_at_the_operating_point(image: Path) -> None:
    t = OP_THRESHS[PATHOLOGIES.index("Pneumothorax")]
    logits = [-12.0] * 18
    logits[PATHOLOGIES.index("Pneumothorax")] = float(np.log(t / (1 - t)))
    got = scores(engine_with(XrvContractModel(logits)), image)
    assert got["Pneumothorax"] == pytest.approx(0.5, abs=1e-4)


def test_real_densenet_forward_contract(image: Path) -> None:
    # torchxrayvision's own DenseNet class (random weights, no download) with
    # operating points set: its forward applies sigmoid and op_norm itself.
    torch.manual_seed(0)
    model = xrv.models.DenseNet(op_threshs=torch.tensor(OP_THRESHS))
    model.classifier.weight.data.zero_()
    model.classifier.bias.data.fill_(-12.0)
    model.pathologies = PATHOLOGIES
    got = scores(engine_with(model), image)
    assert max(got.values()) < 0.01


def test_plain_report_gives_no_medical_directives(image: Path) -> None:
    logits = [-12.0] * 18
    for name in ("Pneumothorax", "Cardiomegaly", "Effusion"):
        logits[PATHOLOGIES.index(name)] = 4.0
    engine = engine_with(XrvContractModel(logits))
    text = engine.generate_plain_report(engine.predict(image)).lower()
    assert "pneumothorax: score" in text
    for phrase in ("seek medical", "immediately", "consult", "recommend", "patient"):
        assert phrase not in text
    assert "not a medical device" in text and "not a diagnosis" in text

    quiet = engine_with(XrvContractModel([-12.0] * 18))
    assert "does not mean the image is normal" in quiet.generate_plain_report(quiet.predict(image))


@pytest.mark.skipif(
    not os.environ.get("XRAY_SAMPLE_IMAGE"), reason="set XRAY_SAMPLE_IMAGE to run the real model"
)
def test_real_pretrained_model_scores_are_not_squeezed_above_one_half() -> None:
    engine = XRayInferenceEngine(device=torch.device("cpu"))
    engine.load_model()
    got = scores(engine, Path(os.environ["XRAY_SAMPLE_IMAGE"]))
    assert len(got) == 18
    assert all(0.0 <= p <= 1.0 for p in got.values())
    assert min(got.values()) < 0.5  # impossible in v0.1.0, where the minimum was 50 %
    if Path(os.environ["XRAY_SAMPLE_IMAGE"]).name == "00000001_000.png":
        # The README shows this image's output; keep it honest.
        assert max(got, key=got.__getitem__) == "Cardiomegaly"
        assert got["Cardiomegaly"] == pytest.approx(0.6601, abs=0.01)
        assert min(got.values()) == pytest.approx(0.0119, abs=0.01)
