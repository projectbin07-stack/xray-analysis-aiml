import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchxrayvision as xrv

from core.report import OPERATING_POINT, risk_level

# ---------------------------------------------------------
# Plain-language descriptions of the label names
# ---------------------------------------------------------
# These explain what each label *refers to*. They are not findings about the image
# and deliberately contain no advice: the model output is not a diagnosis.

LABEL_DESCRIPTIONS = {
    "Pneumothorax": "Label for air in the space around the lung.",
    "Fracture": "Label for a break in a rib or other bone visible on the image.",
    "Pneumonia": "Label for lung opacity of the kind associated with infection.",
    "Cardiomegaly": "Label for an enlarged heart silhouette.",
    "Atelectasis": "Label for part of a lung that is not fully expanded.",
    "Effusion": "Label for fluid in the space around the lung.",
}


class XRayInferenceEngine:
    def __init__(self, device=None):
        self.device = device or self._get_device()
        self.model = None
        self.pathologies = None
        print(f"[INFO] Using device: {self.device}")

    @staticmethod
    def _get_device():
        if torch.cuda.is_available():
            print(f"[INFO] CUDA available. GPU: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        else:
            print("[INFO] CUDA not available. Using CPU.")
            return torch.device("cpu")

    def load_model(self):
        try:
            print("[INFO] Loading DenseNet121 pretrained model...")
            self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.model = self.model.to(self.device)
            self.model.eval()
            self.pathologies = self.model.pathologies
            print("[INFO] Model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Model loading failed: {e}")
            raise

    def preprocess_image(self, image_path, target_size=224):
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError("Image not found or unreadable.")

            img = img.astype(np.float32) / 255.0
            img = xrv.datasets.normalize(img, maxval=1)

            h, w = img.shape
            min_dim = min(h, w)
            start_h = (h - min_dim) // 2
            start_w = (w - min_dim) // 2
            img = img[start_h : start_h + min_dim, start_w : start_w + min_dim]

            img = cv2.resize(img, (target_size, target_size))

            tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
            tensor = tensor.to(self.device)

            return tensor

        except Exception as e:
            print(f"[ERROR] Preprocessing failed: {e}")
            raise

    def predict(self, image_path, top_k=5):
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        image_tensor = self.preprocess_image(image_path)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        # torchxrayvision's pretrained DenseNet already returns
        # op_norm(sigmoid(logits), op_threshs): scores in [0, 1], with 0.5 at each
        # pathology's operating point. Do NOT apply another sigmoid (v0.1.0 did, which
        # squeezed every score into 50-73 %).
        probabilities = outputs.cpu().numpy()[0]

        predictions = list(zip(self.pathologies, probabilities, strict=False))
        predictions_sorted = sorted(predictions, key=lambda x: x[1], reverse=True)

        return {"all_predictions": predictions_sorted, "top_predictions": predictions_sorted[:top_k]}

    def format_results(self, results):
        print("\n" + "=" * 70)
        print("TECHNICAL MODEL OUTPUT")
        print("=" * 70)

        for disease, probability in results["top_predictions"]:
            print(f"{disease:<30} {probability * 100:>6.2f}%   [{risk_level(float(probability))}]")

        print("=" * 70 + "\n")

    def generate_plain_report(self, results, threshold=OPERATING_POINT):
        """Plain-language listing of the scores at or above ``threshold``.

        It describes model output only; it gives no medical advice.
        """
        lines = [
            "=" * 70,
            "MODEL OUTPUT IN PLAIN LANGUAGE (research prototype)",
            "=" * 70,
            "",
        ]
        flagged = [(disease, prob) for disease, prob in results["top_predictions"] if prob >= threshold]

        if not flagged:
            lines.append(f"No pathology score is at or above {threshold * 100:.0f}%.")
            lines.append("This does not mean the image is normal.")
        else:
            lines.append(f"Pathology scores at or above {threshold * 100:.0f}%:")
            lines.append("")
            for disease, prob in flagged:
                lines.append(f"- {disease}: score {prob * 100:.1f}%")
                description = LABEL_DESCRIPTIONS.get(disease)
                if description:
                    lines.append(f"  {description}")

        lines += [
            "",
            "IMPORTANT:",
            "Research and educational software, not a medical device.",
            "These scores are model outputs, not a diagnosis, and must not be used",
            "for diagnosis or treatment decisions.",
            "",
            "=" * 70,
        ]
        return "\n".join(lines)


def main():

    parser = argparse.ArgumentParser(
        description="Chest X-ray classifier scores (research prototype, not a medical device)"
    )

    parser.add_argument("--image", type=str, required=True, help="Path to X-ray image")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print("Image file not found.")
        sys.exit(1)

    device = None if args.device == "auto" else torch.device(args.device)

    try:
        engine = XRayInferenceEngine(device=device)
        engine.load_model()

        results = engine.predict(image_path, top_k=args.top_k)

        engine.format_results(results)

        print(engine.generate_plain_report(results))

    except Exception as e:
        print(f"Fatal Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
