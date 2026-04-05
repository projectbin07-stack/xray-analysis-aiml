import argparse
import cv2
import numpy as np
import torch
import torchxrayvision as xrv
from pathlib import Path
import sys


# ---------------------------------------------------------
# Simple Language Explanations for Patients
# ---------------------------------------------------------

DISEASE_EXPLANATIONS = {
    "Pneumothorax": {
        "simple": "Air may be trapped around the lung, which can partially collapse it.",
        "symptoms": "Chest pain and shortness of breath.",
        "advice": "Seek medical attention immediately."
    },
    "Fracture": {
        "simple": "There may be a broken rib or bone in the chest area.",
        "symptoms": "Pain while breathing or moving.",
        "advice": "Consult a doctor for evaluation."
    },
    "Pneumonia": {
        "simple": "Possible infection in the lungs.",
        "symptoms": "Fever, cough, breathing difficulty.",
        "advice": "Medical consultation recommended."
    },
    "Cardiomegaly": {
        "simple": "The heart may appear enlarged.",
        "symptoms": "Fatigue, shortness of breath.",
        "advice": "Consult a cardiologist."
    },
    "Atelectasis": {
        "simple": "Part of the lung may not be fully expanded.",
        "symptoms": "Breathing discomfort.",
        "advice": "Medical consultation recommended."
    },
    "Effusion": {
        "simple": "Fluid may be collected around the lungs.",
        "symptoms": "Breathing difficulty or chest heaviness.",
        "advice": "Doctor evaluation recommended."
    }
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
            print(f"[INFO] Model loaded successfully.")
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
            img = img[start_h:start_h + min_dim, start_w:start_w + min_dim]

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

        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

        predictions = list(zip(self.pathologies, probabilities))
        predictions_sorted = sorted(predictions, key=lambda x: x[1], reverse=True)

        return {
            "all_predictions": predictions_sorted,
            "top_predictions": predictions_sorted[:top_k]
        }

    def format_results(self, results):
        print("\n" + "=" * 70)
        print("TECHNICAL MODEL OUTPUT")
        print("=" * 70)

        for disease, probability in results["top_predictions"]:
            confidence_pct = probability * 100
            risk = "HIGH" if probability > 0.7 else "MODERATE" if probability > 0.4 else "LOW"
            print(f"{disease:<30} {confidence_pct:>6.2f}%   [{risk}]")

        print("=" * 70 + "\n")

    def generate_patient_report(self, results, threshold=0.5):
        report = []
        report.append("=" * 70)
        report.append("AI-BASED CHEST X-RAY EXPLANATION (Patient Friendly)")
        report.append("=" * 70 + "\n")

        concerning = [
            (disease, prob)
            for disease, prob in results["top_predictions"]
            if prob >= threshold
        ]

        if not concerning:
            report.append("No strong abnormal signs detected by AI.\n")
        else:
            report.append("The AI detected possible signs of:\n")

            for disease, prob in concerning:
                explanation = DISEASE_EXPLANATIONS.get(disease)
                report.append(f"• {disease} ({prob*100:.1f}% confidence)")

                if explanation:
                    report.append(f"  What it means: {explanation['simple']}")
                    report.append(f"  Possible symptoms: {explanation['symptoms']}")
                    report.append(f"  Recommendation: {explanation['advice']}\n")
                else:
                    report.append("  Medical evaluation recommended.\n")

        report.append("IMPORTANT:")
        report.append("This AI system does NOT provide a medical diagnosis.")
        report.append("Please consult a qualified healthcare professional.\n")
        report.append("=" * 70)

        return "\n".join(report)


def main():

    parser = argparse.ArgumentParser(description="Chest X-Ray AI Explanation Tool")

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

        patient_report = engine.generate_patient_report(results)
        print(patient_report)

    except Exception as e:
        print(f"Fatal Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()