# 🧠 AI-Based X-Ray Intelligence & Diagnostic System

An AI-powered medical imaging system for automated chest X-ray analysis, providing disease prediction, risk assessment, and human-friendly clinical insights.

---

## 🚀 Overview

This project leverages deep learning to analyze chest X-ray images and identify potential abnormalities.
It combines **multi-label classification**, **risk scoring**, and **AI-generated explanations** into an interactive web dashboard.

---

## ✨ Features

* 🔍 **Multi-label Disease Prediction**
  Detects multiple possible conditions from a single X-ray

* ⚠️ **Risk Assessment System**
  Classifies cases as **Normal**, **Needs Attention**, or **Critical**

* 📊 **Interactive Dashboard**
  Clean UI with predictions, confidence graphs, and insights

* 🧠 **AI Clinical Explanation (Ollama)**
  Converts technical outputs into:

  * Summary
  * Meaning
  * Recommendation

* 📈 **Model Insights**

  * Prediction distribution
  * Confidence visualization
  * Performance graphs

---

## 📊 Results

### 🖥️ Dashboard

![Dashboard](assets/images/dashboard.png)

---

### 🧪 Sample Outputs

| Sample 1                        | Sample 2                        |
| ------------------------------- | ------------------------------- |
| ![](assets/images/1st test.png) | ![](assets/images/2nd test.png) |

---

## 🧠 Model Details

* **Architecture:** DenseNet121
* **Framework:** PyTorch + TorchXRayVision
* **Type:** Multi-label classification
* **Input:** Chest X-ray images

---

## ⚙️ How to Run

```bash
pip install -r requirements.txt
python app.py
```

Open in browser:

```bash
http://127.0.0.1:8001
```

---

## 🛠️ Tech Stack

* Python
* FastAPI
* PyTorch
* TorchXRayVision
* OpenCV
* Chart.js
* Ollama

---

## 📁 Project Structure

```
xray/
│
├── app.py
├── core/
├── templates/
├── static/
├── assets/images/
├── requirements.txt
└── README.md
```

---

## ⚠️ Disclaimer

This project is intended for **research and educational purposes only**.
It is **not a substitute for professional medical diagnosis**.

---

## 💡 Future Improvements

* Improved model accuracy with custom training
* Support for additional imaging modalities
* Real-time clinical deployment
* Mobile-friendly interface

---

## 👨‍💻 Author

**Navaneeth KG**
Embedded Systems & AI Developer
Founder — Indionics

---

## ⭐ If you like this project

Give it a star ⭐ and support the work!
