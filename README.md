# MoodSnap
# 😊 MoodSnap - Emotion Detection App

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

MoodSnap is a **real-time emotion detection web app** that analyzes facial images using deep learning models. Users can upload images and see predictions along with probabilities and visual charts.  

---

## 🚀 Features

- **Dual Model Support**
  - Custom **CNN Model**
  - Pretrained **BEiT Transformer Model**
- **Emoji + Name Mapping** for easy emotion interpretation
- **Top 3 Predictions** with confidence percentages
- **Probability Bar Charts** for better visualization
- **Interactive Web UI** using **Streamlit**
-

---

## 🎯 Tech Stack

- **Python 3.x**
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/)
- [Streamlit](https://streamlit.io/)
- [Pillow](https://pillow.readthedocs.io/)
- NumPy & Matplotlib

---

## 🛠 Installation & Setup

1. **Clone the repo:**
```bash
git clone https://github.com/yourusername/MoodSnap.git
cd MoodSnap
```
2. **Create a virtual environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```
3. **Install dependencies:**
```bash
pip install -r requirements.txt
```
4. **Run the app:**
```bash
streamlit run app.py
```

---
## 📷 Usage

- Upload an image (jpg, jpeg, or png).
- Select the model you want to use from the sidebar (CNN or BEiT).
- View the predicted emotion and confidence percentage.
- Explore the Top 3 Predictions and probability chart.
---


