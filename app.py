import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
import torch
from transformers import BeitImageProcessor, BeitForImageClassification
import os
import gdown

# ----------------- Download CNN model and encoder if not present -----------------
if not os.path.exists("emotion_model.h5"):
    st.info("Downloading CNN model...")
    url_model = "https://drive.google.com/file/d/1gPfPbTG89asZEO3CEV4A05UxM6dlgU_Q/view?usp=sharing"  # Google Drive or raw GitHub URL
    gdown.download(url_model, "emotion_model.h5", quiet=False)

if not os.path.exists("label_encoder.pkl"):
    st.info("Downloading label encoder...")
    url_encoder = "https://drive.google.com/file/d/1A54IqCWlcNq0J_KJ5nI1wXktKoOrKgNE/view?usp=drive_link"
    gdown.download(url_encoder, "label_encoder.pkl", quiet=False)

# ----------------- Load Models -----------------
# CNN model
cnn_model = load_model("emotion_model.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# BEiT model
processor = BeitImageProcessor.from_pretrained(
    "Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large"
)
beit_model = BeitForImageClassification.from_pretrained(
    "Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large",
    torch_dtype=torch.float32,
    device_map=None
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
beit_model.to(device)

# ----------------- Emoji + Name Mapping -----------------
emotion_map = {
    "angry": "😠 anger",
    "disgust": "🤢 disgust",
    "fear": "😨 fear",
    "happy": "😀 happy",
    "neutral": "😐 neutral",
    "sad": "😢 sad",
    "surprise": "😲 surprise"
}
beit_label_map = {
    "LABEL_0": "😠 anger",
    "LABEL_1": "🤢 disgust",
    "LABEL_2": "😨 fear",
    "LABEL_3": "😀 happy",
    "LABEL_4": "😐 neutral",
    "LABEL_5": "😢 sad",
    "LABEL_6": "😲 surprise"
}

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Emotion Detection App", page_icon="😊", layout="wide")
st.sidebar.title("⚙️ Choose Model")
model_choice = st.sidebar.radio("Select Model:", ["CNN (My Model)", "BEiT (Pretrained Transformer)"])

st.title("😊 Emotion Detection App")
st.write("Upload an image and see the predicted emotion with probabilities.")

uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ----------------- CNN Prediction -----------------
    if model_choice == "CNN (My Model)":
        image_gray = image.convert("L")
        img_array = np.array(image_gray.resize((48, 48)))
        img_array = img_array.reshape(1, 48, 48, 1) / 255.0

        pred = cnn_model.predict(img_array)
        probs = pred[0]
        pred_idx = probs.argmax()
        label_name = le.inverse_transform([pred_idx])[0]
        predicted_label = emotion_map.get(label_name, label_name)
        confidence = probs[pred_idx] * 100

        st.success(f"✅ Predicted Emotion (CNN): {predicted_label}")
        st.info(f"📊 Confidence: {confidence:.2f}%")

        top3_idx = probs.argsort()[-3:][::-1]
        st.write("💡 Top 3 Predictions (CNN):")
        for i in top3_idx:
            name = emotion_map.get(le.inverse_transform([i])[0], le.inverse_transform([i])[0])
            st.write(f"{name}: {probs[i]*100:.2f}%")

        fig, ax = plt.subplots(figsize=(7, 4))
        emotion_names = [emotion_map.get(name, name) for name in le.classes_]
        ax.bar(emotion_names, probs, color="#4CAF50", alpha=0.8)
        ax.set_ylabel("Probability")
        ax.set_xlabel("Emotions")
        ax.set_title("Emotion Probability Distribution (CNN)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # ----------------- BEiT Prediction -----------------
    else:
        inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = beit_model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        pred_idx = probs.argmax().item()
        label_name = beit_model.config.id2label[pred_idx]
        predicted_label = beit_label_map.get(label_name, label_name)
        confidence = probs[pred_idx].item() * 100

        st.success(f"✅ Predicted Emotion (BEiT): {predicted_label}")
        st.info(f"📊 Confidence: {confidence:.2f}%")

        top3_idx = probs.topk(3).indices
        st.write("💡 Top 3 Predictions (BEiT):")
        for i in top3_idx:
            name = beit_label_map.get(beit_model.config.id2label[i.item()], beit_model.config.id2label[i.item()])
            st.write(f"{name}: {probs[i].item()*100:.2f}%")

        fig, ax = plt.subplots(figsize=(7, 4))
        emotion_names = [beit_label_map.get(beit_model.config.id2label[i], beit_model.config.id2label[i]) for i in range(len(probs))]
        ax.bar(emotion_names, probs.cpu().numpy(), color="#4CAF50", alpha=0.8)
        ax.set_ylabel("Probability")
        ax.set_xlabel("Emotions")
        ax.set_title("Emotion Probability Distribution (BEiT)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
