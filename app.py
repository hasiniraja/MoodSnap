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
import requests
from io import BytesIO
import tempfile

# ----------------- Set page config first -----------------
st.set_page_config(page_title="Emotion Detection App", page_icon="😊", layout="wide")

# ----------------- Download CNN model and encoder if not present -----------------
@st.cache_resource
def download_file_from_google_drive(file_id, destination):
    """Download file from Google Drive with progress"""
    try:
        URL = f"https://drive.google.com/uc?id={file_id}&export=download"
        session = requests.Session()
        
        response = session.get(URL, stream=True)
        token = get_confirm_token(response)
        
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)
        
        save_response_content(response, destination)
        return True
    except Exception as e:
        st.error(f"Download failed: {e}")
        return False

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# Check and download files
model_downloaded = False
encoder_downloaded = False

if not os.path.exists("emotion_model.h5"):
    with st.spinner("Downloading CNN model (this may take a few minutes)..."):
        model_downloaded = download_file_from_google_drive("1gPfPbTG89asZEO3CEV4A05UxM6dlgU_Q", "emotion_model.h5")

if not os.path.exists("label_encoder.pkl"):
    with st.spinner("Downloading label encoder..."):
        encoder_downloaded = download_file_from_google_drive("1A54IqCWlcNq0J_KJ5nI1wXktKoOrKgNE", "label_encoder.pkl")

# ----------------- Load Models with Error Handling -----------------
cnn_model = None
le = None
processor = None
beit_model = None

try:
    if os.path.exists("emotion_model.h5") and os.path.getsize("emotion_model.h5") > 0:
        cnn_model = load_model("emotion_model.h5")
        st.sidebar.success("✅ CNN model loaded successfully!")
    else:
        st.sidebar.error("❌ CNN model file not found or empty")
except Exception as e:
    st.sidebar.error(f"❌ Error loading CNN model: {str(e)[:100]}...")

try:
    if os.path.exists("label_encoder.pkl") and os.path.getsize("label_encoder.pkl") > 0:
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        st.sidebar.success("✅ Label encoder loaded successfully!")
    else:
        st.sidebar.error("❌ Label encoder file not found or empty")
except Exception as e:
    st.sidebar.error(f"❌ Error loading label encoder: {str(e)[:100]}...")

# Load BEiT model (from HuggingFace)
try:
    processor = BeitImageProcessor.from_pretrained(
        "Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large"
    )
    beit_model = BeitForImageClassification.from_pretrained(
        "Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large",
        torch_dtype=torch.float32
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beit_model.to(device)
    st.sidebar.success("✅ BEiT model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading BEiT model: {str(e)[:100]}...")

# ----------------- Emoji + Name Mapping -----------------
emotion_map = {
    "angry": "😠 Anger",
    "disgust": "🤢 Disgust", 
    "fear": "😨 Fear",
    "happy": "😀 Happy",
    "neutral": "😐 Neutral",
    "sad": "😢 Sad",
    "surprise": "😲 Surprise"
}

beit_label_map = {
    "LABEL_0": "😠 Anger",
    "LABEL_1": "🤢 Disgust",
    "LABEL_2": "😨 Fear", 
    "LABEL_3": "😀 Happy",
    "LABEL_4": "😐 Neutral",
    "LABEL_5": "😢 Sad",
    "LABEL_6": "😲 Surprise"
}

# ----------------- Streamlit UI -----------------
st.sidebar.title("⚙️ Choose Model")
model_choice = st.sidebar.radio("Select Model:", ["BEiT (Pretrained Transformer)", "CNN (My Model)"])

st.title("😊 Emotion Detection App")
st.write("Upload an image and see the predicted emotion with probabilities.")

# Show download status
if model_downloaded:
    st.info("📥 CNN model downloaded successfully!")
if encoder_downloaded:
    st.info("📥 Label encoder downloaded successfully!")

uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ----------------- BEiT Prediction (always available) -----------------
    if model_choice == "BEiT (Pretrained Transformer)" and beit_model is not None and processor is not None:
        try:
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
        except Exception as e:
            st.error(f"❌ Error during BEiT prediction: {e}")

    # ----------------- CNN Prediction -----------------
    elif model_choice == "CNN (My Model)":
        if cnn_model is not None and le is not None:
            try:
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
                ax.set_ylabel("Emotions")
                ax.set_title("Emotion Probability Distribution (CNN)")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Error during CNN prediction: {e}")
        else:
            st.error("❌ CNN model is not available. Please check if the model downloaded correctly.")
            st.info("You can try using the BEiT model instead, which should always work.")
else:
    st.info("👆 Please upload an image to get started!")

# Add troubleshooting section
with st.expander("ℹ️ Troubleshooting"):
    st.write("""
    **If you're experiencing issues:**
    
    1. Make sure your model files are accessible on Google Drive
    2. Try using the BEiT model (more reliable)
    3. Check that your files are not too large (>100MB)
    4. Refresh the page if downloads fail
    """)