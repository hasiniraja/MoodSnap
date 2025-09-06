import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
import os
import requests
import time

# ----------------- Page config -----------------
st.set_page_config(page_title="Emotion Detection App", page_icon="😊", layout="wide")

# ----------------- Import transformers with error handling -----------------
try:
    import torch
    from transformers import BeitImageProcessor, BeitForImageClassification
    TRANSFORMERS_AVAILABLE = True
    st.sidebar.success("✅ Transformers library imported successfully!")
except ImportError as e:
    st.sidebar.error(f"❌ Transformers import failed: {e}")
    st.sidebar.info("Using CNN model only. Add 'transformers' to requirements.txt for BEiT model.")
    TRANSFORMERS_AVAILABLE = False
    torch = None
    BeitImageProcessor = None
    BeitForImageClassification = None

# ----------------- Download helpers -----------------
@st.cache_resource
def download_file_from_google_drive(file_id, destination):
    """Download file from Google Drive"""
    try:
        URL = f"https://drive.google.com/uc?id={file_id}&export=download"
        session = requests.Session()
        response = session.get(URL, stream=True)
        
        # Check for download confirmation token
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': file_id, 'confirm': value}
                response = session.get(URL, params=params, stream=True)
                break
        
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        st.error(f"Download failed: {e}")
        return False

# ----------------- Download CNN model and encoder -----------------
if not os.path.exists("emotion_model.h5"):
    with st.spinner("Downloading CNN model..."):
        download_file_from_google_drive("1gPfPbTG89asZEO3CEV4A05UxM6dlgU_Q", "emotion_model.h5")

if not os.path.exists("label_encoder.pkl"):
    with st.spinner("Downloading label encoder..."):
        download_file_from_google_drive("1A54IqCWlcNq0J_KJ5nI1wXktKoOrKgNE", "label_encoder.pkl")

# ----------------- Load CNN -----------------
cnn_model = None
le = None
try:
    if os.path.exists("emotion_model.h5") and os.path.getsize("emotion_model.h5") > 0:
        cnn_model = load_model("emotion_model.h5")
    if os.path.exists("label_encoder.pkl") and os.path.getsize("label_encoder.pkl") > 0:
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
    if cnn_model and le:
        st.sidebar.success("✅ CNN model loaded successfully!")
    else:
        st.sidebar.warning("⚠️ CNN model or encoder not available")
except Exception as e:
    st.sidebar.error(f"❌ CNN model failed to load: {str(e)[:100]}...")

# ----------------- Load BEiT (if transformers available) -----------------
processor = None
beit_model = None

if TRANSFORMERS_AVAILABLE:
    try:
        processor = BeitImageProcessor.from_pretrained(
            "Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large"
        )
        beit_model = BeitForImageClassification.from_pretrained(
            "Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large",
            torch_dtype=torch.float32
        )
        beit_model.eval()
        st.sidebar.success("✅ BEiT model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ BEiT model failed to load: {str(e)[:100]}...")
else:
    st.sidebar.info("ℹ️ BEiT model not available (transformers not installed)")

# ----------------- Label maps -----------------
emotion_map = {
    "angry": "😠 Anger", "disgust": "🤢 Disgust", "fear": "😨 Fear",
    "happy": "😀 Happy", "neutral": "😐 Neutral", "sad": "😢 Sad",
    "surprise": "😲 Surprise"
}

beit_label_map = {
    "LABEL_0": "😠 Anger", "LABEL_1": "🤢 Disgust", "LABEL_2": "😨 Fear",
    "LABEL_3": "😀 Happy", "LABEL_4": "😐 Neutral", "LABEL_5": "😢 Sad",
    "LABEL_6": "😲 Surprise"
}

# ----------------- Streamlit UI -----------------
st.sidebar.title("⚙️ Choose Model")

# Only show available models
model_options = []
if beit_model is not None and processor is not None:
    model_options.append("BEiT (Pretrained Transformer)")
if cnn_model is not None and le is not None:
    model_options.append("CNN (My Model)")

if not model_options:
    st.error("❌ No models available. Please check the error messages above.")
    st.info("""
    **Troubleshooting tips:**
    1. Make sure all dependencies are in requirements.txt
    2. Check that model files are accessible on Google Drive
    3. Refresh the page to retry downloading
    """)
    st.stop()

model_choice = st.sidebar.radio("Select Model:", model_options)

st.title("😊 Emotion Detection App")
st.write("Upload an image and see the predicted emotion with probabilities.")

uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ----------------- BEiT Prediction -----------------
    if model_choice == "BEiT (Pretrained Transformer)" and beit_model and processor:
        try:
            with st.spinner("Analyzing with BEiT model..."):
                # Process image
                inputs = processor(images=image.convert("RGB"), return_tensors="pt")
                
                # Make prediction
                with torch.no_grad():
                    outputs = beit_model(**inputs)
                
                # Get probabilities
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                pred_idx = probs.argmax().item()
                label_name = beit_model.config.id2label[pred_idx]
                predicted_emotion = beit_label_map.get(label_name, label_name)
                confidence = probs[pred_idx].item() * 100
                
                st.success(f"✅ Predicted Emotion (BEiT): {predicted_emotion}")
                st.info(f"📊 Confidence: {confidence:.2f}%")
                
                # Show top 3 predictions
                top3_probs, top3_indices = torch.topk(probs, 3)
                st.write("💡 Top 3 Predictions (BEiT):")
                for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
                    emotion_name = beit_label_map.get(beit_model.config.id2label[idx.item()], 
                                                    beit_model.config.id2label[idx.item()])
                    st.write(f"{i+1}. {emotion_name}: {prob.item()*100:.2f}%")
                    
        except Exception as e:
            st.error(f"❌ BEiT prediction failed: {e}")

    # ----------------- CNN Prediction -----------------
    elif model_choice == "CNN (My Model)" and cnn_model and le:
        try:
            with st.spinner("Analyzing with CNN model..."):
                # Preprocess image
                img_gray = image.convert("L").resize((48, 48))
                img_array = np.array(img_gray).reshape(1, 48, 48, 1) / 255.0
                
                # Make prediction
                pred = cnn_model.predict(img_array, verbose=0)[0]
                pred_idx = pred.argmax()
                label_name = le.inverse_transform([pred_idx])[0]
                predicted_emotion = emotion_map.get(label_name, label_name)
                confidence = pred[pred_idx] * 100
                
                st.success(f"✅ Predicted Emotion (CNN): {predicted_emotion}")
                st.info(f"📊 Confidence: {confidence:.2f}%")
                
                # Show top 3 predictions
                top3_indices = pred.argsort()[-3:][::-1]
                st.write("💡 Top 3 Predictions (CNN):")
                for i, idx in enumerate(top3_indices):
                    emotion_name = emotion_map.get(le.inverse_transform([idx])[0], le.inverse_transform([idx])[0])
                    st.write(f"{i+1}. {emotion_name}: {pred[idx]*100:.2f}%")
                
                # Create visualization
                fig, ax = plt.subplots(figsize=(10, 5))
                emotions = [emotion_map.get(cls, cls) for cls in le.classes_]
                ax.bar(emotions, pred, color='lightgreen', alpha=0.8)
                ax.set_ylabel('Probability')
                ax.set_xlabel('Emotions')
                ax.set_title('Emotion Probability Distribution (CNN)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"❌ CNN prediction failed: {e}")

else:
    st.info("👆 Please upload an image to get started!")

# Add requirements info
with st.expander("📋 Required Dependencies"):
    st.code("""
streamlit>=1.27.0
tensorflow>=2.12.0
torch>=2.1.0
transformers>=4.43.0
Pillow>=9.0.0
numpy>=1.26.0
matplotlib>=3.8.1
pandas>=2.1.0
scipy>=1.11.0
scikit-learn>=1.3.0
requests>=2.31.0
    """)
    st.write("Make sure these are in your `requirements.txt` file")