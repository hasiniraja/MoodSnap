import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
import os
import requests

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
cnn_model, le = None, None
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
processor, beit_model = None, None
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

# ----------------- Google Places Helper -----------------
emotion_to_place = {
    "😀 Happy": "cafe",
    "😢 Sad": "park",
    "😠 Anger": "gym",
    "😐 Neutral": "library",
    "😨 Fear": "temple",
    "🤢 Disgust": "museum",
    "😲 Surprise": "amusement_park"
}

def get_places(api_key, location, place_type):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "key": api_key,
        "location": f"{location[0]},{location[1]}",
        "radius": 15000,  # 15 km
        "type": place_type
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "OK":
            st.error(f"Google Places API error: {data.get('status')} - {data.get('error_message', 'No error message')}")
            return []

        results = []
        for place in data.get("results", [])[:5]:
            name = place.get("name")
            address = place.get("vicinity", "Address not available")
            if "geometry" in place and "location" in place["geometry"]:
                lat = place["geometry"]["location"]["lat"]
                lng = place["geometry"]["location"]["lng"]
                maps_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lng}&query_place_id={place.get('place_id', '')}"
                results.append((name, address, maps_url))
        return results

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        return []
    except ValueError as e:
        st.error(f"Failed to parse JSON response: {e}")
        return []

# ----------------- Location Helper -----------------
from streamlit_js_eval import streamlit_js_eval

def get_user_location():
    """Try to detect user location (browser first, then IP, then Delhi)."""

    # 1. Try browser geolocation
    try:
        lat = streamlit_js_eval(
            js_expressions="navigator.geolocation.getCurrentPosition((pos)=>pos.coords.latitude)",
            key="lat"
        )
        lon = streamlit_js_eval(
            js_expressions="navigator.geolocation.getCurrentPosition((pos)=>pos.coords.longitude)",
            key="lon"
        )
        if lat and lon:
            return float(lat), float(lon)
    except Exception as e:
        st.warning(f"⚠️ Browser geolocation failed: {e}")

    # 2. Try IP-based location
    try:
        response = requests.get("https://ipapi.co/json/", timeout=5)
        data = response.json()
        if "latitude" in data and "longitude" in data:
            return float(data["latitude"]), float(data["longitude"])
    except Exception as e:
        st.warning(f"⚠️ IP-based location failed: {e}")

    # 3. Fallback to New Delhi
    return 26.8500, 81.0000

# ----------------- Streamlit UI -----------------
st.sidebar.title("⚙️ Choose Model")
model_options = []
if beit_model is not None and processor is not None:
    model_options.append("BEiT (Pretrained Transformer)")
if cnn_model is not None and le is not None:
    model_options.append("CNN (My Model)")

if not model_options:
    st.error("❌ No models available. Please check errors above.")
    st.stop()

model_choice = st.sidebar.radio("Select Model:", model_options)
st.title("😊 Emotion Detection App + Recommendations")
st.write("Capture a photo with your webcam or upload an image to see the predicted emotion & get recommendations!")

# ----------------- Capture or Upload -----------------
st.subheader("📷 Capture or Upload Image")
camera_photo = st.camera_input("Take a picture")
uploaded_file = st.file_uploader("Or upload an image...", type=["jpg", "png", "jpeg"])

image = None
if camera_photo is not None:
    image = Image.open(camera_photo)
    st.image(image, caption="Captured Image", use_column_width=True)
elif uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# ----------------- Prediction -----------------
predicted_emotion = None
if image is not None:
    if model_choice == "BEiT (Pretrained Transformer)" and beit_model and processor:
        with st.spinner("Analyzing with BEiT model..."):
            inputs = processor(images=image.convert("RGB"), return_tensors="pt")
            with torch.no_grad():
                outputs = beit_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            pred_idx = probs.argmax().item()
            label_name = beit_model.config.id2label[pred_idx]
            predicted_emotion = beit_label_map.get(label_name, label_name)
            st.success(f"✅ Predicted Emotion (BEiT): {predicted_emotion}")
    elif model_choice == "CNN (My Model)" and cnn_model and le:
        with st.spinner("Analyzing with CNN model..."):
            img_gray = image.convert("L").resize((48, 48))
            img_array = np.array(img_gray).reshape(1, 48, 48, 1) / 255.0
            pred = cnn_model.predict(img_array, verbose=0)[0]
            pred_idx = pred.argmax()
            label_name = le.inverse_transform([pred_idx])[0]
            predicted_emotion = emotion_map.get(label_name, label_name)
            st.success(f"✅ Predicted Emotion (CNN): {predicted_emotion}")

# ----------------- Google Maps Recommendations -----------------
if predicted_emotion:
    st.subheader("📍 Smart Recommendations (Google Maps)")
    if "GOOGLE_MAPS_API_KEY" not in st.secrets:
        st.error("Google Maps API key not found. Please add it to your Streamlit secrets.")
        st.sidebar.subheader("Google Maps API Setup")
        st.sidebar.info("""
        To enable location recommendations:
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a project and enable Places API
        3. Generate an API key
        4. Add it to your Streamlit secrets as GOOGLE_MAPS_API_KEY
        """)
    else:
        api_key = st.secrets["GOOGLE_MAPS_API_KEY"]

        # Auto-detect location with manual override
        default_lat, default_lon = get_user_location()
        st.markdown("### 🌍 Location Settings")
        use_auto = st.checkbox("📌 Use auto-detected location", value=True)

        if use_auto:
            user_location = (default_lat, default_lon)
            st.success(f"✅ Auto-detected location: {default_lat}, {default_lon}")
        else:
            lat = st.number_input("Latitude", value=default_lat)
            lon = st.number_input("Longitude", value=default_lon)
            user_location = (lat, lon)

        # Place type mapping
        place_type = emotion_to_place.get(predicted_emotion, "cafe")

        # Fetch places
        places = get_places(api_key, user_location, place_type)

        # Fallback to restaurant if no results
        if not places:
            st.warning("⚠️ No results for this emotion type, trying 'restaurant' instead.")
            places = get_places(api_key, user_location, "restaurant")

        if places:
            st.markdown("#### Recommended Places Nearby")
            for name, address, maps_url in places:
                st.markdown(f"**{name}** – {address} 👉 [View on Maps]({maps_url})")
        else:
            st.info("No places found nearby even after fallback. Try another location.")

# ----------------- Dependencies Info -----------------
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
