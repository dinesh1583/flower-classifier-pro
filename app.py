import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# ------------------------
# Page Configuration
# ------------------------
st.set_page_config(
    page_title="🌸 Flower Classifier Pro",
    page_icon="🌸",
    layout="centered"
)

# ------------------------
# Dark Theme Styling
# ------------------------
st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
            color: white;
        }
        h1 {
            text-align: center;
            color: #FF4B4B;
        }
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌸 Flower Classification using Deep Learning")
st.write("Upload a flower image and the model will predict the type.")

# ------------------------
# Google Drive Model Setup
# ------------------------

FILE_ID = "1W6EIxRVzbgD1er16oUA7-Z0ERRaOwB9D"
MODEL_PATH = "flower_model_pro.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... Please wait ⏳"):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# ------------------------
# Load Model (Cached)
# ------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# ------------------------
# Upload Image
# ------------------------
uploaded_file = st.file_uploader("📤 Upload Flower Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((224, 224))

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image_resized)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]).numpy()

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.markdown("---")
    st.subheader("🔍 Prediction Result")
    st.success(f"🌼 Predicted Flower: {predicted_class}")
    st.info(f"📊 Confidence: {confidence:.2f}%")