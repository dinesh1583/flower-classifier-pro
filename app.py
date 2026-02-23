import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="🌸 Flower Classifier Pro",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Flower Classification using Deep Learning")
st.write("Upload a flower image and the model will predict the type.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("flower_model_pro.h5")

model = load_model()

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

uploaded_file = st.file_uploader("Upload Flower Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))

    st.image(image)

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]).numpy()

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.success(f"Predicted: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")