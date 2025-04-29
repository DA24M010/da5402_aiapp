import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

tf.config.set_visible_devices([], 'GPU')  # Disables GPU

# Load the trained model
model = load_model('./artifacts/detection_model.h5')

# Constants
IMG_SIZE = 120  # Same size used during training
LABELS = ['Negative', 'Positive']

st.set_page_config(page_title="Crack Detection", layout="centered")

st.title("Surface Crack Detection")
st.markdown("Upload an image and let the model predict if it has a crack.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image)
    img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class] * 100

    # Display result
    st.markdown(f"### 🧾 Prediction: `{LABELS[predicted_class]}`")
    st.markdown(f"### 🔍 Confidence: `{confidence:.2f}%`")
