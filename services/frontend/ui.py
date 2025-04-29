import streamlit as st
import requests
import base64
import uuid

backend_url = "http://backend:8000"  # Change if needed

st.title("Surface Crack Detection")

# Initialize session state
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())

# File uploader with key
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key=st.session_state.uploader_key)

# Save uploaded file to session
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file

# Only show image if uploaded_file exists
if st.session_state.uploaded_file is not None:
    st.image(st.session_state.uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Predict button
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            response = requests.post(
                f"{backend_url}/predict/",
                json={"image": base64.b64encode(st.session_state.uploaded_file.getvalue()).decode('utf-8')}
            )
            if response.status_code == 200:
                result = response.json()
                st.session_state.prediction = result["prediction"]
            else:
                st.error("Prediction failed. Try again.")

# After prediction
if st.session_state.prediction is not None:
    st.success(f"Prediction: {st.session_state.prediction}")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Flag Wrong Prediction"):
            # Find correct label (opposite of model's prediction)
            correct_label = 0 if st.session_state.prediction == "Positive" else 1

            feedback_response = requests.post(
                f"{backend_url}/feedback/",
                json={
                    "image": base64.b64encode(st.session_state.uploaded_file.getvalue()).decode('utf-8'),
                    "correct_label": correct_label
                }
            )
            if feedback_response.status_code == 200:
                st.success("Feedback sent! Thank you!")
            else:
                st.error("Failed to send feedback.")

            # Reset everything
            st.session_state.prediction = None
            st.session_state.uploaded_file = None
            st.session_state.uploader_key = str(uuid.uuid4())  # new key to clear uploader
            st.rerun()

    with col2:
        if st.button("Clear"):
            # Reset everything
            st.session_state.prediction = None
            st.session_state.uploaded_file = None
            st.session_state.uploader_key = str(uuid.uuid4())  # new key to clear uploader
            st.rerun()
