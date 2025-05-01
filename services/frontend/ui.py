import streamlit as st
import requests
import base64
import uuid

# Backend URL - change if needed
backend_url = "http://backend:8000"

# Title and Instructions
st.title("Surface Crack Detection")
st.markdown("""
    Upload an image of a surface, and our model will detect any cracks. If the prediction is wrong, you can flag it. 
    Make sure the image is in .jpg, .jpeg, or .png format.
""")

# Initialize session state for prediction and uploaded file
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())

# File uploader with key and a tooltip
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key=st.session_state.uploader_key,
                                 help="Upload an image of the surface to check for cracks.")

# Save uploaded file to session
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file

# Only show the uploaded image if exists, and resize it to fit within the window
if st.session_state.uploaded_file is not None:
    # Resize the image to make sure it doesn't overflow (width set to 700px or any desired value)
    image = st.session_state.uploaded_file
    st.image(image, caption="Uploaded Image", use_container_width=True, width=100)  # Adjust width here

    # Create columns for buttons to ensure they stay aligned
    col1, col2 = st.columns(2)

    with col1:
        # Predict button with spinner and tooltip
        if st.button("Predict", key="predict_button", help="Click to get a prediction on whether the surface has cracks or not"):
            with st.spinner("Predicting..."):
                response = requests.post(
                    f"{backend_url}/predict/",
                    json={"image": base64.b64encode(st.session_state.uploaded_file.getvalue()).decode('utf-8')}
                )
                if response.status_code == 200:
                    result = response.json()
                    result = result["prediction"]
                    if(result == "Positive"):
                        display_res = "Crack Detected !"
                    else:
                        display_res = "No Crack Detected !!"
                    st.session_state.prediction = display_res
                else:
                    st.error("Prediction failed. Please try again later.")

    with col2:
        # Clear button to reset the state
        if st.button("Clear", key="clear_button", help="Clear the uploaded image and prediction result."):
            # Reset everything
            st.session_state.prediction = None
            st.session_state.uploaded_file = None
            st.session_state.uploader_key = str(uuid.uuid4())  # new key to clear uploader
            st.rerun()

# After prediction
if st.session_state.prediction is not None:
    st.success(f"Prediction: {st.session_state.prediction}")

    col1, col2 = st.columns(2)

    with col1:
        # Flag Wrong Prediction button
        if st.button("Flag Wrong Prediction", key="flag_button", help="Flag the prediction as incorrect and provide feedback."):
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
                st.error("Failed to send feedback. Please try again.")

            # Reset everything
            st.session_state.prediction = None
            st.session_state.uploaded_file = None
            st.session_state.uploader_key = str(uuid.uuid4())  # new key to clear uploader
            st.rerun()

# Custom CSS for better visuals
st.markdown("""
    <style>
        .stButton>button {
            background-color: #007BFF;
            color: white;
            border-radius: 5px;
            height: 3em;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .stFileUploader>div {
            border: 2px dashed #007BFF;
            padding: 20px;
            border-radius: 10px;
        }
        .stFileUploader>div:hover {
            background-color: #f0f8ff;
        }
        .stSpinner {
            color: #007BFF;
        }
    </style>
""", unsafe_allow_html=True)

# Handling edge cases (e.g., slow backend or empty file upload)
if uploaded_file is None:
    st.info("No file uploaded yet. Please upload an image of the surface to detect cracks.")
