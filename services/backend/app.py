from fastapi import FastAPI
from fastapi import Response
import torch
from cnnmodel import CNNModel
from pydantic import BaseModel
from utils import predict_image, decode_base64_image, save_feedback_image
import base64
import io
from PIL import Image
import requests
import torchvision.transforms as transforms
import os
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from create_feedback_table import create_feedback_table

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    create_feedback_table()  # Ensure table is created before handling any request

# Define request models
class PredictionRequest(BaseModel):
    image: str  # base64 encoded image

class FeedbackRequest(BaseModel):
    image: str
    correct_label: int

# Transformation (same as during training)
transform = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

MLFLOW_MODEL_SERVER_URL = os.getenv("MLFLOW_MODEL_SERVER_URL")
# Define metrics
REQUEST_COUNT = Counter('inference_requests_total', 'Total number of inference requests')
SUCCESSFUL_PREDICTIONS = Counter('successful_predictions_total', 'Successful predictions')
FAILED_PREDICTIONS = Counter('failed_predictions_total', 'Failed predictions')


# Prediction function
def predict_image(image):
    REQUEST_COUNT.inc()

    # Preprocess the image
    img = transform(image).unsqueeze(0)  # Add batch dimension
    img_list = img.tolist()

    # Prepare payload
    payload = {
        "instances": img_list  # MLflow expects "instances"
    }
    headers = {"Content-Type": "application/json"}

    # Send request to MLflow model server
    response = requests.post(MLFLOW_MODEL_SERVER_URL, json=payload, headers=headers)

    # Check and parse response
    if response.status_code == 200:
        SUCCESSFUL_PREDICTIONS.inc()
        preds = response.json()["predictions"]
        print(preds)
        # Assuming the model outputs a single probability
        return int(preds[0][0] > 0.5)
    else:
        FAILED_PREDICTIONS.inc()
        raise Exception(f"MLflow server error: {response.text}")

# Decode base64 image
def decode_base64_image(base64_string):
    image_bytes = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return img

@app.get("/")
def home():
    return {"message": "Backend is running!"}

@app.post("/predict/")
async def predict(request: PredictionRequest):
    label_map = ['Negative', 'Positive']
    img = decode_base64_image(request.image)
    prediction = predict_image(img)
    return {"prediction": label_map[prediction]}

DB_URI = os.getenv("DATABASE_URL")
@app.post("/feedback/")
async def feedback(request: FeedbackRequest):
    save_feedback_image(request.image, request.correct_label, DB_URI)
    return {"message": "Feedback received!"}

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
