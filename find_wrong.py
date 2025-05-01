import requests
import os
from PIL import Image
import shutil
from torchvision import transforms

# Define image transforms
image_size = 120
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to process each image and send it to the model for prediction
def process_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    img_list = img_tensor.tolist()  # Convert tensor to list

    payload = {
        "inputs": img_list
    }
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(
        "http://127.0.0.1:5001/invocations", 
        json=payload, 
        headers=headers
    )

    if response.status_code == 200:
        preds = response.json()["predictions"]
        return int(preds[0][0] > 0.5), preds[0][0]  # binary_pred, raw_score
    else:
        raise Exception(f"MLflow server error: {response.text}")

# Paths
temp_base = './temp'
incorrect_base = './incorrect_pred'
classes = ['negative']

# Process images
for cls in classes:
    src_folder = os.path.join(temp_base, cls)
    dst_folder = os.path.join(incorrect_base, cls)
    os.makedirs(dst_folder, exist_ok=True)

    true_label = 1 if cls == 'positive' else 0

    for img_filename in os.listdir(src_folder):
        img_path = os.path.join(src_folder, img_filename)
        if os.path.isfile(img_path):
            try:
                pred_label, pred_score = process_image(img_path)
                if pred_label == true_label:
                    os.remove(img_path)
                    print(f"✅ Correct - Deleted: {img_filename}")
                else:
                    shutil.move(img_path, os.path.join(dst_folder, img_filename))
                    print(f"❌ Wrong - Moved to incorrect_pred/{cls}: {img_filename} (Pred: {pred_score:.2f})")
            except Exception as e:
                print(f"Error processing {img_filename}: {e}")
