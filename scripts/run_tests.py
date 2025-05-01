import requests
import os
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# Define image transforms
image_size = 120
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to process each image and send it to the model for prediction
def process_image(image_path):
    # Open image
    img = Image.open(image_path).convert('RGB')
    # Transform image to tensor
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    img_list = img_tensor.tolist()  # Convert tensor to list
    
    # Prepare payload with proper field name for MLflow 2.0
    payload = {
        "inputs": img_list  # Use "inputs" instead of "instances" in MLflow 2.0
    }
    headers = {"Content-Type": "application/json"}
    
    # Send image to MLflow model server
    response = requests.post(
        "http://127.0.0.1:5001/invocations",  # Assuming local MLflow model server
        json=payload, 
        headers=headers
    )
    
    # Check and parse response
    if response.status_code == 200:
        preds = response.json()["predictions"]
        print(preds)
        # Assuming the model outputs a single probability
        return int(preds[0][0] > 0.5), preds[0][0]  # Return both binary prediction and raw probability
    else:
        raise Exception(f"MLflow server error: {response.text}")

# Directory where test images are stored
test_folder = 'test_samples'

# Prepare output file to store results
output_file = 'predictions.txt'

# Store results and images for plotting
image_paths = []
predictions = []
pred_values = []

# Open the file in write mode
with open(output_file, 'w') as f:
    # Loop through all images in the test folder and process them
    for img_filename in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_filename)
        if os.path.isfile(img_path):
            result, pred_value = process_image(img_path)
            result = 'Negative' if result == 0 else 'Positive'
            image_paths.append(img_path)
            predictions.append(result)
            pred_values.append(pred_value)
            # Write the result to the file
            f.write(f"Image: {img_filename}, Prediction: {result}, Raw Prediction Value: {pred_value}\n")

# Plot 4 sample images and their predictions
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Plot up to 4 images
for i in range(min(4, len(image_paths))):
    img = Image.open(image_paths[i]).convert('RGB')
    ax = axes[i]
    ax.imshow(img)
    ax.set_title(f"Prediction: {predictions[i]}\nValue: {pred_values[i]:.2f}")
    ax.axis('off')

# Save the plot
plt.tight_layout()
plt.savefig('./artifacts/unit_test.png')
print("Predictions saved to predictions.txt and plot saved as unit_test.png")
