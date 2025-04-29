import base64
import io
from PIL import Image
import torch
import torchvision.transforms as transforms

# Define image transforms
image_size = 120
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(model, image):
    """Run prediction on a PIL image."""
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        pred = (output.view(-1) > 0.5).float()  # Binary classification threshold
    return int(pred.item())

def decode_base64_image(base64_string):
    """Decode base64 string to a PIL image."""
    image_bytes = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return img

import base64
import psycopg2
from datetime import datetime

def save_feedback_image(base64_string, label, db_url):
# def save_feedback_image(base64_string, label, save_dir="./artifacts/feedback/"):
    """Save base64 image with label into feedback folder."""
    # import os
    # from datetime import datetime
    
    # os.makedirs(save_dir, exist_ok=True)
    
    # image_bytes = base64.b64decode(base64_string)
    # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # filepath = os.path.join(save_dir, f"{timestamp}_label{label}.jpg")
    
    # with open(filepath, "wb") as f:
    #     f.write(image_bytes)
    
    # return filepath

    image_bytes = base64.b64decode(base64_string)
    
    # Connect to the PostgreSQL database
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Prepare the SQL statement to insert the image data
        insert_query = """
        INSERT INTO feedback_images (image_data, label, timestamp)
        VALUES (%s, %s, %s);
        """
        
        # Get current timestamp
        timestamp = datetime.now()
        
        # Execute the insert query
        cursor.execute(insert_query, (psycopg2.Binary(image_bytes), label, timestamp))
        
        # Commit the transaction
        conn.commit()
        
        # Close the cursor and the connection
        cursor.close()
        conn.close()
        
        print(f"Feedback image with label {label} saved successfully at {timestamp}")
    except Exception as e:
        print(f"Error saving feedback to PostgreSQL: {e}")

