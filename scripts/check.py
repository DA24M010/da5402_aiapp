import os
import psycopg2
import base64
from datetime import datetime

from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

def extract_and_save_images():
    try:
        connection = psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", "postgres"),
            port=os.environ.get("POSTGRES_PORT", "5432"),
            database=os.environ.get("POSTGRES_DB", "your_database_name"),
            user=os.environ.get("POSTGRES_USER", "your_user"),
            password=os.environ.get("POSTGRES_PASSWORD", "your_password")
        )

        cursor = connection.cursor()
        cursor.execute("SELECT id, image_data, label, timestamp FROM feedback_images LIMIT 5;")
        rows = cursor.fetchall()

        output_dir = "extracted_images"
        os.makedirs(output_dir, exist_ok=True)

        for row in rows:
            img_id, image_data, label, ts = row
            filename = f"{output_dir}/image_{img_id}_{label}_{ts.strftime('%Y%m%d%H%M%S')}.jpg"

            # If image_data is base64-encoded string, decode it
            if isinstance(image_data, str):
                img_bytes = base64.b64decode(image_data)
            else:
                img_bytes = image_data  # assume it's raw bytes (bytea in Postgres)

            with open(filename, "wb") as f:
                f.write(img_bytes)
            print(f"Saved: {filename}")

    except Exception as e:
        print(f"Error during extraction: {e}")

    finally:
        if 'connection' in locals():
            connection.close()

if __name__ == "__main__":
    extract_and_save_images()
