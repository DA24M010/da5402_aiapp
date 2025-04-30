import os
import base64
import psycopg2
import shutil
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import logging

load_dotenv()

# Set up logging
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / "ingest.log"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

def move_feedback_data(DATA_DIR, conn=None, limit=100):
    if conn is None:
        logging.warning("No DB connection provided, skipping feedback fetch.")
        return False

    moved = False
    try:
        positive_dir = DATA_DIR / "positive"
        negative_dir = DATA_DIR / "negative"

        cur = conn.cursor()
        cur.execute(f"""
            SELECT id, image_data, label, timestamp
            FROM feedback_images
            ORDER BY timestamp ASC
            LIMIT {limit}
        """)
        rows = cur.fetchall()

        if len(rows) < limit:
            logging.info(f"Only {len(rows)} feedback entries found. Skipping.")
            return False

        ids_to_delete = []
        for row in rows:
            row_id, image_data, label, timestamp = row
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
            file_name = f"{timestamp_str}.jpg"
            # If image_data is base64-encoded string, decode it
            if isinstance(image_data, str):
                img_bytes = base64.b64decode(image_data)
            else:
                img_bytes = image_data  # assume it's raw bytes (bytea in Postgres)

            if label == 0:
                out_path = negative_dir / file_name
            elif label == 1:
                out_path = positive_dir / file_name
            else:
                logging.warning(f"Invalid label: {label} for row ID: {row_id}")
                continue

            with open(out_path, "wb") as f:
                f.write(img_bytes)
            moved = True
            ids_to_delete.append(row_id)

        if ids_to_delete:
            cur.execute("DELETE FROM feedback_images WHERE id IN %s", (tuple(ids_to_delete),))
            conn.commit()
            logging.info(f"Moved and deleted {len(ids_to_delete)} entries from feedback_images.")

    except Exception as e:
        logging.error(f"Error while moving feedback data: {e}", exc_info=True)
    finally:
        if conn:
            cur.close()

    return moved
