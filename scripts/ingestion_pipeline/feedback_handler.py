import os
import base64
import psycopg2
import shutil
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def move_feedback_data(DATA_DIR, conn=None, limit = 100):
    if conn is None:
        print("[WARN] No DB connection provided, skipping feedback fetch.")
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
            print(f"[INFO] Only {len(rows)} feedback entries found. Skipping.")
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
                print(f"[SKIP] Invalid label: {label} for row ID: {row_id}")
                continue

            with open(out_path, "wb") as f:
                f.write(img_bytes)
            moved = True
            ids_to_delete.append(row_id)

        if ids_to_delete:
            cur.execute("DELETE FROM feedback_images WHERE id IN %s", (tuple(ids_to_delete),))
            conn.commit()
            print(f"[INFO] Moved and deleted {len(ids_to_delete)} entries from feedback_images.")

    except Exception as e:
        print(f"[ERROR] While moving feedback data: {e}")
    finally:
        if conn:
            cur.close()
    return moved
