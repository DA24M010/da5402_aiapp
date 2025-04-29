from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]  # Go one level up to the scripts folder
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ingestion_pipeline.download_dataset import download_initial_dataset
from ingestion_pipeline.feedback_handler import move_feedback_data
from ingestion_pipeline.dvc_utils import track_and_push_data
from ingestion_pipeline.data_eda import generate_eda

DATA_DIR = Path('./data/raw_data/')

def ingest_data(conn=None, limit =100):
    if not DATA_DIR.exists():
        print("[INFO] raw_data not found. Downloading...")
        download_initial_dataset(DATA_DIR)
    else:
        print("[OK] raw_data already exists")
    
    if conn != None:
        moved = move_feedback_data(DATA_DIR, conn=conn, limit = limit)
        if moved:
            print("Moved")
            # track_and_push_data()
        else:
            print("[INFO] No new feedback data to ingest.")
    else:
        print("[INFO] No feedback_data folder. Skipping.")

    generate_eda(DATA_DIR)

if __name__ == "__main__":
    ingest_data(None)
