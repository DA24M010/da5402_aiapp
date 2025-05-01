from pathlib import Path
import sys
import logging

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ingestion_pipeline.download_dataset import download_initial_dataset
from ingestion_pipeline.feedback_handler import move_feedback_data
from ingestion_pipeline.dvc_utils import track_and_push_data
from ingestion_pipeline.data_eda import generate_eda

# Set up logging
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / "model_rebuild.log"

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

DATA_DIR = Path('./data/raw_data/')

def ingest_data(conn=None, limit=100):
    try:
        if not DATA_DIR.exists():
            logging.info("raw_data not found. Downloading...")
            download_initial_dataset(DATA_DIR)
        else:
            logging.info("raw_data already exists")
        
        if conn is not None:
            moved = move_feedback_data(DATA_DIR, conn=conn, limit=limit)
            if moved:
                logging.info("Feedback data moved successfully")
                track_and_push_data()
                logging.info("DVC track and push completed")
            else:
                logging.info("No new feedback data to ingest.")
        else:
            logging.info("No DB connection provided. Skipping feedback ingestion.")

        generate_eda(DATA_DIR)
        logging.info("EDA report generated successfully")

    except Exception as e:
        logging.exception(f"Error during ingestion pipeline: {e}")

if __name__ == "__main__":
    ingest_data(None)
