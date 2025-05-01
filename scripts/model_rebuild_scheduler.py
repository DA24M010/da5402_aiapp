import os
import psycopg2
import subprocess
from dotenv import load_dotenv
from ingestion_pipeline.run_ingest import ingest_data
from time import sleep
import logging
import traceback

# Setup logging
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    filename="./logs/model_rebuild.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables
load_dotenv()

# Get config from environment
POSTGRES_HOST = os.environ["POSTGRES_HOST"]
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", 5432)
POSTGRES_DB = os.environ["POSTGRES_DB"]
POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
FEEDBACK_MOVE_LIMIT = int(os.environ.get("FEEDBACK_MOVE_LIMIT", 100))


def connect_db():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )


def check_and_ingest():
    conn = None
    cur = None
    try:
        conn = connect_db()
        cur = conn.cursor()

        # Count how many feedback entries are available
        cur.execute("SELECT COUNT(*) FROM feedback_images;")
        count = cur.fetchone()[0]
        logging.info(f"Found {count} feedback entries in DB")

        if count >= FEEDBACK_MOVE_LIMIT:
            logging.info(f"Threshold of {FEEDBACK_MOVE_LIMIT} reached, triggering ingestion...")
            ingest_data(conn, FEEDBACK_MOVE_LIMIT)
            logging.info("Data ingestion complete. Running dvc repro...")

            env = os.environ.copy()
            env["MKL_SERVICE_FORCE_INTEL"] = "1"

            subprocess.run(["dvc", "repro"], check=True, env=env)
            logging.info("dvc repro complete. Stopping old MLflow serve and starting new serve.")

            subprocess.run("pkill -f 'mlflow models serve'", shell=True)
            sleep(2)

            subprocess.Popen([
                "mlflow", "models", "serve", 
                "-m", "models:/surface_crack_detector/Production", 
                "--host", "0.0.0.0", "-p", "5001", "--no-conda"
            ])
            logging.info("New MLflow model serving started.")
        else:
            logging.info(f"Not enough feedback entries to process. Current count: {count}")

    except Exception as e:
        logging.error(f"Error during ingestion and model rebuild: {str(e)}")
        logging.error(traceback.format_exc())
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            logging.info("Database connection closed.")


if __name__ == "__main__":
    check_and_ingest()
