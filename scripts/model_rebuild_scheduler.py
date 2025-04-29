import os
import psycopg2
from dotenv import load_dotenv
from ingestion_pipeline.run_ingest import ingest_data

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
    try:
        conn = connect_db()
        cur = conn.cursor()

        # Count how many feedback entries are available
        cur.execute("SELECT COUNT(*) FROM feedback_images;")
        count = cur.fetchone()[0]
        print(f"[INFO] Found {count} feedback entries in DB")

        if count >= FEEDBACK_MOVE_LIMIT:
            print(f"[INFO] Threshold of {FEEDBACK_MOVE_LIMIT} reached, triggering ingestion...")
            ingest_data(conn, FEEDBACK_MOVE_LIMIT)  # Call the separate ingestion logic using the open connection
        else:
            print("[INFO] Not enough feedback entries to process.")

    except Exception as e:
        print(f"[ERROR] Could not check feedback entries: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()


if __name__ == "__main__":
    check_and_ingest()
