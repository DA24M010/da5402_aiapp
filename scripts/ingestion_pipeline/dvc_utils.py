# ingest_pipeline/dvc_utils.py
import subprocess

def track_and_push_data():
    print("[INFO] Running DVC commands...")
    subprocess.run(["dvc", "add", "data/raw_data"], check=True)
    subprocess.run(["git", "add", "data/raw_data.dvc", "data/.gitignore"], check=True)
    subprocess.run(["git", "commit", "-m", "Ingested new feedback data"], check=True)
    subprocess.run(["dvc", "push"], check=True)
    print("[SUCCESS] Data versioned and pushed to remote.")
