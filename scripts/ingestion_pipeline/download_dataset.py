# ingest_pipeline/download_dataset.py
import shutil
from pathlib import Path
import kagglehub

def download_initial_dataset(DATA_DIR):
    print("[INFO] Downloading dataset from KaggleHub...")
    path = kagglehub.dataset_download("arunrk7/surface-crack-detection")
    print(f"[INFO] Dataset downloaded to: {path}")

    for label in ["Positive", "Negative"]:
        src = Path(path) / label
        dst = DATA_DIR / label.lower()
        dst.mkdir(parents=True, exist_ok=True)
        if src.exists():
            for file in src.glob("*"):
                shutil.move(str(file), dst)
            print(f"[OK] Copied {label} images to {dst}")
        else:
            print(f"[WARN] No '{label}' folder found in downloaded dataset.")
