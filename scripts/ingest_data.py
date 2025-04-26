import os
import shutil
from pathlib import Path
import kagglehub

def download_initial_dataset(data_dir):
    print("[INFO] 'data/' folder not found. Downloading dataset from KaggleHub...")
    path = kagglehub.dataset_download("arunrk7/surface-crack-detection")
    print(f"[INFO] Dataset downloaded to: {path}")

    # Copy Positive and Negative folders from downloaded path to ./data
    for label in ["Positive", "Negative"]:
        src = Path(path) / label
        dst = Path(data_dir) / label.lower()
        dst.mkdir(parents=True, exist_ok=True)
        if src.exists():
            for file in src.glob("*"):
                shutil.move(str(file), dst)
            print(f"[OK] Copied {label} images to {dst}")
        else:
            print(f"[WARN] No '{label}' folder found in downloaded dataset.")

def ensure_main_folders_exist(data_dir):
    for label in ["positive", "negative"]:
        folder = Path(data_dir) / label
        folder.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Ensured folder: {folder}")

def move_images_from_new_data(new_data_dir, data_dir):
    for new_label in ["positive", "negative"]:
        src_folder = Path(new_data_dir) / new_label
        if not src_folder.exists() or not src_folder.is_dir():
            print(f"[SKIP] Folder '{new_label}' not found in new_data/.")
            continue

        dest_folder = Path(data_dir) / new_label.lower()

        image_files = list(src_folder.glob("*"))
        if not image_files:
            print(f"[SKIP] No images found in {src_folder}")
            continue

        for img_path in image_files:
            if img_path.is_file():
                dest_path = dest_folder / img_path.name
                if not dest_path.exists():
                    shutil.move(str(img_path), str(dest_path))
                    print(f"[MOVED] {img_path.name} → {dest_path}")
                else:
                    print(f"[EXISTS] Skipping existing file: {img_path.name}")

        # Clean up folder after moving
        # try:
        #     src_folder.rmdir()
        #     print(f"[CLEANED] Removed empty folder: {src_folder}")
        # except OSError:
        #     print(f"[NOTE] Folder not empty (some files might be skipped): {src_folder}")

def main():
    data_dir = "data"
    new_data_dir = "new_data"

    if not Path(data_dir).exists():
        download_initial_dataset(data_dir)

    ensure_main_folders_exist(data_dir)

    if Path(new_data_dir).exists():
        move_images_from_new_data(new_data_dir, data_dir)
    else:
        print(f"[INFO] No new_data/ folder found. Nothing to ingest.")

if __name__ == "__main__":
    main()
