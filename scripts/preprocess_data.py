import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import shutil

def move_invalid_image(img_path, temp_dir, label):
    temp_label_dir = temp_dir / label
    temp_label_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(img_path), temp_label_dir / img_path.name)

def validate_images_in_folder(folder_path, temp_dir, label):
    invalid_log = []

    for img_file in folder_path.glob("*"):
        if not img_file.is_file():
            continue

        try:
            with Image.open(img_file) as img:
                img.verify()
        except (UnidentifiedImageError, OSError) as e:
            print(f"[INVALID] {img_file} - {e}")
            invalid_log.append(str(img_file))
            move_invalid_image(img_file, temp_dir, label)

    return invalid_log

def main():
    data_dir = Path("data")
    temp_dir = Path("temp")
    labels = ["positive", "negative"]
    all_invalid = []

    for label in labels:
        label_path = data_dir / label
        if not label_path.exists():
            print(f"[SKIP] Folder not found: {label_path}")
            continue
        print(f"[CHECK] Validating images in '{label_path}'...")
        invalids = validate_images_in_folder(label_path, temp_dir, label)
        all_invalid.extend(invalids)

    if all_invalid:
        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)
        with open(log_path / "invalid_images.txt", "w") as f:
            f.write("\n".join(all_invalid))
        print(f"[LOGGED] Invalid images moved to 'temp/' and logged in 'logs/invalid_images.txt'")
    else:
        print(f"[OK] All images are valid.")

if __name__ == "__main__":
    main()
