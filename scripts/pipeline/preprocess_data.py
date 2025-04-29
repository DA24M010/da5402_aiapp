# preprocess_data.py
import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import shutil

def validate_and_resize_image(img_path, output_dir, label, target_size=(120, 120)):
    try:
        with Image.open(img_path) as img:
            img.verify()  # Check if image is valid
        with Image.open(img_path) as img:  # Reopen for resizing (PIL needs reopen after verify)
            img = img.convert('RGB')
            img = img.resize(target_size)
            output_label_dir = output_dir / label
            output_label_dir.mkdir(parents=True, exist_ok=True)
            img.save(output_label_dir / img_path.name)
        return True
    except (UnidentifiedImageError, OSError) as e:
        print(f"[INVALID] {img_path} - {e}")
        return False

def main():
    raw_data_dir = Path("./data/raw_data")      # raw data input
    processed_data_dir = Path("./data/processed_data")  # output folder
    labels = ["positive", "negative"]
    all_invalid = []

    if processed_data_dir.exists():
        shutil.rmtree(processed_data_dir)

    for label in labels:
        label_path = raw_data_dir / label
        if not label_path.exists():
            print(f"[SKIP] Folder not found: {label_path}")
            continue

        print(f"[CHECK] Processing images in '{label_path}'...")
        for img_file in label_path.glob("*"):
            if not img_file.is_file():
                continue

            valid = validate_and_resize_image(img_file, processed_data_dir, label)
            if not valid:
                all_invalid.append(str(img_file))

    log_path = Path("./logs")
    log_path.mkdir(exist_ok=True)

    # Save invalid file list if any
    if all_invalid:
        with open(log_path / "invalid_images.txt", "w") as f:
            f.write("\n".join(all_invalid))
        print(f"[LOGGED] Invalid images moved to 'temp/' and logged in 'logs/invalid_images.txt'")
    else:
        print(f"[OK] All images were valid and resized.")

if __name__ == "__main__":
    main()
