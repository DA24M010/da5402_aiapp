import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import shutil
import logging

# Configure logging
LOG_FILE = "./logs/train_pipeline.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def validate_and_resize_image(img_path, output_dir, label, target_size=(120, 120)):
    """
    Validates and resizes an image to the target size and saves it under the output directory.
    """
    try:
        with Image.open(img_path) as img:
            img.verify()  # Verifies if the image is valid
        with Image.open(img_path) as img:  # Reopen for actual processing
            img = img.convert('RGB')
            img = img.resize(target_size)
            output_label_dir = output_dir / label
            output_label_dir.mkdir(parents=True, exist_ok=True)
            img.save(output_label_dir / img_path.name)
        return True
    except (UnidentifiedImageError, OSError) as e:
        logging.warning(f"Invalid image: {img_path} - {e}")
        return False

def main():
    raw_data_dir = Path("./data/raw_data")
    processed_data_dir = Path("./data/processed_data")
    labels = ["positive", "negative"]
    all_invalid = []

    try:
        if processed_data_dir.exists():
            shutil.rmtree(processed_data_dir)
            logging.info(f"Removed existing processed data directory: {processed_data_dir}")

        for label in labels:
            label_path = raw_data_dir / label
            if not label_path.exists():
                logging.warning(f"Label directory not found: {label_path}")
                continue

            logging.info(f"Processing label directory: {label_path}")
            for img_file in label_path.glob("*"):
                if not img_file.is_file():
                    continue
                valid = validate_and_resize_image(img_file, processed_data_dir, label)
                if not valid:
                    all_invalid.append(str(img_file))

        # Save list of invalid images
        if all_invalid:
            invalid_log_path = Path("./logs/invalid_images.txt")
            with open(invalid_log_path, "w") as f:
                f.write("\n".join(all_invalid))
            logging.info(f"Logged {len(all_invalid)} invalid images to {invalid_log_path}")
        else:
            logging.info("All images were valid and resized successfully.")

    except Exception as e:
        logging.exception(f"An error occurred during preprocessing: {e}")

if __name__ == "__main__":
    main()
