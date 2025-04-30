import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging

# Configure logging
LOG_FILE = "./logs/train_pipeline.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_params():
    """
    Load random seed for dataset split from params.yaml.
    """
    try:
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        return params["split"]["seed"]
    except Exception as e:
        logging.exception("Failed to load split seed from params.yaml")
        raise

def gather_image_paths(data_dir):
    """
    Traverse the processed data directory and collect image paths with corresponding labels.
    """
    records = []
    for label_name, label_val in [("positive", 1), ("negative", 0)]:
        label_folder = Path(data_dir) / label_name
        if not label_folder.exists():
            logging.warning(f"Folder not found: {label_folder}")
            continue

        for img_file in label_folder.glob("*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                records.append({
                    "path": str(img_file),
                    "label": label_val
                })

    df = pd.DataFrame(records)
    logging.info(f"Found {len(df)} images in {data_dir}")
    return df

def split_and_save(df, output_dir, seed=42):
    """
    Split the data into train, validation, and test sets and save to CSV files.
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=seed)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=seed)

        train_df.to_csv(output_dir / "train.csv", index=False)
        val_df.to_csv(output_dir / "val.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)

        logging.info(f"Split saved: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples.")
    except Exception as e:
        logging.exception("Failed during data splitting and saving")

def main():
    try:
        data_dir = "data/processed_data"
        output_dir = "artifacts/metadata"

        logging.info("Starting image path gathering...")
        df = gather_image_paths(data_dir)
        if df.empty:
            logging.error("No images found in data directory.")
            return

        seed = load_params()
        logging.info("Performing stratified train/val/test split...")
        split_and_save(df, output_dir, seed=seed)

    except Exception as e:
        logging.exception("Unhandled exception in train pipeline")

if __name__ == "__main__":
    main()
