import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

import yaml

def load_params():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    return params["split"]["seed"]

def gather_image_paths(data_dir):
    records = []
    for label_name, label_val in [("positive", 1), ("negative", 0)]:
        label_folder = Path(data_dir) / label_name
        if not label_folder.exists():
            print(f"[WARN] Folder not found: {label_folder}")
            continue

        for img_file in label_folder.glob("*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                records.append({
                    "path": str(img_file),
                    "label": label_val
                })
    return pd.DataFrame(records)

def split_and_save(df, output_dir, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=seed)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"[OK] Saved {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples.")

def main():
    data_dir = "data/processed_data"
    output_dir = "artifacts/metadata"

    print("[INFO] Gathering image paths...")
    df = gather_image_paths(data_dir)
    
    seed = load_params()

    if df.empty:
        print("[ERROR] No images found in data directory.")
        return

    print("[INFO] Performing stratified split...")
    split_and_save(df, output_dir, seed = seed)

if __name__ == "__main__":
    main()
