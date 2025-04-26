import os
import random
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def load_metadata(metadata_path):
    return pd.read_csv(metadata_path)

def get_image_dimensions(df):
    widths, heights = [], []
    for path in df["path"]:
        try:
            with Image.open(path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
        except Exception as e:
            print(f"[SKIP] Couldn't open image: {path} | Error: {e}")
    return widths, heights

def plot_histogram(data, title, xlabel, ylabel, out_path):
    plt.figure()
    plt.hist(data, bins=20, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_class_distribution(df, out_path):
    plt.figure()
    df['label'].map({0: "Negative", 1: "Positive"}).value_counts().plot(kind='bar', color=["salmon", "lightgreen"])
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_sample_images(df, label, out_path, max_images=9):
    subset = df[df['label'] == label]
    sample_paths = subset.sample(min(len(subset), max_images))["path"].tolist()

    plt.figure(figsize=(8, 8))
    for idx, path in enumerate(sample_paths):
        try:
            img = Image.open(path)
            plt.subplot(3, 3, idx + 1)
            plt.imshow(img)
            plt.axis('off')
        except:
            continue
    label_str = "Positive" if label == 1 else "Negative"
    plt.suptitle(f"{label_str} Sample Images", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(out_path)
    plt.close()

def save_eda_report(stats, report_path):
    with open(report_path, 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

def main():
    metadata_path = "artifacts/metadata/train.csv"
    artifact_dir = Path("artifacts/eda")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading metadata...")
    df = load_metadata(metadata_path)

    class_counts = df["label"].value_counts().to_dict()
    stats = {
        "Total Samples": len(df),
        "Positive Samples": class_counts.get(1, 0),
        "Negative Samples": class_counts.get(0, 0),
        "Positive/Negative Ratio": round((class_counts.get(1, 0) / (class_counts.get(0, 1) + 1e-5)), 2)
    }

    print("[INFO] Calculating image dimensions...")
    widths, heights = get_image_dimensions(df)
    stats["Avg Width"] = sum(widths) // len(widths)
    stats["Avg Height"] = sum(heights) // len(heights)
    stats["Min Width"] = min(widths)
    stats["Max Width"] = max(widths)
    stats["Min Height"] = min(heights)
    stats["Max Height"] = max(heights)

    print("[INFO] Saving plots...")
    plot_class_distribution(df, artifact_dir / "class_distribution.png")
    plot_sample_images(df, label=0, out_path=artifact_dir / "negative_samples.png")
    plot_sample_images(df, label=1, out_path=artifact_dir / "positive_samples.png")

    print("[INFO] Saving EDA report...")
    save_eda_report(stats, artifact_dir / "eda_report.txt")

    print("[DONE] EDA stage completed.")

if __name__ == "__main__":
    main()
