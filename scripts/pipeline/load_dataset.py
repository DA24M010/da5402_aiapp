import os
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

# Configure logging
LOG_FILE = "./logs/train_pipeline.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class CrackDataset(Dataset):
    """
    Custom Dataset for surface crack detection.
    Expects a CSV file with 'path' and 'label' columns.
    """
    def __init__(self, csv_file, transform=None):
        try:
            self.data = pd.read_csv(csv_file)
            self.transform = transform
            logging.info(f"Loaded dataset from {csv_file} with {len(self.data)} samples.")
        except Exception as e:
            logging.exception(f"Failed to load CSV: {csv_file}")
            raise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        try:
            image = Image.open(row["path"]).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, row["label"]
        except (UnidentifiedImageError, FileNotFoundError) as e:
            logging.error(f"Error loading image at {row['path']}: {e}")
            raise

def load_dataloaders(metadata_path='artifacts/metadata', train_batch_size=32, val_batch_size=32, test_batch_size=1, num_workers=4):
    """
    Load DataLoaders for train/val/test splits.
    """
    try:
        batch_sizes = {
            'train': train_batch_size,
            'val': val_batch_size,
            'test': test_batch_size
        }

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        loaders = {}
        for split in ["train", "val", "test"]:
            csv_path = os.path.join(metadata_path, f"{split}.csv")
            dataset = CrackDataset(csv_file=csv_path, transform=transform)
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_sizes[split],
                shuffle=(split == "train"),
                num_workers=num_workers
            )
            logging.info(f"{split.title()} DataLoader created with batch size {batch_sizes[split]}")

        return loaders
    except Exception as e:
        logging.exception("Failed to initialize DataLoaders")
        raise
