import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import yaml

class CrackDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, row["label"]

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def load_dataloaders(metadata_path = 'artifacts/metadata'):
    params = load_params()
    batch_sizes = {
        "train": params["dataloader"]["train_batch_size"],
        "val": params["dataloader"]["val_batch_size"],
        "test": params["dataloader"]["test_batch_size"]
    }
    num_workers = params["dataloader"]["num_workers"]
    image_size = params["dataloader"].get("image_size", 224)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    loaders = {}
    for split in ["train", "val", "test"]:
        csv_path = f"{metadata_path}/{split}.csv"
        dataset = CrackDataset(
            csv_file=csv_path,
            transform=transform
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_sizes[split],
            shuffle=(split == "train"),
            num_workers=num_workers
        )
    return loaders
