import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

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

def load_dataloaders(metadata_path = 'artifacts/metadata', train_batch_size = 32, val_batch_size = 32, test_batch_size = 1, num_workers = 4):
    batch_sizes = {
        'train' : train_batch_size,
        'val' : val_batch_size, 
        'test' : test_batch_size
    }
    transform = transforms.Compose([
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