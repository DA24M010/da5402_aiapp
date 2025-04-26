import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from load_dataset import load_dataloaders

def show_samples(loader, num_samples=5):
    # Show `num_samples` images from the DataLoader
    for i, (images, labels) in enumerate(loader):
        if i == num_samples:
            break
        
        for j in range(images.size(0)):
            img = images[j].permute(1, 2, 0).numpy()  # Convert to HxWxC format for plotting
            label = labels[j].item()

            plt.imshow(img)
            plt.title(f"Label: {label}")
            plt.axis('off')
            plt.show()

if __name__ == "__main__":
    # Load the DataLoader objects
    loaders = load_dataloaders()

    print("[INFO] Showing samples from train DataLoader...")
    show_samples(loaders["train"], num_samples=3)

    print("[INFO] Showing samples from val DataLoader...")
    show_samples(loaders["val"], num_samples=3)

    print("[INFO] Showing samples from test DataLoader...")
    show_samples(loaders["test"], num_samples=3)
