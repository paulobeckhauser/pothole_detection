import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

class PotholeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset (train or validation).
            transform (callable, optional): Optional transform to be applied to each image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load all image paths and labels
        for batch_folder in os.listdir(root_dir):
            batch_path = os.path.join(root_dir, batch_folder)
            if os.path.isdir(batch_path):
                # Get positive (pothole) images
                pothole_dir = os.path.join(batch_path, "pothole")
                self.image_paths += [os.path.join(pothole_dir, img) for img in os.listdir(pothole_dir)]
                self.labels += [1] * len(os.listdir(pothole_dir))

                # Get negative (background) images
                background_dir = os.path.join(batch_path, "background")
                self.image_paths += [os.path.join(background_dir, img) for img in os.listdir(background_dir)]
                self.labels += [0] * len(os.listdir(background_dir))

        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open the image
        image = Image.open(image_path).convert("RGB")

        # Apply the transform if specified
        if self.transform:
            image = self.transform(image)

        return image, label
    
def get_dataloaders(train_dir, val_dir, batch_size=64, num_workers=4):
    # Define image transformations (augmentation for training, simple scaling for validation)
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Create datasets
    train_dataset = PotholeDataset(root_dir=train_dir, transform=train_transform)
    val_dataset = PotholeDataset(root_dir=val_dir, transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


if __name__ == "__main__":
    train_dir = "/work3/s214598/pothole_detection/data/train"
    val_dir = "/work3/s214598/pothole_detection/data/validation"
    batch_size = 64

    train_loader, val_loader = get_dataloaders(train_dir, val_dir, batch_size=batch_size)

    # Check one batch from the train loader
    for images, labels in train_loader:
        print("Batch of images shape:", images.shape)
        print("Batch of labels:", labels)
        break