import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.labels = LABELS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(self.img_dir) / row['Image Index']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Multi-label encoding
        label_str = row['Finding Labels']
        label_vec = torch.zeros(len(self.labels))
        for i, label in enumerate(self.labels):
            if label in label_str:
                label_vec[i] = 1.0

        return image, label_vec, row['Image Index']


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def get_dataloaders(csv_path, img_dir, batch_size=32, val_split=0.2):
    train_transform, val_transform = get_transforms()

    full_df = pd.read_csv(csv_path)
    
    # Önce mevcut dosyaları filtrele
    img_dir_path = Path(img_dir)
    full_df = full_df[
        full_df['Image Index'].apply(lambda x: (img_dir_path / x).exists())
    ].reset_index(drop=True)
    
    print(f"Toplam kullanılabilir görüntü: {len(full_df)}")

    # Sonra split yap
    val_size = int(len(full_df) * val_split)
    train_df = full_df.iloc[val_size:].reset_index(drop=True)
    val_df   = full_df.iloc[:val_size].reset_index(drop=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_df.to_csv('data/train_split.csv', index=False)
    val_df.to_csv('data/val_split.csv',     index=False)

    train_dataset = ChestXrayDataset('data/train_split.csv', img_dir, train_transform)
    val_dataset   = ChestXrayDataset('data/val_split.csv',   img_dir, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader