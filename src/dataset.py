"""
Dataset loading and preprocessing for NIH Chest X-ray 14.

Key improvements over baseline:
  1. Patient-level split — images are split by Patient ID, not randomly by row,
     preventing data leakage where the same patient appears in train and val.
  2. CLAHE preprocessing — Contrast Limited Adaptive Histogram Equalization
     enhances local contrast in X-rays, widely used in medical imaging.
  3. Stronger augmentation — RandomResizedCrop, ColorJitter, GaussianBlur
     for better generalisation.
  4. Three-way split — train / val / test (70 / 10 / 20 by patient count).
  5. Class weight computation — per-class positive frequency for optional
     use in loss functions.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia',
]

NUM_CLASSES = len(LABELS)


# ─── CLAHE Preprocessing ──────────────────────────────────────────────────────

def apply_clahe(image: Image.Image) -> Image.Image:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Operates on the L (luminance) channel in LAB colour space so that
    colour balance is preserved while local contrast is enhanced.
    Standard practice for chest X-ray preprocessing.

    Args:
        image: PIL RGB image.

    Returns:
        CLAHE-enhanced PIL RGB image.
    """
    img_np = np.array(image, dtype=np.uint8)
    lab    = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)


class CLAHETransform:
    """Torchvision-compatible transform wrapper for apply_clahe."""

    def __call__(self, image: Image.Image) -> Image.Image:
        return apply_clahe(image)

    def __repr__(self) -> str:
        return "CLAHETransform(clipLimit=2.0, tileGridSize=8×8)"


# ─── Patient-Level Split ──────────────────────────────────────────────────────

def patient_level_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.10,
    seed:        int   = 42,
):
    """
    Split a dataset DataFrame at the patient level.

    Shuffles unique Patient IDs then allocates them to train / val / test
    partitions. All images belonging to a patient stay in the same partition,
    eliminating cross-split leakage.

    Args:
        df: Full dataset DataFrame with a 'Patient ID' column.
        train_ratio: Fraction of patients for training (default 0.70).
        val_ratio:   Fraction of patients for validation (default 0.10).
                     Remaining patients go to the test set.
        seed: Random seed for reproducibility.

    Returns:
        (train_df, val_df, test_df) — three DataFrames.
    """
    patient_ids = df['Patient ID'].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)

    n       = len(patient_ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_ids = set(patient_ids[:n_train])
    val_ids   = set(patient_ids[n_train : n_train + n_val])
    test_ids  = set(patient_ids[n_train + n_val :])

    train_df = df[df['Patient ID'].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df['Patient ID'].isin(val_ids)].reset_index(drop=True)
    test_df  = df[df['Patient ID'].isin(test_ids)].reset_index(drop=True)

    return train_df, val_df, test_df


# ─── Dataset ──────────────────────────────────────────────────────────────────

class ChestXrayDataset(Dataset):
    """
    NIH Chest X-ray 14 multi-label dataset.

    Reads image paths and labels from a CSV (or DataFrame) and returns
    (image_tensor, label_vector, image_filename) tuples.

    Args:
        source:    Path to a CSV file or a pandas DataFrame.
        img_dir:   Root directory containing the .png X-ray images.
        transform: torchvision transform pipeline.
    """

    def __init__(self, source, img_dir: str, transform=None):
        if isinstance(source, pd.DataFrame):
            self.df = source.reset_index(drop=True)
        else:
            self.df = pd.read_csv(source)
        self.img_dir   = Path(img_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row      = self.df.iloc[idx]
        img_path = self.img_dir / row['Image Index']
        image    = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Multi-label encoding
        label_str = row['Finding Labels']
        label_vec = torch.zeros(NUM_CLASSES)
        for i, label in enumerate(LABELS):
            if label in label_str:
                label_vec[i] = 1.0

        return image, label_vec, row['Image Index']


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(use_clahe: bool = True):
    """
    Build train and validation transform pipelines.

    Train augmentation:
        CLAHE → RandomResizedCrop(224, scale 0.7–1.0) →
        RandomHorizontalFlip → RandomRotation(±15°) →
        ColorJitter(brightness, contrast) →
        RandomApply(GaussianBlur, p=0.3) →
        ToTensor → Normalize(ImageNet)

    Val/Test: CLAHE → Resize(224) → CenterCrop(224) → ToTensor → Normalize

    Args:
        use_clahe: Apply CLAHE preprocessing (recommended for X-rays).

    Returns:
        (train_transform, val_transform)
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    clahe_step = [CLAHETransform()] if use_clahe else []

    train_transform = transforms.Compose(
        clahe_step + [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    val_transform = transforms.Compose(
        clahe_step + [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    return train_transform, val_transform


# ─── Class Weights ────────────────────────────────────────────────────────────

def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Compute per-class inverse-frequency weights from the training DataFrame.

    Weight_i = N / (2 * pos_i)  (capped at 20 to avoid extreme values).

    These can optionally be passed to a weighted BCE loss as a complement
    to Asymmetric Loss — or used for analysis.

    Args:
        df: Training DataFrame with 'Finding Labels' column.

    Returns:
        Tensor of shape (NUM_CLASSES,).
    """
    n = len(df)
    weights = torch.ones(NUM_CLASSES)
    for i, label in enumerate(LABELS):
        pos = df['Finding Labels'].str.contains(label, regex=False).sum()
        if pos > 0:
            weights[i] = min(n / (2.0 * pos), 20.0)
    return weights


# ─── DataLoaders ──────────────────────────────────────────────────────────────

def get_dataloaders(
    csv_path:   str,
    img_dir:    str,
    batch_size: int   = 32,
    num_workers: int  = 0,
    use_clahe:  bool  = True,
    seed:       int   = 42,
):
    """
    Build train, val, and test DataLoaders with patient-level splits.

    Filters the CSV to only include images that actually exist on disk,
    performs a patient-level 70 / 10 / 20 split, saves split CSVs to
    data/, and returns DataLoaders.

    Args:
        csv_path:    Path to Data_Entry_2017.csv.
        img_dir:     Directory containing X-ray .png files.
        batch_size:  Mini-batch size.
        num_workers: DataLoader worker processes.
        use_clahe:   Apply CLAHE preprocessing.
        seed:        Reproducibility seed for patient shuffle.

    Returns:
        (train_loader, val_loader, test_loader, class_weights)
    """
    train_transform, val_transform = get_transforms(use_clahe=use_clahe)

    full_df = pd.read_csv(csv_path)

    # Filter to images available on disk
    img_dir_path = Path(img_dir)
    mask    = full_df['Image Index'].apply(lambda x: (img_dir_path / x).exists())
    full_df = full_df[mask].reset_index(drop=True)
    print(f"Available images: {len(full_df)}")

    # Patient-level split
    train_df, val_df, test_df = patient_level_split(full_df, seed=seed)
    print(
        f"Patient-level split | "
        f"Train: {len(train_df)} images ({train_df['Patient ID'].nunique()} patients) | "
        f"Val: {len(val_df)} images ({val_df['Patient ID'].nunique()} patients) | "
        f"Test: {len(test_df)} images ({test_df['Patient ID'].nunique()} patients)"
    )

    # Persist splits for evaluation scripts
    os.makedirs('data', exist_ok=True)
    train_df.to_csv('data/train_split.csv', index=False)
    val_df.to_csv('data/val_split.csv',     index=False)
    test_df.to_csv('data/test_split.csv',   index=False)

    # Class weights from training set
    class_weights = compute_class_weights(train_df)

    # Datasets
    train_dataset = ChestXrayDataset(train_df, img_dir, transform=train_transform)
    val_dataset   = ChestXrayDataset(val_df,   img_dir, transform=val_transform)
    test_dataset  = ChestXrayDataset(test_df,  img_dir, transform=val_transform)

    # DataLoaders
    loader_kwargs = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, class_weights
