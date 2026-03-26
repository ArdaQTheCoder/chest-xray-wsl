import pandas as pd
import numpy as np
from pathlib import Path
import torch
from src.model import get_model
from src.cam import load_model
from src.evaluate import evaluate_localization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = get_model(device)
model = load_model('outputs/checkpoints/best_model.pth', model, device)

# Farklı threshold değerleri dene
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]

print("=== THRESHOLD OPTİMİZASYONU (GradCAM) ===\n")
best_threshold = 0.5
best_iou = 0.0

for t in thresholds:
    results = evaluate_localization(
        model=model,
        device=device,
        bbox_csv='data/nih/BBox_List_2017.csv',
        img_dir='data/nih/images',
        method='gradcam',
        threshold=t,
        max_samples=200
    )
    mean_iou = results['iou'].mean()
    print(f"Threshold {t:.1f} → Ortalama IoU: {mean_iou:.4f}\n")

    if mean_iou > best_iou:
        best_iou = mean_iou
        best_threshold = t

print(f"\nEn iyi threshold : {best_threshold}")
print(f"En iyi IoU       : {best_iou:.4f}")