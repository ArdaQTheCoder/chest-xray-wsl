import torch
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from src.cam import generate_cam, preprocess_image
from pytorch_grad_cam.utils.image import show_cam_on_image

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

ORIG_SIZE = 1024  # NIH görüntüleri 1024x1024


def cam_to_binary_mask(cam_map, threshold=0.5):
    """CAM ısı haritasını binary maskeye çevir."""
    binary = (cam_map >= threshold).astype(np.uint8)
    return binary


def bbox_to_mask(x, y, w, h, size=224):
    """Radyolog bbox'ını 224x224 binary maskeye çevir."""
    scale = size / ORIG_SIZE
    x1 = int(x * scale)
    y1 = int(y * scale)
    x2 = int((x + w) * scale)
    y2 = int((y + h) * scale)

    mask = np.zeros((size, size), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def compute_iou(mask_pred, mask_gt):
    """İki binary maske arasında IoU hesapla."""
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union        = np.logical_or(mask_pred, mask_gt).sum()
    if union == 0:
        return 0.0
    return intersection / union


def evaluate_localization(model, device, bbox_csv, img_dir,
                          method='gradcam', threshold=0.5, max_samples=100):
    """
    BBox_List_2017.csv kullanarak CAM lokalizasyon performansını ölç.
    Her görüntü için IoU hesaplar, ortalama IoU döndürür.
    """
    bbox_df = pd.read_csv(bbox_csv)
    img_dir = Path(img_dir)

    results = []

    for idx, row in bbox_df.iterrows():
        if idx >= max_samples:
            break

        img_path = img_dir / row['Image Index']
        if not img_path.exists():
            continue

        # Hangi hastalık sınıfı
        disease = row['Finding Label']
        if disease not in LABELS:
            continue
        class_idx = LABELS.index(disease)

        # CAM üret
        image_tensor, _ = preprocess_image(str(img_path))
        cam_map = generate_cam(model, image_tensor, class_idx, device, method)

        # CAM → binary maske
        pred_mask = cam_to_binary_mask(cam_map, threshold)

        # BBox → binary maske
        gt_mask = bbox_to_mask(
            row['Bbox [x'], row['y'], row['w'], row['h]']
        )

        iou = compute_iou(pred_mask, gt_mask)

        results.append({
            'image':   row['Image Index'],
            'disease': disease,
            'iou':     iou,
            'method':  method,
        })

        print(f"[{idx+1:3d}] {row['Image Index']:25s} | {disease:20s} | IoU: {iou:.4f}")

    results_df = pd.DataFrame(results)

    print(f"\n--- {method.upper()} Sonuçları ---")
    print(f"Toplam örnek : {len(results_df)}")
    print(f"Ortalama IoU : {results_df['iou'].mean():.4f}")
    print(f"\nHastalığa göre IoU:")
    print(results_df.groupby('disease')['iou'].mean().sort_values(ascending=False).to_string())

    return results_df