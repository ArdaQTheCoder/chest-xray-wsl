"""
Comprehensive evaluation for chest X-ray classification and localisation.

Classification metrics:
  - Per-class AUROC with 95% bootstrap confidence intervals (1 000 resamples).
  - Sensitivity (TPR) and Specificity (TNR) at the optimal Youden's J threshold.
  - Macro-averaged AUROC summary.

Localisation metrics:
  - IoU (Intersection over Union) between binarised CAM and radiologist bounding box.
  - Pointing Game accuracy: does the peak CAM activation fall inside the GT bbox?
  - Results reported per disease class and overall.

Both evaluators save results to CSV under outputs/.
"""

import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, roc_curve

from src.dataset import LABELS, ChestXrayDataset
from src.cam import preprocess_image, generate_cam


# ─── Bootstrap CI ─────────────────────────────────────────────────────────────

def _bootstrap_auroc(
    y_true:      np.ndarray,
    y_score:     np.ndarray,
    n_bootstrap: int   = 1000,
    ci:          float = 0.95,
    seed:        int   = 42,
):
    """
    Compute AUROC mean and confidence interval via non-parametric bootstrap.

    Args:
        y_true:      Binary ground-truth labels, shape (N,).
        y_score:     Predicted probabilities, shape (N,).
        n_bootstrap: Number of resampling iterations.
        ci:          Confidence level (default 95%).
        seed:        Random seed.

    Returns:
        (mean_auroc, ci_lower, ci_upper)
    """
    rng  = np.random.default_rng(seed)
    aucs = []
    n    = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt  = y_true[idx]
        ys  = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))

    aucs = np.array(aucs)
    alpha = (1.0 - ci) / 2.0
    return aucs.mean(), np.percentile(aucs, alpha * 100), np.percentile(aucs, (1 - alpha) * 100)


# ─── Classification Evaluation ────────────────────────────────────────────────

def evaluate_classification(
    model,
    device,
    csv_path:     str,
    img_dir:      str,
    batch_size:   int  = 32,
    n_bootstrap:  int  = 1000,
    output_csv:   str  = 'outputs/classification_results.csv',
) -> dict:
    """
    Evaluate multi-label classification performance on a split CSV.

    For each of the 14 disease classes computes:
      - AUROC  with 95% bootstrap CI
      - Optimal threshold (Youden's J = TPR − FPR maximised on the ROC curve)
      - Sensitivity (TPR) and Specificity (TNR) at that threshold

    Args:
        model:       Trained model returning (logits, features).
        device:      torch.device.
        csv_path:    Path to split CSV (e.g. data/test_split.csv).
        img_dir:     Image directory.
        batch_size:  Inference batch size.
        n_bootstrap: Bootstrap resamples for CI.
        output_csv:  Where to save the per-class results table.

    Returns:
        Dict mapping each disease label to its metrics dict.
    """
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = ChestXrayDataset(csv_path, img_dir, transform=val_transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            logits, _ = model(images)
            probs     = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)   # (N, 14)
    all_labels = np.concatenate(all_labels, axis=0)   # (N, 14)

    header = (
        f"\n{'Disease':<25} {'AUROC':>6}  {'95% CI':>15}  "
        f"{'Sensitivity':>12}  {'Specificity':>12}  {'Threshold':>10}"
    )
    print(header)
    print("-" * 90)

    results = {}
    rows    = []

    for i, label in enumerate(LABELS):
        y_true  = all_labels[:, i]
        y_score = all_probs[:, i]

        if len(np.unique(y_true)) < 2:
            print(f"{label:<25} {'N/A':>6}  (only one class present)")
            continue

        auroc            = roc_auc_score(y_true, y_score)
        _, ci_low, ci_hi = _bootstrap_auroc(y_true, y_score, n_bootstrap)

        # Youden's J optimal threshold
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        j_idx         = np.argmax(tpr - fpr)
        opt_threshold = float(thresholds[j_idx])
        sensitivity   = float(tpr[j_idx])
        specificity   = float(1.0 - fpr[j_idx])

        print(
            f"{label:<25} {auroc:>6.4f}  [{ci_low:.4f}-{ci_hi:.4f}]  "
            f"{sensitivity:>12.4f}  {specificity:>12.4f}  {opt_threshold:>10.4f}"
        )

        results[label] = dict(
            auroc=auroc, ci_low=ci_low, ci_high=ci_hi,
            sensitivity=sensitivity, specificity=specificity,
            threshold=opt_threshold,
        )
        rows.append(dict(disease=label, **results[label]))

    macro = np.mean([v['auroc'] for v in results.values()])
    print(f"\n{'Macro AUROC':<25} {macro:.4f}")

    import os
    os.makedirs('outputs', exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Results saved → {output_csv}")

    return results


# ─── Localisation Helpers ─────────────────────────────────────────────────────

def cam_to_binary_mask(cam_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binarise a [0,1] CAM heatmap at the given threshold."""
    return (cam_map >= threshold).astype(np.uint8)


def bbox_to_mask(x: float, y: float, w: float, h: float, size: int = 224) -> np.ndarray:
    """
    Convert a radiologist bounding box (NIH coordinates, 1024×1024 space)
    to a binary mask of the target spatial resolution.

    Args:
        x, y: Top-left corner of the bbox.
        w, h: Width and height.
        size: Target mask side length (default 224).

    Returns:
        Binary mask of shape (size, size).
    """
    orig_size = 1024
    scale = size / orig_size
    x1 = int(x * scale)
    y1 = int(y * scale)
    x2 = int((x + w) * scale)
    y2 = int((y + h) * scale)

    mask = np.zeros((size, size), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Intersection over Union between two binary masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union        = np.logical_or(pred_mask,  gt_mask).sum()
    return float(intersection / union) if union > 0 else 0.0


def pointing_game_hit(cam_map: np.ndarray, x: float, y: float,
                      w: float, h: float, size: int = 224) -> int:
    """
    Pointing Game metric: 1 if the CAM peak falls inside the GT bbox, else 0.

    The Pointing Game (Zhang et al., 2018) is a binary hit/miss metric that
    asks whether the most activated point of the CAM lies within the annotated
    region — a stronger localisation criterion than IoU.

    Args:
        cam_map: CAM heatmap (224×224), values in [0, 1].
        x, y:   Bbox top-left in original 1024×1024 space.
        w, h:   Bbox width and height.
        size:   Spatial resolution of cam_map (default 224).

    Returns:
        1 (hit) or 0 (miss).
    """
    # Peak activation location
    peak_y, peak_x = np.unravel_index(cam_map.argmax(), cam_map.shape)

    # Scale bbox to cam_map resolution
    scale = size / 1024
    x1 = int(x * scale)
    y1 = int(y * scale)
    x2 = int((x + w) * scale)
    y2 = int((y + h) * scale)

    return int(x1 <= peak_x <= x2 and y1 <= peak_y <= y2)


# ─── Localisation Evaluation ──────────────────────────────────────────────────

def evaluate_localization(
    model,
    device,
    bbox_csv:    str,
    img_dir:     str,
    method:      str   = 'gradcam',
    arch:        str   = 'densenet121',
    threshold:   float = 0.5,
    max_samples: int   = 200,
    output_csv:  str   = 'outputs/localization_results.csv',
) -> pd.DataFrame:
    """
    Evaluate CAM localisation quality against radiologist bounding boxes.

    For each annotated image computes:
      - IoU between binarised CAM mask and GT bbox mask.
      - Pointing Game hit/miss.

    Prints per-disease summaries and saves a full results CSV.

    Args:
        model:       Trained model.
        device:      torch.device.
        bbox_csv:    Path to BBox_List_2017.csv.
        img_dir:     Image directory.
        method:      CAM method ('gradcam', 'gradcam++', 'eigencam').
        arch:        Model architecture name (for target layer selection).
        threshold:   Binarisation threshold for IoU computation.
        max_samples: Maximum number of annotated samples to evaluate.
        output_csv:  Where to save full results.

    Returns:
        DataFrame with columns [image, disease, iou, pointing_game, method].
    """
    bbox_df  = pd.read_csv(bbox_csv)
    img_dir  = Path(img_dir)
    model.eval()
    results  = []

    for _, row in bbox_df.iterrows():
        if len(results) >= max_samples:
            break

        img_path = img_dir / row['Image Index']
        if not img_path.exists():
            continue

        disease = row['Finding Label']
        if disease not in LABELS:
            continue
        class_idx = LABELS.index(disease)

        # Bounding box coordinates (NIH CSV has awkward column names)
        bx = float(row['Bbox [x'])
        by = float(row['y'])
        bw = float(row['w'])
        bh = float(row['h]'])

        # Generate CAM
        image_tensor, _ = preprocess_image(str(img_path))
        cam_map = generate_cam(model, image_tensor, class_idx, device, method, arch)

        # IoU
        pred_mask = cam_to_binary_mask(cam_map, threshold)
        gt_mask   = bbox_to_mask(bx, by, bw, bh)
        iou       = compute_iou(pred_mask, gt_mask)

        # Pointing Game
        pg_hit = pointing_game_hit(cam_map, bx, by, bw, bh)

        results.append({
            'image':         row['Image Index'],
            'disease':       disease,
            'iou':           iou,
            'pointing_game': pg_hit,
            'method':        method,
        })

        print(
            f"[{len(results):3d}] {row['Image Index']:25s} | "
            f"{disease:20s} | IoU: {iou:.4f} | PG: {'HIT' if pg_hit else 'miss'}"
        )

    results_df = pd.DataFrame(results)

    print("\n" + "-" * 60)
    print(f"Method: {method.upper()}")
    print(f"Samples evaluated : {len(results_df)}")
    print(f"Mean IoU          : {results_df['iou'].mean():.4f}")
    print(f"Pointing Game Acc : {results_df['pointing_game'].mean():.4f}")
    print(f"\nPer-disease summary:")
    summary = results_df.groupby('disease')[['iou', 'pointing_game']].mean()
    summary.columns = ['Mean IoU', 'PG Accuracy']
    print(summary.sort_values('Mean IoU', ascending=False).to_string())

    import os
    os.makedirs('outputs', exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"\nFull results saved → {output_csv}")

    return results_df
