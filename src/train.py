"""
Training loop for chest X-ray multi-label classification.

Improvements over baseline:
  1. Asymmetric Loss (ASL) — replaces vanilla BCEWithLogitsLoss to handle
     the severe class imbalance of the NIH dataset.
  2. Mixed-Precision Training (AMP) — ~2× speedup on GPUs with negligible
     accuracy loss via float16 forward pass and float32 gradient updates.
  3. Warmup + Cosine Annealing LR — linear warmup for the first few epochs
     stabilises early training, then cosine decay prevents over-shooting.
  4. Gradient Clipping — prevents exploding gradients (max norm = 1.0).
  5. Per-class AUROC logging — prints a breakdown of AUROC per disease at
     the end of each epoch so per-class progress is visible.
  6. Training history CSV — saved to outputs/ after every epoch for
     offline plotting and reporting.
"""

import math
import time
import os

import torch
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import MultilabelAUROC

from src.losses import AsymmetricLoss
from src.dataset import LABELS, NUM_CLASSES


# ─── LR Schedule ──────────────────────────────────────────────────────────────

def get_warmup_cosine_scheduler(
    optimizer,
    warmup_epochs: int,
    total_epochs:  int,
    min_lr_ratio:  float = 0.01,
) -> LambdaLR:
    """
    Linear warmup followed by cosine annealing.

    During warmup: lr scales linearly from 0 → base_lr over warmup_epochs.
    After warmup:  lr follows cosine decay from base_lr → min_lr_ratio * base_lr.

    Args:
        optimizer:     The optimiser to schedule.
        warmup_epochs: Number of warmup epochs.
        total_epochs:  Total training epochs.
        min_lr_ratio:  Final lr as a fraction of base_lr (default 1%).

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)


# ─── Training Loop ────────────────────────────────────────────────────────────

def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs:        int   = 30,
    lr:            float = 1e-4,
    weight_decay:  float = 1e-5,
    warmup_epochs: int   = 3,
    arch:          str   = 'densenet121',
    checkpoint_dir: str  = 'outputs/checkpoints',
):
    """
    Full training loop with validation, checkpointing, and history logging.

    Args:
        model:          Model to train (must return (logits, features)).
        train_loader:   Training DataLoader.
        val_loader:     Validation DataLoader.
        device:         torch.device.
        epochs:         Total training epochs.
        lr:             Initial learning rate (peak after warmup).
        weight_decay:   AdamW weight decay.
        warmup_epochs:  Number of linear-warmup epochs.
        arch:           Architecture name used for checkpoint/history filenames.
        checkpoint_dir: Directory to save the best model checkpoint.

    Returns:
        history: Dict with per-epoch metrics.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    criterion  = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
    optimizer  = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler  = get_warmup_cosine_scheduler(optimizer, warmup_epochs, epochs)
    use_amp    = torch.cuda.is_available()
    scaler     = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Macro AUROC (for checkpointing) + per-class AUROC (for logging)
    auroc_macro    = MultilabelAUROC(num_labels=NUM_CLASSES, average='macro').to(device)
    auroc_perclass = MultilabelAUROC(num_labels=NUM_CLASSES, average=None).to(device)

    best_auroc = 0.0
    history    = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'val_auroc_macro': [], 'lr': [],
    }
    for label in LABELS:
        history[f'auroc_{label}'] = []

    checkpoint_path = os.path.join(checkpoint_dir, f'best_model_{arch}.pth')

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, _ = model(images)
                loss      = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f} | LR: {current_lr:.2e}"
                )

        avg_train_loss = train_loss / len(train_loader)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        auroc_macro.reset()
        auroc_perclass.reset()

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits, _ = model(images)
                    loss      = criterion(logits, labels)

                val_loss += loss.item()
                probs = torch.sigmoid(logits)
                auroc_macro.update(probs, labels.int())
                auroc_perclass.update(probs, labels.int())

        avg_val_loss    = val_loss / len(val_loader)
        val_auroc       = auroc_macro.compute().item()
        per_class_auroc = auroc_perclass.compute()  # (14,)
        elapsed         = time.time() - epoch_start
        current_lr      = optimizer.param_groups[0]['lr']

        # ── Logging ────────────────────────────────────────────────────────────
        print(
            f"\nEpoch {epoch+1}/{epochs}  |  "
            f"Train Loss: {avg_train_loss:.4f}  |  "
            f"Val Loss: {avg_val_loss:.4f}  |  "
            f"Val AUROC (macro): {val_auroc:.4f}  |  "
            f"LR: {current_lr:.2e}  |  "
            f"Time: {elapsed:.1f}s"
        )
        print("  Per-class AUROC:")
        for i, label in enumerate(LABELS):
            auc = per_class_auroc[i].item()
            bar = "█" * int(auc * 20)
            print(f"    {label:<22} {auc:.4f}  {bar}")

        # ── History ────────────────────────────────────────────────────────────
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_auroc_macro'].append(val_auroc)
        history['lr'].append(current_lr)
        for i, label in enumerate(LABELS):
            history[f'auroc_{label}'].append(per_class_auroc[i].item())

        pd.DataFrame(history).to_csv(
            f'outputs/training_history_{arch}.csv', index=False
        )

        # ── LR Step ────────────────────────────────────────────────────────────
        scheduler.step()

        # ── Checkpoint ─────────────────────────────────────────────────────────
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(
                {
                    'epoch':            epoch + 1,
                    'arch':             arch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auroc':        val_auroc,
                    'per_class_auroc':  {
                        label: per_class_auroc[i].item()
                        for i, label in enumerate(LABELS)
                    },
                },
                checkpoint_path,
            )
            print(f"  ✓ Checkpoint saved  (AUROC: {val_auroc:.4f})\n")

    print(f"\nTraining complete. Best Val AUROC: {best_auroc:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History:    outputs/training_history_{arch}.csv")

    return history
