import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import MultilabelAUROC
import time

def train(model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    auroc_metric = MultilabelAUROC(num_labels=14, average='macro').to(device)

    best_auroc = 0.0

    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        start = time.time()

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f'  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} '
                      f'| Loss: {loss.item():.4f}')

        avg_train_loss = train_loss / len(train_loader)

        # --- VAL ---
        model.eval()
        val_loss = 0.0
        auroc_metric.reset()

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits, _ = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                auroc_metric.update(probs, labels.int())

        avg_val_loss = val_loss / len(val_loader)
        val_auroc    = auroc_metric.compute().item()
        elapsed      = time.time() - start

        print(f'\nEpoch {epoch+1}/{epochs} — '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} | '
              f'Val AUROC: {val_auroc:.4f} | '
              f'Time: {elapsed:.1f}s\n')

        scheduler.step(val_auroc)

        # Checkpoint kaydet
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auroc': val_auroc,
            }, 'outputs/checkpoints/best_model.pth')
            print(f'  Checkpoint kaydedildi — AUROC: {val_auroc:.4f}\n')