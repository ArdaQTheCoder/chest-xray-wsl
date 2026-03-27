"""
Model architectures for chest X-ray multi-label classification.

Architectures:
  - DenseNetCBAM   : DenseNet121 backbone + CBAM attention (primary model).
  - EfficientNetModel: EfficientNet-B4 backbone (ablation comparison).

Both models:
  - Return (logits, feature_map) for CAM compatibility.
  - Support Monte Carlo (MC) Dropout uncertainty estimation via enable_mc_dropout().
  - Accept mc_dropout_p for the dropout probability used during both training
    and MC Dropout inference.
"""

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 14

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia',
]


# ─── CBAM Attention ───────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Channel Attention Module from CBAM (ECCV 2018).

    Squeezes spatial dimensions via both average- and max-pooling,
    passes each through a shared MLP, and produces a per-channel
    recalibration weight in [0, 1].
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = self.mlp(self.avg_pool(x).view(b, c))  # (B, C)
        mx  = self.mlp(self.max_pool(x).view(b, c))  # (B, C)
        return self.sigmoid(avg + mx).view(b, c, 1, 1)  # (B, C, 1, 1)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module from CBAM.

    Pools along the channel axis (avg + max), concatenates, and applies
    a single convolution to produce a spatial attention map in [0, 1].
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)       # (B, 1, H, W)
        max_out, _ = x.max(dim=1, keepdim=True)     # (B, 1, H, W)
        concat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        return self.sigmoid(self.conv(concat))       # (B, 1, H, W)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel then spatial)."""

    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_att(x)  # recalibrate channels
        x = x * self.spatial_att(x)  # recalibrate spatial positions
        return x


# ─── Primary Model: DenseNet121 + CBAM ────────────────────────────────────────

class DenseNetCBAM(nn.Module):
    """
    DenseNet121 with CBAM attention inserted after the final dense block.

    Architecture:
        DenseNet121 features  →  ReLU  →  CBAM  →  GAP  →  Dropout  →  Linear(14)

    The CBAM refines the feature map by attending to the most discriminative
    channels and spatial regions before global pooling — particularly beneficial
    for localizing pathological regions in chest X-rays.

    Args:
        pretrained: Load ImageNet-pretrained DenseNet121 weights.
        mc_dropout_p: Dropout probability (used during both training and MC Dropout).
    """

    def __init__(self, pretrained: bool = True, mc_dropout_p: float = 0.5):
        super().__init__()
        weights  = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = models.densenet121(weights=weights)

        self.features   = densenet.features           # (B, 1024, 7, 7) for 224×224
        self.cbam       = CBAM(in_channels=1024, reduction=16)
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(p=mc_dropout_p)
        self.classifier = nn.Linear(1024, NUM_CLASSES)

    def forward(self, x: torch.Tensor):
        feat   = self.features(x)              # (B, 1024, 7, 7)
        feat   = torch.relu(feat)              # DenseNet applies ReLU after features
        feat   = self.cbam(feat)               # (B, 1024, 7, 7)  — attended
        pooled = self.gap(feat).flatten(1)     # (B, 1024)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)       # (B, 14)
        return logits, feat                    # feat exposed for CAM

    def enable_mc_dropout(self) -> None:
        """Switch all Dropout layers to train mode for MC Dropout inference."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


# ─── Comparison Model: EfficientNet-B4 ────────────────────────────────────────

class EfficientNetModel(nn.Module):
    """
    EfficientNet-B4 for ablation comparison against DenseNetCBAM.

    EfficientNet-B4 produces 1792-channel feature maps at 7×7 spatial resolution
    for 224×224 inputs, giving a richer representation than DenseNet121's 1024.

    Args:
        pretrained: Load ImageNet-pretrained EfficientNet-B4 weights.
        mc_dropout_p: Dropout probability.
    """

    def __init__(self, pretrained: bool = True, mc_dropout_p: float = 0.5):
        super().__init__()
        weights      = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        efficientnet = models.efficientnet_b4(weights=weights)

        self.features   = efficientnet.features      # (B, 1792, 7, 7) for 224×224
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(p=mc_dropout_p)
        self.classifier = nn.Linear(1792, NUM_CLASSES)

    def forward(self, x: torch.Tensor):
        feat   = self.features(x)              # (B, 1792, 7, 7)
        pooled = self.gap(feat).flatten(1)     # (B, 1792)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)       # (B, 14)
        return logits, feat

    def enable_mc_dropout(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


# ─── Factory ──────────────────────────────────────────────────────────────────

ARCH_REGISTRY = {
    'densenet121':    DenseNetCBAM,
    'efficientnet_b4': EfficientNetModel,
}


def get_model(
    arch: str = 'densenet121',
    device: str = 'cpu',
    pretrained: bool = True,
    mc_dropout_p: float = 0.5,
) -> nn.Module:
    """
    Instantiate and move a model to the target device.

    Args:
        arch: Architecture key — 'densenet121' or 'efficientnet_b4'.
        device: PyTorch device string or torch.device.
        pretrained: Use ImageNet-pretrained backbone weights.
        mc_dropout_p: Dropout probability for MC Dropout.

    Returns:
        Model on the specified device in eval mode.
    """
    if arch not in ARCH_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch}'. Choose from: {list(ARCH_REGISTRY.keys())}"
        )
    model = ARCH_REGISTRY[arch](pretrained=pretrained, mc_dropout_p=mc_dropout_p)
    return model.to(device)
