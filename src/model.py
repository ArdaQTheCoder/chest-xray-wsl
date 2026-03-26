import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 14

class ChestXrayModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = models.densenet121(weights=weights)

        # Feature extractor — GAP için son conv katmanını ayır
        self.features = densenet.features
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        in_features = densenet.classifier.in_features  # 1024
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, NUM_CLASSES)
        )

    def forward(self, x):
        features = self.features(x)        # (B, 1024, 7, 7)
        pooled   = self.gap(features)      # (B, 1024, 1, 1)
        pooled   = pooled.view(pooled.size(0), -1)  # (B, 1024)
        logits   = self.classifier(pooled) # (B, 14)
        return logits, features            # features CAM için lazım


def get_model(device):
    model = ChestXrayModel(pretrained=True)
    model = model.to(device)
    return model