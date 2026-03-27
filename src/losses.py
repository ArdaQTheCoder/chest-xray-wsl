"""
Asymmetric Loss (ASL) for Multi-Label Classification.

From: "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)
https://arxiv.org/abs/2009.14119

Designed specifically for large-scale, imbalanced multi-label datasets like NIH
Chest X-ray. Addresses two core problems:
  1. Class imbalance — most samples are negative for any given disease.
  2. Hard negative mining — easy negatives dominate gradients and slow learning.

Key ideas:
  - Asymmetric focusing: separate gamma parameters for positives (gamma_pos)
    and negatives (gamma_neg). Defaults (0, 4) down-weight easy negatives heavily.
  - Probability margin shift (clip): shifts negative probabilities by +clip before
    computing loss, effectively discarding very easy negatives (p_neg < clip).
"""

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.

    Args:
        gamma_neg: Focusing exponent for negative samples (default 4).
                   Higher value = harder focus on difficult negatives.
        gamma_pos: Focusing exponent for positive samples (default 0 = BCE for positives).
        clip: Probability margin shift for negatives. Negatives with p < clip
              are treated as p = clip, effectively ignoring very easy negatives.
        eps: Numerical stability constant.
        disable_torch_grad_focal_loss: Skip gradient computation for the
              focusing weight (saves memory, negligible accuracy impact).
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Logits, shape (B, C).
            y: Binary targets, shape (B, C), values in {0, 1}.

        Returns:
            Scalar loss value.
        """
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # Probability margin shift: clip easy negatives
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Binary cross-entropy terms
        los_pos = y       * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric focusing weights
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)

            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt  = pt0 + pt1  # pt for each sample/class

            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)

            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)

            loss = loss * one_sided_w

        return -loss.sum() / x.shape[0]
