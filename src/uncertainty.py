"""
Monte Carlo (MC) Dropout Uncertainty Estimation.

Standard neural networks are overconfident -- they produce a single point
estimate without quantifying how certain that estimate is.  MC Dropout
(Gal & Ghahramani, 2016) approximates Bayesian inference by running N
stochastic forward passes with dropout active during inference.

The variance across passes estimates *epistemic uncertainty* (model uncertainty
due to limited data), which is clinically meaningful: high-uncertainty
predictions should prompt human review rather than autonomous action.

Usage:
    mean, std = mc_dropout_predict(model, image_tensor, device, n_samples=30)
    # mean[0, i] = mean predicted probability for class i
    # std[0,  i] = standard deviation (uncertainty) for class i
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Tuple

from src.dataset import LABELS, NUM_CLASSES


# --- Inference ----------------------------------------------------------------

def mc_dropout_predict(
    model,
    image_tensor: torch.Tensor,
    device,
    n_samples: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run MC Dropout inference for calibrated uncertainty estimation.

    Sets the model to eval mode (disabling BatchNorm running stats updates)
    then calls model.enable_mc_dropout() to reactivate Dropout layers.
    After N stochastic passes it restores full eval mode.

    Args:
        model:        Model with an enable_mc_dropout() method.
        image_tensor: Input tensor, shape (1, 3, 224, 224).
        device:       Computation device.
        n_samples:    Number of stochastic forward passes (≥ 30 recommended).

    Returns:
        mean_probs: Tensor (1, NUM_CLASSES) -- mean predicted probability.
        std_probs:  Tensor (1, NUM_CLASSES) -- std dev (uncertainty proxy).
    """
    model.eval()
    model.enable_mc_dropout()

    all_probs = []
    inp = image_tensor.to(device)

    with torch.no_grad():
        for _ in range(n_samples):
            logits, _ = model(inp)
            probs     = torch.sigmoid(logits)   # (1, 14)
            all_probs.append(probs.cpu())

    all_probs  = torch.stack(all_probs, dim=0)  # (N, 1, 14)
    mean_probs = all_probs.mean(dim=0)           # (1, 14)
    std_probs  = all_probs.std(dim=0)            # (1, 14)

    model.eval()   # restore full eval (dropout disabled)
    return mean_probs, std_probs


# --- Reporting ----------------------------------------------------------------

def uncertainty_report(
    mean_probs: torch.Tensor,
    std_probs:  torch.Tensor,
    threshold:  float = 0.5,
) -> str:
    """
    Format MC Dropout predictions as a human-readable report.

    Confidence tiers:
      HIGH  -- std < 0.05  (model is certain)
      MED   -- std < 0.15  (moderate uncertainty)
      LOW   -- std ≥ 0.15  (high uncertainty, flag for review)

    Args:
        mean_probs: (1, 14) mean predicted probabilities.
        std_probs:  (1, 14) standard deviations.
        threshold:  Decision threshold for POSITIVE/negative status.

    Returns:
        Multi-line string report.
    """
    mean_np = mean_probs[0].numpy()
    std_np  = std_probs[0].numpy()

    sorted_idx = np.argsort(mean_np)[::-1]

    lines = [
        f"\n{'Disease':<25} {'Prob':>6}  {'Std':>6}  {'Conf':>6}  {'Status':>10}",
        "-" * 65,
    ]

    for i in sorted_idx:
        conf   = "HIGH" if std_np[i] < 0.05 else ("MED" if std_np[i] < 0.15 else "LOW")
        status = "POSITIVE" if mean_np[i] >= threshold else "negative"
        lines.append(
            f"{LABELS[i]:<25} {mean_np[i]:>6.3f}  {std_np[i]:>6.3f}  {conf:>6}  {status:>10}"
        )

    return "\n".join(lines)


# --- Visualisation ------------------------------------------------------------

def visualize_uncertainty(
    mean_probs: torch.Tensor,
    std_probs:  torch.Tensor,
    threshold:  float = 0.5,
    top_k:      int   = 14,
    save_path:  str   = None,
) -> None:
    """
    Bar chart of MC Dropout predictions with uncertainty error bars.

    Positive predictions (prob ≥ threshold) are shown in red,
    negative in steelblue.  Error bars show +-1 std.

    Args:
        mean_probs: (1, 14) mean predicted probabilities.
        std_probs:  (1, 14) standard deviations.
        threshold:  Decision threshold line (default 0.5).
        top_k:      Number of classes to display (sorted by probability).
        save_path:  If provided, saves the figure.
    """
    mean_np = mean_probs[0].numpy()
    std_np  = std_probs[0].numpy()

    sorted_idx = np.argsort(mean_np)[::-1][:top_k]
    labels     = [LABELS[i] for i in sorted_idx]
    probs      = mean_np[sorted_idx]
    stds       = std_np[sorted_idx]
    colours    = ['#d62728' if p >= threshold else '#1f77b4' for p in probs]

    fig, ax = plt.subplots(figsize=(10, max(4, top_k * 0.55)))
    x = np.arange(len(labels))
    ax.barh(x, probs, xerr=stds, color=colours, alpha=0.85,
            capsize=4, ecolor='#333333', height=0.65)
    ax.axvline(threshold, color='red', linestyle='--', linewidth=1.2,
               alpha=0.7, label=f'Threshold ({threshold})')
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Predicted Probability', fontsize=11)
    ax.set_title(
        'MC Dropout Predictions with Uncertainty\n'
        '(red = positive, blue = negative, bars = +-1 std)',
        fontsize=12,
    )
    ax.invert_yaxis()
    ax.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')

    plt.show()
    plt.close()
