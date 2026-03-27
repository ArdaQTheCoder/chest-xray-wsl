"""
Class Activation Map (CAM) generation and visualisation.

Supported methods:
  - GradCAM        : Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).
  - GradCAM++      : Improved gradient weighting (Chattopadhay et al., 2018).
  - EigenCAM       : PCA-based, gradient-free (Muhammad & Yeasin, 2020).
  - Integrated Gradients (IG): Attribution via path integral (Sundararajan et al., 2017).
                     Implemented without external dependencies.

Architecture-aware target layers:
  - DenseNet121+CBAM  → model.cbam  (attended 1024-channel feature map)
  - EfficientNet-B4   → model.features[-1]  (1792-channel feature map)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.dataset import LABELS


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, model, device):
    """
    Load a model checkpoint saved by train.py.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        model:           Model instance (architecture must match checkpoint).
        device:          Target device.

    Returns:
        Model in eval mode on device.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


# ─── Image Preprocessing ──────────────────────────────────────────────────────

def preprocess_image(img_path: str):
    """
    Load and preprocess a single image for model inference.

    Applies the same val/test transforms as get_transforms() (no augmentation):
    Resize(256) → CenterCrop(224) → ToTensor → ImageNet-Normalize.

    Note: CLAHE is intentionally omitted here so that the original pixel
    values remain available for visualisation overlays.

    Args:
        img_path: Path to image file.

    Returns:
        (tensor, pil_image) — tensor shape (1, 3, 224, 224), original PIL image.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image  = Image.open(img_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)   # (1, 3, 224, 224)
    return tensor, image


# ─── Target Layer Selection ───────────────────────────────────────────────────

def get_target_layer(model, arch: str = 'densenet121') -> list:
    """
    Return the appropriate CAM target layer for the given architecture.

    The target layer is the last spatial feature map before global pooling:
      - DenseNetCBAM:   model.cbam  — post-attention 1024-ch map (7×7)
      - EfficientNetB4: model.features[-1]  — 1792-ch map (7×7)

    Args:
        model: Instantiated model.
        arch:  Architecture key.

    Returns:
        List containing the target nn.Module (as expected by pytorch_grad_cam).
    """
    if arch == 'densenet121':
        return [model.cbam]
    elif arch == 'efficientnet_b4':
        return [model.features[-1]]
    else:
        raise ValueError(f"Unknown architecture for CAM: '{arch}'")


# ─── Integrated Gradients ─────────────────────────────────────────────────────

def integrated_gradients(
    model,
    image_tensor: torch.Tensor,
    target_class: int,
    device,
    n_steps: int = 50,
) -> np.ndarray:
    """
    Compute an Integrated Gradients attribution map (Sundararajan et al., 2017).

    Approximates the path integral of gradients from a black (zero) baseline
    to the input image using the trapezoidal rule.  Returns a single-channel
    attribution map normalised to [0, 1] by averaging absolute attribution
    across colour channels.

    Args:
        model:        Model in eval mode.
        image_tensor: Input tensor, shape (1, 3, 224, 224).
        target_class: Class index to explain.
        device:       Computation device.
        n_steps:      Number of interpolation steps (more = more accurate).

    Returns:
        Attribution map, shape (224, 224), values in [0, 1].
    """
    model.eval()
    baseline = torch.zeros_like(image_tensor, device=device)
    inp      = image_tensor.to(device)

    grads = []
    alphas = np.linspace(0.0, 1.0, n_steps + 1)

    for alpha in alphas:
        scaled = (baseline + alpha * (inp - baseline)).detach().requires_grad_(True)
        logits, _ = model(scaled)
        score     = logits[0, target_class]
        model.zero_grad()
        score.backward()
        grads.append(scaled.grad.detach().cpu().clone())

    # Trapezoidal approximation of the integral
    grads_t  = torch.stack(grads)                          # (n_steps+1, 1, 3, 224, 224)
    avg_grad = (grads_t[:-1] + grads_t[1:]).mean(dim=0)   # (1, 3, 224, 224)
    ig_map   = ((inp.cpu() - baseline.cpu()) * avg_grad).squeeze(0)  # (3, 224, 224)

    # Collapse channels → single attribution map in [0, 1]
    attr_map = ig_map.abs().mean(dim=0).numpy()            # (224, 224)
    if attr_map.max() > 0:
        attr_map = attr_map / attr_map.max()

    return attr_map


# ─── CAM Generation ───────────────────────────────────────────────────────────

def generate_cam(
    model,
    image_tensor: torch.Tensor,
    class_idx:    int,
    device,
    method: str = 'gradcam',
    arch:   str = 'densenet121',
) -> np.ndarray:
    """
    Generate a Class Activation Map for the specified class.

    Dispatches to the appropriate CAM implementation based on `method`.

    Args:
        model:        Trained model.
        image_tensor: Input tensor (1, 3, 224, 224).
        class_idx:    Target class index (0–13).
        device:       Computation device.
        method:       One of 'gradcam', 'gradcam++', 'eigencam', 'ig'.
        arch:         Architecture key for target layer selection.

    Returns:
        Grayscale CAM array of shape (224, 224) with values in [0, 1].
    """
    if method == 'ig':
        return integrated_gradients(model, image_tensor, class_idx, device)

    cam_classes = {
        'gradcam':   GradCAM,
        'gradcam++': GradCAMPlusPlus,
        'eigencam':  EigenCAM,
    }
    if method not in cam_classes:
        raise ValueError(
            f"Unknown CAM method '{method}'. Choose from: {list(cam_classes.keys()) + ['ig']}"
        )

    target_layers = get_target_layer(model, arch)
    cam           = cam_classes[method](model=model, target_layers=target_layers)
    targets       = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=image_tensor.to(device), targets=targets)
    return grayscale_cam[0]   # (224, 224)


# ─── Visualisation ────────────────────────────────────────────────────────────

def visualize_cam(
    img_path:   str,
    cam_map:    np.ndarray,
    class_name: str,
    method:     str,
    all_probs:  np.ndarray = None,
    save_path:  str        = None,
) -> None:
    """
    Four-panel visualisation: original | heatmap | overlay | (optional) predictions.

    Args:
        img_path:   Path to the original X-ray image.
        cam_map:    CAM heatmap (224×224), values in [0, 1].
        class_name: Name of the predicted class (used in title).
        method:     CAM method name (used in title).
        all_probs:  Optional (14,) array of all class probabilities for
                    a side-by-side prediction bar chart.
        save_path:  If given, saves the figure to this path.
    """
    orig     = cv2.imread(str(img_path))
    orig     = cv2.resize(orig, (224, 224))
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    heatmap     = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay     = cv2.addWeighted(orig_rgb, 0.55, heatmap_rgb, 0.45, 0)

    ncols = 4 if all_probs is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    axes[0].imshow(orig_rgb)
    axes[0].set_title('Original X-ray', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(cam_map, cmap='jet')
    axes[1].set_title(f'{method.upper()} Heatmap\n({class_name})', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=12)
    axes[2].axis('off')

    if all_probs is not None:
        sorted_idx = np.argsort(all_probs)[::-1][:8]
        top_labels = [LABELS[i] for i in sorted_idx]
        top_probs  = all_probs[sorted_idx]
        axes[3].barh(range(len(top_labels)), top_probs, color='steelblue', alpha=0.8)
        axes[3].set_yticks(range(len(top_labels)))
        axes[3].set_yticklabels(top_labels, fontsize=9)
        axes[3].set_xlim(0, 1)
        axes[3].axvline(0.5, color='red', linestyle='--', alpha=0.6)
        axes[3].set_title('Top-8 Predictions', fontsize=12)
        axes[3].invert_yaxis()

    filename = Path(img_path).name
    plt.suptitle(
        f'{filename}  |  {method.upper()}  |  Predicted: {class_name}',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')

    plt.show()


def run_cam_on_image(
    img_path: str,
    model,
    device,
    method: str = 'gradcam',
    arch:   str = 'densenet121',
) -> None:
    """
    Full CAM pipeline for a single image: predict → generate CAM → visualise.

    Args:
        img_path: Path to the X-ray image.
        model:    Loaded model.
        device:   Computation device.
        method:   CAM method.
        arch:     Architecture key.
    """
    image_tensor, _ = preprocess_image(img_path)

    with torch.no_grad():
        logits, _ = model(image_tensor.to(device))
        probs     = torch.sigmoid(logits)[0].cpu().numpy()

    top3 = np.argsort(probs)[::-1][:3]
    print(f'\nPredictions  ({Path(img_path).name}):')
    for i in top3:
        print(f'  {LABELS[i]:<25}  {probs[i]:.4f}')

    top_class_idx  = int(top3[0])
    top_class_name = LABELS[top_class_idx]

    cam_map = generate_cam(model, image_tensor, top_class_idx, device, method, arch)

    save_path = f'outputs/visualizations/{Path(img_path).stem}_{method}.png'
    visualize_cam(img_path, cam_map, top_class_name, method, all_probs=probs,
                  save_path=save_path)
