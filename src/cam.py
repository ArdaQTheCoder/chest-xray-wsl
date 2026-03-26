import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from PIL import Image

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

def load_model(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(img_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
    return tensor, image

def generate_cam(model, image_tensor, class_idx, device, method='gradcam'):
    target_layers = [model.features.denseblock4.denselayer16.conv2]

    cam_methods = {
        'gradcam':     GradCAM,
        'gradcam++':   GradCAMPlusPlus,
        'scorecam':    ScoreCAM,
    }

    CAMClass = cam_methods.get(method, GradCAM)
    cam = CAMClass(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=image_tensor.to(device), targets=targets)
    return grayscale_cam[0]  # (224, 224)

def visualize_cam(img_path, cam_map, class_name, method, save_path=None):
    # Orijinal görüntü
    orig = cv2.imread(str(img_path))
    orig = cv2.resize(orig, (224, 224))
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    # Isı haritası
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = cv2.addWeighted(orig_rgb, 0.6, heatmap_rgb, 0.4, 0)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_rgb, cmap='gray')
    axes[0].set_title('Orijinal Röntgen')
    axes[0].axis('off')

    axes[1].imshow(cam_map, cmap='jet')
    axes[1].set_title(f'{method.upper()} Isı Haritası\n({class_name})')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Kaydedildi: {save_path}')

    plt.show()

def run_cam_on_image(img_path, model, device, method='gradcam'):
    image_tensor, orig_image = preprocess_image(img_path)

    # Tahmin
    with torch.no_grad():
        logits, _ = model(image_tensor.to(device))
        probs = torch.sigmoid(logits)[0]

    # En yüksek 3 tahmini göster
    top3 = torch.topk(probs, 3)
    print(f'\nTahminler ({Path(img_path).name}):')
    for score, idx in zip(top3.values, top3.indices):
        print(f'  {LABELS[idx.item()]:25s} {score.item():.4f}')

    # En yüksek tahmin için CAM üret
    top_class_idx = top3.indices[0].item()
    top_class_name = LABELS[top_class_idx]

    cam_map = generate_cam(model, image_tensor, top_class_idx, device, method)

    save_path = f'outputs/visualizations/{Path(img_path).stem}_{method}.png'
    visualize_cam(img_path, cam_map, top_class_name, method, save_path)