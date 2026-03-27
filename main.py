"""
Chest X-ray Analysis Pipeline -- CLI Entry Point.

Modes
-----
  train            Train a model from scratch (patient-level split, ASL, AMP).
  evaluate         Classification + localisation evaluation on the test split.
  cam              Generate a CAM visualisation for a single image.
  uncertainty      Run MC Dropout uncertainty estimation on a single image.
  threshold-search Find the optimal CAM binarisation threshold via IoU.
  compare          Train both architectures and compare evaluation results.

Examples
--------
  # Train the primary model (DenseNet121 + CBAM)
  python main.py train --arch densenet121 --epochs 30

  # Train the comparison model
  python main.py train --arch efficientnet_b4 --epochs 30

  # Evaluate on test split (classification + localisation)
  python main.py evaluate --arch densenet121

  # Generate GradCAM for one image
  python main.py cam --image data/nih/images/00000001_000.png --method gradcam

  # Uncertainty estimation (30 MC Dropout passes)
  python main.py uncertainty --image data/nih/images/00000001_000.png

  # Find best CAM threshold
  python main.py threshold-search --arch densenet121

  # Launch the Streamlit web demo
  streamlit run app.py
"""

import argparse
import sys
from pathlib import Path

import torch

from src.model import get_model, LABELS
from src.cam import load_model, run_cam_on_image, preprocess_image
from src.uncertainty import mc_dropout_predict, uncertainty_report, visualize_uncertainty


# --- Defaults -----------------------------------------------------------------

DATA_CSV   = 'data/nih/Data_Entry_2017.csv'
IMG_DIR    = 'data/nih/images'
BBOX_CSV   = 'data/nih/BBox_List_2017.csv'
CKPT_DIR   = 'outputs/checkpoints'


def checkpoint_path(arch: str) -> str:
    return str(Path(CKPT_DIR) / f'best_model_{arch}.pth')


# --- Argument Parser ----------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Chest X-ray 14 -- Multi-Label Classification & Explainability',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'mode',
        choices=['train', 'evaluate', 'cam', 'uncertainty', 'threshold-search', 'compare'],
        help='Pipeline mode to run.',
    )
    parser.add_argument(
        '--arch', default='densenet121',
        choices=['densenet121', 'efficientnet_b4'],
        help='Model architecture (default: densenet121).',
    )
    parser.add_argument(
        '--checkpoint', default=None,
        help='Path to checkpoint .pth file (auto-detected from --arch if not given).',
    )
    parser.add_argument('--epochs',     type=int,   default=30,    help='Training epochs.')
    parser.add_argument('--batch-size', type=int,   default=32,    help='Batch size.')
    parser.add_argument('--lr',         type=float, default=1e-4,  help='Learning rate.')
    parser.add_argument('--warmup',     type=int,   default=3,     help='Warmup epochs.')
    parser.add_argument('--no-clahe',   action='store_true',       help='Disable CLAHE.')
    parser.add_argument(
        '--image', default=None,
        help='Path to a single X-ray image (required for cam / uncertainty modes).',
    )
    parser.add_argument(
        '--method', default='gradcam',
        choices=['gradcam', 'gradcam++', 'eigencam', 'ig'],
        help='CAM method (default: gradcam).',
    )
    parser.add_argument(
        '--n-samples', type=int, default=30,
        help='Number of MC Dropout passes for uncertainty estimation.',
    )
    parser.add_argument(
        '--split', default='test', choices=['val', 'test'],
        help='Dataset split to evaluate (default: test).',
    )
    parser.add_argument(
        '--max-samples', type=int, default=200,
        help='Max annotated samples for localisation evaluation.',
    )
    return parser


# --- Mode Handlers ------------------------------------------------------------

def run_train(args, device: torch.device) -> None:
    from src.dataset import get_dataloaders
    from src.train import train

    print(f"Architecture : {args.arch}")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch size   : {args.batch_size}")
    print(f"LR           : {args.lr}")
    print(f"Warmup       : {args.warmup} epochs")
    print(f"CLAHE        : {'off' if args.no_clahe else 'on'}\n")

    train_loader, val_loader, _, _ = get_dataloaders(
        csv_path=DATA_CSV,
        img_dir=IMG_DIR,
        batch_size=args.batch_size,
        use_clahe=not args.no_clahe,
    )

    model = get_model(arch=args.arch, device=device, pretrained=True)
    train(
        model, train_loader, val_loader, device,
        epochs=args.epochs,
        lr=args.lr,
        warmup_epochs=args.warmup,
        arch=args.arch,
        checkpoint_dir=CKPT_DIR,
    )


def run_evaluate(args, device: torch.device) -> None:
    from src.evaluate import evaluate_classification, evaluate_localization

    ckpt = args.checkpoint or checkpoint_path(args.arch)
    model = get_model(arch=args.arch, device=device, pretrained=False)
    model = load_model(ckpt, model, device)

    split_csv = f'data/{args.split}_split.csv'
    if not Path(split_csv).exists():
        print(
            f"Split CSV '{split_csv}' not found.\n"
            "Run `python main.py train` first to generate splits."
        )
        sys.exit(1)

    print("=" * 60)
    print(f"CLASSIFICATION  ({args.split} split)")
    print("=" * 60)
    evaluate_classification(
        model, device, split_csv, IMG_DIR,
        output_csv=f'outputs/classification_{args.arch}_{args.split}.csv',
    )

    print("\n" + "=" * 60)
    print(f"LOCALISATION  (GradCAM, threshold=0.5, max={args.max_samples} samples)")
    print("=" * 60)
    evaluate_localization(
        model, device, BBOX_CSV, IMG_DIR,
        method='gradcam', arch=args.arch,
        max_samples=args.max_samples,
        output_csv=f'outputs/localization_{args.arch}.csv',
    )


def run_cam(args, device: torch.device) -> None:
    if not args.image:
        print("Error: --image is required for cam mode.")
        sys.exit(1)

    ckpt  = args.checkpoint or checkpoint_path(args.arch)
    model = get_model(arch=args.arch, device=device, pretrained=False)
    model = load_model(ckpt, model, device)
    run_cam_on_image(args.image, model, device, method=args.method, arch=args.arch)


def run_uncertainty(args, device: torch.device) -> None:
    if not args.image:
        print("Error: --image is required for uncertainty mode.")
        sys.exit(1)

    ckpt         = args.checkpoint or checkpoint_path(args.arch)
    model        = get_model(arch=args.arch, device=device, pretrained=False)
    model        = load_model(ckpt, model, device)
    image_tensor, _ = preprocess_image(args.image)

    print(f"Running {args.n_samples} MC Dropout passes ...")
    mean_probs, std_probs = mc_dropout_predict(
        model, image_tensor, device, n_samples=args.n_samples
    )

    print(uncertainty_report(mean_probs, std_probs))

    save_path = f'outputs/uncertainty_{Path(args.image).stem}.png'
    visualize_uncertainty(mean_probs, std_probs, save_path=save_path)


def run_threshold_search(args, device: torch.device) -> None:
    from src.evaluate import evaluate_localization

    ckpt  = args.checkpoint or checkpoint_path(args.arch)
    model = get_model(arch=args.arch, device=device, pretrained=False)
    model = load_model(ckpt, model, device)

    thresholds   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    best_thresh  = 0.5
    best_iou     = 0.0

    print("-" * 40)
    print("CAM Threshold Search  (GradCAM)")
    print("-" * 40)

    for t in thresholds:
        results  = evaluate_localization(
            model, device, BBOX_CSV, IMG_DIR,
            method='gradcam', arch=args.arch,
            threshold=t, max_samples=args.max_samples,
            output_csv=f'outputs/threshold_{t:.1f}.csv',
        )
        mean_iou = results['iou'].mean()
        print(f"  Threshold {t:.1f}  ->  Mean IoU: {mean_iou:.4f}")

        if mean_iou > best_iou:
            best_iou    = mean_iou
            best_thresh = t

    print(f"\nBest threshold : {best_thresh}")
    print(f"Best IoU       : {best_iou:.4f}")


def run_compare(args, device: torch.device) -> None:
    """Train both architectures then produce a side-by-side evaluation summary."""
    import pandas as pd
    from src.dataset import get_dataloaders
    from src.train import train
    from src.evaluate import evaluate_classification

    summary_rows = []

    for arch in ['densenet121', 'efficientnet_b4']:
        print(f"\n{'='*60}")
        print(f"Training: {arch}")
        print(f"{'='*60}")

        train_loader, val_loader, _, _ = get_dataloaders(
            csv_path=DATA_CSV, img_dir=IMG_DIR,
            batch_size=args.batch_size, use_clahe=not args.no_clahe,
        )
        model = get_model(arch=arch, device=device, pretrained=True)
        train(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr,
            warmup_epochs=args.warmup, arch=arch,
            checkpoint_dir=CKPT_DIR,
        )

        # Evaluate on val split (test split may not exist yet)
        ckpt   = checkpoint_path(arch)
        model  = get_model(arch=arch, device=device, pretrained=False)
        model  = load_model(ckpt, model, device)
        results = evaluate_classification(
            model, device, 'data/val_split.csv', IMG_DIR,
            output_csv=f'outputs/classification_{arch}_val.csv',
        )
        macro = sum(v['auroc'] for v in results.values()) / len(results)
        summary_rows.append({'Architecture': arch, 'Macro AUROC (val)': round(macro, 4)})

    print("\n" + "=" * 40)
    print("Comparison Summary")
    print("=" * 40)
    print(pd.DataFrame(summary_rows).to_string(index=False))


# --- Main ---------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    dispatch = {
        'train':            run_train,
        'evaluate':         run_evaluate,
        'cam':              run_cam,
        'uncertainty':      run_uncertainty,
        'threshold-search': run_threshold_search,
        'compare':          run_compare,
    }
    dispatch[args.mode](args, device)


if __name__ == '__main__':
    main()
