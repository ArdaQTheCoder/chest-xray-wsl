"""
Chest X-ray Analysis -- Interactive Streamlit Web Demo.

Run:
    streamlit run app.py

Features:
  • Upload any chest X-ray (PNG / JPG / JPEG).
  • Multi-label classification for 14 thoracic diseases.
  • CAM visualisation: GradCAM, GradCAM++, EigenCAM, Integrated Gradients.
  • MC Dropout uncertainty estimation with error bars.
  • Architecture selection: DenseNet121+CBAM or EfficientNet-B4.
  • Full prediction table sortable by probability.
"""

import os
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

# --- Page Config --------------------------------------------------------------

st.set_page_config(
    page_title="Chest X-ray AI Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stMetric { background: #1e2130; border-radius: 8px; padding: 8px; }
    h1 { color: #4fc3f7; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")

    arch = st.selectbox(
        "Architecture",
        ["densenet121", "efficientnet_b4"],
        index=0,
        help="DenseNet121+CBAM is the primary model; EfficientNet-B4 is for comparison.",
    )

    cam_method = st.selectbox(
        "CAM Method",
        ["gradcam", "gradcam++", "eigencam", "ig"],
        index=0,
        help=(
            "gradcam / gradcam++ -- gradient-weighted activation maps.\n"
            "eigencam -- gradient-free PCA-based map.\n"
            "ig -- Integrated Gradients (slower but attribution-theoretically sound)."
        ),
    )

    show_uncertainty = st.checkbox("MC Dropout Uncertainty", value=True,
                                   help="Run N stochastic forward passes to estimate uncertainty.")
    n_mc_samples = st.slider("MC Dropout samples", 10, 50, 30,
                              disabled=not show_uncertainty)
    threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, step=0.05,
                           help="Probability above which a disease is flagged POSITIVE.")

    checkpoint_path = st.text_input(
        "Checkpoint path",
        value=f"outputs/checkpoints/best_model_{arch}.pth",
    )

    st.markdown("---")
    st.markdown("**About this demo**")
    st.markdown(
        "Graduation project -- multi-label chest X-ray classification "
        "with explainable AI and uncertainty quantification."
    )


# --- Model Loading ------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model ...")
def load_cached_model(arch: str, ckpt: str):
    """Load model once; cache across reruns (keyed on arch + ckpt path)."""
    from src.model import get_model
    from src.cam import load_model as _load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_model(arch=arch, device=device, pretrained=False)

    if Path(ckpt).exists():
        model = _load_model(ckpt, model, device)
        status = f"✅ Loaded: `{ckpt}`"
    else:
        status = f"⚠️ Checkpoint not found -- using random weights ({ckpt})"

    return model, device, status


model, device, ckpt_status = load_cached_model(arch, checkpoint_path)

with st.sidebar:
    st.info(ckpt_status)
    if torch.cuda.is_available():
        st.success(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("Running on CPU (inference will be slower)")


# --- Main ---------------------------------------------------------------------

st.title("🫁 Chest X-ray Disease Analysis")
st.markdown(
    "Upload a chest X-ray image to classify **14 thoracic conditions** "
    "with AI-powered explainability and uncertainty estimation."
)

uploaded = st.file_uploader(
    "Upload a chest X-ray (PNG / JPG / JPEG)",
    type=["png", "jpg", "jpeg"],
)

if uploaded is None:
    st.info("👆 Upload a chest X-ray image to begin analysis.")
    st.markdown("""
    **Pipeline highlights**
    | Component | Detail |
    |---|---|
    | Architecture | DenseNet121 + CBAM attention |
    | Comparison | EfficientNet-B4 |
    | Loss | Asymmetric Loss (ICCV 2021) |
    | Training | Mixed-precision + warmup cosine LR |
    | Explainability | GradCAM · GradCAM++ · EigenCAM · Integrated Gradients |
    | Uncertainty | Monte Carlo Dropout |
    | Dataset | NIH Chest X-ray 14 (112 120 images, 14 diseases) |
    """)
    st.stop()

# --- Process Uploaded Image ---------------------------------------------------

# Write to a temp file so OpenCV and preprocess_image can read it
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

try:
    from src.cam import preprocess_image, generate_cam
    from src.uncertainty import mc_dropout_predict, uncertainty_report
    from src.dataset import LABELS

    image_tensor, pil_image = preprocess_image(tmp_path)

    # -- Standard prediction ----------------------------------------------------
    model.eval()
    with torch.no_grad():
        logits, _ = model(image_tensor.to(device))
        probs_det = torch.sigmoid(logits)[0].cpu().numpy()   # deterministic

    # -- MC Dropout uncertainty -------------------------------------------------
    if show_uncertainty:
        with st.spinner(f"Running {n_mc_samples} MC Dropout passes ..."):
            mean_probs_t, std_probs_t = mc_dropout_predict(
                model, image_tensor, device, n_samples=n_mc_samples
            )
        mean_probs = mean_probs_t[0].numpy()
        std_probs  = std_probs_t[0].numpy()
    else:
        mean_probs = probs_det
        std_probs  = np.zeros_like(probs_det)

    sorted_idx     = np.argsort(mean_probs)[::-1]
    top_class_idx  = int(sorted_idx[0])
    top_class_name = LABELS[top_class_idx]

    # -- CAM --------------------------------------------------------------------
    cam_spinner_msg = (
        "Computing Integrated Gradients (this takes ~10-20 s) ..."
        if cam_method == "ig"
        else f"Generating {cam_method.upper()} map ..."
    )
    with st.spinner(cam_spinner_msg):
        cam_map = generate_cam(
            model, image_tensor, top_class_idx, device, cam_method, arch
        )

    # Overlay
    orig_arr   = np.array(pil_image.resize((224, 224))).astype(np.float32) / 255.0
    heatmap_c  = cm.jet(cam_map)[:, :, :3]
    overlay    = (0.55 * orig_arr + 0.45 * heatmap_c).clip(0, 1)

    # -------------------------------------------------------------------------
    # Layout: Original | CAM panels | Predictions
    # -------------------------------------------------------------------------

    col_img, col_cam, col_pred = st.columns([1, 2, 1.5])

    with col_img:
        st.subheader("Original X-ray")
        st.image(pil_image, use_column_width=True)

    with col_cam:
        st.subheader(f"{cam_method.upper()} -- {top_class_name}")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        axes[0].imshow(cam_map, cmap="jet")
        axes[0].set_title("Activation Heatmap", fontsize=11)
        axes[0].axis("off")
        im = axes[0].imshow(cam_map, cmap="jet")
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        axes[1].imshow(overlay)
        axes[1].set_title("Overlay on X-ray", fontsize=11)
        axes[1].axis("off")

        plt.suptitle(
            f"{cam_method.upper()} -- {top_class_name}",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_pred:
        st.subheader("Top Predictions")

        # Horizontal bar chart (top 7)
        top7 = sorted_idx[:7]
        fig, ax = plt.subplots(figsize=(5, 4))
        colours = ["#d62728" if mean_probs[i] >= threshold else "#1f77b4" for i in top7]
        ax.barh(
            range(7),
            mean_probs[top7],
            xerr=std_probs[top7] if show_uncertainty else None,
            color=colours, alpha=0.85, capsize=4, ecolor="#444",
        )
        ax.set_yticks(range(7))
        ax.set_yticklabels([LABELS[i] for i in top7], fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.axvline(threshold, color="red", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"Threshold ({threshold})")
        ax.set_xlabel("Probability")
        ax.legend(fontsize=8)
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # --- Uncertainty report ---------------------------------------------------

    if show_uncertainty:
        with st.expander("MC Dropout Uncertainty Report", expanded=False):
            st.code(uncertainty_report(mean_probs_t, std_probs_t, threshold=threshold))

            fig, ax = plt.subplots(figsize=(10, 5))
            all_idx = np.argsort(mean_probs)[::-1]
            colours_all = [
                "#d62728" if mean_probs[i] >= threshold else "#1f77b4"
                for i in all_idx
            ]
            ax.barh(
                range(len(LABELS)),
                mean_probs[all_idx],
                xerr=std_probs[all_idx],
                color=colours_all, alpha=0.8, capsize=3, ecolor="#444",
            )
            ax.set_yticks(range(len(LABELS)))
            ax.set_yticklabels([LABELS[i] for i in all_idx], fontsize=9)
            ax.set_xlim(0, 1.05)
            ax.axvline(threshold, color="red", linestyle="--", alpha=0.7)
            ax.set_title("All Classes -- MC Dropout Predictions (error bars = +-1 std)")
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # --- Full prediction table ------------------------------------------------

    st.subheader("Full Prediction Table")
    rows = []
    for i, label in enumerate(LABELS):
        rows.append({
            "Disease":      label,
            "Probability":  round(float(mean_probs[i]), 4),
            "Uncertainty":  round(float(std_probs[i]), 4) if show_uncertainty else "--",
            "Status":       "🔴 POSITIVE" if mean_probs[i] >= threshold else "🟢 negative",
        })

    df = pd.DataFrame(rows).sort_values("Probability", ascending=False).reset_index(drop=True)
    st.dataframe(df, use_container_width=True)

    # --- Download results -----------------------------------------------------
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download predictions as CSV",
        data=csv_bytes,
        file_name=f"xray_predictions_{Path(tmp_path).stem}.csv",
        mime="text/csv",
    )

finally:
    os.unlink(tmp_path)
