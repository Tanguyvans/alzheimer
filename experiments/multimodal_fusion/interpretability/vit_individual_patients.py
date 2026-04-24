#!/usr/bin/env python3
"""
Per-patient ViT attention visualization.

Re-runs ViT attention rollout on specific individual subjects (not averaged)
to show inter-subject consistency of the medial-temporal focus pattern.

Picks 2 CN + 2 AD well-classified subjects from fold 0's test set,
loads each subject's actual brain MRI, and overlays its attention map.
"""

import pickle
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from dataset import MultiModalDataset  # noqa: E402
from interpretability.ft_attention import build_paper_model, TABULAR_FEATURES  # noqa: E402
from interpretability.vit_attention import (  # noqa: E402
    patched_attention_forward, attention_rollout, regenerate_fold_splits,
)

EXP = Path("/home/tanguy/medical/alzheimer/experiments/multimodal_fusion")
RESULTS = SCRIPT_DIR / "results_new"


def install_hooks(model):
    for blk in model.mri_backbone.blocks:
        blk.attn.forward = types.MethodType(patched_attention_forward, blk.attn)


def upsample_to_128(grid8):
    t = torch.from_numpy(grid8).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(128, 128, 128), mode="trilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()


def normalize(vol, percentile=(5, 99)):
    lo, hi = np.percentile(vol, percentile)
    return np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1)


def load_brain_for_overlay(scan_path):
    """Load a subject's brain and resample to 128^3 for the overlay."""
    img = nib.load(scan_path)
    vol = img.get_fdata().astype(np.float32)
    zf = [128 / s for s in vol.shape]
    return normalize(zoom(vol, zf, order=1))


def get_subject_attention(model, test_ds, sample_idx, device):
    """Run forward on one sample, return 8^3 rollout grid."""
    mri, tab, _ = test_ds[sample_idx]
    mri = mri.unsqueeze(0).to(device)
    tab = tab.unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(mri, tab)
    attns = [blk.attn.last_attn for blk in model.mri_backbone.blocks]
    roll = attention_rollout(attns)[0].cpu().numpy()  # (512,)
    return roll.reshape(8, 8, 8)


def select_subjects(model, test_ds, loader, device, n_per_class=2):
    """Top n_per_class most confidently correctly classified CN and AD."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for mri, tab, y in loader:
            mri, tab = mri.to(device), tab.to(device)
            out = model(mri, tab)
            logits = out[0] if isinstance(out, tuple) else out
            all_probs.append(torch.softmax(logits, dim=-1).cpu())
            all_labels.append(y)
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = probs.argmax(-1)
    correct = preds == labels
    cn_correct = np.where(correct & (labels == 0))[0]
    ad_correct = np.where(correct & (labels == 1))[0]
    cn_top = cn_correct[np.argsort(-probs[cn_correct, 0])[:n_per_class]]
    ad_top = ad_correct[np.argsort(-probs[ad_correct, 1])[:n_per_class]]
    return cn_top.tolist(), ad_top.tolist(), probs


def plot_patient(ax_row, patient_label, patient_id, prob, bg, heatmap_norm, slices):
    """Plot 3 slices for one patient across a row of 3 axes."""
    for c, (plane, idx) in enumerate(slices):
        ax = ax_row[c]
        if plane == "axial":
            bg_slice = bg[:, :, idx]
            hm_slice = heatmap_norm[:, :, idx]
            title = f"Axial z={idx}"
        elif plane == "coronal":
            bg_slice = bg[:, idx, :]
            hm_slice = heatmap_norm[:, idx, :]
            title = f"Coronal y={idx}"
        else:
            bg_slice = bg[idx, :, :]
            hm_slice = heatmap_norm[idx, :, :]
            title = f"Sagittal x={idx}"
        ax.imshow(bg_slice.T, cmap="gray", origin="lower")
        ax.imshow(hm_slice.T, cmap="hot", alpha=0.55, origin="lower", vmin=0, vmax=1)
        if c == 0:
            ax.set_ylabel(
                f"{patient_label}\n{patient_id}\nP={prob:.2f}",
                fontsize=10, fontweight="bold",
            )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    fold, seed = 0, 42
    _, test_df = regenerate_fold_splits(seed, fold)
    ckpt_path = EXP / "cv_results" / f"seed_{seed}" / f"fold_{fold}" / "model.pth"
    scaler_path = EXP / "cv_results" / f"seed_{seed}" / f"fold_{fold}" / "scaler.pkl"

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    tmp_csv = RESULTS / "_tmp_pat_test.csv"
    test_df.to_csv(tmp_csv, index=False)
    ds = MultiModalDataset(
        str(tmp_csv),
        tabular_features=TABULAR_FEATURES,
        target_shape=(128, 128, 128),
        augment=False,
        normalize_tabular=True,
        scaler=scaler,
        use_paper_preprocessing=True,
        target_spacing=1.75,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    print("[*] Loading model...")
    model = build_paper_model()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    install_hooks(model)

    print("[*] Selecting patients...")
    cn_idx, ad_idx, probs = select_subjects(model, ds, loader, device, n_per_class=2)
    print(f"    CN: {cn_idx}  probs={[f'{probs[i,0]:.3f}' for i in cn_idx]}")
    print(f"    AD: {ad_idx}  probs={[f'{probs[i,1]:.3f}' for i in ad_idx]}")

    patients = []
    for name, indices in [("CN", cn_idx), ("AD", ad_idx)]:
        for i, s_idx in enumerate(indices):
            row = test_df.iloc[s_idx]
            grid = get_subject_attention(model, ds, s_idx, device)
            hmap = upsample_to_128(grid)
            hmap_norm = normalize(hmap, percentile=(5, 99))
            bg = load_brain_for_overlay(row["scan_path"])
            patients.append({
                "label": f"{name} #{i + 1}",
                "subject_id": row["subject_id"],
                "prob": probs[s_idx, 0 if name == "CN" else 1],
                "bg": bg,
                "hmap": hmap_norm,
            })
    tmp_csv.unlink()

    slices = [("axial", 40), ("axial", 50), ("coronal", 72)]
    fig, axes = plt.subplots(4, 3, figsize=(11, 14))
    for r, patient in enumerate(patients):
        plot_patient(axes[r], patient["label"], patient["subject_id"],
                     patient["prob"], patient["bg"], patient["hmap"], slices)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(vmin=0, vmax=1))
    fig.colorbar(sm, cax=cbar_ax, label="Attention (normalized)")

    plt.suptitle(
        "Individual patient ViT attention — 2 CN + 2 AD (fold 0)\n"
        "Each row shows attention from the CLS token to brain regions for one subject",
        fontsize=11, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 0.92, 0.975])
    out_path = RESULTS / "vit_individual_patients.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path.name}")


if __name__ == "__main__":
    main()
