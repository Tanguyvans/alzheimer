#!/usr/bin/env python3
"""
Generate a .docx report from multi-seed results, matching the style
of report_old/resnet3d_fusion_report.docx.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, confusion_matrix

BASE = Path(__file__).parent
MLP_DIR = BASE / "resnet3d_mlp"
XGB_DIR = BASE / "resnet3d_xgboost"
REPORT_DIR = BASE / "report_multi_seed"
REPORT_DIR.mkdir(exist_ok=True)

MAX_SEED = 20


def load_all_seeds():
    all_preds = {}
    seed_counts = {}
    y_true = None

    def _add(method, seed, proba, yt):
        nonlocal y_true
        y_true = yt
        if method not in all_preds:
            all_preds[method] = []
            seed_counts[method] = []
        all_preds[method].append(proba)
        seed_counts[method].append(seed)

    for seed in range(MAX_SEED):
        p = MLP_DIR / "results_early" / f"seed_{seed}"
        if (p / "y_true_test.npy").exists():
            yt = np.load(p / "y_true_test.npy")
            _add("MLP Early", seed, np.load(p / "y_proba_test.npy"), yt)

    for seed in range(MAX_SEED):
        p = MLP_DIR / "results_late_fusion" / f"seed_{seed}"
        if (p / "y_true_test.npy").exists():
            yt = np.load(p / "y_true_test.npy")
            _add("MRI only (MLP)", seed, np.load(p / "y_proba_mri_test.npy"), yt)
            _add("Tab only (MLP)", seed, np.load(p / "y_proba_tab_test.npy"), yt)
            _add("MLP Late Avg", seed, np.load(p / "y_proba_avg_test.npy"), yt)
            _add("MLP Late Wt", seed, np.load(p / "y_proba_weighted_test.npy"), yt)
            _add("MLP Late Stack", seed, np.load(p / "y_proba_stacking_test.npy"), yt)

    for seed in range(MAX_SEED):
        p = XGB_DIR / "results_finetuned" / f"seed_{seed}"
        if (p / "y_true_test.npy").exists():
            yt = np.load(p / "y_true_test.npy")
            _add("XGB Early", seed, np.load(p / "y_proba_test.npy"), yt)

    for seed in range(MAX_SEED):
        p = XGB_DIR / "results_late_fusion" / f"seed_{seed}"
        if (p / "y_true_test.npy").exists():
            yt = np.load(p / "y_true_test.npy")
            _add("MRI only (XGB)", seed, np.load(p / "y_proba_mri_test.npy"), yt)
            _add("Tab only (XGB)", seed, np.load(p / "y_proba_tab_test.npy"), yt)
            _add("XGB Late Avg", seed, np.load(p / "y_proba_avg_test.npy"), yt)
            _add("XGB Late Wt", seed, np.load(p / "y_proba_weighted_test.npy"), yt)
            _add("XGB Late Stack", seed, np.load(p / "y_proba_stacking_test.npy"), yt)

    return y_true, all_preds, seed_counts


def compute_metrics(y_true, all_preds, seed_counts):
    """Compute mean +/- std for each method."""
    rows = []

    # Define display order and labels
    method_info = [
        ("MRI only (XGB)",  "MRI only (ResNet3D)",         "\u2014"),
        ("MRI only (MLP)",  "MRI only (ResNet3D)",         "\u2014"),
        ("Tab only (XGB)",  "Tabular only (XGBoost)",      "\u2014"),
        ("Tab only (MLP)",  "Tabular only (MLP)",          "\u2014"),
        ("MLP Early",       "ResNet3D + MLP concat",       "Early"),
        ("XGB Early",       "ResNet3D emb + Tab \u2192 XGB", "Early"),
        ("XGB Late Avg",    "ResNet3D + XGBoost (Avg)",    "Late"),
        ("XGB Late Wt",     "ResNet3D + XGBoost (Weighted)", "Late"),
        ("XGB Late Stack",  "ResNet3D + XGBoost (Stacking)", "Late"),
        ("MLP Late Avg",    "ResNet3D + MLP (Avg)",        "Late"),
        ("MLP Late Wt",     "ResNet3D + MLP (Weighted)",   "Late"),
        ("MLP Late Stack",  "ResNet3D + MLP (Stacking)",   "Late"),
    ]

    for key, display_name, fusion_type in method_info:
        if key not in all_preds:
            continue
        proba_list = all_preds[key]
        n_seeds = len(proba_list)

        accs, baccs, senss, specs, aucs = [], [], [], [], []
        for y_proba in proba_list:
            y_pred = (y_proba >= 0.5).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            accs.append(accuracy_score(y_true, y_pred) * 100)
            baccs.append(balanced_accuracy_score(y_true, y_pred) * 100)
            senss.append(tp / (tp + fn) * 100 if (tp + fn) > 0 else 0)
            specs.append(tn / (tn + fp) * 100 if (tn + fp) > 0 else 0)
            aucs.append(roc_auc_score(y_true, y_proba))

        rows.append({
            'key': key,
            'Method': display_name,
            'Fusion': fusion_type,
            'N': n_seeds,
            'Acc_mean': np.mean(accs), 'Acc_std': np.std(accs) if n_seeds > 1 else None,
            'BAcc_mean': np.mean(baccs), 'BAcc_std': np.std(baccs) if n_seeds > 1 else None,
            'Sens_mean': np.mean(senss), 'Sens_std': np.std(senss) if n_seeds > 1 else None,
            'Spec_mean': np.mean(specs), 'Spec_std': np.std(specs) if n_seeds > 1 else None,
            'AUC_mean': np.mean(aucs), 'AUC_std': np.std(aucs) if n_seeds > 1 else None,
        })

    return rows


def fmt_pct(mean, std):
    if std is None:
        return f"{mean:.1f}"
    return f"{mean:.1f} \u00b1 {std:.1f}"


def fmt_auc(mean, std):
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} \u00b1 {std:.3f}"


def set_cell_font(cell, text, bold=False, size=8):
    cell.text = ""
    run = cell.paragraphs[0].add_run(text)
    run.font.size = Pt(size)
    run.font.name = "Calibri"
    run.bold = bold
    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER


def generate_report():
    print("Loading predictions...")
    y_true, all_preds, seed_counts = load_all_seeds()
    metrics = compute_metrics(y_true, all_preds, seed_counts)

    # Find best per column
    best_acc = max(r['Acc_mean'] for r in metrics)
    best_bacc = max(r['BAcc_mean'] for r in metrics)
    best_sens = max(r['Sens_mean'] for r in metrics)
    best_spec = max(r['Spec_mean'] for r in metrics)
    best_auc = max(r['AUC_mean'] for r in metrics)

    doc = Document()

    # ── Title ──
    doc.add_heading("ResNet3D Multimodal Fusion", level=1)
    doc.add_paragraph("CN vs AD Classification \u2014 Multi-Seed Test Set Results")

    # ── Dataset ──
    doc.add_heading("Dataset", level=2)
    doc.add_paragraph("Combined trajectory dataset (ADNI + OASIS + NACC)")
    doc.add_paragraph("Train: 4,245 / Val: 910 / Test: 910 samples (78% CN / 22% AD)", style="List Bullet")
    doc.add_paragraph("16 tabular features: demographics, cognitive tests, medical history", style="List Bullet")
    doc.add_paragraph("MRI: 3D brain volumes (128 \u00d7 128 \u00d7 128)", style="List Bullet")

    # ── Backbone ──
    doc.add_heading("Backbone: ResNet3D", level=2)
    doc.add_paragraph(
        "MONAI ResNet50 3D pretrained on 23 medical imaging datasets (MedicalNet). "
        "Outputs 2048-dimensional feature vectors from 3D MRI volumes.",
        style="List Bullet"
    )
    doc.add_paragraph(
        "Fine-tuning: backbone frozen for 3 first epochs, then unfrozen with "
        "differential learning rate (backbone 10x lower). Mixed precision (AMP).",
        style="List Bullet"
    )

    # ── Fusion Strategies ──
    doc.add_heading("Fusion Strategies", level=2)

    doc.add_heading("1. Early Fusion: ResNet3D + MLP", level=3)
    doc.add_paragraph(
        "Both modalities are encoded into feature vectors, concatenated, and fed to a joint MLP classifier. "
        "The entire model is trained end-to-end."
    )
    doc.add_paragraph("MRI branch: ResNet3D (MedicalNet) \u2192 2048-d features", style="List Bullet")
    doc.add_paragraph("Tabular branch: MLP encoder [64, 32] with LayerNorm", style="List Bullet")
    doc.add_paragraph("Fusion: concatenation (2080-d) \u2192 MLP [256, 128] \u2192 classification", style="List Bullet")

    doc.add_heading("2. Early Fusion: ResNet3D embeddings + XGBoost", level=3)
    doc.add_paragraph(
        "ResNet3D is first fine-tuned on MRI classification. Then, 2048-d embeddings are extracted "
        "and concatenated with 16 tabular features. A single XGBoost model is trained on the combined "
        "2064-d feature vector."
    )
    doc.add_paragraph("Phase 1: Fine-tune ResNet3D with linear head (30 epochs)", style="List Bullet")
    doc.add_paragraph("Phase 2: Extract 2048-d embeddings + 16 tabular features \u2192 XGBoost", style="List Bullet")

    doc.add_heading("3. Late Fusion: Separate predictions + probability combination", level=3)
    doc.add_paragraph(
        "Each modality makes its own independent prediction. The MRI branch (fine-tuned ResNet3D with "
        "linear head) produces P(AD|MRI), and the tabular branch (MLP or XGBoost) produces P(AD|tabular). "
        "The two probabilities are combined via:"
    )
    doc.add_paragraph("Average: simple mean of probabilities", style="List Bullet")
    doc.add_paragraph("Weighted average: weights optimized on validation set", style="List Bullet")
    doc.add_paragraph("Stacking: Logistic Regression trained on both branch probabilities", style="List Bullet")

    # ── Results ──
    doc.add_heading("Results", level=2)

    # Insert figures if they exist
    for img_name, caption in [
        ("boxplots.png", "Figure: AUC and Balanced Accuracy distribution across seeds."),
        ("roc_curves.png", "Figure: ROC curves (mean \u00b1 std) across all methods."),
    ]:
        img_path = REPORT_DIR / img_name
        if img_path.exists():
            doc.add_picture(str(img_path), width=Inches(6.0))
            doc.add_paragraph(caption)

    # ── Results Table ──
    doc.add_paragraph(
        "Table: Test set results for all ResNet3D multimodal fusion methods (mean \u00b1 std). "
        "N = number of seeds. Bold values indicate best performance per metric."
    )

    headers = ["Method", "Fusion", "N", "Acc (%)", "Bal Acc (%)", "Sens (%)", "Spec (%)", "AUC"]
    table = doc.add_table(rows=1 + len(metrics), cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    for j, h in enumerate(headers):
        set_cell_font(table.rows[0].cells[j], h, bold=True, size=8)

    # Data rows
    for i, row in enumerate(metrics):
        cells = table.rows[i + 1].cells
        set_cell_font(cells[0], row['Method'], size=8)
        set_cell_font(cells[1], row['Fusion'], size=8)
        set_cell_font(cells[2], str(row['N']), size=8)

        # Acc
        text = fmt_pct(row['Acc_mean'], row['Acc_std'])
        bold = (row['Acc_mean'] == best_acc)
        set_cell_font(cells[3], text, bold=bold, size=8)

        # Bal Acc
        text = fmt_pct(row['BAcc_mean'], row['BAcc_std'])
        bold = (row['BAcc_mean'] == best_bacc)
        set_cell_font(cells[4], text, bold=bold, size=8)

        # Sens
        text = fmt_pct(row['Sens_mean'], row['Sens_std'])
        bold = (row['Sens_mean'] == best_sens)
        set_cell_font(cells[5], text, bold=bold, size=8)

        # Spec
        text = fmt_pct(row['Spec_mean'], row['Spec_std'])
        bold = (row['Spec_mean'] == best_spec)
        set_cell_font(cells[6], text, bold=bold, size=8)

        # AUC
        text = fmt_auc(row['AUC_mean'], row['AUC_std'])
        bold = (row['AUC_mean'] == best_auc)
        set_cell_font(cells[7], text, bold=bold, size=8)

    # ── DeLong ──
    delong_img = REPORT_DIR / "delong_test.png"
    if delong_img.exists():
        doc.add_heading("DeLong Test", level=2)
        doc.add_paragraph(
            "Pairwise DeLong test on mean probabilities. Methods with a single seed "
            "are marked N/A as the comparison is not reliable without repeated runs."
        )
        doc.add_picture(str(delong_img), width=Inches(5.5))

    # ── Confusion Matrices ──
    cm_img = REPORT_DIR / "confusion_matrices.png"
    if cm_img.exists():
        doc.add_heading("Confusion Matrices", level=2)
        doc.add_picture(str(cm_img), width=Inches(6.0))

    # ── GradCAM ──
    gc_img = REPORT_DIR / "gradcam_examples.png"
    if gc_img.exists():
        doc.add_heading("GradCAM", level=2)
        doc.add_paragraph(
            "GradCAM visualizations from the best-AUC seed (MLP Early Fusion). "
            "Red regions indicate high AD-related activation."
        )
        doc.add_picture(str(gc_img), width=Inches(6.0))

    # ── Training Details ──
    doc.add_heading("Training Details", level=2)
    details = [
        ("ResNet3D backbone", "MONAI ResNet50, MedicalNet pretrained (23 datasets)"),
        ("Fine-tuning", "30 epochs, frozen 3 first epochs, differential LR (backbone 10x lower)"),
        ("Mixed precision", "AMP enabled, gradient accumulation (2 steps, effective batch=4)"),
        ("MLP tabular", "LayerNorm, hidden dims [128, 64, 32], dropout=0.3, 100 epochs"),
        ("XGBoost tabular", "max_depth=6, lr=0.1, subsample=0.8, 300 rounds, early stopping=30"),
        ("Class imbalance", "Weighted CrossEntropyLoss (78% CN / 22% AD)"),
        ("Early stopping", "Patience=20, min_epochs=30, monitored: balanced accuracy"),
        ("Optimizer", "AdamW with cosine annealing + warmup (5 epochs)"),
    ]
    t2 = doc.add_table(rows=len(details), cols=2)
    t2.style = "Light Grid Accent 1"
    for i, (comp, conf) in enumerate(details):
        set_cell_font(t2.rows[i].cells[0], comp, bold=True, size=8)
        set_cell_font(t2.rows[i].cells[1], conf, size=8)

    # Save
    out_path = REPORT_DIR / "resnet3d_fusion_report_multi_seed.docx"
    doc.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    generate_report()
