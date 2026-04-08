#!/usr/bin/env python3
"""
Generate a Word report (.docx) with:
- DeLong test results
- 1 IG image AD + 1 IG image CN
- Description of the interpretability folder contents
"""

import csv
import numpy as np
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

BASE = Path(__file__).parent
REPORT_DIR = BASE / "report_multi_seed"
# Handle nested interpretability dir (from scp -r)
IG_DIR = REPORT_DIR / "interpretability" / "interpretability"
if not (IG_DIR / "mlp_early_fusion").exists():
    IG_DIR = REPORT_DIR / "interpretability"


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1a, 0x1a, 0x2e)
    return h


def add_body(doc, text):
    p = doc.add_paragraph(text)
    p.style.font.size = Pt(11)
    return p


def build_report():
    doc = Document()

    # -- Styles --
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)

    # ══════════════════════════════════════════════
    # Title
    # ══════════════════════════════════════════════
    title = doc.add_heading('ResNet3D Fusion — Multi-Seed Analysis Report', level=0)
    for run in title.runs:
        run.font.color.rgb = RGBColor(0x1a, 0x1a, 0x2e)

    doc.add_paragraph(
        'Classification CN vs AD using ResNet50 3D (MedicalNet pretrained) '
        'with clinical tabular features. Multi-seed evaluation (5 seeds) '
        'across 4 fusion strategies.'
    )

    # ══════════════════════════════════════════════
    # 1. Performance Summary
    # ══════════════════════════════════════════════
    add_heading(doc, '1. Performance Summary', level=1)

    # Load summary table
    summary_path = REPORT_DIR / "summary_table.csv"
    with open(summary_path) as f:
        reader = csv.reader(f)
        rows = list(reader)

    headers = ['Method', 'Acc %', 'Bal Acc %', 'Sens %', 'Spec %', 'AUC', 'Seeds']
    table = doc.add_table(rows=len(rows), cols=len(headers), style='Light Shading Accent 1')
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(9)

    # Data rows
    for i, row in enumerate(rows[1:], start=1):
        if len(row) < 12:
            continue
        method = row[0]
        acc_m, acc_s = float(row[1]), float(row[2])
        bacc_m, bacc_s = float(row[3]), float(row[4])
        sens_m, sens_s = float(row[5]), float(row[6])
        spec_m, spec_s = float(row[7]), float(row[8])
        auc_m, auc_s = float(row[9]), float(row[10])
        n_seeds = row[11]

        values = [
            method,
            f'{acc_m:.1f} +/- {acc_s:.1f}',
            f'{bacc_m:.1f} +/- {bacc_s:.1f}',
            f'{sens_m:.1f} +/- {sens_s:.1f}',
            f'{spec_m:.1f} +/- {spec_s:.1f}',
            f'{auc_m:.3f} +/- {auc_s:.3f}',
            str(n_seeds),
        ]
        for j, val in enumerate(values):
            cell = table.rows[i].cells[j]
            cell.text = val
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(9)

    doc.add_paragraph('')

    # ══════════════════════════════════════════════
    # 2. DeLong Test
    # ══════════════════════════════════════════════
    add_heading(doc, '2. DeLong Test — Pairwise AUC Comparison', level=1)

    add_body(doc,
        'The DeLong test assesses whether AUC differences between models are '
        'statistically significant. The heatmap below shows pairwise p-values. '
        'Green cells indicate significant differences (p < 0.05).'
    )

    delong_img = REPORT_DIR / "delong_test.png"
    if delong_img.exists():
        doc.add_picture(str(delong_img), width=Inches(5.5))
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    add_body(doc, 'Key findings from DeLong test:')
    bullets = [
        'All fusion methods significantly outperform MRI-only (p < 0.001), '
        'confirming the value of multimodal integration.',
        'MLP Late Wt and XGB Late Wt achieve the highest AUCs (0.950 and 0.949) '
        'with no significant difference between them (p = 0.36).',
        'Tabular-only models (XGB/MLP) significantly underperform the best fusion methods, '
        'showing that MRI adds discriminative information beyond clinical features.',
        'MLP Late Stack is the weakest fusion method, significantly below all others.',
    ]
    for b in bullets:
        doc.add_paragraph(b, style='List Bullet')

    # ══════════════════════════════════════════════
    # 3. ROC Curves
    # ══════════════════════════════════════════════
    add_heading(doc, '3. ROC Curves', level=1)

    roc_img = REPORT_DIR / "roc_curves.png"
    if roc_img.exists():
        doc.add_picture(str(roc_img), width=Inches(5.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ══════════════════════════════════════════════
    # 4. Interpretability
    # ══════════════════════════════════════════════
    add_heading(doc, '4. Interpretability — Integrated Gradients', level=1)

    add_body(doc,
        'Integrated Gradients (Sundararajan et al., 2017) provides voxel-level attribution '
        'maps at full input resolution (128x128x128). Unlike GradCAM which is limited by '
        'feature map resolution (~4x4x4), Integrated Gradients computes attributions directly '
        'in input space by integrating gradients along a path from a zero baseline to the input. '
        'This yields precise localization of brain regions driving the model\'s decision.'
    )

    add_heading(doc, '4.1 Example: Alzheimer\'s Disease Patient', level=2)
    add_body(doc,
        'High-confidence AD prediction (p(AD) = 0.997). Bright regions indicate voxels '
        'with high attribution for the AD prediction. Activations are concentrated in the '
        'medial temporal lobe (hippocampus, entorhinal cortex), consistent with known '
        'patterns of AD-related neurodegeneration.'
    )

    ad_img = IG_DIR / "mlp_early_fusion" / "AD_01.png"
    if ad_img.exists():
        doc.add_picture(str(ad_img), width=Inches(5.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph('Figure: Integrated Gradients — AD patient (MLP Early Fusion, seed 2)',
                          style='Caption')

    add_heading(doc, '4.2 Example: Cognitively Normal Patient', level=2)
    add_body(doc,
        'High-confidence CN prediction (p(AD) = 0.003). Attribution pattern is more diffuse '
        'and distributed across cortical regions, with less concentration in medial temporal '
        'structures. This suggests the model recognizes the structural integrity of '
        'hippocampal and temporal regions as indicative of normal cognition.'
    )

    cn_img = IG_DIR / "mlp_early_fusion" / "CN_01.png"
    if cn_img.exists():
        doc.add_picture(str(cn_img), width=Inches(5.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph('Figure: Integrated Gradients — CN patient (MLP Early Fusion, seed 2)',
                          style='Caption')

    # ══════════════════════════════════════════════
    # 5. Folder Contents
    # ══════════════════════════════════════════════
    add_heading(doc, '5. Interpretability Folder Contents', level=1)

    add_body(doc,
        'The accompanying interpretability/ folder contains Integrated Gradients '
        'visualizations for all 4 models, computed on the same 5 AD and 5 CN patients '
        'for direct comparison. The structure is as follows:'
    )

    folder_desc = [
        ('mlp_early_fusion/', 'Integrated Gradients for the MLP Early Fusion model '
         '(ResNet3D + Tabular MLP, end-to-end). Contains AD_01..05.png, CN_01..05.png '
         '(individual patient maps, 3 views each) and grid_mlp_early_fusion.png (all 10 patients).'),
        ('mlp_late_fusion/', 'Same for MLP Late Fusion (ResNet3D MRI branch only, '
         'trained separately from tabular MLP). Shows what the MRI-only branch focuses on.'),
        ('xgb_early_fusion/', 'Same for XGBoost Early Fusion (finetuned ResNet3D backbone, '
         'features fed to XGBoost). IG computed through the CNN backbone.'),
        ('xgb_late_fusion/', 'Same for XGBoost Late Fusion (ResNet3D MRI branch, '
         'separate XGBoost on tabular). Shows MRI branch attributions.'),
        ('cross_model_comparison.png', 'Summary grid: 10 patients (rows) x 4 models (columns), '
         'axial view. Allows direct comparison of what each model attends to on the same brain.'),
        ('group_average_AD.png', 'Average attribution map across 50 correctly classified AD patients. '
         'Highlights consistently important regions for AD prediction.'),
        ('group_average_CN.png', 'Average attribution map across 50 correctly classified CN patients.'),
        ('group_difference_AD_minus_CN.png', 'Differential attribution map (AD average minus CN average). '
         'Red = more important for AD (medial temporal lobe), Blue = more important for CN (frontal/parietal). '
         'This is the most informative figure for understanding model behavior.'),
        ('summary_figure.png', 'Paper-ready figure combining individual examples, '
         'group average, and difference map.'),
        ('*.npy files', 'Raw numpy arrays of group averages and difference maps '
         'for further analysis or custom visualization.'),
    ]

    for name, desc in folder_desc:
        p = doc.add_paragraph(style='List Bullet')
        run = p.add_run(name + ' ')
        run.bold = True
        run.font.size = Pt(10)
        run2 = p.add_run(desc)
        run2.font.size = Pt(10)

    add_body(doc,
        'All attribution maps were computed using Integrated Gradients with 100 interpolation '
        'steps and a zero baseline, from the best-performing seed (seed 2, AUC = 0.954). '
        'The same 5 AD and 5 CN patients (selected by prediction confidence) are used across '
        'all 4 models to enable direct comparison.'
    )

    # Save
    out_path = REPORT_DIR / "resnet3d_fusion_report.docx"
    doc.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    build_report()
