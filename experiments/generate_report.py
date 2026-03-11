#!/usr/bin/env python3
"""
Generate ResNet3D fusion experiments report with architecture diagrams and comparison tables.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "report"
OUTPUT_DIR.mkdir(exist_ok=True)


def draw_box(ax, xy, w, h, text, color='#4A90D9', fontsize=9, text_color='white'):
    """Draw a rounded box with centered text."""
    x, y = xy
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='#333333', linewidth=1.2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color)


def draw_arrow(ax, start, end, color='#333333'):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=15,
                            color=color, linewidth=1.5)
    ax.add_patch(arrow)


# ── Colors ──
C_INPUT = '#6C757D'      # grey - inputs
C_MRI = '#2196F3'        # blue - MRI branch
C_TAB = '#FF9800'        # orange - tabular branch
C_FUSION = '#9C27B0'     # purple - fusion
C_OUTPUT = '#4CAF50'     # green - output
C_XGBOOST = '#F44336'    # red - XGBoost


def fig_early_fusion_mlp():
    """Diagram: Early Fusion — ResNet3D + MLP concat → classifier."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('Early Fusion: ResNet3D + MLP', fontsize=14, fontweight='bold', pad=15)

    # MRI branch (top)
    draw_box(ax, (0.3, 3.2), 1.5, 0.8, 'MRI\n(128³)', C_INPUT)
    draw_arrow(ax, (1.8, 3.6), (2.5, 3.6))
    draw_box(ax, (2.5, 3.2), 2.2, 0.8, 'ResNet3D\n(MedicalNet)', C_MRI)
    draw_arrow(ax, (4.7, 3.6), (5.4, 3.6))
    draw_box(ax, (5.4, 3.2), 1.5, 0.8, 'Features\n(2048-d)', C_MRI, text_color='white')

    # Tabular branch (bottom)
    draw_box(ax, (0.3, 1.0), 1.5, 0.8, 'Tabular\n(16 feat.)', C_INPUT)
    draw_arrow(ax, (1.8, 1.4), (2.5, 1.4))
    draw_box(ax, (2.5, 1.0), 2.2, 0.8, 'MLP Encoder\n[64 → 32]', C_TAB)
    draw_arrow(ax, (4.7, 1.4), (5.4, 1.4))
    draw_box(ax, (5.4, 1.0), 1.5, 0.8, 'Features\n(32-d)', C_TAB, text_color='white')

    # Fusion
    draw_arrow(ax, (6.9, 3.2), (7.8, 2.6))
    draw_arrow(ax, (6.9, 1.8), (7.8, 2.3))
    draw_box(ax, (7.8, 1.9), 1.2, 0.8, 'Concat\n(2080-d)', C_FUSION)
    draw_arrow(ax, (9.0, 2.3), (9.3, 2.3))
    draw_box(ax, (9.3, 1.9), 1.5, 0.8, 'MLP\n[256→128]', C_FUSION)
    draw_arrow(ax, (10.8, 2.3), (11.0, 2.3))
    draw_box(ax, (11.0, 1.9), 0.8, 0.8, 'CN\nvs\nAD', C_OUTPUT, fontsize=8)

    # Label
    ax.text(6.0, 0.3, 'Fusion at feature level: concatenation before decision',
            ha='center', va='center', fontsize=10, style='italic', color='#666')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'early_fusion_mlp.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved early_fusion_mlp.png")


def fig_early_fusion_xgboost():
    """Diagram: Early Fusion — ResNet3D embeddings + tabular → XGBoost."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('Early Fusion: ResNet3D embeddings + Tabular → XGBoost', fontsize=14, fontweight='bold', pad=15)

    # MRI branch (top)
    draw_box(ax, (0.3, 3.2), 1.5, 0.8, 'MRI\n(128³)', C_INPUT)
    draw_arrow(ax, (1.8, 3.6), (2.5, 3.6))
    draw_box(ax, (2.5, 3.2), 2.5, 0.8, 'ResNet3D\n(fine-tuned)', C_MRI)
    draw_arrow(ax, (5.0, 3.6), (5.5, 3.6))
    draw_box(ax, (5.5, 3.2), 1.5, 0.8, 'Embeddings\n(2048-d)', C_MRI, text_color='white')

    # Tabular branch (bottom)
    draw_box(ax, (0.3, 1.0), 1.5, 0.8, 'Tabular\n(16 feat.)', C_INPUT)
    draw_arrow(ax, (1.8, 1.4), (5.5, 1.4))
    draw_box(ax, (5.5, 1.0), 1.5, 0.8, 'Raw\nfeatures', C_TAB, text_color='white')

    # Fusion
    draw_arrow(ax, (7.0, 3.2), (7.8, 2.6))
    draw_arrow(ax, (7.0, 1.8), (7.8, 2.3))
    draw_box(ax, (7.8, 1.9), 1.2, 0.8, 'Concat\n(2064-d)', C_FUSION)
    draw_arrow(ax, (9.0, 2.3), (9.3, 2.3))
    draw_box(ax, (9.3, 1.9), 1.5, 0.8, 'XGBoost', C_XGBOOST)
    draw_arrow(ax, (10.8, 2.3), (11.0, 2.3))
    draw_box(ax, (11.0, 1.9), 0.8, 0.8, 'CN\nvs\nAD', C_OUTPUT, fontsize=8)

    # Label
    ax.text(6.0, 0.3, 'Fusion at feature level: embeddings + tabular concatenated → single XGBoost',
            ha='center', va='center', fontsize=10, style='italic', color='#666')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'early_fusion_xgboost.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved early_fusion_xgboost.png")


def fig_late_fusion():
    """Diagram: Late Fusion — separate predictions → combine probas."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Late Fusion: Separate predictions → Probability combination', fontsize=14, fontweight='bold', pad=15)

    # MRI branch (top)
    draw_box(ax, (0.3, 4.0), 1.3, 0.8, 'MRI\n(128³)', C_INPUT)
    draw_arrow(ax, (1.6, 4.4), (2.0, 4.4))
    draw_box(ax, (2.0, 4.0), 2.2, 0.8, 'ResNet3D\n(fine-tuned)', C_MRI)
    draw_arrow(ax, (4.2, 4.4), (4.6, 4.4))
    draw_box(ax, (4.6, 4.0), 1.5, 0.8, 'Linear\nHead', C_MRI)
    draw_arrow(ax, (6.1, 4.4), (6.5, 4.4))
    draw_box(ax, (6.5, 4.0), 1.2, 0.8, 'P(AD)\nMRI', C_MRI, fontsize=9)

    # Tabular branch (bottom) — two variants
    # MLP variant
    draw_box(ax, (0.3, 2.2), 1.3, 0.8, 'Tabular\n(16 feat.)', C_INPUT)
    draw_arrow(ax, (1.6, 2.6), (2.0, 2.6))
    draw_box(ax, (2.0, 2.2), 2.2, 0.8, 'MLP / XGBoost\n(standalone)', C_TAB)
    draw_arrow(ax, (4.2, 2.6), (6.5, 2.6))
    draw_box(ax, (6.5, 2.2), 1.2, 0.8, 'P(AD)\nTab', C_TAB, fontsize=9)

    # Fusion
    draw_arrow(ax, (7.7, 4.0), (8.3, 3.5))
    draw_arrow(ax, (7.7, 3.0), (8.3, 3.3))
    draw_box(ax, (8.3, 2.9), 1.8, 0.8, 'Combine\nProbabilities', C_FUSION)
    draw_arrow(ax, (10.1, 3.3), (10.5, 3.3))
    draw_box(ax, (10.5, 2.9), 0.8, 0.8, 'CN\nvs\nAD', C_OUTPUT, fontsize=8)

    # Fusion methods
    ax.text(9.2, 2.4, 'Average', ha='center', fontsize=8, color='#666',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8E8E8', edgecolor='#999'))
    ax.text(9.2, 1.8, 'Weighted Avg', ha='center', fontsize=8, color='#666',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8E8E8', edgecolor='#999'))
    ax.text(9.2, 1.2, 'Stacking (LogReg)', ha='center', fontsize=8, color='#666',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8E8E8', edgecolor='#999'))

    ax.text(6.0, 0.5, 'Fusion at decision level: each modality predicts independently,\nthen probabilities are combined',
            ha='center', va='center', fontsize=10, style='italic', color='#666')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'late_fusion.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved late_fusion.png")


def fig_comparison_table():
    """Bar chart comparing all methods."""
    methods = [
        'MRI only\n(ResNet3D)',
        'Tab only\n(XGBoost)',
        'Tab only\n(MLP)',
        'Early Fusion\nResNet3D+MLP',
        'Early Fusion\nEmb+XGBoost',
        'Late Avg\n(+XGBoost)',
        'Late Weighted\n(+XGBoost)',
        'Late Stacking\n(+XGBoost)',
        'Late Avg\n(+MLP)',
        'Late Weighted\n(+MLP)',
        'Late Stacking\n(+MLP)',
    ]

    # Using XGBoost MRI-only for shared MRI results
    bal_acc = [82.1, 85.8, 85.2, 88.2, 85.3, 84.6, 88.1, 84.1, 86.3, 88.3, 78.4]
    auc =     [0.907, 0.937, 0.932, 0.952, 0.928, 0.946, 0.948, 0.949, 0.943, 0.947, 0.877]

    colors = [C_MRI, C_XGBOOST, C_TAB, C_FUSION, C_FUSION,
              '#7B1FA2', '#6A1B9A', '#4A148C',
              '#AB47BC', '#9C27B0', '#7B1FA2']

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # Balanced Accuracy
    ax = axes[0]
    bars = ax.bar(range(len(methods)), bal_acc, color=colors, edgecolor='#333', linewidth=0.8)
    ax.set_ylabel('Balanced Accuracy (%)', fontsize=11)
    ax.set_title('Balanced Accuracy Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=7.5, rotation=0)
    ax.set_ylim(40, 95)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.3, label='Random (50%)')
    ax.legend(fontsize=8)
    for i, v in enumerate(bal_acc):
        ax.text(i, v + 0.8, f'{v:.1f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    # AUC
    ax = axes[1]
    bars = ax.bar(range(len(methods)), auc, color=colors, edgecolor='#333', linewidth=0.8)
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_title('AUC Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=7.5, rotation=0)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Random (0.5)')
    ax.legend(fontsize=8)
    for i, v in enumerate(auc):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'comparison_chart.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved comparison_chart.png")


def fig_summary_table():
    """Publication-style summary table as figure."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.axis('off')
    ax.set_title('ResNet3D Multimodal Fusion — Test Set Results (CN vs AD)',
                 fontsize=14, fontweight='bold', pad=20)

    columns = ['Method', 'Fusion Type', 'Acc (%)', 'Bal Acc (%)', 'Sens (%)', 'Spec (%)', 'AUC']

    data = [
        ['MRI only (ResNet3D)',             '—',           '88.0', '82.1', '71.7', '92.6', '0.907'],
        ['Tabular only (XGBoost)',          '—',           '88.4', '85.8', '81.3', '90.3', '0.937'],
        ['Tabular only (MLP)',              '—',           '83.2', '85.2', '88.9', '81.6', '0.932'],
        ['ResNet3D + MLP concat',           'Early',       '90.7', '88.2', '83.8', '92.6', '0.952'],
        ['ResNet3D emb + Tab → XGBoost',    'Early',       '89.6', '85.3', '77.8', '92.8', '0.928'],
        ['ResNet3D + XGBoost (Average)',     'Late',        '90.4', '84.6', '74.2', '94.9', '0.946'],
        ['ResNet3D + XGBoost (Weighted)',    'Late',        '92.5', '88.1', '80.3', '95.9', '0.948'],
        ['ResNet3D + XGBoost (Stacking)',    'Late',        '91.4', '84.1', '71.2', '97.1', '0.949'],
        ['ResNet3D + MLP (Average)',         'Late',        '88.0', '86.3', '83.3', '89.3', '0.943'],
        ['ResNet3D + MLP (Weighted)',        'Late',        '88.9', '88.3', '87.4', '89.3', '0.947'],
        ['ResNet3D + MLP (Stacking)',        'Late',        '88.5', '78.4', '60.6', '96.2', '0.877'],
    ]

    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Header style
    for j in range(len(columns)):
        table[0, j].set_facecolor('#2C3E50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Row colors
    row_colors = {
        0: '#DCEEFB',  # MRI only
        1: '#FFE0B2',  # XGBoost only
        2: '#FFE0B2',  # MLP only
        3: '#E8D5F5',  # Early MLP
        4: '#E8D5F5',  # Early XGBoost
        5: '#F3E5F5',  # Late XGBoost
        6: '#E1BEE7',  # Late XGBoost weighted (best)
        7: '#F3E5F5',  # Late XGBoost stacking
        8: '#F3E5F5',  # Late MLP
        9: '#E1BEE7',  # Late MLP weighted
        10: '#F3E5F5', # Late MLP stacking
    }

    for i in range(len(data)):
        color = row_colors.get(i, 'white')
        for j in range(len(columns)):
            table[i+1, j].set_facecolor(color)

    # Highlight best results
    # Best bal acc: row 9 (MLP weighted 88.3%), col 3
    table[10, 3].set_text_props(fontweight='bold', color='#1B5E20')
    # Best AUC: row 7 (XGBoost stacking 0.949), col 6
    table[8, 6].set_text_props(fontweight='bold', color='#1B5E20')
    # Best acc: row 6 (XGBoost weighted 92.5%), col 2
    table[7, 2].set_text_props(fontweight='bold', color='#1B5E20')

    # Highlight early MLP (best AUC)
    table[4, 6].set_text_props(fontweight='bold', color='#1B5E20')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'summary_table.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved summary_table.png")


if __name__ == '__main__':
    print(f"Generating report in {OUTPUT_DIR}/")
    fig_early_fusion_mlp()
    fig_early_fusion_xgboost()
    fig_late_fusion()
    fig_comparison_table()
    fig_summary_table()
    print(f"\nDone! All figures saved in {OUTPUT_DIR}/")
