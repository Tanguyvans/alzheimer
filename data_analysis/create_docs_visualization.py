#!/usr/bin/env python3
"""
Create Combined Documentation Visualization

Creates a single comprehensive visualization combining key insights from both
ADNI dataset analysis and diagnosis progression analysis for documentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

def main():
    """Create combined visualization for docs"""

    # Setup paths
    BASE_DIR = Path(__file__).parent.parent
    DXSUM_PATH = BASE_DIR / 'dxsum.csv'
    OUTPUT_PATH = Path(__file__).parent / 'combined_analysis.png'
    DOCS_OUTPUT_PATH = BASE_DIR / 'docs' / 'datasets' / 'combined_analysis.png'

    # Color scheme
    COLORS = {
        'CN': '#2ca02c',   # Green
        'MCI': '#1f77b4',  # Blue
        'AD': '#ff7f0e'    # Orange
    }

    DIAGNOSIS_MAP = {1: 'CN', 2: 'MCI', 3: 'AD'}

    print("=" * 80)
    print("CREATING COMBINED DOCUMENTATION VISUALIZATION")
    print("=" * 80)

    # Load diagnosis data
    if not DXSUM_PATH.exists():
        print(f"❌ Error: {DXSUM_PATH} not found")
        return 1

    print(f"✅ Loading data from {DXSUM_PATH}")
    df = pd.read_csv(DXSUM_PATH)
    df['EXAMDATE'] = pd.to_datetime(df['EXAMDATE'], errors='coerce')
    df = df.sort_values(['RID', 'EXAMDATE'])

    # Analyze patient trajectories
    patient_trajectories = {}
    for rid in df['RID'].unique():
        patient_data = df[df['RID'] == rid]
        diagnoses = patient_data['DIAGNOSIS'].dropna().tolist()

        if len(diagnoses) > 0:
            patient_trajectories[rid] = {
                'first_dx': diagnoses[0],
                'last_dx': diagnoses[-1],
                'changed': len(set(diagnoses)) > 1,
                'progression': len(set(diagnoses)) > 1 and diagnoses[-1] > diagnoses[0],
                'improvement': len(set(diagnoses)) > 1 and diagnoses[-1] < diagnoses[0],
                'num_visits': len(patient_data)
            }

    # Create baseline outcomes
    baseline_outcomes = {'CN': {}, 'MCI': {}, 'AD': {}}
    for baseline in ['CN', 'MCI', 'AD']:
        baseline_code = [k for k, v in DIAGNOSIS_MAP.items() if v == baseline][0]
        baseline_patients = [t for t in patient_trajectories.values()
                           if t['first_dx'] == baseline_code and len(df[df['RID'].isin([k for k, v in patient_trajectories.items() if v == t])]) > 1]

        total = sum(1 for t in patient_trajectories.values()
                   if t['first_dx'] == baseline_code)
        to_cn = sum(1 for t in patient_trajectories.values()
                   if t['first_dx'] == baseline_code and t['last_dx'] == 1)
        to_mci = sum(1 for t in patient_trajectories.values()
                    if t['first_dx'] == baseline_code and t['last_dx'] == 2)
        to_ad = sum(1 for t in patient_trajectories.values()
                   if t['first_dx'] == baseline_code and t['last_dx'] == 3)

        baseline_outcomes[baseline] = {
            'total': total,
            'to_CN': to_cn,
            'to_MCI': to_mci,
            'to_AD': to_ad
        }

    # Calculate trajectory statistics
    progressed = sum(1 for t in patient_trajectories.values() if t['progression'])
    improved = sum(1 for t in patient_trajectories.values() if t['improvement'])
    stable = sum(1 for t in patient_trajectories.values() if not t['changed'])
    fluctuated = len(patient_trajectories) - progressed - improved - stable
    total_patients = len(patient_trajectories)

    # Create simplified figure with 4 main panels
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Disease trajectory pie chart (TOP LEFT)
    ax1 = fig.add_subplot(gs[0, 0])
    trajectory_data = [stable, progressed, improved, fluctuated]
    trajectory_labels = [
        f'Stable\n{stable} ({stable/total_patients*100:.1f}%)',
        f'Progression\n{progressed} ({progressed/total_patients*100:.1f}%)',
        f'Improvement\n{improved} ({improved/total_patients*100:.1f}%)',
        f'Fluctuated\n{fluctuated} ({fluctuated/total_patients*100:.1f}%)'
    ]
    trajectory_colors = ['#95e1d3', '#ff6b6b', '#6bcf7f', '#ffd93d']

    wedges, texts, autotexts = ax1.pie(trajectory_data, labels=trajectory_labels,
                                        autopct='', colors=trajectory_colors,
                                        startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax1.set_title('Patient Trajectories (Baseline → Final)\n2,311 Patients',
                 fontweight='bold', fontsize=14, pad=20)

    # 2. Transition matrix heatmap (TOP RIGHT)
    ax2 = fig.add_subplot(gs[0, 1])
    transition_matrix = np.zeros((3, 3))
    for i, baseline in enumerate(['CN', 'MCI', 'AD']):
        data = baseline_outcomes[baseline]
        for j, outcome in enumerate(['CN', 'MCI', 'AD']):
            transition_matrix[i, j] = data[f'to_{outcome}']

    sns.heatmap(transition_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
               xticklabels=['CN', 'MCI', 'AD'], yticklabels=['CN', 'MCI', 'AD'],
               ax=ax2, cbar_kws={'label': 'Number of Patients'},
               linewidths=2, linecolor='black', annot_kws={'fontsize': 13, 'fontweight': 'bold'})
    ax2.set_xlabel('Final Diagnosis', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Baseline Diagnosis', fontsize=13, fontweight='bold')
    ax2.set_title('Diagnosis Transition Matrix', fontsize=14, fontweight='bold', pad=15)

    # 3. MCI Outcomes (BOTTOM LEFT - most important for research)
    ax3 = fig.add_subplot(gs[1, 0])
    mci_data = baseline_outcomes['MCI']
    outcomes = ['CN', 'MCI', 'AD']
    counts = [mci_data[f'to_{dx}'] for dx in outcomes]
    colors_list = [COLORS[dx] for dx in outcomes]
    bars = ax3.bar(outcomes, counts, color=colors_list, alpha=0.7,
                  edgecolor='black', linewidth=2)

    # Highlight stable
    bars[1].set_edgecolor('darkgreen')
    bars[1].set_linewidth(4)

    ax3.set_ylabel('Number of Patients', fontsize=13, fontweight='bold')
    ax3.set_title(f'MCI Baseline Outcomes (n={mci_data["total"]})\nKey Clinical Question',
                 fontweight='bold', fontsize=14)

    for i, v in enumerate(counts):
        percentage = (v / mci_data['total'] * 100) if mci_data['total'] > 0 else 0
        ax3.text(i, v + 15, f'{v}\n({percentage:.1f}%)',
              ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # 4. Key Statistics Panel (BOTTOM RIGHT)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    total_records = len(df)
    avg_visits = df.groupby('RID').size().mean()

    stats_text = "Key Statistics\n" + "="*45 + "\n\n"
    stats_text += f"📊 Dataset Overview:\n"
    stats_text += f"  • Total patients: {total_patients:,}\n"
    stats_text += f"  • Total records: {total_records:,}\n"
    stats_text += f"  • Avg visits/patient: {avg_visits:.1f}\n\n"

    stats_text += f"🔄 Diagnosis Stability:\n"
    stats_text += f"  • Stable: {stable} ({stable/total_patients*100:.1f}%)\n"
    stats_text += f"  • Changed: {progressed+improved+fluctuated}\n"
    stats_text += f"    ({(progressed+improved+fluctuated)/total_patients*100:.1f}%)\n\n"

    stats_text += f"📈 Key Progression Rates:\n"
    mci_to_ad_rate = (baseline_outcomes['MCI']['to_AD'] /
                     baseline_outcomes['MCI']['total'] * 100)
    mci_to_cn_rate = (baseline_outcomes['MCI']['to_CN'] /
                     baseline_outcomes['MCI']['total'] * 100)
    cn_to_mci_rate = (baseline_outcomes['CN']['to_MCI'] /
                     baseline_outcomes['CN']['total'] * 100)

    stats_text += f"  • MCI → AD: {mci_to_ad_rate:.1f}%\n"
    stats_text += f"  • MCI → CN: {mci_to_cn_rate:.1f}% (improvement)\n"
    stats_text += f"  • CN → MCI: {cn_to_mci_rate:.1f}%\n\n"

    stats_text += f"🎯 Most Common Transitions:\n"
    stats_text += f"  1. MCI → AD: {baseline_outcomes['MCI']['to_AD']} cases\n"
    stats_text += f"  2. CN → MCI: {baseline_outcomes['CN']['to_MCI']} cases\n"
    stats_text += f"  3. MCI → CN: {baseline_outcomes['MCI']['to_CN']} cases\n"

    ax4.text(0.05, 0.5, stats_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
            family='monospace', fontweight='bold')

    # Main title
    fig.suptitle('ADNI Dataset Analysis - Comprehensive Overview',
                fontsize=18, fontweight='bold', y=0.995)

    # Save to both locations
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved: {OUTPUT_PATH}")

    # Also save to docs folder
    DOCS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(DOCS_OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved: {DOCS_OUTPUT_PATH}")

    plt.close()

    print("\n" + "=" * 80)
    print("✅ COMBINED VISUALIZATION CREATED SUCCESSFULLY")
    print("=" * 80)

    return 0

if __name__ == "__main__":
    exit(main())
