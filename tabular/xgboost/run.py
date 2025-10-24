#!/usr/bin/env python3
"""
Test trained XGBoost model on clinical_data_all_groups.csv
Load the trained model and evaluate on the full dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Define paths relative to this file
BASE_DIR = Path(__file__).parent.parent.parent  # Go up to alzheimer directory
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'tabular'
MODELS_DIR = OUTPUT_DIR / 'models'
PREDICTIONS_DIR = OUTPUT_DIR / 'predictions'
VISUALIZATIONS_DIR = OUTPUT_DIR / 'visualizations'
REPORTS_DIR = OUTPUT_DIR / 'reports'

# Ensure directories exist
for dir_path in [MODELS_DIR, PREDICTIONS_DIR, VISUALIZATIONS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def load_model(model_name='xgboost_model.pkl'):
    """Load the trained XGBoost model"""
    model_path = MODELS_DIR / model_name
    print(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully")
    return model

def prepare_data(csv_path):
    """Load and prepare clinical data"""
    print(f"\nLoading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {df.shape[0]} samples with {df.shape[1]} columns")

    print("\nClass distribution:")
    print(df['Group'].value_counts())

    # The trained model expects these 13 features:
    expected_features = ['PTGENDER', 'PTDOBYY', 'PTEDUCAT', 'PTRACCAT',
                        'VSWEIGHT', 'VSHEIGHT', 'TRAASCOR', 'TRABSCOR',
                        'CATANIMSC', 'CLOCKSCOR', 'BNTTOTAL', 'DSPANFOR', 'DSPANBAC']

    print(f"\nExpected features: {expected_features}")

    # Check which features are available
    available_features = [f for f in expected_features if f in df.columns]
    missing_features = [f for f in expected_features if f not in df.columns]

    print(f"\nAvailable features: {len(available_features)}/{len(expected_features)}")
    if missing_features:
        print(f"Missing features: {missing_features}")

    # Analyze missing data
    print("\nMissing data in available features:")
    for feat in available_features:
        missing_count = df[feat].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        if missing_pct > 0:
            print(f"  {feat:<15}: {missing_count:>5} ({missing_pct:>6.2f}%)")

    return df, available_features

def preprocess_features(df, feature_cols):
    """Preprocess features to match training format"""
    print("\nPreprocessing features...")

    df_clean = df.copy()

    # Handle categorical columns (PTRACCAT is categorical)
    if 'PTRACCAT' in feature_cols:
        if df_clean['PTRACCAT'].dtype == 'object' or df_clean['PTRACCAT'].isnull().any():
            print("  Encoding PTRACCAT...")
            # Fill missing with mode
            mode_val = df_clean['PTRACCAT'].mode()[0] if len(df_clean['PTRACCAT'].mode()) > 0 else 0
            df_clean['PTRACCAT'] = df_clean['PTRACCAT'].fillna(mode_val)

            # If it's already numeric, keep it, otherwise encode
            if df_clean['PTRACCAT'].dtype == 'object':
                le = LabelEncoder()
                df_clean['PTRACCAT'] = le.fit_transform(df_clean['PTRACCAT'].astype(str))

    # Fill missing values with median for numeric columns
    for col in feature_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            n_missing = df_clean[col].isnull().sum()
            print(f"  Filling {n_missing} missing values in {col} with median: {median_val}")
            df_clean[col] = df_clean[col].fillna(median_val)

    # Map Group to numeric (for evaluation)
    group_map = {"CN": 0, "MCI": 1, "AD": 2}
    df_clean['Group_Numeric'] = df_clean['Group'].map(group_map)

    print("✓ Preprocessing complete")

    return df_clean

def evaluate_predictions(y_true, y_pred, groups):
    """Evaluate model predictions"""

    # Binary classification: AD vs non-AD
    print("\n" + "="*60)
    print("BINARY CLASSIFICATION: AD vs NON-AD")
    print("="*60)

    # Convert to binary: 0/1 (CN/MCI) vs 2 (AD)
    y_true_binary = (y_true == 2).astype(int)
    y_pred_binary = (y_pred == 2).astype(int)

    acc_binary = accuracy_score(y_true_binary, y_pred_binary)
    print(f"Accuracy: {acc_binary:.4f}")

    cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
    print("\nConfusion Matrix:")
    print(f"{'':>12} Predicted")
    print(f"{'':>12} Non-AD  AD")
    print(f"True Non-AD {cm_binary[0,0]:>6} {cm_binary[0,1]:>4}")
    print(f"True AD     {cm_binary[1,0]:>6} {cm_binary[1,1]:>4}")

    # Calculate metrics
    tn, fp, fn, tp = cm_binary.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nPrecision (AD detection): {precision:.4f}")
    print(f"Recall (AD detection):    {recall:.4f}")
    print(f"Specificity (Non-AD):     {specificity:.4f}")
    print(f"F1 Score:                 {f1:.4f}")

    total_ad = np.sum(y_true_binary == 1)
    detected_ad = tp
    print(f"\nAD Cases: {detected_ad}/{total_ad} detected ({detected_ad/total_ad*100:.1f}%)")

    # Multi-class classification
    print("\n" + "="*60)
    print("MULTI-CLASS CLASSIFICATION: CN vs MCI vs AD")
    print("="*60)

    acc_multi = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {acc_multi:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['CN', 'MCI', 'AD'], zero_division=0))

    cm_multi = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"{'':>12} Predicted")
    print(f"{'':>12} CN    MCI   AD")
    print(f"True CN  {cm_multi[0,0]:>6} {cm_multi[0,1]:>5} {cm_multi[0,2]:>5}")
    print(f"True MCI {cm_multi[1,0]:>6} {cm_multi[1,1]:>5} {cm_multi[1,2]:>5}")
    print(f"True AD  {cm_multi[2,0]:>6} {cm_multi[2,1]:>5} {cm_multi[2,2]:>5}")

    # Per-class accuracy
    print("\nPer-class Detection Rate:")
    for i, class_name in enumerate(['CN', 'MCI', 'AD']):
        total = np.sum(y_true == i)
        detected = cm_multi[i, i]
        if total > 0:
            print(f"  {class_name}: {detected}/{total} ({detected/total*100:.1f}%)")

    return {
        'binary': {
            'accuracy': acc_binary,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'confusion_matrix': cm_binary
        },
        'multiclass': {
            'accuracy': acc_multi,
            'confusion_matrix': cm_multi
        }
    }

def main():
    """Main testing function - Focus on MCI predictions"""

    print("="*60)
    print("PREDICT MCI DIRECTION USING TRAINED CN vs AD MODEL")
    print("="*60)

    # Load the trained model (trained on CN=0 vs AD=1)
    model = load_model('xgboost_model.pkl')

    # Load and prepare data using relative path
    data_path = DATA_DIR / 'clinical_data_all_groups.csv'
    df, available_features = prepare_data(str(data_path))

    # Preprocess features
    df_clean = preprocess_features(df, available_features)

    # Extract features
    X = df_clean[available_features].values

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: CN={np.sum(df_clean['Group']=='CN')}, MCI={np.sum(df_clean['Group']=='MCI')}, AD={np.sum(df_clean['Group']=='AD')}")

    # Make predictions (0=CN, 1=AD)
    print("\nMaking predictions...")
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)

    df_clean['Prediction_Numeric'] = y_pred
    df_clean['Prediction'] = df_clean['Prediction_Numeric'].map({0: 'CN', 1: 'AD'})
    df_clean['Probability_CN'] = y_pred_proba[:, 0]
    df_clean['Probability_AD'] = y_pred_proba[:, 1]
    df_clean['Confidence'] = np.maximum(y_pred_proba[:, 0], y_pred_proba[:, 1])

    print("✓ Predictions complete")

    # Focus on MCI predictions
    print("\n" + "="*60)
    print("MCI PATIENT PREDICTIONS")
    print("="*60)

    mci_df = df_clean[df_clean['Group'] == 'MCI'].copy()
    cn_like = np.sum(mci_df['Prediction'] == 'CN')
    ad_like = np.sum(mci_df['Prediction'] == 'AD')

    print(f"\nTotal MCI patients: {len(mci_df)}")
    print(f"\nClassification:")
    print(f"  CN-like (stable):  {cn_like:4d} ({cn_like/len(mci_df)*100:.1f}%)")
    print(f"  AD-like (at risk): {ad_like:4d} ({ad_like/len(mci_df)*100:.1f}%)")

    print(f"\nAD Probability Statistics:")
    print(f"  Mean:   {mci_df['Probability_AD'].mean():.3f}")
    print(f"  Median: {mci_df['Probability_AD'].median():.3f}")
    print(f"  Std:    {mci_df['Probability_AD'].std():.3f}")
    print(f"  Min:    {mci_df['Probability_AD'].min():.3f}")
    print(f"  Max:    {mci_df['Probability_AD'].max():.3f}")

    # Also show CN and AD predictions for reference
    print("\n" + "="*60)
    print("PREDICTIONS FOR ALL GROUPS")
    print("="*60)

    for group in ['CN', 'MCI', 'AD']:
        group_df = df_clean[df_clean['Group'] == group]
        pred_cn = np.sum(group_df['Prediction'] == 'CN')
        pred_ad = np.sum(group_df['Prediction'] == 'AD')
        print(f"\nTrue {group} (n={len(group_df)}):")
        print(f"  Predicted as CN: {pred_cn:4d} ({pred_cn/len(group_df)*100:.1f}%)")
        print(f"  Predicted as AD: {pred_ad:4d} ({pred_ad/len(group_df)*100:.1f}%)")

    # Save all predictions
    output_cols = ['PTID', 'VISCODE', 'Group', 'Prediction', 'Probability_CN', 'Probability_AD', 'Confidence'] + available_features
    all_predictions_path = PREDICTIONS_DIR / 'predictions_all_groups.csv'
    df_clean[output_cols].to_csv(all_predictions_path, index=False)
    print(f"\n💾 All predictions saved to: {all_predictions_path}")

    # Save MCI only
    mci_predictions_path = PREDICTIONS_DIR / 'mci_predictions.csv'
    mci_df[output_cols].to_csv(mci_predictions_path, index=False)
    print(f"💾 MCI predictions saved to: {mci_predictions_path}")

    # Visualizations
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Plot 1: MCI predictions pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    mci_counts = mci_df['Prediction'].value_counts()
    colors = ['#90EE90', '#FFB6C6']
    ax1.pie(mci_counts.values, labels=[f'{l}\n({c})' for l, c in zip(mci_counts.index, mci_counts.values)],
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title(f'MCI Classification (n={len(mci_df)})')

    # Plot 2: AD probability distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(mci_df['Probability_AD'], bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax2.set_xlabel('Probability of AD')
    ax2.set_ylabel('Count')
    ax2.set_title('AD Probability Distribution (MCI)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Predictions by group
    ax3 = fig.add_subplot(gs[0, 2])
    groups = ['CN', 'MCI', 'AD']
    pred_cn = [np.sum(df_clean[df_clean['Group'] == g]['Prediction'] == 'CN') for g in groups]
    pred_ad = [np.sum(df_clean[df_clean['Group'] == g]['Prediction'] == 'AD') for g in groups]
    x = np.arange(len(groups))
    width = 0.35
    ax3.bar(x - width/2, pred_cn, width, label='Predicted CN', color='skyblue')
    ax3.bar(x + width/2, pred_ad, width, label='Predicted AD', color='salmon')
    ax3.set_xlabel('True Group')
    ax3.set_ylabel('Count')
    ax3.set_title('Predictions by True Group')
    ax3.set_xticks(x)
    ax3.set_xticklabels(groups)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: MCI risk stratification
    ax4 = fig.add_subplot(gs[1, 0])
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0.0-0.2\nVery CN-like', '0.2-0.4\nCN-like', '0.4-0.6\nUncertain', '0.6-0.8\nAD-like', '0.8-1.0\nVery AD-like']
    mci_df['Risk_Category'] = pd.cut(mci_df['Probability_AD'], bins=bins, labels=labels)
    risk_counts = mci_df['Risk_Category'].value_counts().sort_index()
    colors_risk = ['green', 'lightgreen', 'yellow', 'orange', 'red']
    ax4.bar(range(len(risk_counts)), risk_counts.values, color=colors_risk)
    ax4.set_xticks(range(len(risk_counts)))
    ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('Count')
    ax4.set_title('MCI Risk Stratification')
    ax4.grid(axis='y', alpha=0.3)

    # Plot 5: Confidence distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(mci_df['Confidence'], bins=20, color='teal', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Confidence')
    ax5.set_ylabel('Count')
    ax5.set_title('Prediction Confidence (MCI)')
    ax5.grid(axis='y', alpha=0.3)

    # Plot 6: Summary stats
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    summary = [
        ['Metric', 'Value'],
        ['Total MCI', f"{len(mci_df)}"],
        ['CN-like', f"{cn_like} ({cn_like/len(mci_df)*100:.1f}%)"],
        ['AD-like', f"{ad_like} ({ad_like/len(mci_df)*100:.1f}%)"],
        ['Mean AD Prob', f"{mci_df['Probability_AD'].mean():.3f}"],
        ['Mean Confidence', f"{mci_df['Confidence'].mean():.3f}"],
    ]
    table = ax6.table(cellText=summary, cellLoc='left', loc='center', colWidths=[0.4, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    viz_path = VISUALIZATIONS_DIR / 'mci_predictions_visualization.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"💾 Visualization saved to: {viz_path}")

    print("\n" + "="*60)
    print("✅ MCI PREDICTION COMPLETE!")
    print("="*60)
    print(f"\nOutputs organized in: {OUTPUT_DIR}")
    print(f"  Models:         {MODELS_DIR}")
    print(f"  Predictions:    {PREDICTIONS_DIR}")
    print(f"  Visualizations: {VISUALIZATIONS_DIR}")
    print(f"  Reports:        {REPORTS_DIR}")

if __name__ == "__main__":
    main()
