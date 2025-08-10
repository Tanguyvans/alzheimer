#!/usr/bin/env python3
"""
Weighted XGBoost for Alzheimer's Prediction
Focus on improving AD detection with class balancing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess ADNI data"""
    
    # Load data
    df = pd.read_csv('/Users/tanguyvans/Desktop/umons/alzheimer/ADNIDenoise/adni_clinical_features.csv')
    print(f"Loaded ADNI clinical data: {df.shape}")
    
    print("\\nOriginal class distribution:")
    print(df['Group'].value_counts())
    
    # Convert to binary: AD = 1, CN+MCI = 0
    group_map = {"AD": 1, "CN": 0, "MCI": 0}
    df['Group'] = df['Group'].map(group_map)
    
    # Convert categorical to numeric
    df['PTGENDER'] = df['PTGENDER'].replace(['Female','Male'], [0,1])
    
    # Label encode other categoricals
    categorical_cols = ['PTETHCAT', 'PTRACCAT', 'APOE Genotype']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    print("\\nBinary class distribution:")
    print(df['Group'].value_counts())
    
    return df

def plot_comparison(results):
    """Plot model comparison"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    models = list(results.keys())
    
    # Accuracy
    accuracies = [results[model]['accuracy'] for model in models]
    ax1.bar(models, accuracies)
    ax1.set_title('Test Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # Precision
    precisions = [results[model]['precision'] for model in models]
    ax2.bar(models, precisions)
    ax2.set_title('Precision (AD Detection)')
    ax2.set_ylabel('Precision')
    ax2.set_ylim(0, 1)
    for i, v in enumerate(precisions):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # Recall
    recalls = [results[model]['recall'] for model in models]
    ax3.bar(models, recalls)
    ax3.set_title('Recall (AD Detection)')
    ax3.set_ylabel('Recall')
    ax3.set_ylim(0, 1)
    for i, v in enumerate(recalls):
        ax3.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # F1 Score
    f1_scores = [results[model]['f1'] for model in models]
    ax4.bar(models, f1_scores)
    ax4.set_title('F1 Score (AD Detection)')
    ax4.set_ylabel('F1 Score')
    ax4.set_ylim(0, 1)
    for i, v in enumerate(f1_scores):
        ax4.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('xgboost_weight_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, model_name, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    print(f"\\n{'='*50}")
    print(f"{model_name.upper()}")
    print(f"{'='*50}")
    
    # Basic metrics
    accuracy = metrics.accuracy_score(y_test, y_pred)
    
    # Handle zero division for precision/recall
    precision = metrics.precision_score(y_test, y_pred, zero_division=0)
    recall = metrics.recall_score(y_test, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (AD detection)")
    print(f"Recall:    {recall:.4f} (AD detection)")
    print(f"F1 Score:  {f1:.4f} (AD detection)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\\nConfusion Matrix:")
    print(f"{'':>12} Predicted")
    print(f"{'':>12} NonAD  AD")
    print(f"True NonAD {cm[0,0]:>6} {cm[0,1]:>4}")
    print(f"True AD    {cm[1,0]:>6} {cm[1,1]:>4}")
    
    # Additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\\nSpecificity: {specificity:.4f} (Non-AD detection)")
    print(f"Sensitivity: {recall:.4f} (AD detection)")
    
    # How many AD cases detected?
    total_ad = np.sum(y_test == 1)
    detected_ad = tp
    print(f"\\nAD Cases: {detected_ad}/{total_ad} detected ({detected_ad/total_ad*100:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'precision': precision, 
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'detected_ad': detected_ad,
        'total_ad': total_ad
    }

def main():
    """Main function"""
    
    print("Weighted XGBoost for Alzheimer's Prediction")
    print("=" * 50)
    
    # Load data
    df = load_and_preprocess_data()
    
    # Prepare features and target
    feature_cols = ["PTGENDER", "AGE", "PTEDUCAT", "PTETHCAT", "PTRACCAT", 
                   "APOE4", "MMSE", "APOE Genotype"]
    
    X = df[feature_cols].values
    y = df['Group'].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
    print(f"\\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Calculate class imbalance
    n_nonAD = np.sum(y_train == 0)
    n_AD = np.sum(y_train == 1)
    imbalance_ratio = n_nonAD / n_AD
    
    print(f"\\nClass Imbalance:")
    print(f"Non-AD: {n_nonAD} samples")
    print(f"AD: {n_AD} samples")  
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # Test different XGBoost weight configurations
    models = {
        'XGBoost Default': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'XGBoost Balanced': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss',
                                        scale_pos_weight=imbalance_ratio),
        'XGBoost 2x Weight': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss',
                                         scale_pos_weight=imbalance_ratio*2),
        'XGBoost 3x Weight': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss',
                                         scale_pos_weight=imbalance_ratio*3)
    }
    
    print(f"\\nTesting scale_pos_weight values:")
    for name, model in models.items():
        if hasattr(model, 'scale_pos_weight') and model.scale_pos_weight is not None:
            print(f"{name}: {model.scale_pos_weight:.2f}")
        else:
            print(f"{name}: default (1.0)")
    
    # Train and evaluate each model
    results = {}
    
    for name, model in models.items():
        print(f"\\nTraining {name}...")
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, name, X_test, y_test)
    
    # Summary comparison
    print(f"\\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    
    summary_df = pd.DataFrame(results).T
    summary_df = summary_df.round(4)
    
    # Sort by F1 score (best balance of precision and recall for AD)
    summary_df = summary_df.sort_values('f1', ascending=False)
    
    print(f"{'Model':<18} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AD Detected':<12}")
    print("-" * 80)
    for idx, (model_name, row) in enumerate(summary_df.iterrows()):
        detected_pct = row['detected_ad']/row['total_ad']*100
        print(f"{model_name:<18} {row['accuracy']:<10.4f} {row['precision']:<10.4f} "
              f"{row['recall']:<10.4f} {row['f1']:<10.4f} {row['detected_ad']:.0f}/{row['total_ad']:.0f} ({detected_pct:.1f}%)")
    
    # Plot comparison
    plot_comparison(results)
    
    # Save results
    summary_df.to_csv('weighted_xgboost_results.csv')
    
    # Best model recommendation
    best_model = summary_df.index[0]
    best_f1 = summary_df.iloc[0]['f1']
    best_detected = summary_df.iloc[0]['detected_ad']
    best_total = summary_df.iloc[0]['total_ad']
    
    print(f"\\n✅ Best performing model: {best_model}")
    print(f"   F1 Score: {best_f1:.4f}")
    print(f"   AD Detection: {best_detected:.0f}/{best_total:.0f} cases ({best_detected/best_total*100:.1f}%)")
    print(f"\\nResults saved to: weighted_xgboost_results.csv")

if __name__ == "__main__":
    main()