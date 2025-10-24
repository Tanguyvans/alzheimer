#!/usr/bin/env python3
"""
XGBoost for Alzheimer's Disease Classification
Binary classification: AD vs CN using clinical features
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
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Define paths relative to this file
BASE_DIR = Path(__file__).parent.parent.parent  # alzheimer directory
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'tabular'
MODELS_DIR = OUTPUT_DIR / 'models'
PREDICTIONS_DIR = OUTPUT_DIR / 'predictions'
VISUALIZATIONS_DIR = OUTPUT_DIR / 'visualizations'

# Ensure output directories exist
for dir_path in [MODELS_DIR, PREDICTIONS_DIR, VISUALIZATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def load_and_preprocess_data():
    """Load and preprocess ADNI data"""

    # Load data using relative path
    data_path = DATA_DIR / 'AD_CN_clinical_data.csv'
    df = pd.read_csv(data_path)
    print(f"Loaded ADNI clinical data: {df.shape}")
    
    print("\\nOriginal class distribution:")
    print(df['Group'].value_counts())
    
    # Convert to binary: AD = 1, CN+MCI = 0
    group_map = {"AD": 1, "CN": 0, "MCI": 0}
    df['Group'] = df['Group'].map(group_map)
    
    # Handle missing values first
    print("\nMissing values before preprocessing:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # Fill missing values with median for numeric, mode for categorical
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
    
    # PTGENDER is already numeric (1.0 for male, other for female)
    # Convert to binary: 1 for male, 0 for female
    df['PTGENDER'] = (df['PTGENDER'] == 1.0).astype(int)
    
    # Label encode other categoricals if they exist
    categorical_cols = ['PTETHCAT', 'PTRACCAT']
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
    
    # Prepare features and target - EXCLUDING MMSCORE to see other important features
    available_feature_cols = ["PTGENDER", "PTDOBYY", "PTEDUCAT", "PTRACCAT", 
                             "VSWEIGHT", "VSHEIGHT", "TRAASCOR", "TRABSCOR",
                             "CATANIMSC", "CLOCKSCOR", "BNTTOTAL", "DSPANFOR", "DSPANBAC"]
    
    # available_feature_cols = ["PTGENDER", "PTDOBYY", "PTEDUCAT", "PTRACCAT", 
    #                         "VSWEIGHT", "VSHEIGHT", "MMSCORE", "TRAASCOR", "TRABSCOR",
    #                         "CATANIMSC", "CLOCKSCOR", "BNTTOTAL", "DSPANFOR", "DSPANBAC"]


    print("\\nEXCLUDING MMSCORE to analyze other predictive features")
    
    # Filter to only include columns that exist in the dataset
    feature_cols = [col for col in available_feature_cols if col in df.columns]
    print(f"\nUsing {len(feature_cols)} features: {feature_cols}")
    
    X = df[feature_cols].values
    y = df['Group'].values
    
    # Train/validation/test split (70/10/20)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp)
    
    print(f"\\nDataset splits:")
    print(f"Train set: {X_train.shape} ({len(y_train)/len(y)*100:.1f}%)")
    print(f"Validation set: {X_val.shape} ({len(y_val)/len(y)*100:.1f}%)")
    print(f"Test set: {X_test.shape} ({len(y_test)/len(y)*100:.1f}%)")
    
    # Check class distribution in each split
    print(f"\\nClass distribution:")
    print(f"Train: CN={np.sum(y_train==0)}, AD={np.sum(y_train==1)}")
    print(f"Val:   CN={np.sum(y_val==0)}, AD={np.sum(y_val==1)}")
    print(f"Test:  CN={np.sum(y_test==0)}, AD={np.sum(y_test==1)}")
    
    # Calculate class imbalance
    n_nonAD = np.sum(y_train == 0)
    n_AD = np.sum(y_train == 1)
    imbalance_ratio = n_nonAD / n_AD
    
    print(f"\\nClass Imbalance:")
    print(f"Non-AD: {n_nonAD} samples")
    print(f"AD: {n_AD} samples")  
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # Train XGBoost model
    print(f"\\nTraining XGBoost model...")
    
    # Create XGBoost classifier
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train with early stopping using validation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    print(f"Best iteration: {model.best_iteration}")
    print(f"Best validation score: {model.best_score:.4f}")
    
    # Evaluate on validation set
    print(f"\\n{'='*60}")
    print("VALIDATION SET PERFORMANCE")
    print(f"{'='*60}")
    val_results = evaluate_model(model, "XGBoost (Validation)", X_val, y_val)
    
    # Evaluate on test set
    print(f"\\n{'='*60}")
    print("TEST SET PERFORMANCE")
    print(f"{'='*60}")
    test_results = evaluate_model(model, "XGBoost (Test)", X_test, y_test)
    
    # Summary of Results
    print(f"\\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    # Compare validation vs test performance
    print(f"\\nValidation vs Test Performance:")
    print(f"{'Metric':<15} {'Validation':<12} {'Test':<12} {'Difference':<12}")
    print("-" * 50)
    print(f"{'Accuracy':<15} {val_results['accuracy']:<12.4f} {test_results['accuracy']:<12.4f} {val_results['accuracy'] - test_results['accuracy']:<12.4f}")
    print(f"{'Precision':<15} {val_results['precision']:<12.4f} {test_results['precision']:<12.4f} {val_results['precision'] - test_results['precision']:<12.4f}")
    print(f"{'Recall':<15} {val_results['recall']:<12.4f} {test_results['recall']:<12.4f} {val_results['recall'] - test_results['recall']:<12.4f}")
    print(f"{'F1 Score':<15} {val_results['f1']:<12.4f} {test_results['f1']:<12.4f} {val_results['f1'] - test_results['f1']:<12.4f}")
    
    # Save results
    results_df = pd.DataFrame({
        'validation': val_results,
        'test': test_results
    }).T
    results_df.to_csv(PREDICTIONS_DIR / 'xgboost_results.csv')

    # Save detailed results
    pd.DataFrame([val_results]).to_csv(PREDICTIONS_DIR / 'xgboost_validation_results.csv', index=False)
    pd.DataFrame([test_results]).to_csv(PREDICTIONS_DIR / 'xgboost_test_results.csv', index=False)
    
    # Feature importance analysis
    print(f"\\n{'='*60}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*60}")
    
    # Get feature importance
    feature_importance = model.feature_importances_
    feature_names = feature_cols
    
    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\\nTop 10 Most Important Features:")
    print("-" * 40)
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<12} : {row['importance']:.4f}")
    
    # Save feature importance
    importance_df.to_csv(PREDICTIONS_DIR / 'feature_importance.csv', index=False)
    print(f"\\nFeature importance saved to: feature_importance.csv")

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    top_10 = importance_df.head(10)
    plt.barh(top_10['feature'][::-1], top_10['importance'][::-1])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Most Important Features (XGBoost)')
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / 'feature_importance_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot confusion matrix (on test set)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\\nConfusion Matrix Analysis (Test Set):")
    
    plt.figure(figsize=(8, 6))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['CN (Control)', 'AD (Alzheimer)'],
               yticklabels=['CN (Control)', 'AD (Alzheimer)'])
    plt.title('Confusion Matrix - XGBoost Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add percentage annotations
    total = np.sum(cm)
    for i in range(2):
        for j in range(2):
            plt.text(j+0.5, i+0.7, f'({cm[i,j]/total*100:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / 'confusion_matrix_xgboost.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save the trained model
    model_path_pkl = MODELS_DIR / 'xgboost_model.pkl'
    with open(model_path_pkl, 'wb') as f:
        pickle.dump(model, f)
    print(f"\\n💾 Model saved to: {model_path_pkl}")

    # Also save using XGBoost's native format
    model_path_json = MODELS_DIR / 'xgboost_model.json'
    model.save_model(str(model_path_json))
    print(f"💾 Model also saved to: {model_path_json} (XGBoost native format)")

    print(f"\\n✅ XGBoost model training completed successfully")
    print(f"   Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"   F1 Score: {test_results['f1']:.4f}")
    print(f"   AD Detection: {test_results['detected_ad']:.0f}/{test_results['total_ad']:.0f} cases ({test_results['detected_ad']/test_results['total_ad']*100:.1f}%)")
    print(f"\\nResults saved to:")
    print(f"  - xgboost_model.pkl (pickled model)")
    print(f"  - xgboost_model.json (XGBoost native format)")
    print(f"  - xgboost_validation_results.csv")
    print(f"  - xgboost_test_results.csv")
    print(f"  - feature_importance.csv")
    print(f"  - confusion_matrix_xgboost.png")
    print(f"  - feature_importance_plot.png")

if __name__ == "__main__":
    main()