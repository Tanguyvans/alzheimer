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
    df = pd.read_csv('/Users/tanguyvans/Desktop/umons/alzheimer/ADNIDenoise/AD_CN_clinical_data.csv')
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
    
    # Train/validation/test split (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
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
    
    # Test different XGBoost weight configurations with more trees and early stopping
    models = {
        'XGBoost Default': XGBClassifier(n_estimators=300, random_state=42, eval_metric='logloss'),
        'XGBoost Balanced': XGBClassifier(n_estimators=300, random_state=42, eval_metric='logloss',
                                        scale_pos_weight=imbalance_ratio),
        'XGBoost 2x Weight': XGBClassifier(n_estimators=300, random_state=42, eval_metric='logloss',
                                         scale_pos_weight=imbalance_ratio*2),
        'XGBoost 3x Weight': XGBClassifier(n_estimators=300, random_state=42, eval_metric='logloss',
                                         scale_pos_weight=imbalance_ratio*3)
    }
    
    print(f"\\nTesting scale_pos_weight values:")
    for name, model in models.items():
        if hasattr(model, 'scale_pos_weight') and model.scale_pos_weight is not None:
            print(f"{name}: {model.scale_pos_weight:.2f}")
        else:
            print(f"{name}: default (1.0)")
    
    # Train and evaluate each model with validation set
    results = {}
    validation_results = {}
    best_model_obj = None
    best_f1 = 0
    
    for name, model in models.items():
        print(f"\\nTraining {name}...")
        
        # Train with early stopping using validation set
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Evaluate on validation set first
        val_results = evaluate_model(model, f"{name} (Validation)", X_val, y_val)
        validation_results[name] = val_results
        
        # Evaluate on test set
        test_results = evaluate_model(model, f"{name} (Test)", X_test, y_test)
        results[name] = test_results
        
        # Keep track of best model based on validation F1 score
        if val_results['f1'] > best_f1:
            best_f1 = val_results['f1']
            best_model_obj = model
    
    # Summary comparison - Validation Results
    print(f"\\n{'='*80}")
    print("VALIDATION SET RESULTS")
    print(f"{'='*80}")
    
    val_summary_df = pd.DataFrame(validation_results).T
    val_summary_df = val_summary_df.round(4)
    val_summary_df = val_summary_df.sort_values('f1', ascending=False)
    
    print(f"{'Model':<18} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AD Detected':<12}")
    print("-" * 80)
    for idx, (model_name, row) in enumerate(val_summary_df.iterrows()):
        detected_pct = row['detected_ad']/row['total_ad']*100
        print(f"{model_name:<18} {row['accuracy']:<10.4f} {row['precision']:<10.4f} "
              f"{row['recall']:<10.4f} {row['f1']:<10.4f} {row['detected_ad']:.0f}/{row['total_ad']:.0f} ({detected_pct:.1f}%)")
    
    # Summary comparison - Test Results
    print(f"\\n{'='*80}")
    print("TEST SET RESULTS")
    print(f"{'='*80}")
    
    test_summary_df = pd.DataFrame(results).T
    test_summary_df = test_summary_df.round(4)
    test_summary_df = test_summary_df.sort_values('f1', ascending=False)
    
    print(f"{'Model':<18} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AD Detected':<12}")
    print("-" * 80)
    for idx, (model_name, row) in enumerate(test_summary_df.iterrows()):
        detected_pct = row['detected_ad']/row['total_ad']*100
        print(f"{model_name:<18} {row['accuracy']:<10.4f} {row['precision']:<10.4f} "
              f"{row['recall']:<10.4f} {row['f1']:<10.4f} {row['detected_ad']:.0f}/{row['total_ad']:.0f} ({detected_pct:.1f}%)")
    
    # Combined comparison table
    print(f"\\n{'='*100}")
    print("VALIDATION vs TEST PERFORMANCE COMPARISON")
    print(f"{'='*100}")
    print(f"{'Model':<18} {'Val F1':<8} {'Test F1':<8} {'Difference':<12} {'Overfitting?':<12}")
    print("-" * 100)
    for model_name in val_summary_df.index:
        val_f1 = val_summary_df.loc[model_name, 'f1']
        test_f1 = test_summary_df.loc[model_name, 'f1']
        diff = val_f1 - test_f1
        overfitting = "Yes" if diff > 0.05 else "No"
        print(f"{model_name:<18} {val_f1:<8.4f} {test_f1:<8.4f} {diff:<12.4f} {overfitting:<12}")
    
    summary_df = test_summary_df  # Use test results for final summary
    
    # Plot comparison
    plot_comparison(results)
    
    # Save results
    test_summary_df.to_csv('weighted_xgboost_test_results.csv')
    val_summary_df.to_csv('weighted_xgboost_validation_results.csv')
    
    # Create combined results file
    combined_results = []
    for model_name in val_summary_df.index:
        val_row = val_summary_df.loc[model_name]
        test_row = test_summary_df.loc[model_name]
        combined_results.append({
            'model': model_name,
            'val_accuracy': val_row['accuracy'],
            'val_precision': val_row['precision'],
            'val_recall': val_row['recall'],
            'val_f1': val_row['f1'],
            'test_accuracy': test_row['accuracy'],
            'test_precision': test_row['precision'],
            'test_recall': test_row['recall'],
            'test_f1': test_row['f1'],
            'f1_difference': val_row['f1'] - test_row['f1'],
            'overfitting': 'Yes' if (val_row['f1'] - test_row['f1']) > 0.05 else 'No'
        })
    
    combined_df = pd.DataFrame(combined_results)
    combined_df.to_csv('weighted_xgboost_combined_results.csv', index=False)
    
    # Feature importance analysis
    if best_model_obj is not None:
        print(f"\\n{'='*60}")
        print("FEATURE IMPORTANCE ANALYSIS (Best Model)")
        print(f"{'='*60}")
        
        # Get feature importance
        feature_importance = best_model_obj.feature_importances_
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
        importance_df.to_csv('feature_importance.csv', index=False)
        print(f"\\nFeature importance saved to: feature_importance.csv")
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        top_10 = importance_df.head(10)
        plt.barh(top_10['feature'][::-1], top_10['importance'][::-1])
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Most Important Features (XGBoost)')
        plt.tight_layout()
        plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot confusion matrix for best model (on test set)
        y_pred_best = best_model_obj.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_best)
        
        print(f"\\nConfusion Matrix Analysis (Test Set):")
        print(f"Model selected based on validation F1 score: {best_f1:.4f}")
        
        plt.figure(figsize=(8, 6))
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['CN (Control)', 'AD (Alzheimer)'],
                   yticklabels=['CN (Control)', 'AD (Alzheimer)'])
        plt.title('Confusion Matrix - Best XGBoost Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add percentage annotations
        total = np.sum(cm)
        for i in range(2):
            for j in range(2):
                plt.text(j+0.5, i+0.7, f'({cm[i,j]/total*100:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig('confusion_matrix_best_model.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Best model recommendation
    best_model = summary_df.index[0]
    best_f1_score = summary_df.iloc[0]['f1']
    best_detected = summary_df.iloc[0]['detected_ad']
    best_total = summary_df.iloc[0]['total_ad']
    
    print(f"\\n✅ Best performing model: {best_model}")
    print(f"   F1 Score: {best_f1_score:.4f}")
    print(f"   AD Detection: {best_detected:.0f}/{best_total:.0f} cases ({best_detected/best_total*100:.1f}%)")
    print(f"\\nResults saved to: weighted_xgboost_results.csv")

if __name__ == "__main__":
    main()