#!/usr/bin/env python3
"""
XGBoost Classifier for Alzheimer's Disease Prediction using Tabular Data
Based on clinical and demographic features from ADNI dataset
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class ADNIXGBoostClassifier:
    """XGBoost classifier for ADNI tabular data"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
        
    def preprocess_data(self, df, target_col='Group', is_training=True):
        """Preprocess the tabular data"""
        
        # Separate features and target
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            X = df
            y = None
            
        # Select numerical and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Remove ID columns and other non-predictive features
        id_cols = ['Image Data ID', 'Subject', 'MRI ID', 'Visit', 'Modality', 
                   'Description', 'Type', 'Acq Date', 'Format', 'Downloaded', 'nii_path']
        
        numeric_cols = [col for col in numeric_cols if col not in id_cols]
        categorical_cols = [col for col in categorical_cols if col not in id_cols + [target_col]]
        
        print(f"Numeric features: {len(numeric_cols)}")
        print(f"Categorical features: {len(categorical_cols)}")
        
        # Create feature matrix
        X_processed = pd.DataFrame()
        
        # Process numeric features
        if numeric_cols:
            X_numeric = X[numeric_cols]
            
            if is_training:
                X_numeric_imputed = self.imputer.fit_transform(X_numeric)
                X_numeric_scaled = self.scaler.fit_transform(X_numeric_imputed)
            else:
                X_numeric_imputed = self.imputer.transform(X_numeric)
                X_numeric_scaled = self.scaler.transform(X_numeric_imputed)
                
            X_numeric_df = pd.DataFrame(X_numeric_scaled, columns=numeric_cols, index=X.index)
            X_processed = pd.concat([X_processed, X_numeric_df], axis=1)
        
        # Process categorical features (one-hot encoding)
        if categorical_cols:
            # For Sex column specifically
            if 'Sex' in categorical_cols:
                X_processed['Sex_M'] = (X['Sex'] == 'M').astype(int)
                categorical_cols.remove('Sex')
            
            # One-hot encode other categorical variables if any
            if categorical_cols:
                X_categorical = pd.get_dummies(X[categorical_cols], prefix=categorical_cols)
                X_processed = pd.concat([X_processed, X_categorical], axis=1)
        
        # Store feature names
        if is_training:
            self.feature_names = X_processed.columns.tolist()
        
        # Encode target variable
        if y is not None and is_training:
            y_encoded = self.label_encoder.fit_transform(y)
        elif y is not None:
            y_encoded = self.label_encoder.transform(y)
        else:
            y_encoded = None
            
        return X_processed, y_encoded
    
    def train(self, X_train, y_train, X_val=None, y_val=None, optimize_hyperparams=True):
        """Train XGBoost model with optional hyperparameter optimization"""
        
        if optimize_hyperparams:
            print("Performing hyperparameter optimization...")
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'gamma': [0, 0.1, 0.2],
                'min_child_weight': [1, 3, 5]
            }
            
            # Base model
            n_classes = len(np.unique(y_train))
            eval_metric = 'logloss' if n_classes == 2 else 'mlogloss'
            
            xgb_base = xgb.XGBClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric=eval_metric
            )
            
            # Grid search with cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            grid_search = GridSearchCV(
                xgb_base, 
                param_grid, 
                cv=cv, 
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            self.model = grid_search.best_estimator_
            
        else:
            # Use default parameters
            # Determine if binary or multiclass
            n_classes = len(np.unique(y_train))
            eval_metric = 'logloss' if n_classes == 2 else 'mlogloss'
            
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric=eval_metric
            )
            
        # Train the model
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=50,
                verbose=True
            )
        else:
            self.model.fit(X_train, y_train)
            
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get classification report
        class_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=class_names)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC AUC for multi-class
        if len(class_names) == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        return results
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        importance_data = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance_data['feature'], importance_data['importance'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_data
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='xgboost_confusion_matrix.png'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        class_names = self.label_encoder.classes_
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - XGBoost Classifier')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath='xgboost_model.pkl'):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'imputer': self.imputer,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='xgboost_model.pkl'):
        """Load a saved model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.imputer = model_data['imputer']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filepath}")

def main():
    """Main training pipeline"""
    
    print("XGBoost Classifier for 3-Class Alzheimer's Disease Prediction (AD/CN/MCI)")
    print("=" * 70)
    
    # Load data
    # Try to load the ADNI clinical features file first
    csv_path = '/Users/tanguyvans/Desktop/umons/alzheimer/ADNIDenoise/adni_clinical_features.csv'
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded data with shape: {df.shape}")
    except FileNotFoundError:
        print(f"CSV file not found at {csv_path}")
        print("Please update the csv_path variable with the correct path to your data.")
        return
    
    # Display basic information
    print("\nData Overview:")
    print(df.info())
    
    if 'Group' in df.columns:
        print("\nClass distribution:")
        print(df['Group'].value_counts())
    
    # Initialize classifier
    clf = ADNIXGBoostClassifier(random_state=42)
    
    # Preprocess data
    X, y = clf.preprocess_data(df, target_col='Group')
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\nDataset splits:")
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    
    # Train model
    print("\nTraining XGBoost model...")
    clf.train(X_train, y_train, X_val, y_val, optimize_hyperparams=False)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = clf.evaluate(X_test, y_test)
    
    print(f"\nTest Accuracy: {results['accuracy']:.4f}")
    print(f"ROC AUC Score: {results['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Plot results
    clf.plot_confusion_matrix(y_test, results['predictions'])
    feature_importance = clf.plot_feature_importance(top_n=15)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(clf.model, X, y, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save model
    clf.save_model('alzheimer_xgboost_model.pkl')
    
    # Save results
    results_df = pd.DataFrame({
        'metric': ['accuracy', 'roc_auc', 'cv_mean', 'cv_std'],
        'value': [results['accuracy'], results['roc_auc'], cv_scores.mean(), cv_scores.std()]
    })
    results_df.to_csv('xgboost_results.csv', index=False)
    print("\nResults saved to xgboost_results.csv")

if __name__ == "__main__":
    main()