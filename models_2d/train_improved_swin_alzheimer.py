#!/usr/bin/env python3
"""
Improved Swin Transformer training for Alzheimer's disease classification
Addresses class imbalance, learning rate, and data augmentation issues
"""

import os
import torch
import numpy as np
import nibabel as nib
import pandas as pd
import re
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import argparse
from glob import glob
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
warnings.filterwarnings('ignore')

def parse_adni_filename(filename):
    """Parse ADNI filename to extract patient information"""
    pattern = r'ADNI_(\d+)_S_(\d+)_MR_([^_]+)_([^_]+)_br_raw_(\d+)_(\d+)_S(\d+)_I(\d+)\.nii\.gz$'
    match = re.match(pattern, filename)
    
    if match:
        site_id = match.group(1)
        subject_id = match.group(2)
        patient_id = f"{site_id}_S_{subject_id}"
        return {'patient_id': patient_id, 'site_id': site_id, 'subject_id': subject_id}
    else:
        simple_match = re.match(r'ADNI_(\d+)_S_(\d+)_MR_.*?\.nii\.gz$', filename)
        if simple_match:
            site_id = simple_match.group(1)
            subject_id = simple_match.group(2)
            patient_id = f"{site_id}_S_{subject_id}"
            return {'patient_id': patient_id, 'site_id': site_id, 'subject_id': subject_id}
    return None

def load_adni_data(data_dir):
    """Load ADNI data and create DataFrame"""
    data_list = []
    
    for diagnosis in ['AD', 'MCI', 'CN']:
        diagnosis_dir = os.path.join(data_dir, diagnosis)
        if not os.path.exists(diagnosis_dir):
            continue
            
        nii_files = glob(os.path.join(diagnosis_dir, '*.nii.gz'))
        print(f"Found {len(nii_files)} files in {diagnosis}")
        
        for file_path in nii_files:
            filename = os.path.basename(file_path)
            parsed_info = parse_adni_filename(filename)
            
            if parsed_info:
                data_list.append({
                    'file_path': file_path,
                    'diagnosis': diagnosis,
                    'filename': filename,
                    'patient_id': parsed_info['patient_id'],
                    'unique_patient_id': parsed_info['patient_id']
                })
    
    df = pd.DataFrame(data_list)
    
    # Use standard 0, 1, 2 mapping for better training
    label_map = {'CN': 0, 'MCI': 1, 'AD': 2}
    df['label'] = df['diagnosis'].map(label_map)
    
    print(f"\nDataset summary:")
    print(f"Total files: {len(df)}")
    print(f"Unique patients: {df['unique_patient_id'].nunique()}")
    print("\nDiagnosis distribution:")
    print(df['diagnosis'].value_counts())
    
    return df

def create_patient_splits(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """Create train/val/test splits at patient level"""
    patients_by_diagnosis = {}
    for diagnosis in df['diagnosis'].unique():
        patients = df[df['diagnosis'] == diagnosis]['unique_patient_id'].unique()
        patients_by_diagnosis[diagnosis] = patients
        print(f"{diagnosis}: {len(patients)} patients")
    
    train_patients, val_patients, test_patients = [], [], []
    
    for diagnosis, patients in patients_by_diagnosis.items():
        train_val_patients, test_pts = train_test_split(
            patients, test_size=test_size, random_state=random_state
        )
        train_pts, val_pts = train_test_split(
            train_val_patients, test_size=val_size/(1-test_size), random_state=random_state
        )
        
        train_patients.extend(train_pts)
        val_patients.extend(val_pts)
        test_patients.extend(test_pts)
    
    train_df = df[df['unique_patient_id'].isin(train_patients)].copy()
    val_df = df[df['unique_patient_id'].isin(val_patients)].copy()
    test_df = df[df['unique_patient_id'].isin(test_patients)].copy()
    
    print(f"\nFinal split:")
    print(f"Train: {len(train_df)} files from {len(train_patients)} patients")
    print(f"Val: {len(val_df)} files from {len(val_patients)} patients")
    print(f"Test: {len(test_df)} files from {len(test_patients)} patients")
    
    return train_df, val_df, test_df

class ImprovedAlzheimerDataset(Dataset):
    """Improved dataset with better augmentation and preprocessing"""
    
    def __init__(self, file_df, processor, slice_selection='hippocampus', 
                 augment=False, min_slice_intensity=0.01):
        self.data = file_df.reset_index(drop=True)
        self.processor = processor
        self.slice_selection = slice_selection
        self.augment = augment
        self.min_slice_intensity = min_slice_intensity
        
    def __len__(self):
        return len(self.data)
    
    def select_slice(self, volume):
        """Select hippocampus region slice with fallback"""
        depth = volume.shape[2]
        
        if self.slice_selection == 'hippocampus':
            hippocampus_start = int(depth * 0.45)
            hippocampus_end = int(depth * 0.55)
            slice_idx = (hippocampus_start + hippocampus_end) // 2
        else:
            slice_idx = depth // 2
            
        return slice_idx
    
    def augment_image(self, pil_image):
        """Apply data augmentation"""
        if not self.augment:
            return pil_image
            
        # Random brightness adjustment
        if np.random.rand() > 0.5:
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(np.random.uniform(0.8, 1.2))
        
        # Random contrast adjustment
        if np.random.rand() > 0.5:
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(np.random.uniform(0.8, 1.2))
            
        return pil_image
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = row['file_path']
        label = int(row['label'])
        
        try:
            nii_img = nib.load(file_path)
            volume = nii_img.get_fdata()
            
            slice_idx = self.select_slice(volume)
            slice_data = volume[:, :, slice_idx]
            
            # Try adjacent slices if intensity too low
            if np.max(slice_data) < self.min_slice_intensity:
                for offset in [1, -1, 2, -2, 3, -3]:
                    new_idx = slice_idx + offset
                    if 0 <= new_idx < volume.shape[2]:
                        slice_data = volume[:, :, new_idx]
                        if np.max(slice_data) >= self.min_slice_intensity:
                            break
            
            # Apply 270-degree rotation
            slice_data = np.rot90(slice_data, 3)
            
            # Improved normalization with contrast enhancement
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
            
            # Apply histogram equalization-like enhancement
            slice_data = np.power(slice_data, 0.8)  # Gamma correction
            
            slice_data = (slice_data * 255).astype(np.uint8)
            
            # Convert to PIL and apply augmentation
            pil_image = Image.fromarray(slice_data).convert('RGB')
            pil_image = self.augment_image(pil_image)
            
            # Process with Swin processor
            processed = self.processor(pil_image, return_tensors="pt")
            
            return {
                'pixel_values': processed['pixel_values'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long)
            }
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            dummy_image = Image.new('RGB', (224, 224), color='black')
            processed = self.processor(dummy_image, return_tensors="pt")
            return {
                'pixel_values': processed['pixel_values'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long)
            }

class WeightedTrainer(Trainer):
    """Custom trainer with class weights"""
    
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../ADNIDenoise')
    parser.add_argument('--output_dir', type=str, default='./improved_swin_results')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)  # Reduced for stability
    parser.add_argument('--learning_rate', type=float, default=1e-5)  # Lower LR
    parser.add_argument('--slice_selection', type=str, default='hippocampus')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    
    args = parser.parse_args()
    
    print("="*60)
    print("IMPROVED SWIN TRANSFORMER ALZHEIMER'S CLASSIFICATION")
    print("="*60)
    print(f"Epochs: {args.num_epochs}, Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}, Augmentation: {args.augment}")
    
    # Load data
    df = load_adni_data(args.data_dir)
    train_df, val_df, test_df = create_patient_splits(df)
    
    # Calculate class weights for imbalanced dataset
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(train_df['label']), 
        y=train_df['label']
    )
    class_weights = torch.FloatTensor(class_weights)
    print(f"\nClass weights: {dict(zip(['CN', 'MCI', 'AD'], class_weights))}")
    
    # Load model
    base_model_name = "microsoft/swin-base-patch4-window7-224"
    print(f"\nLoading model: {base_model_name}")
    
    processor = AutoImageProcessor.from_pretrained(base_model_name)
    model = AutoModelForImageClassification.from_pretrained(
        base_model_name,
        num_labels=3,  # CN=0, MCI=1, AD=2
        ignore_mismatched_sizes=True
    )
    
    # Create datasets with augmentation for training
    train_dataset = ImprovedAlzheimerDataset(train_df, processor, args.slice_selection, augment=args.augment)
    val_dataset = ImprovedAlzheimerDataset(val_df, processor, args.slice_selection, augment=False)
    test_dataset = ImprovedAlzheimerDataset(test_df, processor, args.slice_selection, augment=False)
    
    # Training arguments with better settings
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,  # Warmup for stable training
        weight_decay=0.01,  # Regularization
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        fp16=True,  # Mixed precision for efficiency
    )
    
    # Use weighted trainer
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print(f"\nStarting training with improvements...")
    trainer.train()
    
    # Test evaluation
    print(f"\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test results: {test_results}")
    
    # Generate predictions
    predictions = trainer.predict(test_dataset)
    y_true = test_df['label'].values
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    # Classification report
    class_names = ['CN', 'MCI', 'AD']
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Improved Swin Transformer - Alzheimer\'s Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    accuracy = accuracy_score(y_true, y_pred)
    plt.figtext(0.02, 0.02, f'Test Accuracy: {accuracy:.1%}', fontsize=12, weight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'improved_confusion_matrix.png'), dpi=300)
    plt.show()
    
    # Save detailed results
    results_df = pd.DataFrame({
        'patient_id': test_df['patient_id'].values,
        'true_label': y_true,
        'predicted_label': y_pred,
        'true_diagnosis': [class_names[i] for i in y_true],
        'predicted_diagnosis': [class_names[i] for i in y_pred]
    })
    
    results_path = os.path.join(args.output_dir, 'improved_predictions.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"\n{'='*60}")
    print("IMPROVED TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Final accuracy: {accuracy:.1%}")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()