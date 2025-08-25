#!/usr/bin/env python3
"""
HuggingFace Swin Transformer training for Alzheimer's disease classification from 2D brain slices
Uses patient-level splits to avoid data leakage
Model: fadhilelrizanda/swin_base_alzheimer_mri
"""

import os
import torch
import numpy as np
import nibabel as nib
import pandas as pd
import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
import argparse
from glob import glob
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

def parse_adni_filename(filename):
    """Parse ADNI filename to extract patient information"""
    # Standard ADNI pattern: ADNI_XXX_S_XXXX_MR_YYYYYYY_ZZZZZZZ_br_raw_YYYYMMDD_HHMMSS_S######_I######.nii.gz
    pattern = r'ADNI_(\d+)_S_(\d+)_MR_([^_]+)_([^_]+)_br_raw_(\d+)_(\d+)_S(\d+)_I(\d+)\.nii\.gz$'
    match = re.match(pattern, filename)
    
    if match:
        site_id = match.group(1)
        subject_id = match.group(2)
        timestamp = match.group(3)
        sequence_id = match.group(4)
        image_id = match.group(5)
        patient_id = f"{site_id}_S_{subject_id}"
        return {
            'patient_id': patient_id,
            'site_id': site_id,
            'subject_id': subject_id,
            'timestamp': timestamp,
            'sequence_id': sequence_id,
            'image_id': image_id
        }
    else:
        # Try simpler pattern
        simple_match = re.match(r'ADNI_(\d+)_S_(\d+)_MR_.*?\.nii\.gz$', filename)
        if simple_match:
            site_id = simple_match.group(1)
            subject_id = simple_match.group(2)
            patient_id = f"{site_id}_S_{subject_id}"
            return {
                'patient_id': patient_id,
                'site_id': site_id,
                'subject_id': subject_id,
                'timestamp': 'unknown',
                'sequence_id': 'unknown',
                'image_id': 'unknown'
            }
    return None

def load_adni_data(data_dir):
    """Load ADNI data and create DataFrame"""
    data_list = []
    
    for diagnosis in ['AD', 'MCI', 'CN']:
        diagnosis_dir = os.path.join(data_dir, diagnosis)
        if not os.path.exists(diagnosis_dir):
            print(f"Warning: {diagnosis_dir} does not exist")
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
    
    # Keep 4-class structure matching pretrained model:
    # 0: mild demented (MCI), 1: moderate demented (empty), 2: non demented (CN), 3: very mild demented (AD)
    label_map = {'CN': 2, 'MCI': 0, 'AD': 3}  # Class 1 (moderate) will be empty
    df['label'] = df['diagnosis'].map(label_map)
    
    print(f"\nDataset summary:")
    print(f"Total files: {len(df)}")
    print(f"Unique patients: {df['unique_patient_id'].nunique()}")
    print("\nDiagnosis distribution:")
    print(df['diagnosis'].value_counts())
    print("\nLabel distribution:")
    print(df['label'].value_counts())
    
    return df

def create_patient_splits(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """Create train/val/test splits at patient level to avoid data leakage"""
    
    # Get unique patients per diagnosis
    patients_by_diagnosis = {}
    for diagnosis in df['diagnosis'].unique():
        patients = df[df['diagnosis'] == diagnosis]['unique_patient_id'].unique()
        patients_by_diagnosis[diagnosis] = patients
        print(f"{diagnosis}: {len(patients)} patients")
    
    train_patients = []
    val_patients = []
    test_patients = []
    
    # Split patients by diagnosis
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
    
    # Create data splits
    train_df = df[df['unique_patient_id'].isin(train_patients)].copy()
    val_df = df[df['unique_patient_id'].isin(val_patients)].copy()
    test_df = df[df['unique_patient_id'].isin(test_patients)].copy()
    
    print(f"\nFinal split:")
    print(f"Train: {len(train_df)} files from {len(train_patients)} patients")
    print(f"Val: {len(val_df)} files from {len(val_patients)} patients")
    print(f"Test: {len(test_df)} files from {len(test_patients)} patients")
    
    return train_df, val_df, test_df

class AlzheimerSwinDataset(Dataset):
    """Dataset for loading brain slices for Swin Transformer model"""
    
    def __init__(self, file_df, processor, slice_selection='hippocampus', num_slices=1, augment=False, min_slice_intensity=0.01):
        """
        Args:
            file_df: DataFrame with file paths and labels
            processor: Hugging Face AutoImageProcessor
            slice_selection: 'middle', 'hippocampus', 'max_intensity', or 'random'
            num_slices: Number of slices to extract per volume (1 or 3)
            augment: Whether to apply data augmentation
        """
        self.data = file_df.reset_index(drop=True)
        self.processor = processor
        self.slice_selection = slice_selection
        self.num_slices = num_slices
        self.augment = augment
        self.min_slice_intensity = min_slice_intensity
        
    def __len__(self):
        # If using multi-slices, multiply dataset size
        return len(self.data) * self.num_slices
    
    def select_slices(self, volume):
        """Select slices based on strategy and number requested"""
        depth = volume.shape[2]
        
        if self.num_slices == 1:
            # Single slice selection
            if self.slice_selection == 'hippocampus':
                hippocampus_start = int(depth * 0.45)
                hippocampus_end = int(depth * 0.55)
                slice_idx = (hippocampus_start + hippocampus_end) // 2
                return [slice_idx]
            else:
                slice_idx = depth // 2
                return [slice_idx]
                
        elif self.num_slices == 3:
            # Multi-slice selection
            if self.slice_selection == 'hippocampus':
                # Hippocampus region with 3 slices
                hippocampus_start = int(depth * 0.42)
                hippocampus_end = int(depth * 0.58)
                middle = (hippocampus_start + hippocampus_end) // 2
                spacing = (hippocampus_end - hippocampus_start) // 4
                return [hippocampus_start + spacing, middle, hippocampus_end - spacing]
            else:
                # Middle region with 3 slices
                middle = depth // 2
                spacing = depth // 10
                return [middle - spacing, middle, middle + spacing]
        
        return [depth // 2]
    
    def __getitem__(self, idx):
        # Calculate which file and which slice within that file
        file_idx = idx // self.num_slices
        slice_idx_in_set = idx % self.num_slices
        
        row = self.data.iloc[file_idx]
        file_path = row['file_path']
        label = int(row['label'])
        
        try:
            # Load the full 3D NIfTI volume
            nii_img = nib.load(file_path)
            volume = nii_img.get_fdata()
            
            # Select slices
            slice_indices = self.select_slices(volume)
            slice_idx = slice_indices[slice_idx_in_set] if slice_idx_in_set < len(slice_indices) else slice_indices[0]
            slice_data = volume[:, :, slice_idx]
            
            # Check if slice has enough intensity
            if np.max(slice_data) < self.min_slice_intensity:
                # Try adjacent slices
                for offset in [1, -1, 2, -2, 3, -3]:
                    new_idx = slice_idx + offset
                    if 0 <= new_idx < volume.shape[2]:
                        slice_data = volume[:, :, new_idx]
                        if np.max(slice_data) >= self.min_slice_intensity:
                            break
            
            # Apply 270-degree rotation for HuggingFace compatibility
            slice_data = np.rot90(slice_data, 3)  # 270 degrees clockwise
            
            # Normalize to 0-255 and convert to RGB
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
            slice_data = (slice_data * 255).astype(np.uint8)
            
            # Convert to PIL Image (required by Swin processor)
            pil_image = Image.fromarray(slice_data).convert('RGB')
            
            # Process with Swin processor
            processed = self.processor(pil_image, return_tensors="pt")
            
            return {
                'pixel_values': processed['pixel_values'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long)
            }
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return dummy data
            dummy_image = Image.new('RGB', (224, 224), color='black')
            processed = self.processor(dummy_image, return_tensors="pt")
            return {
                'pixel_values': processed['pixel_values'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long)
            }

class WeightedTrainer(Trainer):
    """Custom trainer with class weights to handle imbalanced data"""
    
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.class_weights is not None:
            # Ensure class weights are on the same device as the model
            device_weights = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=device_weights)
        else:
            loss_fct = nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
    }

def main():
    parser = argparse.ArgumentParser(description='Train Swin Transformer on Alzheimer\'s MRI data')
    parser.add_argument('--data_dir', type=str, default='../ADNIDenoise', 
                      help='Path to ADNI data directory')
    parser.add_argument('--output_dir', type=str, default='./swin_alzheimer_results',
                      help='Output directory for model and results')
    parser.add_argument('--num_epochs', type=int, default=2,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Training batch size')
    parser.add_argument('--slice_selection', type=str, default='hippocampus',
                      choices=['middle', 'hippocampus', 'max_intensity', 'random'],
                      help='Slice selection strategy')
    parser.add_argument('--num_slices', type=int, default=1, choices=[1, 3],
                      help='Number of slices per volume (1 or 3)')
    parser.add_argument('--use_class_weights', action='store_true',
                      help='Use class weights to handle imbalanced data')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                      help='Learning rate')
    
    args = parser.parse_args()
    
    print("="*60)
    print("IMPROVED SWIN TRANSFORMER ALZHEIMER'S CLASSIFICATION")
    print("="*60)
    print(f"Model: microsoft/swin-base-patch4-window7-224")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Slice selection: {args.slice_selection}")
    print(f"Number of slices per volume: {args.num_slices}")
    print(f"Use class weights: {args.use_class_weights}")
    print(f"Learning rate: {args.learning_rate}")
    
    if args.num_slices > 1:
        print(f"Note: Using {args.num_slices} slices will multiply dataset size by {args.num_slices}x")
    
    # Force CPU training for stability (avoid MPS issues)
    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("Forcing CPU training for stability")
    
    # Load data
    df = load_adni_data(args.data_dir)
    train_df, val_df, test_df = create_patient_splits(df)
    
    # Calculate class weights if requested
    class_weights = None
    if args.use_class_weights:
        # Get class counts in training data
        train_labels = train_df['label'].values
        unique_classes = np.array([0, 2, 3])  # Only classes we have data for
        
        # Compute weights for existing classes
        existing_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=train_labels
        )
        
        # Create 4-class weight tensor (with 0 weight for class 1)
        weights = np.zeros(4)
        weights[0] = existing_weights[0]  # MCI weight
        weights[1] = 0.0  # No weight for moderate (class 1)
        weights[2] = existing_weights[1]  # CN weight  
        weights[3] = existing_weights[2]  # AD weight
        
        class_weights = torch.FloatTensor(weights)
        print(f"\nClass weights computed:")
        print(f"  MCI (0): {weights[0]:.3f}")
        print(f"  Moderate (1): {weights[1]:.3f} (no training data)")
        print(f"  CN (2): {weights[2]:.3f}")
        print(f"  AD (3): {weights[3]:.3f}")
    
    # Load model and processor
    base_model_name = "microsoft/swin-base-patch4-window7-224"
    print(f"\nLoading base Swin model: {base_model_name}")
    print("Using 4-class structure matching pretrained model:")
    print("  0: Mild Demented (MCI)")
    print("  1: Moderate Demented (empty - no training data)")  
    print("  2: Non Demented (CN)")
    print("  3: Very Mild Demented (AD)")
    
    processor = AutoImageProcessor.from_pretrained(base_model_name)
    model = AutoModelForImageClassification.from_pretrained(
        base_model_name,
        num_labels=4,  # 4-class structure
        ignore_mismatched_sizes=True
    )
    
    # Create datasets with multi-slice support
    train_dataset = AlzheimerSwinDataset(train_df, processor, args.slice_selection, args.num_slices)
    val_dataset = AlzheimerSwinDataset(val_df, processor, args.slice_selection, args.num_slices)
    test_dataset = AlzheimerSwinDataset(test_df, processor, args.slice_selection, args.num_slices)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} slices ({len(train_df)} volumes)")
    print(f"  Val: {len(val_dataset)} slices ({len(val_df)} volumes)")
    print(f"  Test: {len(test_dataset)} slices ({len(test_df)} volumes)")
    
    # Setup training arguments with MPS stability fixes
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,  # Reduced logging frequency
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,  # Add warmup for stability
        weight_decay=0.01,  # Add regularization
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_num_workers=0,  # Set to 0 for MPS compatibility
        use_mps_device=False,  # Force CPU/CUDA to avoid MPS issues
    )
    
    # Initialize trainer (weighted if class weights provided)
    if class_weights is not None:
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        print("Using WeightedTrainer with class weights")
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        print("Using standard Trainer (no class weights)")
    
    print(f"\nStarting training...")
    print(f"Training on {len(train_dataset)} samples")
    print(f"Validating on {len(val_dataset)} samples")
    
    # Train the model
    trainer.train()
    
    # Evaluate on test set
    print(f"\nEvaluating on test set ({len(test_dataset)} samples)...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test results: {test_results}")
    
    # Generate predictions for detailed analysis
    predictions = trainer.predict(test_dataset)
    y_true = test_df['label'].values
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    # Classification report - 4-class mapping
    idx_to_class = {
        0: 'MCI (Mild Demented)', 
        1: 'Moderate Demented (Empty)', 
        2: 'CN (Non Demented)', 
        3: 'AD (Very Mild Demented)'
    }
    
    # Get unique labels present in predictions (could include class 1 now)
    unique_labels = sorted(list(set(y_true.tolist() + y_pred.tolist())))
    class_names = [idx_to_class.get(i, f'Class_{i}') for i in unique_labels]
    
    # Count predictions for class 1 (moderate) if any
    class_1_predictions = np.sum(y_pred == 1)
    if class_1_predictions > 0:
        print(f"\nNote: {class_1_predictions} samples predicted as class 1 (Moderate Demented)")
        print("This is expected since we're using the pretrained 4-class structure.")
    
    print(f"\nClassification Report:")
    print(f"Class mapping: {idx_to_class}")
    print(classification_report(y_true, y_pred, labels=unique_labels, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Swin Transformer - Alzheimer\'s Classification\nConfusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add accuracy to plot
    accuracy = accuracy_score(y_true, y_pred)
    plt.figtext(0.02, 0.02, f'Test Accuracy: {accuracy:.1%}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    results_df = pd.DataFrame({
        'patient_id': test_df['patient_id'].values,
        'true_label': y_true,
        'predicted_label': y_pred,
        'true_diagnosis': [idx_to_class.get(i, f'Class_{i}') for i in y_true],
        'predicted_diagnosis': [idx_to_class.get(i, f'Class_{i}') for i in y_pred]
    })
    
    results_path = os.path.join(args.output_dir, 'test_predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nTest predictions saved to: {results_path}")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Final test accuracy: {accuracy:.1%}")
    print(f"Model saved in: {args.output_dir}")
    print(f"Confusion matrix: {os.path.join(args.output_dir, 'confusion_matrix.png')}")

if __name__ == "__main__":
    main()