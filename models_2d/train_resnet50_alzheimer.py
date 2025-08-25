#!/usr/bin/env python3
"""
Fine-tune ResNet50 Alzheimer Model for 3-Class Classification
Uses pretrained evanrsl/resnet-Alzheimer model for AD/MCI/CN classification
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import re
from glob import glob
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Hugging Face imports
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from transformers import TrainingArguments, Trainer
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    HF_AVAILABLE = True
except ImportError as e:
    print(f"Hugging Face transformers not available: {e}")
    print("Install with: pip install transformers")
    HF_AVAILABLE = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def extract_patient_info_from_filename(filename):
    """Extract patient information from ADNI filename"""
    match = re.match(r'ADNI_(\d+)_S_(\d+)_MR_.*?_(\d{14}).*?_S(\d+)_I(\d+)\.nii\.gz$', filename)
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


def scan_adni_directory(base_path):
    """Scan ADNI directory and create a catalog of all MRI files"""
    print(f"Scanning ADNI directory: {base_path}")
    
    all_files = []
    
    for diagnosis in ['AD', 'MCI', 'CN']:
        diagnosis_path = os.path.join(base_path, diagnosis)
        
        if not os.path.exists(diagnosis_path):
            print(f"Warning: Directory {diagnosis_path} not found!")
            continue
            
        print(f"Scanning {diagnosis} directory...")
        
        nii_files = glob(os.path.join(diagnosis_path, '*.nii.gz'))
        print(f"Found {len(nii_files)} .nii.gz files in {diagnosis}")
        
        for file_path in nii_files:
            filename = os.path.basename(file_path)
            patient_info = extract_patient_info_from_filename(filename)
            
            if patient_info:
                file_record = {
                    'diagnosis': diagnosis,
                    'filename': filename,
                    'file_path': file_path,
                    'label': {'AD': 0, 'MCI': 1, 'CN': 2}[diagnosis],
                    **patient_info
                }
                all_files.append(file_record)
            else:
                print(f"Warning: Could not parse filename: {filename}")
    
    df = pd.DataFrame(all_files)
    print(f"Total files cataloged: {len(df)}")
    
    if len(df) > 0:
        print(f"\\nFiles by diagnosis:\\n{df['diagnosis'].value_counts()}")
        print(f"\\nUnique patients by diagnosis:\\n{df.groupby('diagnosis')['patient_id'].nunique()}")
    
    return df


def create_patient_split(df, test_size=0.2, val_size=0.1, random_state=42):
    """Create patient-level splits"""
    
    # Create unique patient identifiers
    df['unique_patient_id'] = df['diagnosis'] + '_' + df['patient_id']
    
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
    
    print(f"\\nFinal split:")
    print(f"Train: {len(train_df)} files from {len(train_patients)} patients")
    print(f"Val: {len(val_df)} files from {len(val_patients)} patients")
    print(f"Test: {len(test_df)} files from {len(test_patients)} patients")
    
    return train_df, val_df, test_df


class ResNetAlzheimerDataset(Dataset):
    """Dataset for loading brain slices for ResNet Alzheimer model"""
    
    def __init__(self, file_df, processor, slice_selection='hippocampus', augment=False, min_slice_intensity=0.01):
        """
        Args:
            file_df: DataFrame with file paths and labels
            processor: Hugging Face image processor
            slice_selection: 'middle', 'hippocampus', 'max_intensity', or 'random'
            augment: Whether to apply data augmentation
        """
        self.data = file_df.reset_index(drop=True)
        self.processor = processor
        self.slice_selection = slice_selection
        self.augment = augment
        self.min_slice_intensity = min_slice_intensity
        
    def __len__(self):
        return len(self.data)
    
    def select_slice(self, volume):
        """Select which slice to extract based on strategy"""
        depth = volume.shape[2]
        
        if self.slice_selection == 'middle':
            slice_idx = depth // 2
        elif self.slice_selection == 'hippocampus':
            # Hippocampus region - key for Alzheimer's detection
            hippocampus_start = int(depth * 0.45)
            hippocampus_end = int(depth * 0.55)
            slice_idx = (hippocampus_start + hippocampus_end) // 2
        elif self.slice_selection == 'max_intensity':
            # Find slice with maximum mean intensity
            max_intensity = 0
            slice_idx = depth // 2
            start = int(depth * 0.2)
            end = int(depth * 0.8)
            
            for idx in range(start, end):
                slice_intensity = np.mean(volume[:, :, idx])
                if slice_intensity > max_intensity:
                    max_intensity = slice_intensity
                    slice_idx = idx
        elif self.slice_selection == 'random':
            start = int(depth * 0.2)
            end = int(depth * 0.8)
            slice_idx = np.random.randint(start, end)
        else:
            slice_idx = depth // 2
            
        return slice_idx
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = row['file_path']
        label = int(row['label'])
        
        try:
            # Load the full 3D NIfTI volume
            nii_img = nib.load(file_path)
            volume = nii_img.get_fdata()
            
            # Select slice
            slice_idx = self.select_slice(volume)
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
            
            # Convert to PIL Image (required by processor)
            pil_image = Image.fromarray(slice_data).convert('RGB')
            
            # Process with image processor
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


class ResNetAlzheimerClassifier(nn.Module):
    """3-class classifier using pretrained ResNet50 Alzheimer model"""
    
    def __init__(self, model_name="evanrsl/resnet-Alzheimer", num_classes=3):
        super(ResNetAlzheimerClassifier, self).__init__()
        
        # Load pretrained ResNet50 Alzheimer model
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name, 
            num_labels=num_classes,
            ignore_mismatched_sizes=True,  # Allow resizing classifier head from 4 to 3 classes
        )
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.3)
        self.num_classes = num_classes
        
    def forward(self, pixel_values, labels=None):
        # Get outputs from base model
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        
        # Note: The model already includes loss calculation when labels are provided
        return outputs
    
    def freeze_backbone(self, freeze=True):
        """Freeze or unfreeze the ResNet backbone"""
        # ResNet model structure in transformers
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:  # Everything except final classifier
                param.requires_grad = not freeze


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy
    }


def train_resnet_model(model, train_dataset, val_dataset, output_dir="./resnet50_alzheimer_results", 
                      num_epochs=20, batch_size=16, learning_rate=1e-4):
    """Train the ResNet50 model using Hugging Face Trainer"""
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,  # Quick warmup for pretrained model
        weight_decay=0.1,  # Strong regularization
        logging_dir=f'{output_dir}/logs',
        logging_steps=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        learning_rate=learning_rate,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        gradient_accumulation_steps=1,
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
    )
    
    # Create optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 
         'lr': learning_rate * 0.1},  # Lower LR for backbone
        {'params': [p for n, p in model.named_parameters() if 'classifier' in n], 
         'lr': learning_rate}  # Higher LR for classifier
    ])
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)  # Custom optimizer
    )
    
    print("Starting training...")
    trainer.train()
    
    return trainer


def evaluate_model(model, test_dataset, class_names=['AD', 'MCI', 'CN']):
    """Evaluate model on test set"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values)
            predictions = torch.argmax(outputs.logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\\nTest Accuracy: {accuracy:.4f}")
    
    print("\\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('ResNet50 Alzheimer Model Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('resnet50_alzheimer_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy


def main():
    if not HF_AVAILABLE:
        print("Hugging Face transformers not available. Please install it first.")
        return
    
    parser = argparse.ArgumentParser(description='Fine-tune ResNet50 Alzheimer Model')
    parser.add_argument('--adni_dir', default='../ADNIDenoise', help='Path to ADNI directory')
    parser.add_argument('--slice_selection', default='hippocampus', 
                        choices=['middle', 'hippocampus', 'max_intensity', 'random'],
                        help='Slice selection strategy')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--freeze_backbone', action='store_true', 
                        help='Freeze ResNet backbone for first few epochs')
    
    args = parser.parse_args()
    
    # Load processor and model
    print("Loading pretrained ResNet50 Alzheimer model...")
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetAlzheimerClassifier("evanrsl/resnet-Alzheimer", num_classes=3).to(device)
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Scan ADNI directory
    print(f"\\nScanning ADNI directory: {args.adni_dir}")
    df = scan_adni_directory(args.adni_dir)
    
    if df.empty:
        print("Error: No files found in ADNI directory!")
        return
    
    # Create patient-level splits
    print("\\nCreating patient-level splits...")
    train_df, val_df, test_df = create_patient_split(df)
    
    print(f"\\nUsing slice selection strategy: {args.slice_selection}")
    
    # Create datasets
    train_dataset = ResNetAlzheimerDataset(train_df, processor, slice_selection=args.slice_selection, augment=True)
    val_dataset = ResNetAlzheimerDataset(val_df, processor, slice_selection=args.slice_selection, augment=False)
    test_dataset = ResNetAlzheimerDataset(test_df, processor, slice_selection=args.slice_selection, augment=False)
    
    print(f"\\nDataset sizes:")
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    
    # Optionally freeze backbone
    if args.freeze_backbone:
        print("\\nFreezing ResNet backbone for initial training...")
        model.freeze_backbone(True)
    
    # Train model
    trainer = train_resnet_model(
        model, train_dataset, val_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Evaluate on test set
    print("\\nEvaluating on test set...")
    test_accuracy = evaluate_model(model, test_dataset)
    
    # Save final model
    model.model.save_pretrained("./resnet50_alzheimer_3class_final")
    processor.save_pretrained("./resnet50_alzheimer_3class_final")
    
    print(f"\\nTraining completed!")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Model saved to: ./resnet50_alzheimer_3class_final")
    
    # Print model details
    print(f"\\nModel details:")
    print(f"  Base model: evanrsl/resnet-Alzheimer (ResNet50)")
    print(f"  Original training: 4 classes, adapted to 3 classes")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Slice selection: {args.slice_selection}")


if __name__ == "__main__":
    main()