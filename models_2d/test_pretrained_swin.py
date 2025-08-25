#!/usr/bin/env python3
"""
Test pretrained Swin Transformer on Alzheimer's data WITHOUT training
Uses the fadhilelrizanda/swin_base_alzheimer_mri model directly with 270° rotation
"""

import os
import torch
import numpy as np
import nibabel as nib
import pandas as pd
import re
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
from glob import glob
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
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
    """Load ADNI data"""
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
    
    # Map to pretrained model classes (original labels)
    # Assuming pretrained model: 0=MCI, 1=Moderate, 2=CN, 3=AD
    label_map = {'CN': 2, 'MCI': 0, 'AD': 3}
    df['label'] = df['diagnosis'].map(label_map)
    
    return df

def create_patient_splits(df, test_size=0.2, random_state=42):
    """Create patient-level test split"""
    patients_by_diagnosis = {}
    for diagnosis in df['diagnosis'].unique():
        patients = df[df['diagnosis'] == diagnosis]['unique_patient_id'].unique()
        patients_by_diagnosis[diagnosis] = patients
        print(f"{diagnosis}: {len(patients)} patients")
    
    test_patients = []
    
    for diagnosis, patients in patients_by_diagnosis.items():
        _, test_pts = train_test_split(patients, test_size=test_size, random_state=random_state)
        test_patients.extend(test_pts)
    
    test_df = df[df['unique_patient_id'].isin(test_patients)].copy()
    
    print(f"\nTest split: {len(test_df)} files from {len(test_patients)} patients")
    for diagnosis in ['CN', 'MCI', 'AD']:
        count = len(test_df[test_df['diagnosis'] == diagnosis])
        print(f"  {diagnosis}: {count}")
    
    return test_df

def select_hippocampus_slice(volume):
    """Select hippocampus region slice"""
    depth = volume.shape[2]
    hippocampus_start = int(depth * 0.45)
    hippocampus_end = int(depth * 0.55)
    slice_idx = (hippocampus_start + hippocampus_end) // 2
    return slice_idx

def preprocess_volume(file_path, processor):
    """Load and preprocess a single volume"""
    try:
        # Load volume
        nii_img = nib.load(file_path)
        volume = nii_img.get_fdata()
        
        # Select hippocampus slice
        slice_idx = select_hippocampus_slice(volume)
        slice_data = volume[:, :, slice_idx]
        
        # Check intensity and try adjacent slices if needed
        if np.max(slice_data) < 0.01:
            for offset in [1, -1, 2, -2, 3, -3]:
                new_idx = slice_idx + offset
                if 0 <= new_idx < volume.shape[2]:
                    slice_data = volume[:, :, new_idx]
                    if np.max(slice_data) >= 0.01:
                        break
        
        # Apply 270-degree rotation for HuggingFace compatibility
        slice_data = np.rot90(slice_data, 3)  # 270 degrees clockwise
        
        # Normalize to 0-255
        slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
        slice_data = (slice_data * 255).astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(slice_data).convert('RGB')
        
        # Process with model processor
        processed = processor(pil_image, return_tensors="pt")
        
        return processed['pixel_values']
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def test_pretrained_model():
    """Test the pretrained model without training"""
    
    print("="*60)
    print("TESTING PRETRAINED SWIN TRANSFORMER ON ALZHEIMER'S DATA")
    print("="*60)
    print("Model: microsoft/swin-base-patch4-window7-224")
    print("Rotation: 270° clockwise for HuggingFace compatibility")
    print("Slice selection: Hippocampus region")
    
    # Load data
    data_dir = '../ADNIDenoise'
    df = load_adni_data(data_dir)
    test_df = create_patient_splits(df)
    
    # Load pretrained model
    model_name = "microsoft/swin-base-patch4-window7-224"
    print(f"\nLoading pretrained model: {model_name}")
    
    processor = AutoImageProcessor.from_pretrained(model_name)
    
    # Load model with original ImageNet classes first
    model = AutoModelForImageClassification.from_pretrained(model_name)
    
    print(f"Model loaded with {model.config.num_labels} classes")
    print("Note: Using ImageNet pretrained weights - no Alzheimer-specific training")
    
    # Set model to evaluation mode
    model.eval()
    
    # Make predictions
    predictions = []
    labels = []
    patient_ids = []
    
    print(f"\nProcessing {len(test_df)} test samples...")
    
    for idx, row in test_df.iterrows():
        file_path = row['file_path']
        label = row['label']
        patient_id = row['patient_id']
        
        # Preprocess volume
        pixel_values = preprocess_volume(file_path, processor)
        
        if pixel_values is not None:
            with torch.no_grad():
                outputs = model(pixel_values)
                logits = outputs.logits
                
                # Get predicted class
                predicted_class = torch.argmax(logits, dim=1).item()
                
                predictions.append(predicted_class)
                labels.append(label)
                patient_ids.append(patient_id)
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} samples")
    
    # Convert to numpy arrays
    y_true = np.array(labels)
    y_pred = np.array(predictions)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print("RESULTS - PRETRAINED MODEL (NO TRAINING)")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy:.1%}")
    print(f"Processed {len(predictions)} samples")
    
    # Map class indices to names for display
    idx_to_class = {
        0: 'MCI (Mild Demented)', 
        1: 'Moderate Demented', 
        2: 'CN (Non Demented)', 
        3: 'AD (Very Mild Demented)'
    }
    
    # Get unique labels for classification report
    unique_labels = sorted(list(set(y_true.tolist() + y_pred.tolist())))
    class_names = [idx_to_class.get(i, f'Class_{i}') for i in unique_labels]
    
    print(f"\nPrediction distribution:")
    unique_preds, pred_counts = np.unique(y_pred, return_counts=True)
    for pred_class, count in zip(unique_preds, pred_counts):
        print(f"  {idx_to_class.get(pred_class, f'Class_{pred_class}')}: {count}")
    
    print(f"\nTrue label distribution:")
    unique_true, true_counts = np.unique(y_true, return_counts=True)
    for true_class, count in zip(unique_true, true_counts):
        print(f"  {idx_to_class.get(true_class, f'Class_{true_class}')}: {count}")
    
    # Classification report
    print(f"\nClassification Report:")
    try:
        print(classification_report(y_true, y_pred, labels=unique_labels, target_names=class_names))
    except Exception as e:
        print(f"Error generating classification report: {e}")
        
        # Manual calculation
        for i, label in enumerate(unique_labels):
            label_name = idx_to_class.get(label, f'Class_{label}')
            correct = np.sum((y_true == label) & (y_pred == label))
            total_true = np.sum(y_true == label)
            total_pred = np.sum(y_pred == label)
            
            precision = correct / total_pred if total_pred > 0 else 0
            recall = correct / total_true if total_true > 0 else 0
            
            print(f"{label_name}: Precision={precision:.3f}, Recall={recall:.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Pretrained Swin - No Training\nTest Accuracy: {accuracy:.1%}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('pretrained_swin_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    results_df = pd.DataFrame({
        'patient_id': patient_ids,
        'true_label': y_true,
        'predicted_label': y_pred,
        'true_diagnosis': [idx_to_class.get(i, f'Class_{i}') for i in y_true],
        'predicted_diagnosis': [idx_to_class.get(i, f'Class_{i}') for i in y_pred]
    })
    
    results_df.to_csv('pretrained_swin_predictions.csv', index=False)
    print(f"\nResults saved to: pretrained_swin_predictions.csv")
    print(f"Confusion matrix saved to: pretrained_swin_confusion_matrix.png")

if __name__ == "__main__":
    test_pretrained_model()