#!/usr/bin/env python3
"""
Optimized CNN+LSTM Sequence Model for Alzheimer's Disease Classification
Enhancements to achieve >80% validation accuracy
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
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class OptimizedCNN_LSTM_Classifier(nn.Module):
    """
    Optimized CNN + LSTM model with improvements:
    - ResNet-inspired CNN blocks
    - Attention mechanism
    - Better regularization
    - Larger capacity
    """
    
    def __init__(self, num_classes=3, cnn_features=512, lstm_hidden=256, num_layers=2, dropout=0.3):
        super(OptimizedCNN_LSTM_Classifier, self).__init__()
        
        # Enhanced CNN feature extractor with residual connections
        self.cnn_extractor = nn.Sequential(
            # Block 1 - Initial feature extraction
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Block 2 - Residual-inspired block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3 - Deeper feature extraction
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4 - High-level features
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Global Average Pooling instead of fixed size
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # Feature reduction
            nn.Linear(512, cnn_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Bidirectional LSTM with more capacity
        self.lstm = nn.LSTM(
            input_size=cnn_features,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism for LSTM outputs
        lstm_output_size = lstm_hidden * 2  # *2 for bidirectional
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_hidden),
            nn.Tanh(),
            nn.Linear(lstm_hidden, 1)
        )
        
        # Enhanced classifier with more capacity
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(lstm_output_size, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, c, h, w = x.shape
        
        # Reshape to process all slices together
        x = x.view(batch_size * seq_len, c, h, w)
        
        # Extract CNN features for each slice
        cnn_features = self.cnn_extractor(x)  # (batch_size * seq_len, cnn_features)
        
        # Reshape back to sequences
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        
        # Process sequence with LSTM
        lstm_out, _ = self.lstm(cnn_features)
        
        # Apply attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum of LSTM outputs
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)  # (batch_size, lstm_hidden * 2)
        
        # Final classification
        output = self.classifier(attended_output)
        
        return output


class EnhancedScanSequenceDataset(Dataset):
    """
    Enhanced dataset with data augmentation and better preprocessing
    """
    
    def __init__(self, csv_data, base_path, sequence_length=20, augment=False):
        self.base_path = base_path
        self.sequence_length = sequence_length
        self.augment = augment
        
        # Group slices by scan_id
        self.scans = []
        for scan_id in csv_data['scan_id'].unique():
            scan_data = csv_data[csv_data['scan_id'] == scan_id].copy()
            
            # Sort by slice index to maintain order
            scan_data = scan_data.sort_values('slice_index')
            
            if len(scan_data) >= sequence_length:
                # Take the first sequence_length slices
                scan_data = scan_data.head(sequence_length)
                self.scans.append(scan_data)
        
        print(f"Enhanced dataset created with {len(self.scans)} scan sequences")
        
    def __len__(self):
        return len(self.scans)
    
    def __getitem__(self, idx):
        scan_data = self.scans[idx]
        
        # Get label (should be same for all slices in the scan)
        label = int(scan_data.iloc[0]['label'])
        
        # Load all slices for this scan
        slice_sequence = []
        
        for _, row in scan_data.iterrows():
            slice_path = row['slice_path']
            
            # Handle both absolute and relative paths
            if not os.path.exists(slice_path):
                relative_path = row['relative_path']
                slice_path = os.path.join(self.base_path, relative_path)
            
            try:
                # Load NIfTI slice
                nii_img = nib.load(slice_path)
                data = nii_img.get_fdata()
                
                # Ensure we have RGB channels
                if len(data.shape) == 3 and data.shape[2] == 3:
                    data = torch.FloatTensor(data).permute(2, 0, 1)
                else:
                    if len(data.shape) == 2:
                        data = torch.FloatTensor(data).unsqueeze(0).repeat(3, 1, 1)
                    else:
                        data = torch.FloatTensor(data[:, :, 0]).unsqueeze(0).repeat(3, 1, 1)
                
                # Resize to 224x224
                data = F.interpolate(data.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
                data = data.squeeze(0)
                
                # Enhanced normalization
                # Histogram equalization-like normalization
                data_flat = data.flatten()
                data_sorted = torch.sort(data_flat)[0]
                # Use 1st and 99th percentile for robust normalization
                p1, p99 = data_sorted[int(0.01 * len(data_sorted))], data_sorted[int(0.99 * len(data_sorted))]
                data = torch.clamp((data - p1) / (p99 - p1 + 1e-8), 0, 1)
                
                # Data augmentation for training
                if self.augment and torch.rand(1) < 0.5:
                    # Random brightness adjustment
                    brightness_factor = 0.8 + 0.4 * torch.rand(1)  # 0.8 to 1.2
                    data = torch.clamp(data * brightness_factor, 0, 1)
                    
                    # Random contrast adjustment
                    if torch.rand(1) < 0.3:
                        contrast_factor = 0.8 + 0.4 * torch.rand(1)  # 0.8 to 1.2
                        data = torch.clamp((data - 0.5) * contrast_factor + 0.5, 0, 1)
                
                slice_sequence.append(data)
                
            except Exception as e:
                print(f"Error loading slice {slice_path}: {e}")
                # Use zero tensor as fallback
                slice_sequence.append(torch.zeros(3, 224, 224))
        
        # Stack slices into sequence tensor
        sequence_tensor = torch.stack(slice_sequence, dim=0)  # (seq_len, 3, 224, 224)
        
        return sequence_tensor, label


def create_scan_level_patient_split(df, test_size=0.2, val_size=0.1, random_state=42):
    """Create patient-level splits at the scan level"""
    
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
        
        print(f"{diagnosis} split: {len(train_pts)} train, {len(val_pts)} val, {len(test_pts)} test patients")
    
    # Create data splits based on patients
    train_df = df[df['unique_patient_id'].isin(train_patients)].copy()
    val_df = df[df['unique_patient_id'].isin(val_patients)].copy()
    test_df = df[df['unique_patient_id'].isin(test_patients)].copy()
    
    return train_df, val_df, test_df


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Enhanced training with better optimization"""
    
    # Class weights for imbalanced data
    class_weights = torch.tensor([1.0, 1.2, 0.8]).to(device)  # Slightly weight MCI more
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Use different learning rates for different parts
    cnn_params = [p for name, p in model.named_parameters() if 'cnn_extractor' in name]
    lstm_params = [p for name, p in model.named_parameters() if 'lstm' in name or 'attention' in name]
    classifier_params = [p for name, p in model.named_parameters() if 'classifier' in name]
    
    optimizer = optim.AdamW([
        {'params': cnn_params, 'lr': learning_rate * 0.5, 'weight_decay': 1e-4},
        {'params': lstm_params, 'lr': learning_rate, 'weight_decay': 1e-5},
        {'params': classifier_params, 'lr': learning_rate * 1.5, 'weight_decay': 1e-4}
    ])
    
    # Cosine annealing scheduler with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    patience = 0
    max_patience = 12
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for sequences, labels in pbar:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}')
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping with best model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience = 0
            print(f"✅ New best validation accuracy: {val_acc:.2f}%")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies


def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Classification report
    class_names = ['CN', 'MCI', 'AD']
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('CNN-LSTM Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add percentages as text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            percentage = cm[i, j] / cm[i].sum() * 100
            plt.text(j + 0.5, i + 0.7, f'{percentage:.1f}%', 
                    ha='center', va='center', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig('cnn_lstm_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'cnn_lstm_confusion_matrix.png'")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Optimized CNN+LSTM Sequence Model')
    parser.add_argument('--csv_path', default='../full_brain_slices_dataset/full_brain_slices.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=6)  # Smaller batch for larger model
    parser.add_argument('--learning_rate', type=float, default=0.0008)
    parser.add_argument('--sequence_length', type=int, default=20)
    
    args = parser.parse_args()
    
    # Load data
    print("Loading slice data from NEW comprehensive dataset...")
    df = pd.read_csv(args.csv_path)
    print(f"Dataset loaded: {len(df)} slices from {df['patient_id'].nunique()} patients")
    
    # Show dataset statistics
    print(f"\nDataset statistics:")
    print(f"Total slices: {len(df)}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print(f"Unique scans: {df['scan_id'].nunique()}")
    print(f"\nSlices by diagnosis:")
    print(df['diagnosis'].value_counts())
    print(f"\nPatients by diagnosis:")
    print(df.groupby('diagnosis')['patient_id'].nunique())
    
    # Create patient-level splits
    print("\nCreating patient-level splits...")
    train_df, val_df, test_df = create_scan_level_patient_split(df)
    
    # Create enhanced datasets with augmentation for training
    train_dataset = EnhancedScanSequenceDataset(train_df, os.path.dirname(args.csv_path), 
                                               args.sequence_length, augment=True)
    val_dataset = EnhancedScanSequenceDataset(val_df, os.path.dirname(args.csv_path), 
                                             args.sequence_length, augment=False)
    test_dataset = EnhancedScanSequenceDataset(test_df, os.path.dirname(args.csv_path), 
                                              args.sequence_length, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Create optimized model
    model = OptimizedCNN_LSTM_Classifier(
        num_classes=3, 
        cnn_features=512, 
        lstm_hidden=256, 
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    print(f"\nOptimized model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    print("\nStarting optimized CNN+LSTM training...")
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, num_epochs=args.epochs, learning_rate=args.learning_rate
    )
    
    # Save model
    torch.save(model.state_dict(), 'best_optimized_cnn_lstm.pth')
    print("Model saved as 'best_optimized_cnn_lstm.pth'")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy = evaluate_model(model, test_loader)
    
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
    print(f"Best Validation Accuracy: {max(val_accs):.2f}%")
    print(f"\nDataset improvement from old dataset:")
    print(f"  Old: 136 patients, ~91K slices")
    print(f"  New: {df['patient_id'].nunique()} patients, {len(df)} slices")
    print(f"  Improvement: {df['patient_id'].nunique()/136:.1f}x more patients!")


if __name__ == "__main__":
    main()