#!/usr/bin/env python3
"""
Patient-Level Split Dual CNN Training for Whole Brain Slices
Ensures no patient appears in both train and test sets to prevent data leakage
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

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class DualCNN(nn.Module):
    """
    Dual CNN architecture following the GitHub repo approach
    CNN1: Fine-grained local features (small kernels)
    CNN2: Broader patterns (larger kernels)
    """
    
    def __init__(self, num_classes=3, dropout=0.5):
        super(DualCNN, self).__init__()
        
        # CNN1: Fine-grained local features (small kernels)
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # CNN2: Broader patterns (larger kernels)
        self.cnn2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Calculate feature map size after conv layers (assuming 224x224 input)
        # After 3 max pools (2x2): 224 -> 112 -> 56 -> 28
        self.feature_size = 128 * 28 * 28  # 128 channels, 28x28 spatial
        
        # Fusion and classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_size * 2, 512),  # *2 because we concat CNN1 and CNN2
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Forward through both CNNs
        features1 = self.cnn1(x)
        features2 = self.cnn2(x)
        
        # Flatten features
        features1 = features1.view(features1.size(0), -1)
        features2 = features2.view(features2.size(0), -1)
        
        # Concatenate features from both CNNs
        combined_features = torch.cat([features1, features2], dim=1)
        
        # Classification
        output = self.classifier(combined_features)
        return output


class BrainSliceDataset(Dataset):
    """Dataset for brain slices with patient-level splitting"""
    
    def __init__(self, csv_data, base_path):
        self.data = csv_data.reset_index(drop=True)
        self.base_path = base_path
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        slice_path = row['slice_path']
        label = int(row['label'])
        
        try:
            # Load NIfTI slice
            nii_img = nib.load(slice_path)
            data = nii_img.get_fdata()
            
            # Ensure we have RGB channels (should be 3D: H x W x 3)
            if len(data.shape) == 3 and data.shape[2] == 3:
                # Convert to tensor and rearrange to (C, H, W)
                data = torch.FloatTensor(data).permute(2, 0, 1)
            else:
                # Handle grayscale or other formats
                if len(data.shape) == 2:
                    # Grayscale: repeat to 3 channels
                    data = torch.FloatTensor(data).unsqueeze(0).repeat(3, 1, 1)
                else:
                    # Take first channel and repeat
                    data = torch.FloatTensor(data[:, :, 0]).unsqueeze(0).repeat(3, 1, 1)
            
            # Resize to 224x224 for CNN
            data = F.interpolate(data.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
            data = data.squeeze(0)
            
            # Normalize to [0, 1]
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            
            return data, label
            
        except Exception as e:
            print(f"Error loading {slice_path}: {e}")
            # Return zero tensor as fallback
            return torch.zeros(3, 224, 224), label


def create_patient_split(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Create patient-level train/val/test splits to prevent data leakage
    Ensures same patient doesn't appear in multiple splits
    """
    
    # Get unique patients per diagnosis
    patients_by_diagnosis = {}
    for diagnosis in df['diagnosis'].unique():
        patients = df[df['diagnosis'] == diagnosis]['patient_id'].unique()
        patients_by_diagnosis[diagnosis] = patients
        print(f"{diagnosis}: {len(patients)} patients")
    
    train_patients = []
    val_patients = []
    test_patients = []
    
    # Split patients by diagnosis to maintain class balance
    for diagnosis, patients in patients_by_diagnosis.items():
        # First split: train+val vs test
        train_val_patients, test_pts = train_test_split(
            patients, test_size=test_size, random_state=random_state
        )
        
        # Second split: train vs val
        train_pts, val_pts = train_test_split(
            train_val_patients, test_size=val_size/(1-test_size), random_state=random_state
        )
        
        train_patients.extend(train_pts)
        val_patients.extend(val_pts)
        test_patients.extend(test_pts)
        
        print(f"{diagnosis} split: {len(train_pts)} train, {len(val_pts)} val, {len(test_pts)} test patients")
    
    # Create data splits based on patient assignments
    train_df = df[df['patient_id'].isin(train_patients)].copy()
    val_df = df[df['patient_id'].isin(val_patients)].copy()
    test_df = df[df['patient_id'].isin(test_patients)].copy()
    
    print(f"\nFinal split:")
    print(f"Train: {len(train_df)} slices from {len(train_patients)} patients")
    print(f"Val: {len(val_df)} slices from {len(val_patients)} patients") 
    print(f"Test: {len(test_df)} slices from {len(test_patients)} patients")
    
    # Check class distribution
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n{split_name} class distribution:")
        class_counts = split_df['diagnosis'].value_counts()
        for diagnosis, count in class_counts.items():
            print(f"  {diagnosis}: {count} slices")
    
    return train_df, val_df, test_df


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """Train the dual CNN model"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
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
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
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
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        scheduler.step()
    
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
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to class names
    class_names = ['CN', 'MCI', 'AD']
    
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Patient-Level Split Dual CNN')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('patient_split_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Dual CNN with Patient-Level Split')
    parser.add_argument('--csv_path', default='../whole_brain_slices_dataset/whole_brain_slices.csv', help='Path to slices CSV')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading slice data...")
    df = pd.read_csv(args.csv_path)
    print(f"Total slices: {len(df)}")
    
    # Create patient-level splits
    print("\nCreating patient-level splits...")
    train_df, val_df, test_df = create_patient_split(df)
    
    # Create datasets and loaders
    train_dataset = BrainSliceDataset(train_df, os.path.dirname(args.csv_path))
    val_dataset = BrainSliceDataset(val_df, os.path.dirname(args.csv_path))
    test_dataset = BrainSliceDataset(test_df, os.path.dirname(args.csv_path))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = DualCNN(num_classes=3).to(device)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    print("\nStarting training...")
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, num_epochs=args.epochs, learning_rate=args.learning_rate
    )
    
    # Save model
    torch.save(model.state_dict(), 'best_patient_split_dual_cnn.pth')
    print("Model saved as 'best_patient_split_dual_cnn.pth'")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy = evaluate_model(model, test_loader)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('patient_split_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()