#!/usr/bin/env python3
"""
Pretrained CNN Models for Alzheimer's Disease Classification
Using ResNet50, EfficientNet, or DenseNet with the new comprehensive dataset
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
import torchvision.models as models
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PretrainedClassifier(nn.Module):
    """
    Pretrained model with custom classification head for Alzheimer's
    """
    
    def __init__(self, model_name='resnet50', num_classes=3, dropout=0.5):
        super(PretrainedClassifier, self).__init__()
        
        # Load pretrained model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Identity()  # Remove original FC layer
            
        elif model_name == 'densenet121':
            self.model = models.densenet121(pretrained=True)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
            
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=True)
            num_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Identity()
            
        elif model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            num_features = self.model.classifier[0].in_features
            self.model.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.7),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
        
        self.model_name = model_name
        
    def forward(self, x):
        # Extract features
        features = self.model(x)
        # Classify
        output = self.classifier(features)
        return output
    
    def freeze_backbone(self, freeze=True):
        """Freeze or unfreeze the pretrained layers"""
        for param in self.model.parameters():
            param.requires_grad = not freeze


class BrainSliceDataset(Dataset):
    """Dataset for individual brain slices with pretrained model preprocessing"""
    
    def __init__(self, csv_data, base_path, augment=False):
        self.data = csv_data.reset_index(drop=True)
        self.base_path = base_path
        self.augment = augment
        
        # Standard ImageNet preprocessing
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Augmentation transforms
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        slice_path = row['slice_path']
        label = int(row['label'])
        
        # Handle paths
        if not os.path.exists(slice_path):
            relative_path = row['relative_path']
            slice_path = os.path.join(self.base_path, relative_path)
        
        try:
            # Load NIfTI slice
            nii_img = nib.load(slice_path)
            data = nii_img.get_fdata()
            
            # Convert to 3-channel tensor
            if len(data.shape) == 3 and data.shape[2] == 3:
                data = data
            else:
                if len(data.shape) == 2:
                    data = np.stack([data, data, data], axis=-1)
                else:
                    data = np.stack([data[:, :, 0], data[:, :, 0], data[:, :, 0]], axis=-1)
            
            # Normalize to 0-1
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            
            # Convert to tensor
            data = torch.FloatTensor(data).permute(2, 0, 1)
            
            # Resize to 224x224
            data = F.interpolate(data.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
            data = data.squeeze(0)
            
            # Apply augmentation if training
            if self.augment:
                # Convert to PIL for augmentation
                from torchvision.transforms import ToPILImage, ToTensor
                to_pil = ToPILImage()
                to_tensor = ToTensor()
                
                pil_img = to_pil(data)
                pil_img = self.augment_transform(pil_img)
                data = to_tensor(pil_img)
            
            # Apply ImageNet normalization
            data = self.normalize(data)
            
            return data, label
            
        except Exception as e:
            print(f"Error loading {slice_path}: {e}")
            # Return normalized zero tensor
            zero_tensor = torch.zeros(3, 224, 224)
            return self.normalize(zero_tensor), label


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
    
    print(f"\nFinal split:")
    print(f"Train: {len(train_df)} slices from {len(train_patients)} patients")
    print(f"Val: {len(val_df)} slices from {len(val_patients)} patients")
    print(f"Test: {len(test_df)} slices from {len(test_patients)} patients")
    
    return train_df, val_df, test_df


def train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=0.001, freeze_epochs=5):
    """Train the model with gradual unfreezing"""
    
    # Calculate class weights
    print("\nCalculating class weights...")
    train_labels = []
    for _, labels in tqdm(train_loader, desc="Collecting labels"):
        train_labels.extend(labels.numpy())
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Different learning rates for pretrained and new layers
    pretrained_params = model.model.parameters()
    classifier_params = model.classifier.parameters()
    
    optimizer = optim.Adam([
        {'params': pretrained_params, 'lr': learning_rate * 0.1},
        {'params': classifier_params, 'lr': learning_rate}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    patience = 0
    max_patience = 10
    
    # Initial training with frozen backbone
    print(f"\nTraining with frozen backbone for {freeze_epochs} epochs...")
    model.freeze_backbone(True)
    
    for epoch in range(num_epochs):
        # Unfreeze backbone after freeze_epochs
        if epoch == freeze_epochs:
            print("\nUnfreezing backbone layers...")
            model.freeze_backbone(False)
            # Reduce learning rate for pretrained layers
            for param_group in optimizer.param_groups:
                if param_group['lr'] > learning_rate * 0.1:
                    param_group['lr'] = learning_rate * 0.5
        
        # Training
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Print per-class accuracy for validation
        val_cm = confusion_matrix(val_labels, val_predictions)
        class_accuracies = val_cm.diagonal() / val_cm.sum(axis=1)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}')
        print(f'  Per-class Val Acc: AD={class_accuracies[0]:.2f}, MCI={class_accuracies[1]:.2f}, CN={class_accuracies[2]:.2f}')
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping
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
        for data, labels in tqdm(test_loader, desc="Evaluating"):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    class_names = ['AD', 'MCI', 'CN']  # Match label encoding
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model.model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'{model.model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Pretrained CNN Model')
    parser.add_argument('--csv_path', default='../full_brain_slices_dataset/full_brain_slices.csv')
    parser.add_argument('--model', default='resnet50', choices=['resnet50', 'densenet121', 'efficientnet_b0', 'vgg16'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--freeze_epochs', type=int, default=5, help='Number of epochs to train with frozen backbone')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading slice data from NEW comprehensive dataset...")
    df = pd.read_csv(args.csv_path)
    print(f"Dataset loaded: {len(df)} slices from {df['patient_id'].nunique()} patients")
    
    # Show dataset statistics
    print(f"\nDataset statistics:")
    print(f"Slices by diagnosis:")
    print(df['diagnosis'].value_counts())
    print(f"\nPatients by diagnosis:")
    print(df.groupby('diagnosis')['patient_id'].nunique())
    
    # Create patient-level splits
    print("\nCreating patient-level splits...")
    train_df, val_df, test_df = create_patient_split(df)
    
    # Check class distribution in splits
    print("\nClass distribution in splits:")
    print("Train:", train_df['diagnosis'].value_counts().to_dict())
    print("Val:", val_df['diagnosis'].value_counts().to_dict())
    print("Test:", test_df['diagnosis'].value_counts().to_dict())
    
    # Create datasets
    train_dataset = BrainSliceDataset(train_df, os.path.dirname(args.csv_path), augment=True)
    val_dataset = BrainSliceDataset(val_df, os.path.dirname(args.csv_path), augment=False)
    test_dataset = BrainSliceDataset(test_df, os.path.dirname(args.csv_path), augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    print(f"\nCreating {args.model} model...")
    model = PretrainedClassifier(model_name=args.model, num_classes=3, dropout=args.dropout).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train
    print("\nStarting training...")
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, 
        num_epochs=args.epochs, 
        learning_rate=args.learning_rate,
        freeze_epochs=args.freeze_epochs
    )
    
    # Save model
    model_filename = f'best_{args.model}.pth'
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as '{model_filename}'")
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_accuracy = evaluate_model(model, test_loader)
    
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
    print(f"Best Validation Accuracy: {max(val_accs):.2f}%")
    print(f"\nDataset improvement:")
    print(f"  Using {df['patient_id'].nunique()} patients (4.8x more than old dataset)")
    print(f"  Total {len(df)} slices")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.axvline(x=args.freeze_epochs, color='r', linestyle='--', label='Unfreeze')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{args.model}_training_curves.png', dpi=300, bbox_inches='tight')
    print(f"Training curves saved as '{args.model}_training_curves.png'")


if __name__ == "__main__":
    main()