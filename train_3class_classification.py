#!/usr/bin/env python3
"""
3-Class Alzheimer's Disease Classification Model
Classifies brain MRI scans into AD (Alzheimer's), MCI (Mild Cognitive Impairment), and CN (Cognitively Normal)

Following methodology from Nature paper for neuroimaging-based classification.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ADNIDataset(Dataset):
    """Dataset class for ADNI 3-class classification"""
    
    def __init__(self, data_dir, csv_file=None, transform=None, target_size=(128, 128, 128)):
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        self.samples = []
        
        # Load data paths and labels
        self._load_data_paths()
        
    def _load_data_paths(self):
        """Load file paths and create labels for 3-class classification"""
        
        # AD samples
        ad_dir = os.path.join(self.data_dir, 'AD')
        if os.path.exists(ad_dir):
            ad_files = [f for f in os.listdir(ad_dir) if f.endswith('.nii.gz')]
            for file in ad_files:
                self.samples.append((os.path.join(ad_dir, file), 0))  # AD = 0
        
        # CN samples  
        cn_dir = os.path.join(self.data_dir, 'CN')
        if os.path.exists(cn_dir):
            cn_files = [f for f in os.listdir(cn_dir) if f.endswith('.nii.gz')]
            for file in cn_files:
                self.samples.append((os.path.join(cn_dir, file), 1))  # CN = 1
        
        # MCI samples
        mci_dir = os.path.join(self.data_dir, 'MCI') 
        if os.path.exists(mci_dir):
            mci_files = [f for f in os.listdir(mci_dir) if f.endswith('.nii.gz')]
            for file in mci_files:
                self.samples.append((os.path.join(mci_dir, file), 2))  # MCI = 2
        
        print(f"Dataset loaded: {len(self.samples)} total samples")
        labels = [label for _, label in self.samples]
        print(f"AD: {labels.count(0)}, CN: {labels.count(1)}, MCI: {labels.count(2)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        try:
            # Load NIfTI image
            nii_img = nib.load(file_path)
            data = nii_img.get_fdata()
            
            # Normalize intensity values
            data = (data - data.mean()) / (data.std() + 1e-8)
            
            # Resize to target size
            data = self._resize_volume(data, self.target_size)
            
            # Convert to tensor and add channel dimension
            data = torch.FloatTensor(data).unsqueeze(0)  # Shape: (1, D, H, W)
            
            # Apply transforms if specified
            if self.transform:
                data = self.transform(data)
            
            return data, label
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a zero tensor if loading fails
            return torch.zeros((1,) + self.target_size), label
    
    def _resize_volume(self, volume, target_size):
        """Resize 3D volume to target size using simple interpolation"""
        from scipy.ndimage import zoom
        
        current_size = volume.shape
        zoom_factors = [t/c for t, c in zip(target_size, current_size)]
        
        try:
            resized = zoom(volume, zoom_factors, order=1)
            return resized
        except:
            # Fallback: pad or crop to target size
            return self._pad_or_crop(volume, target_size)
    
    def _pad_or_crop(self, volume, target_size):
        """Fallback method to pad or crop volume to target size"""
        current_size = np.array(volume.shape)
        target_size = np.array(target_size)
        
        # Calculate padding/cropping for each dimension
        result = np.zeros(target_size)
        
        # Calculate start indices
        start_idx = np.maximum(0, (target_size - current_size) // 2)
        end_idx = start_idx + np.minimum(current_size, target_size)
        
        # Calculate source indices
        src_start = np.maximum(0, (current_size - target_size) // 2)
        src_end = src_start + np.minimum(current_size, target_size)
        
        # Copy data
        result[start_idx[0]:end_idx[0], 
               start_idx[1]:end_idx[1], 
               start_idx[2]:end_idx[2]] = volume[src_start[0]:src_end[0],
                                                 src_start[1]:src_end[1], 
                                                 src_start[2]:src_end[2]]
        return result

class ResNet3D(nn.Module):
    """3D ResNet for Alzheimer's classification"""
    
    def __init__(self, num_classes=3, input_channels=1):
        super(ResNet3D, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First block with potential downsampling
        layers.append(BasicBlock3D(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class BasicBlock3D(nn.Module):
    """Basic 3D ResNet block"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip_connection(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(dataloader), 100. * correct / total

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    return running_loss / len(dataloader), 100. * correct / total, all_predictions, all_targets

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - 3-Class AD Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Configuration
    data_dir = '/Users/tanguyvans/Desktop/umons/alzheimer/ADNIDenoise'
    csv_file = '/Users/tanguyvans/Desktop/umons/alzheimer/ADNIDenoise/AD_CN_df.csv'
    
    # Hyperparameters
    batch_size = 8  # Reduced for memory efficiency
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-4
    target_size = (96, 96, 96)  # Reduced for memory efficiency
    
    # Class names
    class_names = ['AD', 'CN', 'MCI']
    
    print("Loading dataset...")
    dataset = ADNIDataset(data_dir, csv_file, target_size=target_size)
    
    # Split dataset (80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = ResNet3D(num_classes=3, input_channels=1)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_acc = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_ad_mci_cn_model.pth')
            print(f"New best model saved! Validation accuracy: {val_acc:.2f}%")
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Best Val Acc: {best_val_acc:.2f}%")
    
    # Load best model for final evaluation
    checkpoint = torch.load('best_ad_mci_cn_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation
    print("\nFinal evaluation on validation set:")
    val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(val_targets, val_preds, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(val_targets, val_preds, class_names, 'confusion_matrix_3class.png')
    
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
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves_3class.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final validation accuracy: {val_acc:.2f}%")

if __name__ == "__main__":
    main()