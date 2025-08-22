#!/usr/bin/env python3
"""
MedCLIP for Alzheimer's Disease Classification
Uses medical-specific pretrained features for brain slice classification
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

# Check PyTorch availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Always import torchvision components
import torchvision.models as models
from torchvision import transforms

try:
    from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
    print("MedCLIP imported successfully")
except ImportError as e:
    print(f"MedCLIP import failed: {e}")
    print("Falling back to standard ResNet approach")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MedCLIPClassifier(nn.Module):
    """
    MedCLIP-based classifier for Alzheimer's disease
    Uses medical-specific pretrained features or falls back to ResNet
    """
    
    def __init__(self, num_classes=3, freeze_backbone=True):
        super(MedCLIPClassifier, self).__init__()
        
        self.use_medclip = False
        
        try:
            # Try to load MedCLIP vision model
            self.medclip_model = MedCLIPVisionModelViT()
            self.processor = MedCLIPProcessor()
            self.use_medclip = True
            print("Using MedCLIP backbone")
            
            # Freeze MedCLIP if specified
            if freeze_backbone:
                for param in self.medclip_model.parameters():
                    param.requires_grad = False
            
            # Get MedCLIP output dimensions
            backbone_dim = 768  # ViT-B/16 output dimension
            
        except Exception as e:
            print(f"MedCLIP initialization failed: {e}")
            print("Falling back to ResNet backbone")
            
            # Fallback to ResNet
            self.resnet = models.resnet50(pretrained=True)
            self.resnet.fc = nn.Identity()  # Remove final layer
            
            if freeze_backbone:
                for param in self.resnet.parameters():
                    param.requires_grad = False
            
            backbone_dim = 2048  # ResNet50 output dimension
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(backbone_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, 3, 224, 224)
        
        if self.use_medclip:
            # Extract MedCLIP features
            with torch.no_grad() if not self.training else torch.enable_grad():
                backbone_features = self.medclip_model(x)  # (batch_size, 768)
        else:
            # Extract ResNet features
            with torch.no_grad() if not self.training else torch.enable_grad():
                backbone_features = self.resnet(x)  # (batch_size, 2048)
        
        # Classification
        output = self.classifier(backbone_features)
        
        return output


class MedCLIPBrainSliceDataset(Dataset):
    """Dataset for brain slices with MedCLIP or standard preprocessing"""
    
    def __init__(self, csv_data, base_path, augment=False, use_medclip=True):
        self.data = csv_data.reset_index(drop=True)
        self.base_path = base_path
        self.augment = augment
        self.use_medclip = use_medclip
        
        if self.use_medclip:
            try:
                # MedCLIP processor handles normalization
                self.processor = MedCLIPProcessor()
            except:
                self.use_medclip = False
                print("MedCLIP processor failed, using standard transforms")
        
        if not self.use_medclip:
            # Custom transforms that don't rely on NumPy
            self.resize = transforms.Resize((224, 224))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                               std=[0.229, 0.224, 0.225])
        
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
            # Try to load NIfTI slice
            try:
                nii_img = nib.load(slice_path)
                data = nii_img.get_fdata()
                
                # Convert to PIL Image
                if len(data.shape) == 3 and data.shape[2] == 3:
                    # Already RGB
                    data = data
                else:
                    # Convert grayscale to RGB
                    if len(data.shape) == 2:
                        data = np.stack([data, data, data], axis=-1)
                    else:
                        data = np.stack([data[:, :, 0], data[:, :, 0], data[:, :, 0]], axis=-1)
                
                # Normalize to 0-255 range for PIL
                data = ((data - data.min()) / (data.max() - data.min() + 1e-8) * 255).astype(np.uint8)
                
                # Convert to PIL Image
                from PIL import Image
                pil_image = Image.fromarray(data)
                
            except Exception as nifti_error:
                print(f"NIfTI loading failed for {slice_path}: {nifti_error}")
                # Create a dummy black image
                from PIL import Image
                pil_image = Image.new('RGB', (224, 224), (0, 0, 0))
            
            if self.use_medclip:
                # Process with MedCLIP processor
                processed = self.processor(images=pil_image, return_tensors="pt")
                processed_tensor = processed["pixel_values"].squeeze(0)  # Remove batch dimension
                return processed_tensor, label
            else:
                # Custom preprocessing for ResNet without NumPy
                # Resize image
                pil_image = self.resize(pil_image)
                
                # Convert PIL to tensor manually
                tensor = torch.tensor(list(pil_image.getdata()), dtype=torch.float32)
                tensor = tensor.view(pil_image.size[1], pil_image.size[0], 3)  # H, W, C
                tensor = tensor.permute(2, 0, 1)  # C, H, W
                tensor = tensor / 255.0  # Normalize to [0, 1]
                
                # Apply normalization
                tensor = self.normalize(tensor)
                return tensor, label
            
        except Exception as e:
            print(f"Error loading {slice_path}: {e}")
            # Return zero tensor with appropriate preprocessing
            from PIL import Image
            
            if self.use_medclip:
                # Create dummy image for MedCLIP
                dummy_image = Image.new('RGB', (224, 224), (0, 0, 0))
                processed = self.processor(images=dummy_image, return_tensors="pt")
                return processed["pixel_values"].squeeze(0), label
            else:
                # Create dummy tensor for ResNet without NumPy
                tensor = torch.zeros(3, 224, 224, dtype=torch.float32)
                tensor = self.normalize(tensor)
                return tensor, label


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


def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001):
    """Train MedCLIP model"""
    
    criterion = nn.CrossEntropyLoss()
    
    # Only optimize trainable parameters (classifier head if MedCLIP is frozen)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    patience = 0
    max_patience = 7
    
    for epoch in range(num_epochs):
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
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}')
        
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
    
    backbone_name = "MedCLIP" if model.use_medclip else "ResNet"
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\n{backbone_name} Test Accuracy: {accuracy:.4f}")
    
    class_names = ['CN', 'MCI', 'AD']
    print(f"\n{backbone_name} Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{backbone_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'{backbone_name.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Train MedCLIP Model')
    parser.add_argument('--csv_path', default='../whole_brain_slices_dataset/whole_brain_slices.csv')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--unfreeze_medclip', action='store_true', help='Fine-tune MedCLIP layers')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading slice data for MedCLIP...")
    df = pd.read_csv(args.csv_path)
    
    # Create patient-level splits
    print("\nCreating patient-level splits...")
    train_df, val_df, test_df = create_patient_split(df)
    
    # Detect if MedCLIP is available
    try:
        test_model = MedCLIPVisionModelViT()
        use_medclip = True
        print("MedCLIP backbone available")
    except:
        use_medclip = False
        print("Using ResNet backbone instead of MedCLIP")
    
    # Create datasets
    train_dataset = MedCLIPBrainSliceDataset(train_df, os.path.dirname(args.csv_path), augment=True, use_medclip=use_medclip)
    val_dataset = MedCLIPBrainSliceDataset(val_df, os.path.dirname(args.csv_path), augment=False, use_medclip=use_medclip)
    test_dataset = MedCLIPBrainSliceDataset(test_df, os.path.dirname(args.csv_path), augment=False, use_medclip=use_medclip)
    
    # Use num_workers=0 to avoid multiprocessing issues with NumPy
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create model (MedCLIP or ResNet)
    model = MedCLIPClassifier(
        num_classes=3, 
        freeze_backbone=not args.unfreeze_medclip
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_name = "MedCLIP" if model.use_medclip else "ResNet"
    print(f"\n{backbone_name} model created with {total_params:,} total parameters")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train
    print(f"\nStarting {backbone_name} training...")
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, num_epochs=args.epochs, learning_rate=args.learning_rate
    )
    
    # Save model
    model_filename = f'best_{backbone_name.lower()}.pth'
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as '{model_filename}'")
    
    # Evaluate
    print(f"\nEvaluating {backbone_name} on test set...")
    test_accuracy = evaluate_model(model, test_loader)
    
    print(f"\nFinal {backbone_name} Test Accuracy: {test_accuracy:.4f}")
    print(f"Best {backbone_name} Validation Accuracy: {max(val_accs):.2f}%")


if __name__ == "__main__":
    main()