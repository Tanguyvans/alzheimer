import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Configuration
class Config:
    BATCH_SIZE = 2
    EPOCHS = 3
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4
    DATA_PATH = "npy_seg/"
    CSV_PATH = "db.csv"
    MODEL_SAVE_PATH = "./models"
    
# Dataset class
class AlzheimerDataset(Dataset):
    def __init__(self, data_path, dataframe, transform=None):
        self.data_path = data_path
        self.df = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image ID and label
        adni_id = self.df.iloc[idx]['adni_id']
        label = self.df.iloc[idx]['label_encoded']
        
        # Load NPY file
        img_path = os.path.join(self.data_path, f"{adni_id}.npy")
        image = np.load(img_path)
        
        # Convert to tensor and add channel dimension if needed
        image = torch.FloatTensor(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)
            
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        
        # Debug: afficher la forme du tenseur
        print(f"Image shape: {image.shape}")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Modified ResNet3D
class ResNet3DClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNet3DClassifier, self).__init__()
        self.resnet = r3d_18(pretrained=True)
        
        # Modify first conv layer to accept single channel
        self.resnet.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7),
                                       stride=(1, 2, 2), padding=(1, 3, 3),
                                       bias=False)
        
        # Modify final FC layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        print(f"Batch shape: {images.shape}")  # Devrait être (2, 1, 50, 128, 128)
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
        
    return running_loss/len(dataloader), 100.*correct/total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss/len(dataloader), 100.*correct/total

def test(model, test_loader, device, le):
    """Evaluate model on test data and show predictions vs true values."""
    model.eval()
    all_preds = []
    all_labels = []
    
    print("\nStarting testing phase...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Convert numeric labels back to original classes
    pred_classes = le.inverse_transform(all_preds)
    true_classes = le.inverse_transform(all_labels)
    
    # Print results
    print("\nTest Results:")
    print("Predicted\tTrue")
    print("-" * 30)
    for pred, true in zip(pred_classes, true_classes):
        print(f"{pred}\t\t{true}")
    
    # Calculate accuracy
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    
    return pred_classes, true_classes

def main():
    config = Config()
    
    # Create model save directory
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    
    # Load and prepare data
    df = pd.read_csv(config.CSV_PATH)
    
    # Print class distribution
    print("\nClass Distribution:")
    print(df['research_group'].value_counts())
    print("\nUnique classes:", df['research_group'].unique())
    
    # Encode labels
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['research_group'])
    num_classes = len(le.classes_)
    print(f"\nNumber of classes: {num_classes}")
    print("Classes mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
    
    # Split data into train, val, and test
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, 
                                            stratify=df['label_encoded'])
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, 
                                       stratify=train_val_df['label_encoded'])
    
    print("\nDataset splits:")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Create datasets
    train_dataset = AlzheimerDataset(config.DATA_PATH, train_df)
    val_dataset = AlzheimerDataset(config.DATA_PATH, val_df)
    test_dataset = AlzheimerDataset(config.DATA_PATH, test_df)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                          shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                           shuffle=False, num_workers=config.NUM_WORKERS)
    
    # Initialize model
    model = ResNet3DClassifier(num_classes=num_classes)
    model = model.to(config.DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                    patience=5, factor=0.5)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                          optimizer, config.DEVICE)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                      os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth'))
            
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Test best model
    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth')))
    pred_classes, true_classes = test(model, test_loader, config.DEVICE, le)
    
    # Save test results
    results_df = pd.DataFrame({
        'adni_id': test_df['adni_id'].values,
        'predicted': pred_classes,
        'true': true_classes
    })
    results_df.to_csv('test_results.csv', index=False)
    print("\nResults saved to test_results.csv")

if __name__ == "__main__":
    main() 