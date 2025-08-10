# Alzheimer's Disease Classification Using 3D CNNs

This repository contains a deep learning pipeline for classifying brain MRI scans into three categories:
- **AD** (Alzheimer's Disease)
- **MCI** (Mild Cognitive Impairment) 
- **CN** (Cognitively Normal)

## 🧠 Dataset

The project uses the **ADNIDenoise** dataset containing 2,495 preprocessed brain MRI scans:
- **AD**: 433 scans
- **CN**: 746 scans
- **MCI**: 1,316 scans

All images are already skull-stripped and registered to MNI space.

## 📊 Results

Our 3D ResNet model achieved excellent performance:

```
Overall Accuracy: 95.19%

Class-wise Performance:
              precision    recall  f1-score   support
          AD       0.96      0.90      0.93        84
          CN       0.94      0.96      0.95       153
         MCI       0.95      0.96      0.96       262
```

## 🚀 Quick Start

### Prerequisites

```bash
# Activate virtual environment
source env/bin/activate

# Ensure dependencies are installed
pip install torch torchvision nibabel numpy pandas scikit-learn matplotlib tqdm
```

### 1. Data Preprocessing

First, analyze and prepare your dataset:

```bash
python data_preprocessing_3class.py
```

This script will:
- Analyze the dataset structure and scan properties
- Create metadata files for all three classes
- Generate train/validation splits (80/20)
- Create sample visualizations
- Output: `metadata_3class.csv`, `train_split.csv`, `val_split.csv`

### 2. Training the Model

Train the 3D ResNet model:

```bash
python train_3class_classification.py
```

Training details:
- **Architecture**: 3D ResNet with residual connections
- **Input size**: 96×96×96 voxels (memory-efficient)
- **Batch size**: 8
- **Optimizer**: AdamW with learning rate scheduling
- **Training time**: ~50 epochs

The script will:
- Train the model with automatic validation
- Save the best model as `best_ad_mci_cn_model.pth`
- Generate training curves and confusion matrix
- Display detailed classification metrics

### 3. Using the Trained Model

To load and use the trained model:

```python
import torch
from train_3class_classification import ResNet3D

# Load model
model = ResNet3D(num_classes=3, input_channels=1)
checkpoint = torch.load('best_ad_mci_cn_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Model is ready for inference
```

## 📁 Project Structure

```
alzheimer/
├── ADNIDenoise/            # Main dataset
│   ├── AD/                 # Alzheimer's Disease scans
│   ├── CN/                 # Cognitively Normal scans
│   └── MCI/                # Mild Cognitive Impairment scans
├── preprocessing/          # Preprocessing pipeline modules
├── mni_template/           # MNI brain template for registration
├── train_3class_classification.py    # Main training script
├── data_preprocessing_3class.py      # Data preparation script
├── best_ad_mci_cn_model.pth         # Trained model weights
├── confusion_matrix_3class.png       # Results visualization
└── training_curves_3class.png        # Training history

```

## 🔧 Advanced Usage

### Custom Training Configuration

Edit the hyperparameters in `train_3class_classification.py`:

```python
# Hyperparameters
batch_size = 8          # Increase if you have more GPU memory
num_epochs = 50         # Number of training epochs
learning_rate = 1e-4    # Initial learning rate
target_size = (96, 96, 96)  # Input volume size
```

### Data Augmentation

The pipeline includes basic intensity normalization. You can add more augmentations by modifying the `transform` parameter in the dataset class.

## 📈 Model Architecture

The model uses a 3D ResNet architecture optimized for medical imaging:

```
3D ResNet Architecture:
- Initial Conv3D + BatchNorm + ReLU + MaxPool
- 4 Residual blocks with increasing channels (64→128→256→512)
- Adaptive 3D pooling for consistent output size
- Dropout (0.5) for regularization
- Final FC layer for 3-class classification
```

## 🤝 Citation

If you use this code, please cite:
- ADNI dataset: [adni.loni.usc.edu](http://adni.loni.usc.edu/)
- Based on methodology from Nature Scientific Reports paper on neuroimaging classification

## 📝 Notes

- All brain scans are preprocessed (skull-stripped and MNI-registered)
- The model processes full 3D volumes, not 2D slices
- Memory usage: ~8GB GPU memory with batch_size=8
- Training time: Approximately 2-3 hours on a modern GPU