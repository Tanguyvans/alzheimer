#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for 3-Class AD Classification
Prepares ADNI dataset for training AD/MCI/CN classification model
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    """Preprocessor for ADNI neuroimaging data"""
    
    def __init__(self, data_dir, target_size=(128, 128, 128)):
        self.data_dir = data_dir
        self.target_size = target_size
        self.samples_info = []
        
    def load_dataset_info(self):
        """Load and analyze dataset information"""
        
        # Scan each directory for file counts and basic info
        for class_name in ['AD', 'CN', 'MCI']:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                files = [f for f in os.listdir(class_dir) if f.endswith('.nii.gz')]
                print(f"{class_name}: {len(files)} files")
                
                for file in files[:5]:  # Sample first 5 files for analysis
                    file_path = os.path.join(class_dir, file)
                    try:
                        nii = nib.load(file_path)
                        shape = nii.get_fdata().shape
                        spacing = nii.header.get_zooms()
                        self.samples_info.append({
                            'class': class_name,
                            'file': file,
                            'shape': shape,
                            'spacing': spacing
                        })
                    except Exception as e:
                        print(f"Error reading {file}: {e}")
        
        return self.samples_info
    
    def analyze_dataset(self):
        """Analyze dataset characteristics"""
        
        print("Dataset Analysis")
        print("=" * 50)
        
        # Load sample info
        info = self.load_dataset_info()
        
        if not info:
            print("No valid samples found!")
            return
        
        # Analyze image dimensions
        shapes = [sample['shape'] for sample in info]
        spacings = [sample['spacing'] for sample in info]
        
        print(f"\nImage Shapes (sample of {len(shapes)} files):")
        unique_shapes = list(set([str(s) for s in shapes]))
        for shape in unique_shapes:
            count = sum(1 for s in shapes if str(s) == shape)
            print(f"  {shape}: {count} files")
        
        print(f"\nVoxel Spacings (sample of {len(spacings)} files):")
        unique_spacings = list(set([str(s[:3]) for s in spacings]))  # First 3 dimensions
        for spacing in unique_spacings[:5]:  # Show first 5
            count = sum(1 for s in spacings if str(s[:3]) == spacing)
            print(f"  {spacing}: {count} files")
        
        # Class distribution
        print("\nClass Distribution:")
        for class_name in ['AD', 'CN', 'MCI']:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                file_count = len([f for f in os.listdir(class_dir) if f.endswith('.nii.gz')])
                print(f"  {class_name}: {file_count} samples")
    
    def preprocess_volume(self, volume):
        """Preprocess a single 3D volume"""
        
        # 1. Intensity normalization (z-score)
        mean_val = np.mean(volume)
        std_val = np.std(volume)
        if std_val > 0:
            volume = (volume - mean_val) / std_val
        
        # 2. Clip extreme values
        volume = np.clip(volume, -5, 5)
        
        # 3. Resize to target size
        if volume.shape != self.target_size:
            zoom_factors = [t/c for t, c in zip(self.target_size, volume.shape)]
            volume = zoom(volume, zoom_factors, order=1)
        
        return volume.astype(np.float32)
    
    def create_metadata_file(self):
        """Create comprehensive metadata file for 3-class classification"""
        
        metadata = []
        
        for class_name in ['AD', 'CN', 'MCI']:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                files = [f for f in os.listdir(class_dir) if f.endswith('.nii.gz')]
                
                for file in files:
                    # Extract subject ID from filename
                    parts = file.split('_')
                    subject_id = '_'.join(parts[1:3]) if len(parts) >= 3 else 'unknown'
                    
                    metadata.append({
                        'filename': file,
                        'filepath': os.path.join(class_name, file),
                        'subject_id': subject_id,
                        'class': class_name,
                        'label': {'AD': 0, 'CN': 1, 'MCI': 2}[class_name]
                    })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(metadata)
        csv_path = os.path.join(self.data_dir, 'metadata_3class.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"\nMetadata saved to: {csv_path}")
        print(f"Total samples: {len(df)}")
        print("\nClass distribution:")
        print(df['class'].value_counts())
        
        return df
    
    def create_train_val_split(self, test_size=0.2, random_state=42):
        """Create stratified train/validation split"""
        
        # Load metadata
        metadata_file = os.path.join(self.data_dir, 'metadata_3class.csv')
        if not os.path.exists(metadata_file):
            print("Creating metadata file first...")
            df = self.create_metadata_file()
        else:
            df = pd.read_csv(metadata_file)
        
        # Create stratified split
        X = df[['filename', 'filepath', 'subject_id']]
        y = df['label']
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        # Create train/val metadata
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        
        # Add class names back
        label_to_class = {0: 'AD', 1: 'CN', 2: 'MCI'}
        train_df['class'] = train_df['label'].map(label_to_class)
        val_df['class'] = val_df['label'].map(label_to_class)
        
        # Save splits
        train_df.to_csv(os.path.join(self.data_dir, 'train_split.csv'), index=False)
        val_df.to_csv(os.path.join(self.data_dir, 'val_split.csv'), index=False)
        
        print(f"\nTrain/validation split created:")
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        
        print("\nTraining set distribution:")
        print(train_df['class'].value_counts())
        
        print("\nValidation set distribution:")
        print(val_df['class'].value_counts())
        
        return train_df, val_df
    
    def visualize_sample_images(self, n_samples=2):
        """Visualize sample images from each class"""
        
        fig, axes = plt.subplots(3, n_samples*3, figsize=(15, 9))
        fig.suptitle('Sample Brain Scans from Each Class', fontsize=16)
        
        row_idx = 0
        for class_name in ['AD', 'CN', 'MCI']:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                files = [f for f in os.listdir(class_dir) if f.endswith('.nii.gz')][:n_samples]
                
                for col_idx, file in enumerate(files):
                    try:
                        # Load image
                        file_path = os.path.join(class_dir, file)
                        nii = nib.load(file_path)
                        data = nii.get_fdata()
                        
                        # Show sagittal, coronal, and axial slices
                        mid_sag = data[data.shape[0]//2, :, :]
                        mid_cor = data[:, data.shape[1]//2, :]
                        mid_ax = data[:, :, data.shape[2]//2]
                        
                        slices = [mid_sag, mid_cor, mid_ax]
                        slice_names = ['Sagittal', 'Coronal', 'Axial']
                        
                        for slice_idx, (slice_data, slice_name) in enumerate(zip(slices, slice_names)):
                            ax = axes[row_idx, col_idx*3 + slice_idx]
                            im = ax.imshow(slice_data.T, cmap='gray', origin='lower')
                            ax.set_title(f'{class_name} - {slice_name}')
                            ax.axis('off')
                            
                    except Exception as e:
                        print(f"Error visualizing {file}: {e}")
            
            row_idx += 1
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'sample_images.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_dataset_statistics(self):
        """Get comprehensive dataset statistics"""
        
        stats = {
            'total_samples': 0,
            'class_counts': {},
            'file_sizes': [],
            'image_shapes': [],
            'voxel_spacings': []
        }
        
        for class_name in ['AD', 'CN', 'MCI']:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                files = [f for f in os.listdir(class_dir) if f.endswith('.nii.gz')]
                stats['class_counts'][class_name] = len(files)
                stats['total_samples'] += len(files)
                
                # Sample a few files for detailed analysis
                for file in files[:10]:  # Analyze first 10 files per class
                    file_path = os.path.join(class_dir, file)
                    try:
                        # File size
                        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                        stats['file_sizes'].append(file_size)
                        
                        # Image properties
                        nii = nib.load(file_path)
                        stats['image_shapes'].append(nii.get_fdata().shape)
                        stats['voxel_spacings'].append(nii.header.get_zooms())
                        
                    except Exception as e:
                        print(f"Error analyzing {file}: {e}")
        
        return stats

def main():
    """Main preprocessing pipeline"""
    
    data_dir = '/Users/tanguyvans/Desktop/umons/alzheimer/ADNIDenoise'
    
    print("Alzheimer's Disease 3-Class Classification - Data Preprocessing")
    print("=" * 70)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(data_dir, target_size=(128, 128, 128))
    
    # 1. Analyze dataset
    print("\n1. Analyzing dataset...")
    preprocessor.analyze_dataset()
    
    # 2. Get detailed statistics
    print("\n2. Getting dataset statistics...")
    stats = preprocessor.get_dataset_statistics()
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Class distribution: {stats['class_counts']}")
    
    if stats['file_sizes']:
        print(f"Average file size: {np.mean(stats['file_sizes']):.2f} MB")
        print(f"File size range: {np.min(stats['file_sizes']):.2f} - {np.max(stats['file_sizes']):.2f} MB")
    
    # 3. Create metadata file
    print("\n3. Creating metadata file...")
    preprocessor.create_metadata_file()
    
    # 4. Create train/validation split
    print("\n4. Creating train/validation split...")
    train_df, val_df = preprocessor.create_train_val_split()
    
    # 5. Visualize sample images
    print("\n5. Creating sample visualizations...")
    try:
        preprocessor.visualize_sample_images()
    except Exception as e:
        print(f"Visualization error: {e}")
    
    print("\n" + "="*70)
    print("Data preprocessing completed successfully!")
    print(f"Files created in {data_dir}:")
    print("- metadata_3class.csv (complete dataset metadata)")
    print("- train_split.csv (training set)")
    print("- val_split.csv (validation set)")
    print("- sample_images.png (sample visualizations)")
    print("\nDataset is now ready for training!")

if __name__ == "__main__":
    main()