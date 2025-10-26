#!/usr/bin/env python3
"""
Official prediction script using the exact ADNI_unimodal_models repository structure
"""

import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

# Add the cloned repo to path
repo_path = Path(__file__).parent / 'ADNI_unimodal_models'
sys.path.insert(0, str(repo_path))

from encoders.image_encoders import DenseNet3D
from models.classifier import CustomClassifier
from models.model import UnimodalModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_preprocess_nifti(nifti_path: str, target_shape=(128, 128, 128)):
    """
    Load and preprocess NIfTI file

    Args:
        nifti_path: Path to NIfTI file
        target_shape: Target shape for resizing

    Returns:
        Preprocessed tensor [1, 1, D, H, W]
    """
    logger.info(f"Loading NIfTI: {nifti_path}")

    # Load NIfTI
    nii = nib.load(nifti_path)
    volume = nii.get_fdata()

    logger.info(f"Original shape: {volume.shape}")

    # Normalize to [0, 1]
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    # Resize to target shape
    if volume.shape != target_shape:
        zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
        volume = zoom(volume, zoom_factors, order=1)
        logger.info(f"Resized to: {volume.shape}")

    # Convert to tensor and add batch + channel dimensions
    tensor = torch.from_numpy(volume).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]

    return tensor


def load_model(checkpoint_path: str, device='cpu'):
    """
    Load the UnimodalModel from checkpoint using official structure

    Args:
        checkpoint_path: Path to .ckpt file
        device: Device to use

    Returns:
        Loaded model and class names
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get hyperparameters
    hparams = checkpoint.get('hyper_parameters', {})
    classes = hparams.get('classes', ['AD', 'CN'])
    num_classes = len(classes)

    logger.info(f"Classes: {classes}")
    logger.info(f"Number of classes: {num_classes}")

    # Create encoder (DenseNet3D)
    image_encoder = DenseNet3D(freeze=False, include_head=False)

    # Create classifier
    classifier = CustomClassifier(
        hidden_dim=0,  # No hidden layer
        activation_fun=nn.ReLU(),
        num_class=num_classes,
        task="multiclass",
        dropout_rate=0
    )

    # Load state dict first to determine input dimension
    state_dict = checkpoint['state_dict']

    # Remove 'model.' prefix if present
    if all(k.startswith('model.') for k in state_dict.keys()):
        logger.info("Stripping 'model.' prefix from state_dict")
        state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}

    # Initialize classifier with correct input dimension from checkpoint
    fc2_weight = state_dict['classifier.fc2.weight']
    input_dim = fc2_weight.shape[1]
    logger.info(f"Detected input dimension: {input_dim}")

    # Initialize the classifier layers
    classifier.set_input_dim(input_dim, device)

    # Create UnimodalModel
    encoders = {"image": image_encoder}
    model = UnimodalModel(encoders=encoders, classifier=classifier)

    # Load weights
    model.load_state_dict(state_dict, strict=True)
    logger.info("Model loaded successfully")

    # Set to eval mode
    model.eval()
    model.to(device)

    return model, classes


def predict(model, input_tensor, classes, device='cpu'):
    """
    Make prediction

    Args:
        model: Loaded UnimodalModel
        input_tensor: Preprocessed tensor [1, 1, D, H, W]
        classes: List of class names
        device: Device to use

    Returns:
        Predicted class and confidence scores
    """
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        # Model expects a dict with 'image' key
        batch = {"image": input_tensor}
        logits = model(batch)

        # Get probabilities
        probabilities = torch.softmax(logits, dim=1)

        # Get prediction
        pred_idx = probabilities.argmax(dim=1).item()
        predicted_class = classes[pred_idx]

        # Get confidence scores
        confidence_scores = {
            classes[i]: float(probabilities[0, i])
            for i in range(len(classes))
        }

    return predicted_class, confidence_scores


def main():
    parser = argparse.ArgumentParser(description='Official ADNI prediction using repository code')

    parser.add_argument('--model', '-m', type=str,
                       default='models/best_model_densenet3D_epoch=14-step=3510.ckpt',
                       help='Path to model checkpoint')

    parser.add_argument('--data', '-d', type=str, required=True,
                       help='Path to NIfTI file')

    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')

    args = parser.parse_args()

    # Validate paths
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)

    if not Path(args.data).exists():
        logger.error(f"Data not found: {args.data}")
        sys.exit(1)

    try:
        print("\n" + "="*80)
        print("🧠 OFFICIAL ADNI UNIMODAL MODEL PREDICTION")
        print("="*80 + "\n")

        # Load model
        model, classes = load_model(args.model, device=args.device)

        # Load and preprocess data
        input_tensor = load_and_preprocess_nifti(args.data)

        # Make prediction
        logger.info("Making prediction...")
        predicted_class, confidence_scores = predict(model, input_tensor, classes, device=args.device)

        # Display results
        print("\n" + "="*80)
        print("🎯 PREDICTION RESULTS")
        print("="*80)
        print(f"\n✅ Predicted Class: {predicted_class}")
        print(f"\n📈 Confidence Scores:")
        for class_name, score in confidence_scores.items():
            bar = "█" * int(score * 50)
            print(f"  {class_name:6s}: {score:6.2%} {bar}")
        print("\n" + "="*80 + "\n")

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
