#!/usr/bin/env python3
"""
Batch prediction using official ADNI_unimodal_models DenseNet3D
Processes AD and CN folders and generates accuracy report
"""

import sys
import argparse
import logging
import csv
import time
from pathlib import Path
from typing import List, Dict

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
    """Load and preprocess NIfTI file"""
    nii = nib.load(nifti_path)
    volume = nii.get_fdata()

    # Normalize
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    # Resize
    if volume.shape != target_shape:
        zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
        volume = zoom(volume, zoom_factors, order=1)

    # Convert to tensor
    tensor = torch.from_numpy(volume).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]

    return tensor


def load_model(checkpoint_path: str, device='cpu'):
    """Load the UnimodalModel from checkpoint"""
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get hyperparameters
    hparams = checkpoint.get('hyper_parameters', {})
    classes = hparams.get('classes', ['AD', 'CN'])
    if hasattr(classes, 'tolist'):
        classes = classes.tolist()
    else:
        classes = list(classes)

    num_classes = len(classes)

    # Create encoder
    image_encoder = DenseNet3D(freeze=False, include_head=False)

    # Create classifier
    classifier = CustomClassifier(
        hidden_dim=0,
        activation_fun=nn.ReLU(),
        num_class=num_classes,
        task="multiclass",
        dropout_rate=0
    )

    # Load state dict
    state_dict = checkpoint['state_dict']

    # Remove 'model.' prefix
    if all(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}

    # Initialize classifier
    fc2_weight = state_dict['classifier.fc2.weight']
    input_dim = fc2_weight.shape[1]
    classifier.set_input_dim(input_dim, device)

    # Create model
    encoders = {"image": image_encoder}
    model = UnimodalModel(encoders=encoders, classifier=classifier)

    # Load weights
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    logger.info("Model loaded successfully")

    return model, classes


def find_nifti_files(folder_path: Path) -> List[Path]:
    """Find all NIfTI files in folder, excluding hidden files"""
    nifti_files = []

    nifti_files.extend(folder_path.glob('*.nii'))
    nifti_files.extend(folder_path.glob('*.nii.gz'))
    nifti_files.extend(folder_path.glob('*/*.nii'))
    nifti_files.extend(folder_path.glob('*/*.nii.gz'))

    # Filter out hidden files
    nifti_files = [f for f in nifti_files if not f.name.startswith('._') and not f.name.startswith('.')]

    return sorted(nifti_files)


def predict_folder(model, classes, folder_path: Path, ground_truth_label: str, device='cpu') -> List[Dict]:
    """Predict all files in a folder"""
    logger.info(f"\nProcessing folder: {folder_path}")

    nifti_files = find_nifti_files(folder_path)

    if not nifti_files:
        logger.warning(f"No NIfTI files found in {folder_path}")
        return []

    logger.info(f"Found {len(nifti_files)} NIfTI files")

    results = []

    for i, nifti_file in enumerate(nifti_files, 1):
        try:
            logger.info(f"[{i}/{len(nifti_files)}] Processing: {nifti_file.name}")

            start_time = time.time()

            # Load and preprocess
            input_tensor = load_and_preprocess_nifti(str(nifti_file))
            input_tensor = input_tensor.to(device)

            # Predict
            with torch.no_grad():
                batch = {"image": input_tensor}
                logits = model(batch)
                probabilities = torch.softmax(logits, dim=1)

                pred_idx = probabilities.argmax(dim=1).item()
                predicted_class = classes[pred_idx]

                confidence_scores = {
                    classes[i]: float(probabilities[0, i])
                    for i in range(len(classes))
                }

            elapsed_time = time.time() - start_time

            result = {
                'file_path': str(nifti_file),
                'file_name': nifti_file.name,
                'ground_truth': ground_truth_label,
                'predicted_class': predicted_class,
                'confidence_AD': confidence_scores.get('AD', 0.0),
                'confidence_CN': confidence_scores.get('CN', 0.0),
                'inference_time_sec': elapsed_time,
                'correct': predicted_class == ground_truth_label
            }

            results.append(result)

            logger.info(f"  ✓ Predicted: {predicted_class} (AD: {confidence_scores.get('AD', 0):.2%}, CN: {confidence_scores.get('CN', 0):.2%})")

        except Exception as e:
            logger.error(f"  ✗ Failed to process {nifti_file.name}: {str(e)}")
            result = {
                'file_path': str(nifti_file),
                'file_name': nifti_file.name,
                'ground_truth': ground_truth_label,
                'predicted_class': 'ERROR',
                'confidence_AD': 0.0,
                'confidence_CN': 0.0,
                'inference_time_sec': 0.0,
                'correct': False,
                'error': str(e)
            }
            results.append(result)

    return results


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate accuracy metrics"""
    valid_results = [r for r in results if r['predicted_class'] != 'ERROR']

    if not valid_results:
        return {}

    total = len(valid_results)
    correct = sum(1 for r in valid_results if r['correct'])
    accuracy = correct / total if total > 0 else 0

    metrics = {
        'total': total,
        'correct': correct,
        'accuracy': accuracy
    }

    # Per-class metrics
    for class_name in ['AD', 'CN', 'MCI']:
        class_results = [r for r in valid_results if r['ground_truth'] == class_name]
        if class_results:
            class_correct = sum(1 for r in class_results if r['correct'])
            metrics[f'{class_name}_total'] = len(class_results)
            metrics[f'{class_name}_correct'] = class_correct
            metrics[f'{class_name}_accuracy'] = class_correct / len(class_results)

    return metrics


def save_results(results: List[Dict], output_path: str):
    """Save results to CSV"""
    if not results:
        logger.warning("No results to save")
        return

    fieldnames = ['file_name', 'file_path', 'ground_truth', 'predicted_class',
                  'confidence_AD', 'confidence_CN', 'inference_time_sec', 'correct']

    if any('error' in r for r in results):
        fieldnames.append('error')

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Results saved to: {output_path}")


def generate_summary_report(results: List[Dict], output_path: str):
    """Generate a summary report with overall statistics"""
    valid_results = [r for r in results if r['predicted_class'] != 'ERROR']

    if not valid_results:
        logger.warning("No valid results to summarize")
        return

    # Group results by ground truth
    groups = {}
    for r in valid_results:
        prog_type = r['ground_truth']
        if prog_type not in groups:
            groups[prog_type] = []
        groups[prog_type].append(r)

    # Calculate statistics for each group
    summary_data = []

    for prog_type in sorted(groups.keys()):
        group_results = groups[prog_type]
        total = len(group_results)

        # Count predictions
        pred_ad = sum(1 for r in group_results if r['predicted_class'] == 'AD')
        pred_cn = sum(1 for r in group_results if r['predicted_class'] == 'CN')

        # Calculate percentages
        pct_ad = (pred_ad / total * 100) if total > 0 else 0
        pct_cn = (pred_cn / total * 100) if total > 0 else 0

        # Average confidences
        avg_conf_ad = np.mean([r['confidence_AD'] for r in group_results])
        avg_conf_cn = np.mean([r['confidence_CN'] for r in group_results])

        # Accuracy
        correct = sum(1 for r in group_results if r.get('correct', False))
        accuracy = (correct / total * 100) if total > 0 else 0

        summary_data.append({
            'group': prog_type,
            'total_patients': total,
            'predicted_AD': pred_ad,
            'predicted_CN': pred_cn,
            'percent_AD': pct_ad,
            'percent_CN': pct_cn,
            'avg_confidence_AD': avg_conf_ad,
            'avg_confidence_CN': avg_conf_cn,
            'accuracy': accuracy
        })

    # Save to CSV
    fieldnames = ['group', 'total_patients', 'predicted_AD', 'predicted_CN',
                  'percent_AD', 'percent_CN', 'avg_confidence_AD', 'avg_confidence_CN', 'accuracy']

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_data)

    logger.info(f"Summary report saved to: {output_path}")

    return summary_data


def main():
    parser = argparse.ArgumentParser(description='Batch prediction using DenseNet3D')

    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to parent folder containing AD and CN subfolders')

    parser.add_argument('--output', '-o', type=str, default='batch_predictions_official.csv',
                       help='Output CSV file')

    parser.add_argument('--model', '-m', type=str,
                       default='models/best_model_densenet3D_epoch=14-step=3510.ckpt',
                       help='Path to model checkpoint')

    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path not found: {args.input}")
        sys.exit(1)

    ad_folder = input_path / 'AD'
    cn_folder = input_path / 'CN'
    mci_folder = input_path / 'MCI'

    if not ad_folder.exists() and not cn_folder.exists() and not mci_folder.exists():
        logger.error(f"No diagnostic folders (AD/CN/MCI) found in {input_path}")
        sys.exit(1)

    try:
        print("\n" + "="*80)
        print("🧠 BATCH ALZHEIMER'S PREDICTION - DenseNet3D (Official)")
        print("="*80 + "\n")

        # Load model
        model, classes = load_model(args.model, device=args.device)

        all_results = []

        # Process AD folder
        if ad_folder.exists():
            print(f"\n📁 Processing AD folder...")
            print("-" * 80)
            ad_results = predict_folder(model, classes, ad_folder, 'AD', device=args.device)
            all_results.extend(ad_results)

        # Process CN folder
        if cn_folder.exists():
            print(f"\n📁 Processing CN folder...")
            print("-" * 80)
            cn_results = predict_folder(model, classes, cn_folder, 'CN', device=args.device)
            all_results.extend(cn_results)

        # Process MCI folder
        if mci_folder.exists():
            print(f"\n📁 Processing MCI folder...")
            print("-" * 80)
            mci_results = predict_folder(model, classes, mci_folder, 'MCI', device=args.device)
            all_results.extend(mci_results)

        # Calculate metrics
        metrics = calculate_metrics(all_results)

        if metrics:
            print("\n" + "="*80)
            print("📊 SUMMARY METRICS")
            print("="*80)
            print(f"\nTotal predictions: {metrics['total']}")
            print(f"Correct predictions: {metrics['correct']}")
            print(f"Overall accuracy: {metrics['accuracy']:.2%}")

            if 'AD_total' in metrics:
                print(f"\nAD class:")
                print(f"  Total: {metrics['AD_total']}")
                print(f"  Correct: {metrics['AD_correct']}")
                print(f"  Accuracy: {metrics['AD_accuracy']:.2%}")

            if 'CN_total' in metrics:
                print(f"\nCN class:")
                print(f"  Total: {metrics['CN_total']}")
                print(f"  Correct: {metrics['CN_correct']}")
                print(f"  Accuracy: {metrics['CN_accuracy']:.2%}")

            if 'MCI_total' in metrics:
                print(f"\nMCI class:")
                print(f"  Total: {metrics['MCI_total']}")
                print(f"  Correct: {metrics['MCI_correct']}")
                print(f"  Accuracy: {metrics['MCI_accuracy']:.2%}")

        # Save detailed results
        print("\n" + "="*80)
        save_results(all_results, args.output)

        # Generate and save summary report
        summary_output = args.output.replace('.csv', '_summary.csv')
        print("\n📊 Generating summary report...")
        summary_data = generate_summary_report(all_results, summary_output)

        # Print summary to console
        if summary_data:
            print("\n" + "="*80)
            print("📈 PREDICTION SUMMARY BY GROUP")
            print("="*80)
            print(f"\n{'Group':<10} {'Total':>7} {'→AD':>6} {'→CN':>6} {'%AD':>7} {'%CN':>7} {'Accuracy':>10}")
            print("-" * 80)
            for row in summary_data:
                print(f"{row['group']:<10} {row['total_patients']:>7} "
                      f"{row['predicted_AD']:>6} {row['predicted_CN']:>6} "
                      f"{row['percent_AD']:>6.1f}% {row['percent_CN']:>6.1f}% "
                      f"{row['accuracy']:>9.1f}%")

        print("="*80 + "\n")

    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
