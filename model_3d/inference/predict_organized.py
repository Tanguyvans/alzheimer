#!/usr/bin/env python3
"""
Batch Prediction on Organized ADNI Dataset using DenseNet3D

Processes all patient groups from organized dataset:
- MCI→AD: MCI patients who progressed to Alzheimer's
- MCI→CN: MCI patients who reverted to cognitively normal
- MCI→MCI: MCI patients who remained stable
- CN→CN: Healthy patients who remained healthy
- AD→AD: AD patients who remained with AD

Generates detailed predictions CSV and summary statistics.
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

    # Filter out hidden files
    nifti_files = [f for f in nifti_files if not f.name.startswith('._') and not f.name.startswith('.')]

    return sorted(nifti_files)


def extract_patient_id(filename: str) -> str:
    """Extract patient ID from filename (e.g., '002_S_0729_bl_...' -> '002_S_0729')"""
    parts = filename.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:3])  # e.g., 002_S_0729
    return filename


def predict_folder(model, classes, folder_path: Path, progression_label: str, device='cpu') -> List[Dict]:
    """Predict all files in a folder"""
    logger.info(f"\nProcessing folder: {folder_path.name}")

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

            # Extract patient ID
            patient_id = extract_patient_id(nifti_file.name)

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

            # Map progression to expected outcome
            # MCI->AD patients should ideally predict as AD
            # MCI->CN patients should ideally predict as CN
            # MCI->MCI is ambiguous (could be either)
            if progression_label == 'MCI->AD':
                expected = 'AD'
            elif progression_label == 'MCI->CN':
                expected = 'CN'
            else:  # MCI->MCI
                expected = 'MCI'  # We'll mark as "uncertain"

            correct = (predicted_class == expected) if expected != 'MCI' else None

            result = {
                'patient_id': patient_id,
                'file_name': nifti_file.name,
                'file_path': str(nifti_file),
                'true_progression': progression_label,
                'expected_class': expected,
                'predicted_class': predicted_class,
                'confidence_AD': confidence_scores.get('AD', 0.0),
                'confidence_CN': confidence_scores.get('CN', 0.0),
                'inference_time_sec': elapsed_time,
                'correct': correct
            }

            results.append(result)

            logger.info(f"  ✓ Predicted: {predicted_class} (AD: {confidence_scores.get('AD', 0):.2%}, CN: {confidence_scores.get('CN', 0):.2%})")

        except Exception as e:
            logger.error(f"  ✗ Failed to process {nifti_file.name}: {str(e)}")
            result = {
                'patient_id': extract_patient_id(nifti_file.name),
                'file_name': nifti_file.name,
                'file_path': str(nifti_file),
                'true_progression': progression_label,
                'expected_class': '',
                'predicted_class': 'ERROR',
                'confidence_AD': 0.0,
                'confidence_CN': 0.0,
                'inference_time_sec': 0.0,
                'correct': False,
                'error': str(e)
            }
            results.append(result)

    return results


def calculate_mci_metrics(results: List[Dict]) -> Dict:
    """Calculate MCI progression prediction metrics"""
    valid_results = [r for r in results if r['predicted_class'] != 'ERROR']

    if not valid_results:
        return {}

    metrics = {}

    # Overall stats
    metrics['total'] = len(valid_results)

    # MCI->AD metrics (should predict AD)
    mci_to_ad = [r for r in valid_results if r['true_progression'] == 'MCI->AD']
    if mci_to_ad:
        predicted_ad = sum(1 for r in mci_to_ad if r['predicted_class'] == 'AD')
        metrics['MCI->AD_total'] = len(mci_to_ad)
        metrics['MCI->AD_predicted_AD'] = predicted_ad
        metrics['MCI->AD_predicted_CN'] = len(mci_to_ad) - predicted_ad
        metrics['MCI->AD_accuracy'] = predicted_ad / len(mci_to_ad)
        metrics['MCI->AD_avg_confidence_AD'] = np.mean([r['confidence_AD'] for r in mci_to_ad])

    # MCI->CN metrics (should predict CN)
    mci_to_cn = [r for r in valid_results if r['true_progression'] == 'MCI->CN']
    if mci_to_cn:
        predicted_cn = sum(1 for r in mci_to_cn if r['predicted_class'] == 'CN')
        metrics['MCI->CN_total'] = len(mci_to_cn)
        metrics['MCI->CN_predicted_CN'] = predicted_cn
        metrics['MCI->CN_predicted_AD'] = len(mci_to_cn) - predicted_cn
        metrics['MCI->CN_accuracy'] = predicted_cn / len(mci_to_cn)
        metrics['MCI->CN_avg_confidence_CN'] = np.mean([r['confidence_CN'] for r in mci_to_cn])

    # MCI->MCI metrics (distribution)
    mci_to_mci = [r for r in valid_results if r['true_progression'] == 'MCI->MCI']
    if mci_to_mci:
        predicted_ad = sum(1 for r in mci_to_mci if r['predicted_class'] == 'AD')
        predicted_cn = sum(1 for r in mci_to_mci if r['predicted_class'] == 'CN')
        metrics['MCI->MCI_total'] = len(mci_to_mci)
        metrics['MCI->MCI_predicted_AD'] = predicted_ad
        metrics['MCI->MCI_predicted_CN'] = predicted_cn
        metrics['MCI->MCI_percent_AD'] = predicted_ad / len(mci_to_mci)
        metrics['MCI->MCI_percent_CN'] = predicted_cn / len(mci_to_mci)

    return metrics


def save_results(results: List[Dict], output_path: str):
    """Save results to CSV"""
    if not results:
        logger.warning("No results to save")
        return

    fieldnames = ['patient_id', 'file_name', 'true_progression', 'expected_class',
                  'predicted_class', 'confidence_AD', 'confidence_CN',
                  'inference_time_sec', 'correct', 'file_path']

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

    # Group results by progression type
    groups = {}
    for r in valid_results:
        prog_type = r['true_progression']
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

        # Accuracy (only for groups with expected outcome)
        accuracy = None
        if 'correct' in group_results[0] and group_results[0]['correct'] is not None:
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
            'accuracy': accuracy if accuracy is not None else 'N/A'
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
    parser = argparse.ArgumentParser(description='Batch Prediction on Organized ADNI Dataset using DenseNet3D')

    parser.add_argument('--input', '-i', type=str,
                       default='/Volumes/KINGSTON/ADNI-MCI-organized',
                       help='Path to organized dataset folder (contains AD->AD, CN->CN, MCI->AD, MCI->CN, MCI->MCI)')

    parser.add_argument('--output', '-o', type=str, default='organized_predictions.csv',
                       help='Output CSV file for detailed predictions')

    parser.add_argument('--model', '-m', type=str,
                       default='models/best_model_densenet3D_epoch=14-step=3510.ckpt',
                       help='Path to model checkpoint')

    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to use')

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path not found: {args.input}")
        sys.exit(1)

    # Define all folders
    folders = {
        'MCI->AD': input_path / 'MCI->AD',
        'MCI->CN': input_path / 'MCI->CN',
        'MCI->MCI': input_path / 'MCI->MCI',
        'CN->CN': input_path / 'CN->CN',
        'AD->AD': input_path / 'AD->AD'
    }

    # Check if at least one folder exists
    existing_folders = {k: v for k, v in folders.items() if v.exists()}
    if not existing_folders:
        logger.error(f"No patient folders found in {input_path}")
        sys.exit(1)

    try:
        print("\n" + "="*80)
        print("🧠 ORGANIZED DATASET PREDICTION - DenseNet3D")
        print("="*80 + "\n")

        # Load model
        model, classes = load_model(args.model, device=args.device)

        all_results = []

        # Process each folder
        folder_descriptions = {
            'AD->AD': 'AD patients (stable)',
            'CN->CN': 'CN patients (stable healthy)',
            'MCI->AD': 'MCI patients who progressed to AD',
            'MCI->CN': 'MCI patients who reverted to CN',
            'MCI->MCI': 'MCI patients (stable)'
        }

        for folder_name, folder_path in existing_folders.items():
            description = folder_descriptions.get(folder_name, folder_name)
            print(f"\n📁 Processing {folder_name}: {description}...")
            print("-" * 80)
            results = predict_folder(model, classes, folder_path, folder_name, device=args.device)
            all_results.extend(results)

        # Print completion message
        print("\n" + "="*80)
        print(f"✅ Processed {len(all_results)} total scans")
        print("="*80)

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
            print(f"\n{'Group':<15} {'Total':>7} {'→AD':>6} {'→CN':>6} {'%AD':>7} {'%CN':>7} {'Accuracy':>10}")
            print("-" * 80)
            for row in summary_data:
                acc_str = f"{row['accuracy']:.1f}%" if isinstance(row['accuracy'], (int, float)) else row['accuracy']
                print(f"{row['group']:<15} {row['total_patients']:>7} "
                      f"{row['predicted_AD']:>6} {row['predicted_CN']:>6} "
                      f"{row['percent_AD']:>6.1f}% {row['percent_CN']:>6.1f}% "
                      f"{acc_str:>10}")

            print("\n" + "="*80)
            print("💡 INTERPRETATION GUIDE:")
            print("="*80)
            print("  • CN→CN: Should predict CN (healthy patients staying healthy)")
            print("  • AD→AD: Should predict AD (AD patients remaining with AD)")
            print("  • MCI→AD: Should predict AD (MCI patients who progressed)")
            print("  • MCI→CN: Should predict CN (MCI patients who improved)")
            print("  • MCI→MCI: No clear expectation (stable MCI, distribution informative)")
            print("")

        print("="*80 + "\n")

    except Exception as e:
        logger.error(f"MCI progression prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
