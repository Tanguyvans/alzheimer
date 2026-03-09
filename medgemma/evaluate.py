#!/usr/bin/env python3
"""
Evaluate MedGemma on Alzheimer's Classification

Supports:
- Zero-shot evaluation (no fine-tuning)
- Fine-tuned model evaluation (with LoRA adapters)

IMPORTANT: Uses bf16 (no quantization) for inference - quantization breaks generation.
"""

import torch
import yaml
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix, classification_report
)
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
import sys
sys.path.insert(0, 'medgemma')
from utils.slice_extractor import MultiViewSliceExtractor


TABULAR_FEATURES = {
    'AGE': 'Age',
    'PTGENDER': 'Gender',
    'PTEDUCAT': 'Years of education',
    'CATANIMSC': 'Category fluency (animals)',
    'TRAASCOR': 'Trail Making Test A (seconds)',
    'TRABSCOR': 'Trail Making Test B (seconds)',
    'DSPANFOR': 'Digit span forward',
    'DSPANBAC': 'Digit span backward',
    'BNTTOTAL': 'Boston Naming Test score',
}


def format_clinical_info(row):
    """Format tabular features as readable string."""
    info_parts = []
    for feature, name in TABULAR_FEATURES.items():
        if feature not in row or pd.isna(row[feature]):
            continue
        value = row[feature]
        if feature == 'PTGENDER':
            value = 'Male' if value == 1 else 'Female'
        elif feature == 'AGE':
            value = f"{value:.0f} years"
        elif feature == 'PTEDUCAT':
            value = f"{value:.0f} years"
        elif feature in ['TRAASCOR', 'TRABSCOR']:
            value = f"{value:.0f}s"
        elif isinstance(value, float):
            value = f"{value:.1f}"
        info_parts.append(f"- {name}: {value}")
    return "\n".join(info_parts) if info_parts else "No clinical data available"


def parse_prediction(output_text):
    """Parse model output to get CN or AD prediction."""
    upper = output_text.upper().strip()

    # Direct match
    if upper == "CN" or upper == "AD":
        return upper

    # Look for classification in output
    if "AD" in upper and "CN" not in upper:
        return "AD"
    if "CN" in upper and "AD" not in upper:
        return "CN"

    return "?"


def load_model(checkpoint_path=None, use_quantization=False):
    """
    Load MedGemma model for evaluation.

    Args:
        checkpoint_path: Path to LoRA checkpoint (None for zero-shot)
        use_quantization: Whether to use quantization (NOT recommended for inference)

    Returns:
        model, processor
    """
    print(f"Loading MedGemma...")

    processor = AutoProcessor.from_pretrained("google/medgemma-1.5-4b-it")

    if use_quantization:
        print("WARNING: Quantization may break generation. Use bf16 for reliable results.")
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForImageTextToText.from_pretrained(
            "google/medgemma-1.5-4b-it",
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        # Use bf16 - this works reliably
        model = AutoModelForImageTextToText.from_pretrained(
            "google/medgemma-1.5-4b-it",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    # Load LoRA adapters if checkpoint provided
    if checkpoint_path:
        print(f"Loading LoRA adapters from: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()  # Merge for faster inference

    model.eval()
    print("Model loaded!")

    return model, processor


def evaluate(
    model,
    processor,
    test_csv,
    use_tabular=True,
    max_samples=None,
    verbose=True
):
    """
    Evaluate model on test set.

    Args:
        model: MedGemma model
        processor: MedGemma processor
        test_csv: Path to test CSV
        use_tabular: Include clinical features in prompt
        max_samples: Limit samples for testing
        verbose: Print individual predictions

    Returns:
        Dict with metrics
    """
    df = pd.read_csv(test_csv)
    if max_samples:
        df = df.head(max_samples)

    # Setup extractor
    extractor = MultiViewSliceExtractor(
        n_coronal=2,
        n_axial=2,
        coronal_region=(0.45, 0.55),
        axial_region=(0.35, 0.45),
        output_size=448
    )

    # Prompt templates
    prompt_with_clinical = """Patient clinical information:
{clinical_info}

Brain MRI: 2 coronal slices (hippocampus) + 2 axial slices (ventricles).

Classify this patient as:
- CN: Cognitively Normal
- AD: Alzheimer's Disease

Respond with only: CN or AD"""

    prompt_simple = """Brain MRI: 2 coronal slices (hippocampus) + 2 axial slices (ventricles).

Classify as CN (Cognitively Normal) or AD (Alzheimer's Disease).

Respond with only: CN or AD"""

    y_true = []
    y_pred = []
    outputs = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        true_label = "CN" if row['label'] == 0 else "AD"
        y_true.append(true_label)

        # Extract slices
        try:
            slices = extractor.extract_all(row['scan_path'])
        except Exception as e:
            print(f"Error loading {row['scan_path']}: {e}")
            y_pred.append("?")
            outputs.append("ERROR")
            continue

        # Build prompt
        if use_tabular:
            clinical_info = format_clinical_info(row)
            prompt = prompt_with_clinical.format(clinical_info=clinical_info)
        else:
            prompt = prompt_simple

        # Build messages
        user_content = [{"type": "image", "image": img} for img in slices]
        user_content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": user_content}]

        # Process
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=slices, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)

        input_len = inputs["input_ids"].shape[1]
        output_text = processor.decode(output_ids[0, input_len:], skip_special_tokens=True).strip()
        outputs.append(output_text)

        # Parse prediction
        pred = parse_prediction(output_text)
        y_pred.append(pred)

        if verbose:
            status = "correct" if pred == true_label else "WRONG"
            print(f"  {idx+1}: True={true_label}, Pred={pred} ({status}) | Output: '{output_text}'")

    # Compute metrics
    # Filter out invalid predictions for metrics
    valid_mask = [p != "?" for p in y_pred]
    y_true_valid = [y for y, v in zip(y_true, valid_mask) if v]
    y_pred_valid = [p for p, v in zip(y_pred, valid_mask) if v]

    if len(y_true_valid) == 0:
        print("No valid predictions!")
        return {}

    # Map to numeric
    label_map = {"CN": 0, "AD": 1}
    y_true_num = [label_map[y] for y in y_true_valid]
    y_pred_num = [label_map[y] for y in y_pred_valid]

    metrics = {
        "accuracy": accuracy_score(y_true_num, y_pred_num),
        "balanced_accuracy": balanced_accuracy_score(y_true_num, y_pred_num),
        "f1_macro": f1_score(y_true_num, y_pred_num, average="macro"),
        "f1_cn": f1_score(y_true_num, y_pred_num, pos_label=0),
        "f1_ad": f1_score(y_true_num, y_pred_num, pos_label=1),
        "precision_cn": precision_score(y_true_num, y_pred_num, pos_label=0),
        "precision_ad": precision_score(y_true_num, y_pred_num, pos_label=1),
        "recall_cn": recall_score(y_true_num, y_pred_num, pos_label=0),  # Specificity for AD
        "recall_ad": recall_score(y_true_num, y_pred_num, pos_label=1),  # Sensitivity for AD
        "invalid_predictions": len(y_true) - len(y_true_valid),
        "total_samples": len(y_true),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true_num, y_pred_num)
    metrics["confusion_matrix"] = cm

    return metrics, y_true, y_pred, outputs


def main():
    parser = argparse.ArgumentParser(description="Evaluate MedGemma on Alzheimer's Classification")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to LoRA checkpoint")
    parser.add_argument("--test_csv", type=str,
                        default="experiments/multimodal_fusion/data/combined_trajectory/test.csv")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for testing")
    parser.add_argument("--no_tabular", action="store_true", help="Don't include clinical features")
    parser.add_argument("--quiet", action="store_true", help="Don't print individual predictions")
    args = parser.parse_args()

    print("="*60)
    print("MedGemma Alzheimer's Classification Evaluation")
    print("="*60)

    if args.checkpoint:
        print(f"Mode: Fine-tuned (checkpoint: {args.checkpoint})")
    else:
        print("Mode: Zero-shot")

    # Load model
    model, processor = load_model(args.checkpoint, use_quantization=False)

    # Evaluate
    metrics, y_true, y_pred, outputs = evaluate(
        model, processor,
        args.test_csv,
        use_tabular=not args.no_tabular,
        max_samples=args.max_samples,
        verbose=not args.quiet
    )

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Invalid Predictions: {metrics['invalid_predictions']}")
    print(f"\nAccuracy: {metrics['accuracy']*100:.1f}%")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']*100:.1f}%")
    print(f"F1 Macro: {metrics['f1_macro']*100:.1f}%")
    print(f"\nPer-class metrics:")
    print(f"  CN - F1: {metrics['f1_cn']*100:.1f}%, Precision: {metrics['precision_cn']*100:.1f}%, Recall: {metrics['recall_cn']*100:.1f}%")
    print(f"  AD - F1: {metrics['f1_ad']*100:.1f}%, Precision: {metrics['precision_ad']*100:.1f}%, Recall: {metrics['recall_ad']*100:.1f}%")
    print(f"\nConfusion Matrix:")
    print(f"           Pred CN  Pred AD")
    print(f"  True CN    {metrics['confusion_matrix'][0,0]:4d}     {metrics['confusion_matrix'][0,1]:4d}")
    print(f"  True AD    {metrics['confusion_matrix'][1,0]:4d}     {metrics['confusion_matrix'][1,1]:4d}")


if __name__ == "__main__":
    main()
