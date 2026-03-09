#!/usr/bin/env python3
"""
MedGemma Inference and Evaluation for Alzheimer's Classification

Supports:
- Zero-shot evaluation (base MedGemma)
- Fine-tuned model evaluation
- Single sample prediction

Usage:
    # Zero-shot evaluation
    python inference.py --mode zero-shot --samples 10

    # Evaluate fine-tuned model
    python inference.py --mode evaluate --checkpoint ./checkpoints/final

    # Predict single scan
    python inference.py --mode predict --scan /path/to/scan.nii.gz
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from dataset import MedGemmaAlzheimerDataset, DEFAULT_PROMPT
from utils.slice_extractor import SliceExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_prediction(output_text: str) -> int:
    """
    Parse model output text to class label.

    Args:
        output_text: Raw model output string

    Returns:
        0 for CN, 1 for AD, -1 for invalid/unparseable
    """
    output = output_text.strip().upper()

    # Check for exact matches first
    if output == "AD":
        return 1
    elif output == "CN":
        return 0

    # Check for partial matches
    if "AD" in output and "CN" not in output:
        return 1
    elif "CN" in output and "AD" not in output:
        return 0

    # Check for related terms
    if any(term in output for term in ["ALZHEIMER", "DEMENTIA", "DISEASE"]):
        return 1
    elif any(term in output for term in ["NORMAL", "HEALTHY", "COGNITIVELY NORMAL"]):
        return 0

    # Unable to parse
    logger.warning(f"Unable to parse prediction: '{output_text}'")
    return -1


def load_model(
    config: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
    use_quantization: bool = True
) -> Tuple[Any, Any]:
    """
    Load MedGemma model and processor.

    Args:
        config: Configuration dict
        checkpoint_path: Path to fine-tuned checkpoint (None for base model)
        use_quantization: Whether to use 4-bit quantization

    Returns:
        model, processor
    """
    model_name = config.get("model", {}).get("name", "google/medgemma-1.5-4b-it")

    logger.info(f"Loading model: {model_name}")
    if checkpoint_path:
        logger.info(f"Loading fine-tuned weights from: {checkpoint_path}")

    # Setup quantization
    bnb_config = None
    if use_quantization:
        quant_config = config.get("model", {}).get("quantization", {})
        compute_dtype_str = quant_config.get("bnb_4bit_compute_dtype", "bfloat16")
        compute_dtype = getattr(torch, compute_dtype_str, torch.bfloat16)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config.get("load_in_4bit", True),
            bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name)

    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load fine-tuned adapter if provided
    if checkpoint_path:
        checkpoint = Path(checkpoint_path)

        # Check if it's a PEFT adapter directory
        adapter_config_path = checkpoint / "adapter_config.json"
        if adapter_config_path.exists():
            logger.info("Loading PEFT adapter...")
            model = PeftModel.from_pretrained(model, checkpoint)
        else:
            # Try loading as full model
            logger.info("Loading full model checkpoint...")
            model = AutoModelForImageTextToText.from_pretrained(
                checkpoint,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

    model.eval()

    return model, processor


def predict_single(
    model,
    processor,
    slices: List[Image.Image],
    config: Dict[str, Any]
) -> Tuple[str, int]:
    """
    Make prediction for a single sample.

    Args:
        model: MedGemma model
        processor: MedGemma processor
        slices: List of PIL Images
        config: Configuration dict

    Returns:
        raw_output: Model's text output
        prediction: Parsed class label (0=CN, 1=AD, -1=invalid)
    """
    prompt = config.get("inference", {}).get("prompt", DEFAULT_PROMPT)

    # Build messages
    user_content = []
    for img in slices:
        user_content.append({"type": "image", "image": img})
    user_content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": user_content}]

    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Process inputs
    inputs = processor(
        text=text,
        images=slices,
        return_tensors="pt"
    )

    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    gen_config = config.get("inference", {})
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_config.get("max_new_tokens", 10),
            do_sample=gen_config.get("do_sample", False),
            temperature=gen_config.get("temperature", 1.0),
        )

    # Decode output (only the new tokens)
    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0, input_len:]
    raw_output = processor.decode(generated_tokens, skip_special_tokens=True)

    # Parse to class
    prediction = parse_prediction(raw_output)

    return raw_output, prediction


def evaluate(
    model,
    processor,
    csv_path: str,
    config: Dict[str, Any],
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate model on dataset.

    Args:
        model: MedGemma model
        processor: MedGemma processor
        csv_path: Path to evaluation CSV
        config: Configuration dict
        max_samples: Maximum samples to evaluate (None for all)

    Returns:
        Dict with metrics and predictions
    """
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)

    slice_config = config.get("data", {}).get("slice_extraction", {})
    extractor = SliceExtractor(
        view=slice_config.get("view", "coronal"),
        n_slices=slice_config.get("n_slices", 5),
        region_start=slice_config.get("region_start", 0.40),
        region_end=slice_config.get("region_end", 0.60),
        output_size=slice_config.get("output_size", 896)
    )

    predictions = []
    labels = []
    raw_outputs = []
    invalid_count = 0

    logger.info(f"Evaluating on {len(df)} samples...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        try:
            # Extract slices
            slices = extractor.extract_from_nifti(row['scan_path'])

            # Predict
            raw_output, pred = predict_single(model, processor, slices, config)

            raw_outputs.append(raw_output)
            labels.append(int(row['label']))

            if pred == -1:
                invalid_count += 1
                # Default to most common class on invalid
                predictions.append(0)
            else:
                predictions.append(pred)

        except Exception as e:
            logger.error(f"Error processing {row['scan_path']}: {e}")
            predictions.append(0)
            labels.append(int(row['label']))
            raw_outputs.append("ERROR")
            invalid_count += 1

    # Calculate metrics
    predictions = np.array(predictions)
    labels = np.array(labels)

    accuracy = accuracy_score(labels, predictions)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary')
    f1_macro = f1_score(labels, predictions, average='macro')
    cm = confusion_matrix(labels, predictions)

    # Calculate sensitivity/specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    results = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "f1_score": f1,
        "f1_macro": f1_macro,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": cm.tolist(),
        "invalid_predictions": invalid_count,
        "total_samples": len(df),
        "predictions": predictions.tolist(),
        "labels": labels.tolist(),
        "raw_outputs": raw_outputs,
    }

    return results


def print_results(results: Dict[str, Any]):
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTotal Samples: {results['total_samples']}")
    print(f"Invalid Predictions: {results['invalid_predictions']}")
    print(f"\nMetrics:")
    print(f"  Accuracy:          {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Balanced Accuracy: {results['balanced_accuracy']:.4f} ({results['balanced_accuracy']*100:.2f}%)")
    print(f"  F1 Score (AD):     {results['f1_score']:.4f}")
    print(f"  F1 Score (Macro):  {results['f1_macro']:.4f}")
    print(f"  Sensitivity (AD):  {results['sensitivity']:.4f}")
    print(f"  Specificity (CN):  {results['specificity']:.4f}")

    cm = np.array(results['confusion_matrix'])
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              CN    AD")
    print(f"  Actual CN   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"  Actual AD   {cm[1,0]:4d}  {cm[1,1]:4d}")
    print("=" * 60)


def zero_shot_evaluation(
    config: Dict[str, Any],
    config_path: str,
    max_samples: int = 10
):
    """Run zero-shot evaluation on base MedGemma."""
    logger.info("Running zero-shot evaluation...")

    model, processor = load_model(config, checkpoint_path=None)

    config_dir = Path(config_path).parent
    test_csv = str(config_dir / config.get("data", {}).get("test_csv", "../data/combined/mri_cn_ad_test.csv"))

    results = evaluate(model, processor, test_csv, config, max_samples)
    print_results(results)

    return results


def fine_tuned_evaluation(
    config: Dict[str, Any],
    config_path: str,
    checkpoint_path: str,
    max_samples: Optional[int] = None
):
    """Evaluate fine-tuned model."""
    logger.info("Evaluating fine-tuned model...")

    model, processor = load_model(config, checkpoint_path=checkpoint_path)

    config_dir = Path(config_path).parent
    test_csv = str(config_dir / config.get("data", {}).get("test_csv", "../data/combined/mri_cn_ad_test.csv"))

    results = evaluate(model, processor, test_csv, config, max_samples)
    print_results(results)

    # Save results
    output_dir = Path(checkpoint_path).parent if checkpoint_path else config_dir
    results_path = output_dir / "evaluation_results.yaml"

    # Convert numpy types for YAML
    save_results = {k: v for k, v in results.items() if k not in ['predictions', 'labels', 'raw_outputs']}

    with open(results_path, 'w') as f:
        yaml.dump(save_results, f, default_flow_style=False)

    logger.info(f"Results saved to {results_path}")

    return results


def predict_scan(
    config: Dict[str, Any],
    scan_path: str,
    checkpoint_path: Optional[str] = None
):
    """Predict for a single MRI scan."""
    logger.info(f"Predicting for: {scan_path}")

    model, processor = load_model(config, checkpoint_path=checkpoint_path)

    slice_config = config.get("data", {}).get("slice_extraction", {})
    extractor = SliceExtractor(
        view=slice_config.get("view", "coronal"),
        n_slices=slice_config.get("n_slices", 5),
        region_start=slice_config.get("region_start", 0.40),
        region_end=slice_config.get("region_end", 0.60),
        output_size=slice_config.get("output_size", 896)
    )

    slices = extractor.extract_from_nifti(scan_path)
    raw_output, prediction = predict_single(model, processor, slices, config)

    label_map = {0: "CN (Cognitively Normal)", 1: "AD (Alzheimer's Disease)", -1: "Invalid"}

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Scan: {scan_path}")
    print(f"Model output: {raw_output}")
    print(f"Prediction: {label_map.get(prediction, 'Unknown')}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="MedGemma inference and evaluation for Alzheimer's classification"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["zero-shot", "evaluate", "predict"],
        help="Inference mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to fine-tuned checkpoint (for evaluate/predict modes)"
    )
    parser.add_argument(
        "--scan",
        type=str,
        default=None,
        help="Path to MRI scan (for predict mode)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate"
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent / args.config

    if not config_path.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(str(config_path))

    logger.info("=" * 60)
    logger.info("MedGemma Inference")
    logger.info(f"Mode: {args.mode}")
    logger.info("=" * 60)

    if args.mode == "zero-shot":
        zero_shot_evaluation(config, str(config_path), args.samples or 10)

    elif args.mode == "evaluate":
        if not args.checkpoint:
            logger.error("--checkpoint required for evaluate mode")
            sys.exit(1)
        fine_tuned_evaluation(config, str(config_path), args.checkpoint, args.samples)

    elif args.mode == "predict":
        if not args.scan:
            logger.error("--scan required for predict mode")
            sys.exit(1)
        predict_scan(config, args.scan, args.checkpoint)


if __name__ == "__main__":
    main()
