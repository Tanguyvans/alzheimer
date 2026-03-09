#!/usr/bin/env python3
"""
Test MedGemma with multi-view slices (2 coronal + 2 axial) + tabular data.
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import sys
sys.path.insert(0, 'medgemma')
from utils.slice_extractor import MultiViewSliceExtractor
import pandas as pd
import numpy as np

# Tabular feature formatting
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


def main():
    print("="*60)
    print("MULTI-VIEW + TABULAR TEST")
    print("2 Coronal + 2 Axial + Clinical Data")
    print("="*60)

    # Load model
    print("\nLoading MedGemma (bf16)...")
    processor = AutoProcessor.from_pretrained("google/medgemma-1.5-4b-it")
    model = AutoModelForImageTextToText.from_pretrained(
        "google/medgemma-1.5-4b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded!")

    # Setup extractor with good settings
    extractor = MultiViewSliceExtractor(
        n_coronal=2,
        n_axial=2,
        coronal_region=(0.45, 0.55),
        axial_region=(0.35, 0.45),
        output_size=448
    )

    # Load test data
    df = pd.read_csv('experiments/multimodal_fusion/data/combined_trajectory/test.csv')

    print("\n" + "="*60)
    print("TESTING ON 10 SAMPLES")
    print("="*60)

    correct = 0
    total = 10

    for i in range(total):
        row = df.iloc[i]
        true_label = "CN" if row['label'] == 0 else "AD"

        # Extract slices
        all_slices = extractor.extract_all(row['scan_path'])

        # Format clinical info
        clinical_info = format_clinical_info(row)

        # Build prompt with clinical data
        prompt = f"""Patient clinical information:
{clinical_info}

These MRI images show 2 coronal views (hippocampus region) and 2 axial views (showing ventricles).

Based on the clinical data and MRI findings (hippocampal atrophy, ventricular enlargement, cortical thinning), classify this patient as either cognitively normal (CN) or having Alzheimer's disease (AD).

Respond with only: CN or AD"""

        # Build message
        user_content = [{"type": "image", "image": img} for img in all_slices]
        user_content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": user_content}]

        # Process
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=all_slices, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

        input_len = inputs["input_ids"].shape[1]
        output_text = processor.decode(outputs[0, input_len:], skip_special_tokens=True).strip()

        # Parse prediction
        output_upper = output_text.upper()
        if "AD" in output_upper and "CN" not in output_upper:
            pred = "AD"
        elif "CN" in output_upper and "AD" not in output_upper:
            pred = "CN"
        else:
            pred = "?"

        is_correct = pred == true_label
        correct += int(is_correct)

        print(f"\nSample {i+1}: True={true_label}")
        print(f"  Clinical: Age={row.get('AGE', 'N/A')}, Gender={'M' if row.get('PTGENDER')==1 else 'F'}")
        print(f"  Output: '{output_text}'")
        print(f"  Prediction: {pred} {'✓' if is_correct else '✗'}")

    print("\n" + "="*60)
    print(f"ACCURACY: {correct}/{total} ({100*correct/total:.0f}%)")
    print("="*60)


if __name__ == "__main__":
    main()
