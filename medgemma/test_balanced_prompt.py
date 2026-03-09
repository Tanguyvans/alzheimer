#!/usr/bin/env python3
"""
Test a balanced prompt that includes clinical data.
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import sys
sys.path.insert(0, 'medgemma')
from utils.slice_extractor import MultiViewSliceExtractor
import pandas as pd

TABULAR_FEATURES = {
    'AGE': 'Age',
    'PTGENDER': 'Gender',
    'PTEDUCAT': 'Years of education',
    'CATANIMSC': 'Category fluency (animals)',
    'TRAASCOR': 'Trail Making Test A (seconds)',
    'TRABSCOR': 'Trail Making Test B (seconds)',
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
    upper = output_text.upper()

    # Look for explicit classification patterns
    if "**AD**" in upper or "CLASSIFICATION: AD" in upper or "FINAL: AD" in upper:
        return "AD"
    if "**CN**" in upper or "CLASSIFICATION: CN" in upper or "FINAL: CN" in upper:
        return "CN"
    if "CONSISTENT WITH ALZHEIMER" in upper or "SUGGESTS AD" in upper or "INDICATES AD" in upper:
        return "AD"
    if "COGNITIVELY NORMAL" in upper or "NORMAL AGING" in upper or "CONSISTENT WITH CN" in upper:
        return "CN"

    # Check last lines for final answer
    lines = [l.strip() for l in output_text.strip().split('\n') if l.strip()]
    for line in reversed(lines[-3:]):
        line_upper = line.upper()
        if "AD" in line_upper and "CN" not in line_upper:
            return "AD"
        if "CN" in line_upper and "AD" not in line_upper:
            return "CN"

    return "?"


def main():
    print("="*60)
    print("TESTING BALANCED PROMPT WITH CLINICAL DATA")
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

    # Setup extractor
    extractor = MultiViewSliceExtractor(
        n_coronal=2,
        n_axial=2,
        coronal_region=(0.45, 0.55),
        axial_region=(0.35, 0.45),
        output_size=448
    )

    # Balanced prompt
    balanced_prompt = """You are a neuroradiologist assessing a patient for possible Alzheimer's disease.

Patient clinical information:
{clinical_info}

Brain MRI findings to evaluate:
- These images show 2 coronal views (hippocampus region) and 2 axial views (ventricles)

Assessment criteria:
1. Hippocampal volume relative to age-matched norms
2. Ventricular size relative to age-matched norms
3. Cortical sulcal widening
4. Integration with cognitive test results

Important: Many elderly patients have some degree of age-related atrophy. Only classify as AD if findings exceed normal age-related changes AND are consistent with the clinical presentation.

Classification:
- CN (Cognitively Normal): Imaging within normal limits for age, cognitive tests normal/mildly affected
- AD (Alzheimer's Disease): Disproportionate medial temporal atrophy, significant cognitive impairment

Based on imaging AND clinical data, what is your assessment?
End with: Classification: CN or Classification: AD"""

    # Load test data
    df = pd.read_csv('experiments/multimodal_fusion/data/combined_trajectory/test.csv')

    # Get 10 CN and 10 AD samples
    cn_samples = df[df['label'] == 0].head(10)
    ad_samples = df[df['label'] == 1].head(10)

    cn_correct = 0
    ad_correct = 0

    # Test CN samples
    print("\n--- CN Samples (10) ---")
    for i, (_, row) in enumerate(cn_samples.iterrows()):
        all_slices = extractor.extract_all(row['scan_path'])
        clinical_info = format_clinical_info(row)
        prompt = balanced_prompt.format(clinical_info=clinical_info)

        user_content = [{"type": "image", "image": img} for img in all_slices]
        user_content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": user_content}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=all_slices, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=400, do_sample=False)

        input_len = inputs["input_ids"].shape[1]
        output_text = processor.decode(outputs[0, input_len:], skip_special_tokens=True).strip()

        pred = parse_prediction(output_text)
        is_correct = pred == "CN"
        cn_correct += int(is_correct)

        status = "correct" if is_correct else "WRONG"
        print(f"  CN Sample {i+1}: Pred={pred} ({status})")
        if not is_correct:
            print(f"    Clinical: Age={row.get('AGE', 'N/A')}, TrailA={row.get('TRAASCOR', 'N/A')}")

    # Test AD samples
    print("\n--- AD Samples (10) ---")
    for i, (_, row) in enumerate(ad_samples.iterrows()):
        all_slices = extractor.extract_all(row['scan_path'])
        clinical_info = format_clinical_info(row)
        prompt = balanced_prompt.format(clinical_info=clinical_info)

        user_content = [{"type": "image", "image": img} for img in all_slices]
        user_content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": user_content}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=all_slices, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=400, do_sample=False)

        input_len = inputs["input_ids"].shape[1]
        output_text = processor.decode(outputs[0, input_len:], skip_special_tokens=True).strip()

        pred = parse_prediction(output_text)
        is_correct = pred == "AD"
        ad_correct += int(is_correct)

        status = "correct" if is_correct else "WRONG"
        print(f"  AD Sample {i+1}: Pred={pred} ({status})")
        if not is_correct:
            print(f"    Clinical: Age={row.get('AGE', 'N/A')}, TrailA={row.get('TRAASCOR', 'N/A')}")

    # Summary
    cn_acc = cn_correct / 10 * 100
    ad_acc = ad_correct / 10 * 100
    bal_acc = (cn_acc + ad_acc) / 2
    total_acc = (cn_correct + ad_correct) / 20 * 100

    print("\n" + "="*60)
    print("SUMMARY - Balanced Prompt with Clinical Data")
    print("="*60)
    print(f"  CN Accuracy: {cn_correct}/10 ({cn_acc:.0f}%)")
    print(f"  AD Accuracy: {ad_correct}/10 ({ad_acc:.0f}%)")
    print(f"  Balanced Accuracy: {bal_acc:.1f}%")
    print(f"  Total Accuracy: {total_acc:.0f}%")


if __name__ == "__main__":
    main()
