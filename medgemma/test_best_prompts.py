#!/usr/bin/env python3
"""
Test the best-performing prompts on both CN and AD samples.
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import sys
sys.path.insert(0, 'medgemma')
from utils.slice_extractor import MultiViewSliceExtractor
import pandas as pd

PROMPTS = {
    "pathology_focused": """You are examining brain MRI images from an elderly patient.
These show 2 coronal views (hippocampus region) and 2 axial views (ventricles).

Look carefully for signs of neurodegeneration:
1. Hippocampal volume - is there asymmetric or bilateral atrophy?
2. Temporal horn enlargement - are the temporal horns of lateral ventricles dilated?
3. Medial temporal lobe - is there widening of the choroidal fissure?
4. Cortical sulci - is there widening suggesting cortical atrophy?
5. Ventricular size - is there ventricular enlargement (ventriculomegaly)?

Based on these findings, is this brain more consistent with:
- Normal aging (CN)
- Alzheimer's disease (AD)

Provide your analysis and final classification.""",

    "binary_direct": """Brain MRI: 2 coronal + 2 axial slices.

This patient has either:
- CN: Cognitively Normal - age-appropriate brain volume
- AD: Alzheimer's Disease - pathological atrophy

Key AD indicators: hippocampal atrophy, enlarged ventricles, cortical thinning.

Look at the hippocampus size and ventricle size. Classify as CN or AD.
Respond with your assessment and final answer.""",
}


def parse_prediction(output_text):
    """Parse model output to get CN or AD prediction."""
    upper = output_text.upper()

    # Look for explicit classification patterns
    if "**AD**" in upper or "**AD (" in upper or ": AD" in upper.replace(" ", ""):
        return "AD"
    if "**CN**" in upper or "**CN (" in upper or ": CN" in upper.replace(" ", ""):
        return "CN"

    # Check ending
    lines = output_text.strip().split('\n')
    last_line = lines[-1].upper()
    if "AD" in last_line and "CN" not in last_line:
        return "AD"
    if "CN" in last_line and "AD" not in last_line:
        return "CN"

    # Fallback: count mentions
    ad_count = upper.count(" AD ") + upper.count(" AD.") + upper.count("(AD)")
    cn_count = upper.count(" CN ") + upper.count(" CN.") + upper.count("(CN)")

    if ad_count > cn_count:
        return "AD"
    elif cn_count > ad_count:
        return "CN"

    return "?"


def main():
    print("="*60)
    print("TESTING BEST PROMPTS ON CN AND AD")
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

    # Load test data
    df = pd.read_csv('experiments/multimodal_fusion/data/combined_trajectory/test.csv')

    # Get 10 CN and 10 AD samples
    cn_samples = df[df['label'] == 0].head(10)
    ad_samples = df[df['label'] == 1].head(10)

    results = {prompt_name: {"cn_correct": 0, "ad_correct": 0} for prompt_name in PROMPTS}

    for prompt_name, prompt in PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"TESTING PROMPT: {prompt_name}")
        print(f"{'='*60}")

        # Test CN samples
        print("\n--- CN Samples (10) ---")
        for i, (_, row) in enumerate(cn_samples.iterrows()):
            all_slices = extractor.extract_all(row['scan_path'])

            user_content = [{"type": "image", "image": img} for img in all_slices]
            user_content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": user_content}]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, images=all_slices, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)

            input_len = inputs["input_ids"].shape[1]
            output_text = processor.decode(outputs[0, input_len:], skip_special_tokens=True).strip()

            pred = parse_prediction(output_text)
            is_correct = pred == "CN"
            results[prompt_name]["cn_correct"] += int(is_correct)

            status = "correct" if is_correct else "WRONG"
            print(f"  CN Sample {i+1}: Pred={pred} ({status})")

        # Test AD samples
        print("\n--- AD Samples (10) ---")
        for i, (_, row) in enumerate(ad_samples.iterrows()):
            all_slices = extractor.extract_all(row['scan_path'])

            user_content = [{"type": "image", "image": img} for img in all_slices]
            user_content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": user_content}]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, images=all_slices, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)

            input_len = inputs["input_ids"].shape[1]
            output_text = processor.decode(outputs[0, input_len:], skip_special_tokens=True).strip()

            pred = parse_prediction(output_text)
            is_correct = pred == "AD"
            results[prompt_name]["ad_correct"] += int(is_correct)

            status = "correct" if is_correct else "WRONG"
            print(f"  AD Sample {i+1}: Pred={pred} ({status})")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for prompt_name, res in results.items():
        cn_acc = res["cn_correct"] / 10 * 100
        ad_acc = res["ad_correct"] / 10 * 100
        bal_acc = (cn_acc + ad_acc) / 2
        total_acc = (res["cn_correct"] + res["ad_correct"]) / 20 * 100

        print(f"\n{prompt_name}:")
        print(f"  CN Accuracy: {res['cn_correct']}/10 ({cn_acc:.0f}%)")
        print(f"  AD Accuracy: {res['ad_correct']}/10 ({ad_acc:.0f}%)")
        print(f"  Balanced Accuracy: {bal_acc:.1f}%")
        print(f"  Total Accuracy: {total_acc:.0f}%")


if __name__ == "__main__":
    main()
