#!/usr/bin/env python3
"""
Analyze MedGemma's bias in AD detection.

Tests different prompt strategies to understand why AD cases are misclassified.
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import sys
sys.path.insert(0, 'medgemma')
from utils.slice_extractor import MultiViewSliceExtractor
import pandas as pd

def main():
    print("="*60)
    print("ANALYZING AD DETECTION BIAS")
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

    # Setup extractor
    extractor = MultiViewSliceExtractor(
        n_coronal=2,
        n_axial=2,
        coronal_region=(0.45, 0.55),
        axial_region=(0.35, 0.45),
        output_size=448
    )

    # Load test data - get clear AD cases
    df = pd.read_csv('experiments/multimodal_fusion/data/combined_trajectory/test.csv')

    # Get 3 AD samples
    ad_samples = df[df['label'] == 1].head(3)

    # Different prompts to test
    prompts = {
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

        "comparative": """Compare these brain MRI images to what you would expect in a healthy elderly brain.

Images show: 2 coronal slices (hippocampus) and 2 axial slices (ventricles).

In Alzheimer's disease, you would typically see:
- Hippocampal atrophy (often asymmetric, left > right)
- Enlarged temporal horns
- Widened choroidal fissure
- Enlarged lateral ventricles
- Parietal and temporal cortical atrophy

Describe what you observe and whether these findings are present.
Final classification: CN (cognitively normal) or AD (Alzheimer's disease)?""",

        "binary_direct": """Brain MRI: 2 coronal + 2 axial slices.

This patient has either:
- CN: Cognitively Normal - age-appropriate brain volume
- AD: Alzheimer's Disease - pathological atrophy

Key AD indicators: hippocampal atrophy, enlarged ventricles, cortical thinning.

Look at the hippocampus size and ventricle size. Classify as CN or AD.
Respond with your assessment and final answer.""",
    }

    for sample_idx, (_, row) in enumerate(ad_samples.iterrows()):
        print(f"\n{'='*60}")
        print(f"AD SAMPLE {sample_idx + 1}")
        print(f"Subject: {row.get('subject_id', 'N/A')}")
        print(f"Age: {row.get('AGE', 'N/A')}")
        print(f"{'='*60}")

        # Extract slices
        all_slices = extractor.extract_all(row['scan_path'])

        for prompt_name, prompt in prompts.items():
            print(f"\n--- Prompt: {prompt_name} ---")

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
                outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)

            input_len = inputs["input_ids"].shape[1]
            output_text = processor.decode(outputs[0, input_len:], skip_special_tokens=True).strip()

            # Check for AD mention
            has_ad = "AD" in output_text.upper() and "ALZHEIMER" in output_text.upper() or \
                     output_text.upper().strip().endswith("AD") or \
                     ": AD" in output_text.upper()

            print(f"Output (truncated): {output_text[:500]}...")
            print(f"Contains AD classification: {has_ad}")


if __name__ == "__main__":
    main()
