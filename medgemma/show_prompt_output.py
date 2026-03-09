#!/usr/bin/env python3
"""
Show exactly what prompt we send and what output we get.
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
    return "\n".join(info_parts)

def main():
    print("="*70)
    print("PROMPT AND OUTPUT EXAMPLES")
    print("="*70)

    # Load model
    print("\nLoading MedGemma (bf16)...")
    processor = AutoProcessor.from_pretrained("google/medgemma-1.5-4b-it")
    model = AutoModelForImageTextToText.from_pretrained(
        "google/medgemma-1.5-4b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    extractor = MultiViewSliceExtractor(
        n_coronal=2, n_axial=2,
        coronal_region=(0.45, 0.55),
        axial_region=(0.35, 0.45),
        output_size=448
    )

    df = pd.read_csv('experiments/multimodal_fusion/data/combined_trajectory/test.csv')

    # Test 2 CN and 2 AD
    samples = pd.concat([
        df[df['label'] == 0].head(2),
        df[df['label'] == 1].head(2)
    ])

    for idx, (_, row) in enumerate(samples.iterrows()):
        true_label = "CN" if row['label'] == 0 else "AD"
        clinical_info = format_clinical_info(row)

        prompt = f"""Patient clinical information:
{clinical_info}

Brain MRI: 2 coronal slices (hippocampus) + 2 axial slices (ventricles).

Classify this patient as:
- CN: Cognitively Normal
- AD: Alzheimer's Disease

Respond with only: CN or AD"""

        print(f"\n{'='*70}")
        print(f"SAMPLE {idx+1} - True Label: {true_label}")
        print(f"{'='*70}")
        print(f"\n--- PROMPT SENT TO MODEL ---")
        print(prompt)
        print(f"\n--- MODEL OUTPUT ---")

        slices = extractor.extract_all(row['scan_path'])
        user_content = [{"type": "image", "image": img} for img in slices]
        user_content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": user_content}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=slices, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

        input_len = inputs["input_ids"].shape[1]
        output_text = processor.decode(outputs[0, input_len:], skip_special_tokens=True).strip()

        print(f"'{output_text}'")


if __name__ == "__main__":
    main()
