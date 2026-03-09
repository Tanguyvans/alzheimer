#!/usr/bin/env python3
"""
Test base MedGemma model (no fine-tuning) on Alzheimer's classification.

Usage:
    python medgemma/test_base_model.py
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import sys
sys.path.insert(0, 'medgemma')
from utils.slice_extractor import SliceExtractor
import pandas as pd

def main():
    print("="*60)
    print("BASE MEDGEMMA TEST (No Fine-tuning)")
    print("="*60)

    print("\nLoading model in bf16...")
    processor = AutoProcessor.from_pretrained("google/medgemma-1.5-4b-it")
    model = AutoModelForImageTextToText.from_pretrained(
        "google/medgemma-1.5-4b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded!")

    # Get test samples
    df = pd.read_csv('experiments/multimodal_fusion/data/combined_trajectory/test.csv')
    extractor = SliceExtractor(
        view="coronal",
        n_slices=3,
        region_start=0.40,
        region_end=0.60,
        output_size=448
    )

    prompt = (
        "These are 3 coronal MRI slices of the hippocampus region. "
        "Based on hippocampal morphology, classify this patient as either "
        "cognitively normal (CN) or having Alzheimer's disease (AD). "
        "Respond with only: CN or AD"
    )

    print("\n" + "="*60)
    print("TESTING ON 10 SAMPLES")
    print("="*60)

    correct = 0
    total = 10

    for i in range(total):
        row = df.iloc[i]
        true_label = "CN" if row['label'] == 0 else "AD"

        # Extract slices
        slices = extractor.extract_from_nifti(row['scan_path'])

        # Build message
        user_content = [{"type": "image", "image": img} for img in slices]
        user_content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": user_content}]

        # Process
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=slices, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

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

        print(f"\nSample {i+1}:")
        print(f"  True label: {true_label}")
        print(f"  Model output: '{output_text}'")
        print(f"  Parsed: {pred} {'✓' if is_correct else '✗'}")

    print("\n" + "="*60)
    print(f"ACCURACY: {correct}/{total} ({100*correct/total:.0f}%)")
    print("="*60)


if __name__ == "__main__":
    main()
