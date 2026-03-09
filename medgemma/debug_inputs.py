#!/usr/bin/env python3
"""
Debug script to visualize exactly what we send to MedGemma.

Shows:
1. The MRI slices we extract
2. The full prompt
3. The full model output (more tokens)
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'medgemma')
from utils.slice_extractor import SliceExtractor
import pandas as pd

def main():
    print("="*60)
    print("DEBUG: What we send to MedGemma")
    print("="*60)

    # Get one sample
    df = pd.read_csv('experiments/multimodal_fusion/data/combined_trajectory/test.csv')
    row = df.iloc[0]
    true_label = "CN" if row['label'] == 0 else "AD"

    print(f"\nSample info:")
    print(f"  Subject: {row['subject_id']}")
    print(f"  True label: {true_label}")
    print(f"  Scan path: {row['scan_path']}")

    # Extract slices
    extractor = SliceExtractor(
        view="coronal",
        n_slices=3,
        region_start=0.40,
        region_end=0.60,
        output_size=448
    )
    slices = extractor.extract_from_nifti(row['scan_path'])

    # Plot the slices
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'MRI Slices sent to MedGemma - True Label: {true_label}', fontsize=14)

    for i, (ax, img) in enumerate(zip(axes, slices)):
        ax.imshow(img)
        ax.set_title(f'Slice {i+1}\nSize: {img.size}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('medgemma/debug_slices.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved slices to: medgemma/debug_slices.png")

    # Show the prompt
    prompt = (
        "These are 3 coronal MRI slices of the hippocampus region. "
        "Based on hippocampal morphology, classify this patient as either "
        "cognitively normal (CN) or having Alzheimer's disease (AD). "
        "Respond with only: CN or AD"
    )

    print(f"\n" + "="*60)
    print("PROMPT WE SEND:")
    print("="*60)
    print(prompt)

    # Load model and generate
    print(f"\n" + "="*60)
    print("LOADING MODEL...")
    print("="*60)

    processor = AutoProcessor.from_pretrained("google/medgemma-1.5-4b-it")
    model = AutoModelForImageTextToText.from_pretrained(
        "google/medgemma-1.5-4b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Build message
    user_content = [{"type": "image", "image": img} for img in slices]
    user_content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": user_content}]

    # Show full formatted input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"\n" + "="*60)
    print("FULL FORMATTED INPUT (last 500 chars):")
    print("="*60)
    print(text[-500:])

    # Process and generate with MORE tokens
    inputs = processor(text=text, images=slices, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(f"\n" + "="*60)
    print("INPUT SHAPES:")
    print("="*60)
    print(f"  input_ids: {inputs['input_ids'].shape}")
    print(f"  pixel_values: {inputs['pixel_values'].shape}")

    print(f"\n" + "="*60)
    print("GENERATING (max 100 tokens)...")
    print("="*60)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    output_text = processor.decode(outputs[0, input_len:], skip_special_tokens=True)

    print(f"\n" + "="*60)
    print("FULL MODEL OUTPUT:")
    print("="*60)
    print(f"'{output_text}'")
    print("="*60)


if __name__ == "__main__":
    main()
