#!/usr/bin/env python3
"""
Test MedGemma with multi-view slices (coronal + axial).
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'medgemma')
from utils.slice_extractor import MultiViewSliceExtractor
import pandas as pd

def main():
    print("="*60)
    print("MULTI-VIEW TEST: Coronal + Axial")
    print("="*60)

    # Get test sample
    df = pd.read_csv('experiments/multimodal_fusion/data/combined_trajectory/test.csv')
    row = df.iloc[0]
    true_label = "CN" if row['label'] == 0 else "AD"

    # Extract multi-view slices
    extractor = MultiViewSliceExtractor(
        n_coronal=2,
        n_axial=2,
        coronal_region=(0.45, 0.55),  # Tighter hippocampus region
        axial_region=(0.40, 0.60),
        output_size=448
    )

    coronal_slices, axial_slices = extractor.extract_from_nifti(row['scan_path'])
    all_slices = coronal_slices + axial_slices

    print(f"\nExtracted: {len(coronal_slices)} coronal + {len(axial_slices)} axial = {len(all_slices)} total")

    # Plot all slices
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'Multi-View Slices - True Label: {true_label}', fontsize=14)

    for i, (ax, img) in enumerate(zip(axes[:2], coronal_slices)):
        ax.imshow(img)
        ax.set_title(f'Coronal {i+1}')
        ax.axis('off')

    for i, (ax, img) in enumerate(zip(axes[2:], axial_slices)):
        ax.imshow(img)
        ax.set_title(f'Axial {i+1}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('medgemma/multiview_slices.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: medgemma/multiview_slices.png")

    # Test with model
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

    # Updated prompt for multi-view
    prompt = (
        "These are brain MRI slices showing 2 coronal views (showing the hippocampus) "
        "and 2 axial views (showing ventricles and overall brain structure). "
        "Based on hippocampal atrophy, ventricular enlargement, and cortical thinning, "
        "classify this patient as either cognitively normal (CN) or having Alzheimer's disease (AD). "
        "Respond with only: CN or AD"
    )

    print(f"\nPrompt:\n{prompt}")

    # Build message
    user_content = [{"type": "image", "image": img} for img in all_slices]
    user_content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": user_content}]

    # Process and generate
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=all_slices, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(f"\n" + "="*60)
    print("GENERATING...")
    print("="*60)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    output_text = processor.decode(outputs[0, input_len:], skip_special_tokens=True)

    print(f"\nTrue label: {true_label}")
    print(f"Model output: '{output_text}'")

    # Test on more samples
    print(f"\n" + "="*60)
    print("TESTING ON 10 SAMPLES")
    print("="*60)

    correct = 0
    for i in range(10):
        row = df.iloc[i]
        true_label = "CN" if row['label'] == 0 else "AD"

        all_slices = extractor.extract_all(row['scan_path'])

        user_content = [{"type": "image", "image": img} for img in all_slices]
        user_content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": user_content}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=all_slices, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

        input_len = inputs["input_ids"].shape[1]
        output_text = processor.decode(outputs[0, input_len:], skip_special_tokens=True).strip()

        # Parse
        output_upper = output_text.upper()
        if "AD" in output_upper and "CN" not in output_upper:
            pred = "AD"
        elif "CN" in output_upper and "AD" not in output_upper:
            pred = "CN"
        else:
            pred = "?"

        is_correct = pred == true_label
        correct += int(is_correct)
        print(f"Sample {i+1}: True={true_label}, Pred={pred}, Output='{output_text}' {'✓' if is_correct else '✗'}")

    print(f"\nAccuracy: {correct}/10 ({100*correct/10:.0f}%)")


if __name__ == "__main__":
    main()
