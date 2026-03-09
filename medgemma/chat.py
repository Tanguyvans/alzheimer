#!/usr/bin/env python3
"""
Interactive chat with MedGemma for exploring model outputs.

Usage:
    python medgemma/chat.py                        # Chat without images
    python medgemma/chat.py --sample 0             # Load sample 0 from test set
    python medgemma/chat.py --scan path.nii.gz     # Load NIfTI scan (extracts slices)
    python medgemma/chat.py --image photo.png      # Load PNG/JPG image directly
    python medgemma/chat.py --image img1.png img2.png  # Load multiple images
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import sys
import argparse
import os
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
    'DSPANFOR': 'Digit span forward',
    'DSPANBAC': 'Digit span backward',
    'BNTTOTAL': 'Boston Naming Test score',
}

def format_clinical_info(row):
    info_parts = []
    for feature, name in TABULAR_FEATURES.items():
        if feature not in row or pd.isna(row[feature]):
            continue
        value = row[feature]

        # Skip invalid/missing values (commonly coded as -4, -1, 999, etc.)
        if isinstance(value, (int, float)) and value < 0:
            continue

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


def load_image(path):
    """Load an image file and convert to RGB."""
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with MedGemma")
    parser.add_argument("--sample", type=int, default=None, help="Load sample index from test set")
    parser.add_argument("--scan", type=str, default=None, help="Path to NIfTI scan (extracts slices)")
    parser.add_argument("--image", type=str, nargs='+', default=None, help="Path to image(s) - PNG, JPG, etc.")
    parser.add_argument("--max_tokens", type=int, default=500, help="Max tokens to generate")
    args = parser.parse_args()

    print("="*70)
    print("MedGemma Interactive Chat")
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
    print("Model loaded!")

    # Setup extractor for NIfTI files
    extractor = MultiViewSliceExtractor(
        n_coronal=2, n_axial=2,
        coronal_region=(0.45, 0.55),
        axial_region=(0.35, 0.45),
        output_size=448
    )

    # Load images if specified
    images = None
    clinical_info = None
    true_label = None

    if args.image:
        # Load PNG/JPG images directly
        images = []
        for img_path in args.image:
            print(f"Loading image: {img_path}")
            img = load_image(img_path)
            images.append(img)
            print(f"  Size: {img.size}, Mode: {img.mode}")
        print(f"\nLoaded {len(images)} image(s)")

    elif args.sample is not None:
        df = pd.read_csv('experiments/multimodal_fusion/data/combined_trajectory/test.csv')
        row = df.iloc[args.sample]
        true_label = "CN" if row['label'] == 0 else "AD"
        clinical_info = format_clinical_info(row)
        print(f"\nLoaded sample {args.sample}:")
        print(f"  True label: {true_label}")
        print(f"  Scan: {row['scan_path']}")
        print(f"\nClinical info:")
        print(clinical_info)
        images = extractor.extract_all(row['scan_path'])
        print(f"\nExtracted {len(images)} slices (2 coronal + 2 axial)")

    elif args.scan:
        print(f"\nLoading NIfTI scan: {args.scan}")
        images = extractor.extract_all(args.scan)
        print(f"Extracted {len(images)} slices (2 coronal + 2 axial)")

    print("\n" + "="*70)
    print("Commands:")
    print("  Type your message to chat with MedGemma")
    print("  'quit' or 'exit' to exit")
    print("  'load N' to load sample N from test set (includes clinical data)")
    print("  'img path.png' to load an image file")
    print("  'clinical' to show clinical data (if loaded via --sample or load N)")
    print("  'prompt' to send images with clinical data prompt")
    print("  'info' to show current images info")
    print("  'clear' to clear loaded images")
    print("="*70)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if user_input.lower() == 'clear':
            images = None
            clinical_info = None
            true_label = None
            print("Cleared loaded images.")
            continue

        if user_input.lower() == 'info':
            if images:
                print(f"  Loaded: {len(images)} image(s)")
                for i, img in enumerate(images):
                    print(f"    [{i}] Size: {img.size}")
                if true_label:
                    print(f"  True label: {true_label}")
                if clinical_info:
                    print(f"  Clinical info:\n{clinical_info}")
            else:
                print("  No images loaded. Use 'load N' or 'img path.png'")
            continue

        if user_input.lower().startswith('load '):
            try:
                idx = int(user_input.split()[1])
                df = pd.read_csv('experiments/multimodal_fusion/data/combined_trajectory/test.csv')
                row = df.iloc[idx]
                true_label = "CN" if row['label'] == 0 else "AD"
                clinical_info = format_clinical_info(row)
                images = extractor.extract_all(row['scan_path'])
                print(f"Loaded sample {idx}:")
                print(f"  True label: {true_label}")
                print(f"  Clinical info:\n{clinical_info}")
                print(f"  Extracted {len(images)} slices")
            except Exception as e:
                print(f"Error loading sample: {e}")
            continue

        if user_input.lower().startswith('img '):
            try:
                img_path = user_input[4:].strip()
                img = load_image(img_path)
                if images is None:
                    images = []
                images.append(img)
                print(f"Added image: {img_path} (Size: {img.size})")
                print(f"Total images: {len(images)}")
            except Exception as e:
                print(f"Error loading image: {e}")
            continue

        if user_input.lower() == 'clinical':
            if clinical_info:
                print(f"Clinical data:\n{clinical_info}")
                if true_label:
                    print(f"\nTrue label: {true_label}")
            else:
                print("No clinical data loaded. Use 'load N' to load a sample with clinical data.")
            continue

        if user_input.lower() == 'prompt':
            if not images:
                print("No images loaded. Load images first.")
                continue
            # Build prompt with clinical data and MRI context
            mri_info = """MRI acquisition details:
- Modality: T1-weighted MPRAGE
- Preprocessing: Skull-stripped, registered to MNI template
- Views: Coronal slices (hippocampus region) and axial slices (ventricles)"""

            if clinical_info:
                user_input = f"""Patient clinical information:
{clinical_info}

{mri_info}

Based on these brain MRI images and clinical data, assess for signs of neurodegeneration.
Is this patient likely cognitively normal (CN) or showing signs of Alzheimer's disease (AD)?"""
            else:
                user_input = f"""{mri_info}

Based on these brain MRI images, assess for signs of neurodegeneration.
Is this patient likely cognitively normal (CN) or showing signs of Alzheimer's disease (AD)?"""
            print(f"Sending prompt:\n{user_input}\n")

        # MRI context info
        mri_context = """MRI acquisition details:
- Modality: T1-weighted MPRAGE
- Preprocessing: Skull-stripped, registered to MNI template
- Views: Coronal slices (hippocampus region) and axial slices (ventricles)"""

        # Automatically prepend clinical info and MRI context if available
        if clinical_info and images and user_input.lower() != 'prompt':
            full_prompt = f"""Patient clinical information:
{clinical_info}

{mri_context}

IMPORTANT: Analyze BOTH the MRI images AND the clinical test scores above.
The cognitive test scores are critical for diagnosis:
- Trail Making Test A: Normal <30s, impaired >45s
- Trail Making Test B: Normal <75s, impaired >180s
- Category fluency: Normal 15-25, impaired <12

User question: {user_input}"""
        elif images and user_input.lower() != 'prompt':
            full_prompt = f"""{mri_context}

User question: {user_input}"""
        else:
            full_prompt = user_input

        # Debug: show what we're sending
        print(f"\n[DEBUG] Sending prompt:\n{'-'*40}\n{full_prompt}\n{'-'*40}")

        # Build message
        if images:
            user_content = [{"type": "image", "image": img} for img in images]
            user_content.append({"type": "text", "text": full_prompt})
        else:
            user_content = [{"type": "text", "text": full_prompt}]

        messages = [{"role": "user", "content": user_content}]

        # Process
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if images:
            inputs = processor(text=text, images=images, return_tensors="pt")
        else:
            inputs = processor(text=text, return_tensors="pt")

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        print("\nMedGemma: ", end="", flush=True)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False
            )

        input_len = inputs["input_ids"].shape[1]
        output_text = processor.decode(outputs[0, input_len:], skip_special_tokens=True).strip()
        print(output_text)


if __name__ == "__main__":
    main()
