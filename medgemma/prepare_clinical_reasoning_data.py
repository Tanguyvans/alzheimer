#!/usr/bin/env python3
"""
Prepare training data for MedGemma clinical reasoning fine-tuning.

Focuses on teaching correct interpretation of cognitive test scores,
not MRI-specific findings (which we don't have ground truth for).
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
import sys
sys.path.insert(0, 'medgemma')

# Normative values from literature
NORMS = {
    'TRAASCOR': {  # Trail Making Test A
        'name': 'Trail Making Test A',
        'unit': 'seconds',
        'normal_cutoff': 30,  # Tombaugh 2004: mean ~29s
        'impaired_cutoff': 45,
        'interpretation': lambda x: 'normal' if x < 30 else ('mildly impaired' if x < 45 else 'impaired')
    },
    'TRABSCOR': {  # Trail Making Test B
        'name': 'Trail Making Test B',
        'unit': 'seconds',
        'normal_cutoff': 75,   # Tombaugh 2004: mean ~75s
        'impaired_cutoff': 180,  # Systematic review cutoff
        'interpretation': lambda x: 'normal' if x < 75 else ('impaired' if x < 180 else 'severely impaired')
    },
    'CATANIMSC': {  # Category Fluency - Animals
        'name': 'Category fluency (animals)',
        'unit': 'words',
        'normal_cutoff': 15,  # <15 suggests impairment
        'good_cutoff': 21,    # >21 probably normal
        'interpretation': lambda x: 'normal' if x >= 15 else 'impaired'
    },
    'DSPANFOR': {  # Digit Span Forward
        'name': 'Digit Span Forward',
        'unit': 'score',
        'normal_cutoff': 5,
        'interpretation': lambda x: 'normal' if x >= 5 else 'impaired'
    },
    'DSPANBAC': {  # Digit Span Backward
        'name': 'Digit Span Backward',
        'unit': 'score',
        'normal_cutoff': 4,
        'interpretation': lambda x: 'normal' if x >= 4 else 'impaired'
    },
    'BNTTOTAL': {  # Boston Naming Test
        'name': 'Boston Naming Test',
        'unit': 'score',
        'normal_cutoff': 24,  # Out of 30
        'interpretation': lambda x: 'normal' if x >= 24 else 'impaired'
    },
}


def interpret_score(feature: str, value: float) -> Dict:
    """Interpret a cognitive test score using normative data."""
    if feature not in NORMS:
        return None
    if pd.isna(value) or value < 0:  # Missing data
        return None

    norm = NORMS[feature]
    interpretation = norm['interpretation'](value)

    # Build reference string
    if feature == 'TRAASCOR':
        reference = f"normal <{norm['normal_cutoff']}s"
    elif feature == 'TRABSCOR':
        reference = f"normal <{norm['normal_cutoff']}s, impaired >{norm['impaired_cutoff']}s"
    elif feature == 'CATANIMSC':
        reference = f"normal >{norm['normal_cutoff']}"
    else:
        reference = f"normal >={norm['normal_cutoff']}"

    return {
        'name': norm['name'],
        'value': value,
        'unit': norm['unit'],
        'interpretation': interpretation,
        'reference': reference
    }


def generate_clinical_assessment(row: pd.Series, label: str) -> str:
    """Generate a clinical reasoning assessment based on test scores."""

    # Interpret each available score
    interpretations = []
    impaired_tests = []

    for feature in ['TRAASCOR', 'TRABSCOR', 'CATANIMSC', 'DSPANFOR', 'DSPANBAC', 'BNTTOTAL']:
        if feature in row:
            result = interpret_score(feature, row[feature])
            if result:
                interpretations.append(result)
                if 'impaired' in result['interpretation']:
                    impaired_tests.append(result)

    # Build the assessment text
    lines = ["Clinical Assessment:", "", "**Cognitive Test Interpretation:**"]

    for interp in interpretations:
        if interp['unit'] == 'seconds':
            value_str = f"{interp['value']:.0f}s"
        elif interp['unit'] == 'score':
            value_str = f"{interp['value']:.0f}"
        else:
            value_str = f"{interp['value']:.1f}"

        lines.append(f"- {interp['name']}: {value_str} - {interp['interpretation']} ({interp['reference']})")

    # Pattern analysis
    lines.extend(["", "**Pattern Analysis:**"])

    n_impaired = len(impaired_tests)
    has_trail_impairment = any('Trail' in t['name'] and 'impaired' in t['interpretation'] for t in impaired_tests)
    has_severe_trail = any('Trail' in t['name'] and 'severe' in t['interpretation'] for t in impaired_tests)
    has_fluency_impairment = any('fluency' in t['name'].lower() for t in impaired_tests)
    has_naming_impairment = any('Naming' in t['name'] for t in impaired_tests)

    if label == "AD":
        if n_impaired >= 2:
            # Clear impairment pattern
            if has_severe_trail:
                lines.append("Significant Trail Making impairment indicates executive dysfunction.")
            elif has_trail_impairment:
                lines.append("Trail Making impairment suggests executive dysfunction.")
            if has_fluency_impairment:
                lines.append("Reduced category fluency suggests semantic memory impairment.")
            if has_naming_impairment:
                lines.append("Boston Naming impairment indicates word-finding difficulties.")
            lines.append("This cognitive pattern is consistent with Alzheimer's disease.")
            lines.extend(["", "**Conclusion:** Cognitive profile indicates Alzheimer's disease.", "", "Diagnosis: AD"])
        elif n_impaired == 1:
            # Mild impairment
            lines.append("Mild cognitive impairment detected in testing.")
            lines.append("Combined with clinical presentation, findings suggest early Alzheimer's disease.")
            lines.extend(["", "**Conclusion:** Cognitive profile suggests early Alzheimer's disease.", "", "Diagnosis: AD"])
        else:
            # No impairment on tests but AD diagnosis
            lines.append("Cognitive test scores relatively preserved.")
            lines.append("However, clinical diagnosis of Alzheimer's disease established through comprehensive evaluation")
            lines.append("(may include biomarkers, longitudinal decline, or clinical symptoms not captured in these tests).")
            lines.extend(["", "**Conclusion:** AD diagnosis based on comprehensive clinical evaluation.", "", "Diagnosis: AD"])
    else:  # CN
        if n_impaired == 0:
            lines.append("All cognitive tests within normal limits for age.")
            lines.append("No pattern suggestive of neurodegenerative process.")
        elif n_impaired == 1:
            lines.append("Isolated test finding within expected variation for age.")
            lines.append("No consistent pattern suggestive of neurodegeneration.")
        else:
            lines.append("Some test variations noted, but pattern not consistent with Alzheimer's disease.")
        lines.extend(["", "**Conclusion:** Cognitive profile within normal limits.", "", "Diagnosis: CN"])

    return "\n".join(lines)


def generate_user_prompt(row: pd.Series) -> str:
    """Generate the user prompt with clinical data."""

    lines = ["Patient clinical data:"]

    # Demographics
    if 'AGE' in row and not pd.isna(row['AGE']) and row['AGE'] > 0:
        lines.append(f"- Age: {row['AGE']:.0f} years")
    if 'PTGENDER' in row and not pd.isna(row['PTGENDER']):
        gender = 'Male' if row['PTGENDER'] == 1 else 'Female'
        lines.append(f"- Gender: {gender}")
    if 'PTEDUCAT' in row and not pd.isna(row['PTEDUCAT']) and row['PTEDUCAT'] > 0:
        lines.append(f"- Education: {row['PTEDUCAT']:.0f} years")

    # Cognitive tests
    for feature in ['TRAASCOR', 'TRABSCOR', 'CATANIMSC', 'DSPANFOR', 'DSPANBAC', 'BNTTOTAL']:
        if feature in row and not pd.isna(row[feature]) and row[feature] >= 0:
            name = NORMS[feature]['name']
            value = row[feature]
            if NORMS[feature]['unit'] == 'seconds':
                lines.append(f"- {name}: {value:.0f} seconds")
            else:
                lines.append(f"- {name}: {value:.0f}")

    lines.extend(["", "Assess this patient for cognitive impairment and provide your diagnosis."])

    return "\n".join(lines)


def prepare_training_example(row: pd.Series, scan_path: str) -> Dict:
    """Prepare a single training example in the conversation format."""

    label = "CN" if row['label'] == 0 else "AD"

    user_prompt = generate_user_prompt(row)
    assistant_response = generate_clinical_assessment(row, label)

    return {
        'scan_path': scan_path,
        'messages': [
            {
                'role': 'user',
                'content': user_prompt
            },
            {
                'role': 'assistant',
                'content': assistant_response
            }
        ],
        'label': label
    }


def prepare_dataset(csv_path: str, output_path: str):
    """Prepare full training dataset."""

    df = pd.read_csv(csv_path)

    examples = []
    for idx, row in df.iterrows():
        example = prepare_training_example(row, row['scan_path'])
        examples.append(example)

    # Save as JSON lines
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"Prepared {len(examples)} training examples")
    print(f"Saved to: {output_path}")

    # Show example
    print("\n" + "="*60)
    print("EXAMPLE TRAINING DATA:")
    print("="*60)
    print("\n--- USER PROMPT ---")
    print(examples[0]['messages'][0]['content'])
    print("\n--- ASSISTANT RESPONSE ---")
    print(examples[0]['messages'][1]['content'])

    return examples


if __name__ == "__main__":
    # Prepare train, val, test sets
    data_dir = Path("experiments/multimodal_fusion/data/combined_trajectory")
    output_dir = Path("medgemma/data")
    output_dir.mkdir(exist_ok=True)

    for split in ['train', 'val', 'test']:
        csv_path = data_dir / f"{split}.csv"
        output_path = output_dir / f"{split}_clinical_reasoning.jsonl"

        print(f"\nProcessing {split}...")
        prepare_dataset(str(csv_path), str(output_path))
