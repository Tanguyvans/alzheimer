"""
MedGemma Dataset for Alzheimer's Classification

Loads MRI volumes, extracts multi-view slices (coronal + axial), and formats
data for MedGemma instruction-tuned model fine-tuning.

Multi-view approach:
- 2 coronal slices: hippocampus region (45-55% of y-axis)
- 2 axial slices: ventricle region (35-45% of z-axis)
"""

import torch
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image
import logging

from utils.slice_extractor import MultiViewSliceExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Default classification prompt with multi-view and clinical data
DEFAULT_PROMPT = """Patient clinical information:
{clinical_info}

Brain MRI: 2 coronal slices (hippocampus) + 2 axial slices (ventricles).

Classify this patient as:
- CN: Cognitively Normal
- AD: Alzheimer's Disease

Respond with only: CN or AD"""

# Simple prompt without clinical data
SIMPLE_PROMPT = """Brain MRI: 2 coronal slices (hippocampus) + 2 axial slices (ventricles).

Classify as CN (Cognitively Normal) or AD (Alzheimer's Disease).

Respond with only: CN or AD"""

# Tabular feature names and descriptions
TABULAR_FEATURES = {
    'AGE': 'Age',
    'PTGENDER': 'Gender',  # 1=Male, 2=Female
    'PTEDUCAT': 'Years of education',
    'CATANIMSC': 'Category fluency (animals)',
    'TRAASCOR': 'Trail Making Test A (seconds)',
    'TRABSCOR': 'Trail Making Test B (seconds)',
    'DSPANFOR': 'Digit span forward',
    'DSPANBAC': 'Digit span backward',
    'BNTTOTAL': 'Boston Naming Test score',
    'BMI': 'BMI',
}


class MedGemmaAlzheimerDataset(Dataset):
    """
    Dataset for MedGemma fine-tuning on Alzheimer's classification.

    Loads MRI NIfTI files, extracts multi-view slices (coronal + axial), and formats as
    conversation messages for instruction-tuned VLM training.
    """

    def __init__(
        self,
        csv_path: str,
        processor: Any,
        n_coronal: int = 2,
        n_axial: int = 2,
        coronal_region: Tuple[float, float] = (0.45, 0.55),
        axial_region: Tuple[float, float] = (0.35, 0.45),
        output_size: int = 448,
        is_training: bool = True,
        max_samples: Optional[int] = None,
        use_tabular: bool = True,
        tabular_features: Optional[List[str]] = None
    ):
        """
        Args:
            csv_path: Path to CSV with 'scan_path', 'label', 'group' columns
            processor: MedGemma processor (AutoProcessor)
            n_coronal: Number of coronal slices (hippocampus view)
            n_axial: Number of axial slices (ventricle view)
            coronal_region: Start/end fraction for coronal slices
            axial_region: Start/end fraction for axial slices
            output_size: Output slice size (448 for memory efficiency)
            is_training: Whether this is training data (includes labels)
            max_samples: Limit number of samples (for testing)
            use_tabular: Whether to include tabular features in prompt
            tabular_features: List of tabular feature column names to use
        """
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.is_training = is_training
        self.use_tabular = use_tabular
        self.tabular_features = tabular_features or list(TABULAR_FEATURES.keys())
        self.output_size = output_size

        if max_samples is not None:
            self.df = self.df.head(max_samples)

        # Initialize multi-view slice extractor
        self.slice_extractor = MultiViewSliceExtractor(
            n_coronal=n_coronal,
            n_axial=n_axial,
            coronal_region=coronal_region,
            axial_region=axial_region,
            output_size=output_size
        )

        # Class mapping
        self.label_to_text = {0: "CN", 1: "AD"}
        self.text_to_label = {"CN": 0, "AD": 1}

        self._log_stats()

    def _log_stats(self):
        """Log dataset statistics."""
        logger.info(f"Loaded {len(self.df)} samples")
        logger.info(f"Using tabular features: {self.use_tabular}")
        if self.use_tabular:
            available = [f for f in self.tabular_features if f in self.df.columns]
            logger.info(f"  Available features: {available}")
        if 'label' in self.df.columns:
            label_counts = self.df['label'].value_counts().sort_index()
            for label, count in label_counts.items():
                label_name = self.label_to_text.get(label, str(label))
                logger.info(f"  {label_name}: {count} ({100*count/len(self.df):.1f}%)")

    def _format_clinical_info(self, row: pd.Series) -> str:
        """Format tabular features as readable clinical info string."""
        info_parts = []

        for feature in self.tabular_features:
            if feature not in row or pd.isna(row[feature]):
                continue

            value = row[feature]
            name = TABULAR_FEATURES.get(feature, feature)

            # Format specific features
            if feature == 'PTGENDER':
                value = 'Male' if value == 1 else 'Female'
            elif feature == 'AGE':
                value = f"{value:.0f} years"
            elif feature == 'PTEDUCAT':
                value = f"{value:.0f} years"
            elif feature in ['TRAASCOR', 'TRABSCOR']:
                value = f"{value:.0f}s"
            elif feature == 'BMI':
                value = f"{value:.1f}"
            elif isinstance(value, float):
                value = f"{value:.1f}"

            info_parts.append(f"- {name}: {value}")

        return "\n".join(info_parts) if info_parts else "No clinical data available"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a processed sample ready for MedGemma.

        Returns:
            Dict with 'input_ids', 'attention_mask', 'pixel_values', 'labels'
        """
        row = self.df.iloc[idx]
        scan_path = row['scan_path']
        label = int(row['label']) if 'label' in row else None
        label_text = self.label_to_text.get(label, "") if label is not None else ""

        # Extract multi-view slices from MRI
        try:
            slices = self.slice_extractor.extract_all(scan_path)
        except Exception as e:
            logger.warning(f"Error loading {scan_path}: {e}. Using black images.")
            # Return black images on error
            slices = [
                Image.new('RGB', (self.output_size,) * 2, color=0)
                for _ in range(self.slice_extractor.total_slices)
            ]

        # Build prompt with or without tabular data
        if self.use_tabular:
            clinical_info = self._format_clinical_info(row)
            prompt = DEFAULT_PROMPT.format(clinical_info=clinical_info)
        else:
            prompt = SIMPLE_PROMPT

        # Build conversation messages
        messages = self._build_messages(slices, label_text, prompt)

        # Process with MedGemma processor
        processed = self._process_messages(messages, slices)

        return processed

    def _build_messages(
        self,
        slices: List[Image.Image],
        label_text: str,
        prompt: str
    ) -> List[Dict[str, Any]]:
        """
        Build conversation messages in MedGemma format.

        Args:
            slices: List of PIL Images
            label_text: "CN" or "AD" label text
            prompt: The text prompt to use (may include clinical info)

        Returns:
            List of message dicts with 'role' and 'content'
        """
        # Build user content with images and text prompt
        user_content = []

        # Add all images
        for i, img in enumerate(slices):
            user_content.append({"type": "image", "image": img})

        # Add text prompt
        user_content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "user", "content": user_content}
        ]

        # Add assistant response for training
        if self.is_training and label_text:
            messages.append({
                "role": "assistant",
                "content": label_text
            })

        return messages

    def _process_messages(
        self,
        messages: List[Dict[str, Any]],
        slices: List[Image.Image]
    ) -> Dict[str, torch.Tensor]:
        """
        Process messages using MedGemma processor.

        Args:
            messages: Conversation messages
            slices: List of PIL Images

        Returns:
            Dict with tokenized inputs and pixel values
        """
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not self.is_training
        )

        # Process text and images together
        processed = self.processor(
            text=text,
            images=slices,
            return_tensors="pt",
            padding=False
        )

        # Squeeze batch dimension (will be batched by DataLoader)
        result = {
            "input_ids": processed["input_ids"].squeeze(0),
            "attention_mask": processed["attention_mask"].squeeze(0),
        }

        if "pixel_values" in processed:
            result["pixel_values"] = processed["pixel_values"].squeeze(0)

        # For training, create labels (same as input_ids, with padding tokens masked)
        if self.is_training:
            result["labels"] = result["input_ids"].clone()

        return result


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching MedGemma samples.

    Handles variable-length sequences with padding.
    For multi-image inputs, concatenates pixel_values along batch dimension.
    """
    # Find max length in batch
    max_len = max(item["input_ids"].size(0) for item in batch)

    # Initialize padded tensors
    batch_size = len(batch)
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

    has_labels = "labels" in batch[0]
    if has_labels:
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    # Concatenate pixel values along batch dimension
    # Each item has shape (num_images, C, H, W), concatenate to (total_images, C, H, W)
    pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)

    # Fill in values with right-padding
    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        if has_labels:
            labels[i, :seq_len] = item["labels"]

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
    }

    if has_labels:
        result["labels"] = labels

    return result


def get_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    processor: Any,
    config: Dict,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        test_csv: Path to test CSV
        processor: MedGemma processor
        config: Configuration dict with slice extraction and training params
        num_workers: Number of dataloader workers

    Returns:
        train_loader, val_loader, test_loader
    """
    slice_config = config.get("data", {}).get("slice_extraction", {})
    training_config = config.get("training", {})

    # Common dataset kwargs for multi-view extraction
    dataset_kwargs = {
        "processor": processor,
        "n_coronal": slice_config.get("n_coronal", 2),
        "n_axial": slice_config.get("n_axial", 2),
        "coronal_region": tuple(slice_config.get("coronal_region", [0.45, 0.55])),
        "axial_region": tuple(slice_config.get("axial_region", [0.35, 0.45])),
        "output_size": slice_config.get("output_size", 448),
        "use_tabular": config.get("data", {}).get("use_tabular", True),
    }

    # Create datasets
    train_dataset = MedGemmaAlzheimerDataset(
        train_csv,
        is_training=True,
        **dataset_kwargs
    )

    val_dataset = MedGemmaAlzheimerDataset(
        val_csv,
        is_training=True,  # Include labels for evaluation
        **dataset_kwargs
    )

    test_dataset = MedGemmaAlzheimerDataset(
        test_csv,
        is_training=True,  # Include labels for evaluation
        **dataset_kwargs
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.get("per_device_train_batch_size", 1),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.get("per_device_eval_batch_size", 1),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.get("per_device_eval_batch_size", 1),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    logger.info(f"Train: {len(train_dataset)} samples")
    logger.info(f"Val: {len(val_dataset)} samples")
    logger.info(f"Test: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import sys

    print("MedGemma Alzheimer Dataset Test")
    print("=" * 50)
    print("\nThis module requires a MedGemma processor to test.")
    print("Use train.py or inference.py for full functionality.")
    print("\nDataset expects CSV with columns:")
    print("  - scan_path: path to .nii.gz file")
    print("  - label: 0 (CN) or 1 (AD)")
    print("  - group: 'CN' or 'AD'")
