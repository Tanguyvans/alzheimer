#!/usr/bin/env python3
"""
MedGemma 1.5 4B QLoRA Fine-tuning for Alzheimer's Classification

Fine-tunes MedGemma on CN vs AD classification using coronal hippocampus
slices extracted from 3D MRI volumes.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --max_samples 100  # Quick test
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml
import wandb
import numpy as np
from tqdm import tqdm

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig

from dataset import MedGemmaAlzheimerDataset, collate_fn, DEFAULT_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_quantization(config: Dict[str, Any]) -> BitsAndBytesConfig:
    """Create BitsAndBytes quantization config for QLoRA."""
    quant_config = config.get("model", {}).get("quantization", {})

    # Map string dtype to torch dtype
    compute_dtype_str = quant_config.get("bnb_4bit_compute_dtype", "bfloat16")
    compute_dtype = getattr(torch, compute_dtype_str, torch.bfloat16)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config.get("load_in_4bit", True),
        bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    logger.info("Quantization config:")
    logger.info(f"  4-bit: {bnb_config.load_in_4bit}")
    logger.info(f"  Double quant: {bnb_config.bnb_4bit_use_double_quant}")
    logger.info(f"  Quant type: {bnb_config.bnb_4bit_quant_type}")
    logger.info(f"  Compute dtype: {compute_dtype}")

    return bnb_config


def setup_lora(config: Dict[str, Any]) -> LoraConfig:
    """Create LoRA configuration."""
    lora_config = config.get("model", {}).get("lora", {})

    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 16),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        target_modules=lora_config.get("target_modules", "all-linear"),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    logger.info("LoRA config:")
    logger.info(f"  Rank (r): {peft_config.r}")
    logger.info(f"  Alpha: {peft_config.lora_alpha}")
    logger.info(f"  Dropout: {peft_config.lora_dropout}")
    logger.info(f"  Target modules: {peft_config.target_modules}")

    return peft_config


def load_model_and_processor(
    config: Dict[str, Any],
    bnb_config: BitsAndBytesConfig
) -> tuple:
    """Load MedGemma model and processor."""
    model_name = config.get("model", {}).get("name", "google/medgemma-1.5-4b-it")
    device_map = config.get("hardware", {}).get("device_map", "auto")

    logger.info(f"Loading model: {model_name}")
    logger.info(f"Device map: {device_map}")

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name)

    # Load model with quantization
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    logger.info(f"Model loaded. Parameters: {model.num_parameters():,}")

    return model, processor


def setup_peft_model(model, lora_config: LoraConfig):
    """Prepare model for QLoRA training."""
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters
    model = get_peft_model(model, lora_config)

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")

    return model


def create_training_data(
    csv_path: str,
    processor,
    config: Dict[str, Any],
    max_samples: Optional[int] = None
) -> MedGemmaAlzheimerDataset:
    """Create training dataset."""
    data_config = config.get("data", {})
    slice_config = data_config.get("slice_extraction", {})

    dataset = MedGemmaAlzheimerDataset(
        csv_path=csv_path,
        processor=processor,
        n_slices=slice_config.get("n_slices", 5),
        region_start=slice_config.get("region_start", 0.40),
        region_end=slice_config.get("region_end", 0.60),
        output_size=slice_config.get("output_size", 896),
        prompt=config.get("inference", {}).get("prompt", DEFAULT_PROMPT),
        is_training=True,
        max_samples=max_samples,
        use_tabular=data_config.get("use_tabular", False),
        tabular_features=data_config.get("tabular_features", None)
    )

    return dataset


def formatting_func(example):
    """Format function for SFTTrainer - returns text strings."""
    # This is called by SFTTrainer to format examples
    # For VLMs with images, we need a different approach
    return example


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

    predictions, labels = eval_pred

    # For generative models, we need to decode and parse predictions
    # This is a placeholder - actual implementation in evaluate()
    return {}


def train(
    config: Dict[str, Any],
    config_path: str,
    max_samples: Optional[int] = None
):
    """Main training function."""

    # Setup W&B
    wandb_config = config.get("wandb", {})
    if wandb_config.get("enabled", True):
        wandb.init(
            project=wandb_config.get("project", "medgemma-alzheimer"),
            name=wandb_config.get("name", "medgemma_cn_ad_qlora"),
            tags=wandb_config.get("tags", []),
            config=config
        )

    # Setup quantization and LoRA
    bnb_config = setup_quantization(config)
    lora_config = setup_lora(config)

    # Load model and processor
    model, processor = load_model_and_processor(config, bnb_config)

    # Setup PEFT model
    model = setup_peft_model(model, lora_config)

    # Resolve CSV paths relative to config file
    config_dir = Path(config_path).parent
    data_config = config.get("data", {})

    train_csv = str(config_dir / data_config.get("train_csv", "../data/combined/mri_cn_ad_train.csv"))
    val_csv = str(config_dir / data_config.get("val_csv", "../data/combined/mri_cn_ad_val.csv"))

    logger.info(f"Train CSV: {train_csv}")
    logger.info(f"Val CSV: {val_csv}")

    # Create datasets
    train_dataset = create_training_data(train_csv, processor, config, max_samples)
    val_dataset = create_training_data(val_csv, processor, config, max_samples)

    # Training arguments
    training_config = config.get("training", {})
    output_dir = str(config_dir / training_config.get("output_dir", "./checkpoints"))

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=training_config.get("num_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 16),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        learning_rate=training_config.get("learning_rate", 2e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        bf16=training_config.get("bf16", True),
        fp16=training_config.get("fp16", False),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        logging_steps=training_config.get("logging_steps", 10),
        eval_strategy=training_config.get("eval_strategy", "epoch"),
        save_strategy=training_config.get("save_strategy", "epoch"),
        save_total_limit=training_config.get("save_total_limit", 3),
        load_best_model_at_end=training_config.get("load_best_model_at_end", True),
        metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=False,  # Lower loss is better
        report_to="wandb" if wandb_config.get("enabled", True) else "none",
        seed=training_config.get("seed", 42),
        remove_unused_columns=False,
        dataset_text_field="",  # Not used with custom collator
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    logger.info("Training configuration:")
    logger.info(f"  Epochs: {training_args.num_train_epochs}")
    logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  Gradient checkpointing: {training_args.gradient_checkpointing}")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    logger.info(f"  Output dir: {output_dir}")

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        data_collator=collate_fn,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    final_path = Path(output_dir) / "final"
    logger.info(f"Saving final model to {final_path}")
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))

    # Save LoRA adapter separately
    adapter_path = Path(output_dir) / "adapter"
    logger.info(f"Saving LoRA adapter to {adapter_path}")
    model.save_pretrained(str(adapter_path))

    if wandb_config.get("enabled", True):
        wandb.finish()

    logger.info("Training complete!")

    return trainer


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune MedGemma 1.5 4B for Alzheimer's classification"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of training samples (for testing)"
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to script directory
        config_path = Path(__file__).parent / args.config

    if not config_path.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(str(config_path))

    logger.info("=" * 60)
    logger.info("MedGemma 1.5 4B QLoRA Fine-tuning")
    logger.info("Task: Alzheimer's Classification (CN vs AD)")
    logger.info("=" * 60)

    # Run training
    train(config, str(config_path), args.max_samples)


if __name__ == "__main__":
    main()
