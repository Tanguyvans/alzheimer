#!/usr/bin/env python3
"""
Experiment Runner for Loss Function Comparison

Runs multiple training configurations to compare different loss functions
and hyperparameters for Alzheimer's disease classification.

Usage:
    python run_experiments.py --base-config configs/config_combined.yaml
    python run_experiments.py --base-config configs/config_combined.yaml --experiments focal weighted_ce
"""

import os
import yaml
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime
from copy import deepcopy
import pandas as pd

# Define experiment configurations
EXPERIMENTS = {
    # Baseline: Weighted Cross-Entropy (current best)
    'weighted_ce': {
        'name': 'Weighted Cross-Entropy',
        'training': {
            'loss_type': 'cross_entropy',
            'use_weighted_loss': True,
        }
    },

    # Focal Loss variants
    'focal_g2_a75': {
        'name': 'Focal Loss (gamma=2.0, alpha=0.75)',
        'training': {
            'loss_type': 'focal',
            'focal_gamma': 2.0,
            'focal_alpha': 0.75,
            'use_weighted_loss': True,
        }
    },
    'focal_g2_a80': {
        'name': 'Focal Loss (gamma=2.0, alpha=0.80)',
        'training': {
            'loss_type': 'focal',
            'focal_gamma': 2.0,
            'focal_alpha': 0.80,
            'use_weighted_loss': True,
        }
    },
    'focal_g3_a75': {
        'name': 'Focal Loss (gamma=3.0, alpha=0.75)',
        'training': {
            'loss_type': 'focal',
            'focal_gamma': 3.0,
            'focal_alpha': 0.75,
            'use_weighted_loss': True,
        }
    },

    # Unweighted baselines
    'ce_unweighted': {
        'name': 'Cross-Entropy (unweighted)',
        'training': {
            'loss_type': 'cross_entropy',
            'use_weighted_loss': False,
        }
    },
    'focal_unweighted': {
        'name': 'Focal Loss (unweighted)',
        'training': {
            'loss_type': 'focal',
            'focal_gamma': 2.0,
            'focal_alpha': 0.75,
            'use_weighted_loss': False,
        }
    },

    # Monitor balanced accuracy
    'focal_bal_acc': {
        'name': 'Focal Loss + Monitor Balanced Acc',
        'training': {
            'loss_type': 'focal',
            'focal_gamma': 2.0,
            'focal_alpha': 0.75,
            'use_weighted_loss': True,
        },
        'callbacks': {
            'early_stopping': {
                'enabled': True,
                'patience': 25,
                'monitor': 'val_balanced_accuracy',
            }
        }
    },
}


def load_config(config_path: str) -> dict:
    """Load base configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    """Deep merge override into base config."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config: dict, path: str):
    """Save configuration to YAML file."""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def run_experiment(config_path: str, experiment_name: str) -> dict:
    """Run a single training experiment and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {experiment_name}")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")

    # Run training
    cmd = ['python3', 'train.py', '--config', config_path]
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"ERROR: Experiment {experiment_name} failed!")
        return None

    # Find the latest results file
    results_dir = Path('results')
    result_files = sorted(results_dir.glob('*/results.json'), key=os.path.getmtime, reverse=True)

    if result_files:
        with open(result_files[0], 'r') as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description='Run loss function experiments')
    parser.add_argument('--base-config', type=str, default='configs/config_combined.yaml',
                        help='Base configuration file')
    parser.add_argument('--experiments', nargs='+', default=None,
                        help='Specific experiments to run (default: all)')
    parser.add_argument('--output-dir', type=str, default='experiment_results',
                        help='Directory to save experiment results')
    parser.add_argument('--list', action='store_true',
                        help='List available experiments and exit')
    args = parser.parse_args()

    # List experiments
    if args.list:
        print("\nAvailable experiments:")
        print("-" * 40)
        for key, exp in EXPERIMENTS.items():
            print(f"  {key}: {exp['name']}")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base config
    base_config = load_config(args.base_config)

    # Determine which experiments to run
    experiments_to_run = args.experiments if args.experiments else list(EXPERIMENTS.keys())

    # Run experiments
    results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for exp_key in experiments_to_run:
        if exp_key not in EXPERIMENTS:
            print(f"WARNING: Unknown experiment '{exp_key}', skipping...")
            continue

        exp_config = EXPERIMENTS[exp_key]

        # Merge configs
        config = merge_configs(base_config, exp_config)
        config['experiment']['name'] = f"{base_config['experiment']['name']}_{exp_key}"

        # Save temporary config
        temp_config_path = output_dir / f'config_{exp_key}.yaml'
        save_config(config, str(temp_config_path))

        # Run experiment
        exp_result = run_experiment(str(temp_config_path), exp_config['name'])

        if exp_result:
            results.append({
                'experiment': exp_key,
                'name': exp_config['name'],
                'accuracy': exp_result.get('accuracy', 0),
                'balanced_accuracy': exp_result.get('balanced_accuracy', 0),
                'epochs_trained': exp_result.get('epochs_trained', 0),
                'best_val_acc': exp_result.get('best_val_acc', 0),
            })

    # Summary
    if results:
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)

        df = pd.DataFrame(results)
        df = df.sort_values('balanced_accuracy', ascending=False)

        print(df.to_string(index=False))

        # Save summary
        summary_path = output_dir / f'summary_{timestamp}.csv'
        df.to_csv(summary_path, index=False)
        print(f"\nResults saved to: {summary_path}")

        # Best experiment
        best = df.iloc[0]
        print(f"\nBest experiment: {best['name']}")
        print(f"  Accuracy: {best['accuracy']:.2f}%")
        print(f"  Balanced Accuracy: {best['balanced_accuracy']:.2f}%")


if __name__ == '__main__':
    main()
