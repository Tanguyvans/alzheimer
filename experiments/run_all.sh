#!/bin/bash
# Run all ResNet3D experiments sequentially
set -e

source /home/tanguy/medical/alzheimer/env/bin/activate

echo "=========================================="
echo "1/4 — XGBoost Early Fusion (train_finetuned)"
echo "=========================================="
cd /home/tanguy/medical/alzheimer/experiments/resnet3d_xgboost
python train_finetuned.py --config config.yaml
echo "DONE: XGBoost Early Fusion"

echo "=========================================="
echo "2/4 — XGBoost Late Fusion"
echo "=========================================="
cd /home/tanguy/medical/alzheimer/experiments/resnet3d_xgboost
python train_late_fusion.py --config config.yaml
echo "DONE: XGBoost Late Fusion"

echo "=========================================="
echo "3/4 — MLP Early Fusion"
echo "=========================================="
cd /home/tanguy/medical/alzheimer/experiments/resnet3d_mlp
python train.py --config config.yaml --output-dir results_early
echo "DONE: MLP Early Fusion"

echo "=========================================="
echo "4/4 — MLP Late Fusion"
echo "=========================================="
cd /home/tanguy/medical/alzheimer/experiments/resnet3d_mlp
python train_late_fusion.py --config config.yaml
echo "DONE: MLP Late Fusion"

echo "=========================================="
echo "ALL RUNS COMPLETE"
echo "=========================================="
