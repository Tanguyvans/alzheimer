#!/bin/bash
#SBATCH --job-name=resnet3d_fusion
#SBATCH --output=log_lucia_run_all.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --time=6:30:00
#SBATCH --account=mmfusion

echo "----------------- Setting Python Environment ------------------"
module load devel/python/3.9.13
source ~/alzheimer/env/bin/activate

echo "--------------- Running All Experiments ---------------"
echo -n "Started: "
date

# 1/4 — XGBoost Early Fusion
echo "=========================================="
echo "1/4 — XGBoost Early Fusion"
echo "=========================================="
cd ~/alzheimer/experiments/resnet3d_xgboost
srun python train_finetuned.py --config config.yaml
echo "DONE: XGBoost Early Fusion"

# 2/4 — XGBoost Late Fusion
echo "=========================================="
echo "2/4 — XGBoost Late Fusion"
echo "=========================================="
cd ~/alzheimer/experiments/resnet3d_xgboost
srun python train_late_fusion.py --config config.yaml
echo "DONE: XGBoost Late Fusion"

# 3/4 — MLP Early Fusion
echo "=========================================="
echo "3/4 — MLP Early Fusion"
echo "=========================================="
cd ~/alzheimer/experiments/resnet3d_mlp
srun python train.py --config config.yaml --output-dir results_early
echo "DONE: MLP Early Fusion"

# 4/4 — MLP Late Fusion
echo "=========================================="
echo "4/4 — MLP Late Fusion"
echo "=========================================="
cd ~/alzheimer/experiments/resnet3d_mlp
srun python train_late_fusion.py --config config.yaml
echo "DONE: MLP Late Fusion"

# Analysis
echo "=========================================="
echo "5/5 — Analysis (DeLong, Confusion, ROC, XAI)"
echo "=========================================="
cd ~/alzheimer/experiments
srun python analyze_results.py --gradcam
echo "DONE: Analysis"

echo "=========================================="
echo "ALL COMPLETE"
echo -n "Finished: "
date
