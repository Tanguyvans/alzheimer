#!/bin/bash
# Run all 4 ResNet3D experiments with 15 seeds each
# Usage: nohup bash run_all_seeds.sh > run_all_seeds.log 2>&1 &
set -e

source /home/tanguy/medical/alzheimer/env/bin/activate

SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14)
N_SEEDS=${#SEEDS[@]}

echo "============================================"
echo "Multi-seed experiment run"
echo "Seeds: ${SEEDS[*]}"
echo "Total runs: $((N_SEEDS * 4))"
echo "Started: $(date)"
echo "============================================"

# ── 1/4 — XGBoost Early Fusion ──
echo ""
echo "=========================================="
echo "1/4 — XGBoost Early Fusion ($N_SEEDS seeds)"
echo "=========================================="
cd /home/tanguy/medical/alzheimer/experiments/resnet3d_xgboost
for SEED in "${SEEDS[@]}"; do
    echo "[$(date +%H:%M:%S)] Seed $SEED ..."
    python train_finetuned.py --config config.yaml \
        --output-dir "results_finetuned/seed_${SEED}" \
        --seed "$SEED"
    echo "  -> Done (seed $SEED)"
done
echo "DONE: XGBoost Early Fusion (all seeds)"

# ── 2/4 — XGBoost Late Fusion ──
echo ""
echo "=========================================="
echo "2/4 — XGBoost Late Fusion ($N_SEEDS seeds)"
echo "=========================================="
cd /home/tanguy/medical/alzheimer/experiments/resnet3d_xgboost
for SEED in "${SEEDS[@]}"; do
    echo "[$(date +%H:%M:%S)] Seed $SEED ..."
    python train_late_fusion.py --config config.yaml \
        --output-dir "results_late_fusion/seed_${SEED}" \
        --seed "$SEED"
    echo "  -> Done (seed $SEED)"
done
echo "DONE: XGBoost Late Fusion (all seeds)"

# ── 3/4 — MLP Early Fusion ──
echo ""
echo "=========================================="
echo "3/4 — MLP Early Fusion ($N_SEEDS seeds)"
echo "=========================================="
cd /home/tanguy/medical/alzheimer/experiments/resnet3d_mlp
for SEED in "${SEEDS[@]}"; do
    echo "[$(date +%H:%M:%S)] Seed $SEED ..."
    python train.py --config config.yaml \
        --output-dir "results_early/seed_${SEED}" \
        --seed "$SEED"
    echo "  -> Done (seed $SEED)"
done
echo "DONE: MLP Early Fusion (all seeds)"

# ── 4/4 — MLP Late Fusion ──
echo ""
echo "=========================================="
echo "4/4 — MLP Late Fusion ($N_SEEDS seeds)"
echo "=========================================="
cd /home/tanguy/medical/alzheimer/experiments/resnet3d_mlp
for SEED in "${SEEDS[@]}"; do
    echo "[$(date +%H:%M:%S)] Seed $SEED ..."
    python train_late_fusion.py --config config.yaml \
        --output-dir "results_late_fusion/seed_${SEED}" \
        --seed "$SEED"
    echo "  -> Done (seed $SEED)"
done
echo "DONE: MLP Late Fusion (all seeds)"

# ── Analysis ──
echo ""
echo "=========================================="
echo "Running multi-seed analysis..."
echo "=========================================="
cd /home/tanguy/medical/alzheimer/experiments
python analyze_multi_seed.py --gradcam

echo ""
echo "============================================"
echo "ALL COMPLETE: $(date)"
echo "============================================"
