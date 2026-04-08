#!/bin/bash
# Resume MLP Late Fusion: seeds 2, 3, 4
# Previous run crashed at seed 2 epoch 4 (OOM)
# Usage: nohup bash run_remaining_mlp_late.sh > run_remaining_mlp_late.log 2>&1 &
set -e

source /home/maxglo/tanguy/env/bin/activate

SEEDS=(2 3 4)

echo "============================================"
echo "MLP Late Fusion — remaining seeds: ${SEEDS[*]}"
echo "Started: $(date)"
echo "============================================"

cd /home/maxglo/tanguy/alzheimer/experiments/resnet3d_mlp
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "[$(date +%H:%M:%S)] Seed $SEED ..."
    python train_late_fusion.py --config config.yaml \
        --output-dir "results_late_fusion/seed_${SEED}" \
        --seed "$SEED"
    echo "  -> Done (seed $SEED)"
done
echo ""
echo "DONE: MLP Late Fusion (seeds ${SEEDS[*]})"

# Analysis
echo ""
echo "=========================================="
echo "Running multi-seed analysis..."
echo "=========================================="
cd /home/maxglo/tanguy/alzheimer/experiments
python analyze_multi_seed.py --gradcam

echo ""
echo "============================================"
echo "ALL COMPLETE: $(date)"
echo "============================================"
