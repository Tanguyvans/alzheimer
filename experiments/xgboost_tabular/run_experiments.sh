#!/bin/bash
# Run XGBoost experiments
#
# Usage:
#   ./run_experiments.sh /path/to/ALL_classes_clinical.csv
#
# This will train and evaluate both:
#   1. Binary classification: CN vs (AD + MCI→AD)
#   2. 4-class classification: CN | MCI stable | MCI→AD | AD

set -e

# Check if input CSV is provided
if [ -z "$1" ]; then
    echo "Usage: ./run_experiments.sh /path/to/ALL_classes_clinical.csv"
    echo ""
    echo "Expected CSV columns:"
    echo "  - Subject: Patient ID"
    echo "  - Group: Original diagnosis (CN, MCI, AD)"
    echo "  - DX: Final diagnosis after follow-up (CN, MCI, AD)"
    echo "  - Clinical features (MMSCORE, CDGLOBAL, etc.)"
    exit 1
fi

INPUT_CSV="$1"
SEED="${2:-42}"

# Check if file exists
if [ ! -f "$INPUT_CSV" ]; then
    echo "Error: File not found: $INPUT_CSV"
    exit 1
fi

echo "========================================"
echo "XGBoost Experiments"
echo "========================================"
echo "Input CSV: $INPUT_CSV"
echo "Random seed: $SEED"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create results directory
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "Experiment 1: Binary Classification"
echo "CN vs (AD + MCI→AD)"
echo "========================================"
python3 "$SCRIPT_DIR/train_binary.py" \
    --input-csv "$INPUT_CSV" \
    --output-dir "$RESULTS_DIR/binary" \
    --seed "$SEED"

echo ""
echo "========================================"
echo "Experiment 2: 4-Class Classification"
echo "CN | MCI stable | MCI→AD | AD"
echo "========================================"
python3 "$SCRIPT_DIR/train_multiclass.py" \
    --input-csv "$INPUT_CSV" \
    --output-dir "$RESULTS_DIR/multiclass" \
    --seed "$SEED"

echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - Binary:     $RESULTS_DIR/binary/"
echo "  - Multiclass: $RESULTS_DIR/multiclass/"
echo ""
echo "Each directory contains:"
echo "  - xgboost_model.json: Trained model"
echo "  - scaler.pkl: Feature scaler"
echo "  - metrics.json: Evaluation metrics"
echo "  - predictions.csv: Test set predictions"
echo "  - feature_importance.csv: Feature importance scores"
echo "  - confusion_matrix.png: Confusion matrix plot"
