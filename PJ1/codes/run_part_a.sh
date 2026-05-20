#!/bin/bash
# Part A: train MLP baseline, then evaluate on test set (for Huawei Cloud Notebook)
set -e
cd "$(dirname "$0")"
echo "[1/2] Training MLP (5 epochs, may take 1-2 hours)..."
python test_train.py
echo ""
echo "[2/2] Evaluating on MNIST test set..."
python test_model.py
echo "Done. Check results/ for learning curve and summary."
