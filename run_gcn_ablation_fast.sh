#!/usr/bin/env bash
# Quick GCN ablation runner (≈1–1.5h total if run sequentially on one GPU).
# Runs the five short experiments discussed in the ablation plan with consistent logging.

set -euo pipefail

DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-200}"
EVAL_STEPS="${EVAL_STEPS:-10}"
RUNS="${RUNS:-1}"

run_exp() {
  local tag="$1"; shift
  echo "=== [${tag}] starting ==="
  pixi run python train_ddi_reference.py \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --eval_steps "${EVAL_STEPS}" \
    --runs "${RUNS}" \
    "$@"
  echo "=== [${tag}] done ==="
}

# E1: 2-layer, no dropout (baseline)
run_exp "E1_L2_H256_do0.0_lr0.005_bs64k" \
  --num_layers 2 --hidden_channels 256 --dropout 0.0 \
  --lr 0.005 --batch_size 65536

# E2: 2-layer, dropout 0.25
run_exp "E2_L2_H256_do0.25_lr0.005_bs64k" \
  --num_layers 2 --hidden_channels 256 --dropout 0.25 \
  --lr 0.005 --batch_size 65536

# E3: 3-layer, zero-drop, low LR
run_exp "E3_L3_H192_do0.0_lr0.001_bs32k" \
  --num_layers 3 --hidden_channels 192 --dropout 0.0 \
  --lr 0.001 --batch_size 32768

# E4: 3-layer, light dropout
run_exp "E4_L3_H192_do0.1_lr0.001_bs32k" \
  --num_layers 3 --hidden_channels 192 --dropout 0.1 \
  --lr 0.001 --batch_size 32768

# E5: 3-layer, zero-drop, LR bump
run_exp "E5_L3_H192_do0.0_lr0.0015_bs32k" \
  --num_layers 3 --hidden_channels 192 --dropout 0.0 \
  --lr 0.0015 --batch_size 32768
