#!/bin/bash
# 이 스크립트가 하는 일:
# loss 함수만 다르게 해서 같은 서브셋으로 학습하는 A/B 비교 실험을 돌린다.
# 사용법: bash scripts/04b_ab_loss.sh <실험이름> <loss옵션들>
#   예) bash scripts/04b_ab_loss.sh A_weighted --loss=weighted
#       bash scripts/04b_ab_loss.sh B_huber --loss=huber --huber_delta=0.01

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

NAME="$1"
shift  # 첫 인자를 빼고 나머지를 loss 옵션으로 넘김

OUT="results/ab_loss/${NAME}"
mkdir -p "$OUT"

mace_run_train \
  --name="${NAME}" \
  --train_file="data/processed/train_sub2000.xyz" \
  --valid_file="data/processed/valid_sub800.xyz" \
  --foundation_model="models/mace_mp_small_float32.model" \
  --multiheads_finetuning=False \
  --energy_key="energy" \
  --forces_key="forces" \
  --E0s="average" \
  --batch_size=4 \
  --max_num_epochs=5 \
  --ema --ema_decay=0.99 \
  --energy_weight=1 --forces_weight=100 \
  --default_dtype=float32 \
  --device=cpu \
  --seed=42 \
  "$@" \
  --model_dir="${OUT}" \
  --checkpoints_dir="${OUT}/checkpoints" \
  --results_dir="${OUT}" \
  --log_dir="${OUT}"
