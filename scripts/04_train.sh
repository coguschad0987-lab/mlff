#!/bin/bash
# 이 스크립트가 하는 일:
# mace-mp small 파운데이션 모델을 data/processed/train.xyz(SiN)로 fine-tuning한다.
# MPS보다 CPU가 더 빠른 것으로 벤치마크되어(e3nn 텐서 연산이 MPS에 최적화가 덜 되어 있음) --device=cpu로 실행한다.

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

mace_run_train \
  --name="mace_v1" \
  --train_file="data/processed/train.xyz" \
  --valid_file="data/processed/valid.xyz" \
  --foundation_model="models/mace_mp_small_float32.model" \
  --multiheads_finetuning=False \
  --energy_key="energy" \
  --forces_key="forces" \
  --E0s="average" \
  --batch_size=4 \
  --max_num_epochs=1 \
  --ema --ema_decay=0.99 \
  --energy_weight=1 --forces_weight=100 \
  --default_dtype=float32 \
  --device=cpu \
  --seed=42 \
  --model_dir="models" \
  --checkpoints_dir="models/checkpoints" \
  --results_dir="results/logs" \
  --log_dir="results/logs"
