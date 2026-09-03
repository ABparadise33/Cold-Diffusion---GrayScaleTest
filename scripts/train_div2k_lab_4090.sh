#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
config="$repo_root/configs/div2k_lab_sat1_50k.yaml"
train_dir="${DIV2K_TRAIN_DIR:-$repo_root/data/DIV2K/DIV2K_train_HR}"
val_dir="${DIV2K_VAL_DIR:-$repo_root/data/DIV2K/DIV2K_valid_HR}"
batch_size="${DIV2K_BATCH_SIZE:-16}"
grad_accum="${DIV2K_GRAD_ACCUM:-1}"
num_workers="${DIV2K_NUM_WORKERS:-4}"
dry_run="${DIV2K_DRY_RUN:-0}"

if [[ "$dry_run" != "1" && ! -x "$python_bin" ]]; then
  echo "ERROR: run bash scripts/setup_cuda_4090.sh first." >&2
  exit 1
fi
if [[ "$dry_run" != "1" && ( ! -d "$train_dir" || ! -d "$val_dir" ) ]]; then
  echo "ERROR: DIV2K train/val data not found." >&2
  echo "Run: .venv/bin/python tools/prepare_div2k.py --delete-archives" >&2
  exit 1
fi

cd "$repo_root"
echo "Starting DIV2K Lab Cold colorization: saturation=1.0, steps=20"
echo "batch=$batch_size grad_accum=$grad_accum effective_batch=$((batch_size * grad_accum)) workers=$num_workers"

command=(
  "$python_bin" -u train.py
  --config "$config"
  --train-dir "$train_dir"
  --val-dir "$val_dir"
  --device cuda
  --batch-size "$batch_size"
  --grad-accum "$grad_accum"
  --num-workers "$num_workers"
)

if [[ -n "${DIV2K_OUTPUT_DIR:-}" ]]; then
  command+=(--output-dir "$DIV2K_OUTPUT_DIR")
fi

if [[ "$dry_run" == "1" ]]; then
  printf 'DRY RUN:'
  printf ' %q' "${command[@]}" "$@"
  printf '\n'
else
  "${command[@]}" "$@"
fi

echo "DIV2K LAB SATURATION 1.0 TRAINING COMPLETE"
