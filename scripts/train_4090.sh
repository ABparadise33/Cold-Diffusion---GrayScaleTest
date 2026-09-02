#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
raw_dir="${COLD_RAW_DIR:-$repo_root/data/UIEB/raw-890}"
reference_dir="${COLD_REFERENCE_DIR:-$repo_root/data/UIEB/reference-890}"
split_file="${COLD_SPLIT_FILE:-$repo_root/splits/uieb_seed42.json}"
batch_size="${COLD_BATCH_SIZE:-16}"
grad_accum="${COLD_GRAD_ACCUM:-1}"
num_workers="${COLD_NUM_WORKERS:-4}"

if [[ ! -x "$python_bin" ]]; then
  echo "ERROR: run bash scripts/setup_cuda_4090.sh first." >&2
  exit 1
fi
if [[ ! -d "$raw_dir" || ! -d "$reference_dir" || ! -f "$split_file" ]]; then
  echo "ERROR: UIEB data or split not found. Run .venv/bin/python tools/prepare_uieb.py first." >&2
  exit 1
fi

cd "$repo_root"
echo "Starting Cold Gray 50k on CUDA"
echo "batch=$batch_size grad_accum=$grad_accum effective_batch=$((batch_size * grad_accum)) workers=$num_workers"

command=(
  "$python_bin" -u train.py
  --config configs/cold_gray_50k.yaml
  --raw-dir "$raw_dir"
  --reference-dir "$reference_dir"
  --split-file "$split_file"
  --device cuda
  --batch-size "$batch_size"
  --grad-accum "$grad_accum"
  --num-workers "$num_workers"
)

if [[ -n "${COLD_OUTPUT_DIR:-}" ]]; then
  command+=(--output-dir "$COLD_OUTPUT_DIR")
fi

"${command[@]}" "$@"
