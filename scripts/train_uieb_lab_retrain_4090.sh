#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
raw_dir="${UIEB_RAW_DIR:-$repo_root/data/UIEB/raw-890}"
reference_dir="${UIEB_REFERENCE_DIR:-$repo_root/data/UIEB/reference-890}"
split_file="${UIEB_SPLIT_FILE:-$repo_root/splits/uieb_seed42.json}"
output_dir="${UIEB_RETRAIN_OUTPUT_DIR:-$repo_root/outputs/uieb_lab_retrain_50k}"
batch_size="${COLD_BATCH_SIZE:-16}"
grad_accum="${COLD_GRAD_ACCUM:-1}"
num_workers="${COLD_NUM_WORKERS:-4}"
dry_run="${UIEB_RETRAIN_DRY_RUN:-0}"
resume_requested=0

for argument in "$@"; do
  if [[ "$argument" == "--resume" || "$argument" == "--resume-if-exists" ]]; then
    resume_requested=1
  fi
done

if [[ "$dry_run" != "1" && ! -x "$python_bin" ]]; then
  echo "ERROR: run bash scripts/setup_cuda_4090.sh first." >&2
  exit 1
fi
if [[ "$dry_run" != "1" && ( ! -d "$raw_dir" || ! -d "$reference_dir" || ! -f "$split_file" ) ]]; then
  echo "ERROR: UIEB data is missing. Run .venv/bin/python tools/prepare_uieb.py first." >&2
  exit 1
fi
if [[ "$dry_run" != "1" && -f "$output_dir/checkpoints/latest.pt" && "$resume_requested" == "0" ]]; then
  echo "ERROR: a new-run checkpoint already exists: $output_dir/checkpoints/latest.pt" >&2
  echo "Use --resume auto to continue it, or set a different UIEB_RETRAIN_OUTPUT_DIR." >&2
  exit 1
fi

cd "$repo_root"
echo "Fresh UIEB Lab regression training"
echo "No old checkpoint is loaded unless --resume is explicitly supplied."
echo "Validation image: outputs/uieb_lab_retrain_50k/samples/step_XXXXXX.png every 1000 steps"

command=(
  "$python_bin" -u train.py
  --config configs/uieb_lab_retrain_50k.yaml
  --raw-dir "$raw_dir"
  --reference-dir "$reference_dir"
  --split-file "$split_file"
  --output-dir "$output_dir"
  --device cuda
  --batch-size "$batch_size"
  --grad-accum "$grad_accum"
  --num-workers "$num_workers"
)

if [[ "$dry_run" == "1" ]]; then
  printf 'DRY RUN:'
  printf ' %q' "${command[@]}" "$@"
  printf '\n'
else
  "${command[@]}" "$@"
fi
