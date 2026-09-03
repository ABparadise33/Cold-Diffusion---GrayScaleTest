#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
checkpoint="${DIV2K_CHECKPOINT:-$repo_root/outputs/div2k_lab_sat1_50k/checkpoints/best.pt}"
raw_dir="${UIEB_RAW_DIR:-$repo_root/data/UIEB/raw-890}"
reference_dir="${UIEB_REFERENCE_DIR:-$repo_root/data/UIEB/reference-890}"
split_file="${UIEB_SPLIT_FILE:-$repo_root/splits/uieb_seed42.json}"
output_dir="${DIV2K_EVAL_OUTPUT_DIR:-$repo_root/evaluation/div2k_lab_sat_1.00x_uieb}"
dry_run="${EVAL_DRY_RUN:-0}"

if [[ "$dry_run" != "1" && ! -x "$python_bin" ]]; then
  echo "ERROR: run bash scripts/setup_cuda_4090.sh first." >&2
  exit 1
fi
if [[ "$dry_run" != "1" && ! -f "$checkpoint" ]]; then
  echo "ERROR: checkpoint not found: $checkpoint" >&2
  exit 1
fi
if [[ "$dry_run" != "1" && ( ! -d "$raw_dir" || ! -d "$reference_dir" || ! -f "$split_file" ) ]]; then
  echo "ERROR: UIEB data or split is missing. Run .venv/bin/python tools/prepare_uieb.py first." >&2
  exit 1
fi

cd "$repo_root"
echo "Evaluating DIV2K Lab saturation 1.0 on UIEB Test 90"
command=(
  "$python_bin" -u evaluate.py
  --checkpoint "$checkpoint"
  --raw-dir "$raw_dir"
  --reference-dir "$reference_dir"
  --split-file "$split_file"
  --split test
  --device cuda
  --original-size
  --batch-size 1
  --extended-metrics
  --extended-metric-size 256
  --output-dir "$output_dir"
)

if [[ "$dry_run" == "1" ]]; then
  printf 'DRY RUN:'
  printf ' %q' "${command[@]}" "$@"
  printf '\n'
else
  "${command[@]}" "$@"
fi

echo "DIV2K LAB UIEB EVALUATION COMPLETE"
