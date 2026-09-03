#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
selection="${1:-lab}"

case "$selection" in
  lab)
    checkpoint_dirs=("div2k_lab_sat1_50k")
    output_names=("div2k_lab_sat_1.00x_div2k_val")
    ;;
  rgb)
    checkpoint_dirs=("div2k_rgb_sat1_50k")
    output_names=("div2k_rgb_sat_1.00x_div2k_val")
    ;;
  both)
    checkpoint_dirs=("div2k_lab_sat1_50k" "div2k_rgb_sat1_50k")
    output_names=("div2k_lab_sat_1.00x_div2k_val" "div2k_rgb_sat_1.00x_div2k_val")
    ;;
  *)
    echo "Usage: bash scripts/evaluate_div2k_validation_4090.sh {lab|rgb|both}" >&2
    exit 2
    ;;
esac
shift

valid_dir="${DIV2K_VAL_DIR:-$repo_root/data/DIV2K/DIV2K_valid_HR}"
split_file="${DIV2K_VAL_SPLIT_FILE:-$repo_root/splits/div2k_valid_all.json}"
tile_size="${DIV2K_EVAL_TILE_SIZE:-512}"
tile_overlap="${DIV2K_EVAL_TILE_OVERLAP:-64}"
preview_count="${DIV2K_EVAL_PREVIEW_COUNT:-6}"
dry_run="${EVAL_DRY_RUN:-0}"

if [[ "$dry_run" != "1" && ! -x "$python_bin" ]]; then
  echo "ERROR: run bash scripts/setup_cuda_4090.sh first." >&2
  exit 1
fi
if [[ "$dry_run" != "1" && ( ! -d "$valid_dir" || ! -f "$split_file" ) ]]; then
  echo "ERROR: DIV2K validation data is missing: $valid_dir" >&2
  echo "Run .venv/bin/python tools/prepare_div2k.py first." >&2
  exit 1
fi

cd "$repo_root"
echo "Evaluating ${#checkpoint_dirs[@]} model(s) on all 100 full-resolution DIV2K validation images"

for index in "${!checkpoint_dirs[@]}"; do
  checkpoint="$repo_root/outputs/${checkpoint_dirs[$index]}/checkpoints/best.pt"
  output_dir="$repo_root/evaluation/${output_names[$index]}"
  if [[ "$dry_run" != "1" && ! -f "$checkpoint" ]]; then
    echo "ERROR: checkpoint not found: $checkpoint" >&2
    exit 1
  fi

  echo "[$((index + 1))/${#checkpoint_dirs[@]}] ${checkpoint_dirs[$index]}"
  command=(
    "$python_bin" -u evaluate.py
    --checkpoint "$checkpoint"
    --raw-dir "$valid_dir"
    --reference-dir "$valid_dir"
    --split-file "$split_file"
    --split test
    --device cuda
    --original-size
    --batch-size 1
    --tile-size "$tile_size"
    --tile-overlap "$tile_overlap"
    --preview-count "$preview_count"
    --preview-max-side 512
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
done

echo "DIV2K VALIDATION EVALUATION COMPLETE"
