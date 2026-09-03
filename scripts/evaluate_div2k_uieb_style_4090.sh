#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
selection="${1:-1.0}"

case "$selection" in
  1|1.0)
    checkpoint_dirs=("div2k_uieb_style_lab_sat_1.00x_50k")
    labels=("1.00x")
    ;;
  1.25)
    checkpoint_dirs=("div2k_uieb_style_lab_sat_1.25x_50k")
    labels=("1.25x")
    ;;
  1.5)
    checkpoint_dirs=("div2k_uieb_style_lab_sat_1.50x_50k")
    labels=("1.50x")
    ;;
  2|2.0)
    checkpoint_dirs=("div2k_uieb_style_lab_sat_2.00x_50k")
    labels=("2.00x")
    ;;
  all)
    checkpoint_dirs=(
      "div2k_uieb_style_lab_sat_1.00x_50k"
      "div2k_uieb_style_lab_sat_1.25x_50k"
      "div2k_uieb_style_lab_sat_1.50x_50k"
      "div2k_uieb_style_lab_sat_2.00x_50k"
    )
    labels=("1.00x" "1.25x" "1.50x" "2.00x")
    ;;
  *)
    echo "Usage: bash scripts/evaluate_div2k_uieb_style_4090.sh {1.0|1.25|1.5|2.0|all}" >&2
    exit 2
    ;;
esac
shift

valid_dir="${DIV2K_VAL_DIR:-$repo_root/data/DIV2K/DIV2K_valid_HR}"
split_file="${DIV2K_VAL_SPLIT_FILE:-$repo_root/splits/div2k_valid_all.json}"
tile_size="${DIV2K_EVAL_TILE_SIZE:-512}"
tile_overlap="${DIV2K_EVAL_TILE_OVERLAP:-64}"
dry_run="${EVAL_DRY_RUN:-0}"

if [[ "$dry_run" != "1" && ! -x "$python_bin" ]]; then
  echo "ERROR: run bash scripts/setup_cuda_4090.sh first." >&2
  exit 1
fi
if [[ "$dry_run" != "1" && ( ! -d "$valid_dir" || ! -f "$split_file" ) ]]; then
  echo "ERROR: DIV2K validation data is missing: $valid_dir" >&2
  exit 1
fi

cd "$repo_root"
for index in "${!checkpoint_dirs[@]}"; do
  checkpoint="$repo_root/outputs/${checkpoint_dirs[$index]}/checkpoints/best.pt"
  output_dir="$repo_root/evaluation/div2k_uieb_style_lab_sat_${labels[$index]}_div2k_val"
  if [[ "$dry_run" != "1" && ! -f "$checkpoint" ]]; then
    echo "ERROR: checkpoint not found: $checkpoint" >&2
    exit 1
  fi
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
    --preview-count 6
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
