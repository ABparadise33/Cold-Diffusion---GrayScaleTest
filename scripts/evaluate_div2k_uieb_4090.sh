#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
selection="${1:-all}"

case "$selection" in
  1|1.0)
    checkpoint_dirs=("div2k_rgb_sat1_50k")
    labels=("1.00x")
    ;;
  1.25)
    checkpoint_dirs=("div2k_rgb_sat1_25_50k")
    labels=("1.25x")
    ;;
  1.5)
    checkpoint_dirs=("div2k_rgb_sat1_5_50k")
    labels=("1.50x")
    ;;
  2|2.0)
    checkpoint_dirs=("div2k_rgb_sat2_50k")
    labels=("2.00x")
    ;;
  all)
    checkpoint_dirs=(
      "div2k_rgb_sat1_50k"
      "div2k_rgb_sat1_25_50k"
      "div2k_rgb_sat1_5_50k"
      "div2k_rgb_sat2_50k"
    )
    labels=("1.00x" "1.25x" "1.50x" "2.00x")
    ;;
  *)
    echo "Usage: bash scripts/evaluate_div2k_uieb_4090.sh {1.0|1.25|1.5|2.0|all}" >&2
    exit 2
    ;;
esac
shift

raw_dir="${UIEB_RAW_DIR:-$repo_root/data/UIEB/raw-890}"
reference_dir="${UIEB_REFERENCE_DIR:-$repo_root/data/UIEB/reference-890}"
split_file="${UIEB_SPLIT_FILE:-$repo_root/splits/uieb_seed42.json}"
dry_run="${EVAL_DRY_RUN:-0}"

if [[ "$dry_run" != "1" && ! -x "$python_bin" ]]; then
  echo "ERROR: run bash scripts/setup_cuda_4090.sh first." >&2
  exit 1
fi
if [[ "$dry_run" != "1" && ( ! -d "$raw_dir" || ! -d "$reference_dir" || ! -f "$split_file" ) ]]; then
  echo "ERROR: UIEB data or split is missing. Run .venv/bin/python tools/prepare_uieb.py first." >&2
  exit 1
fi

cd "$repo_root"
echo "Starting ${#checkpoint_dirs[@]} DIV2K saturation evaluation run(s) on UIEB Test 90"

for index in "${!checkpoint_dirs[@]}"; do
  checkpoint_dir="${checkpoint_dirs[$index]}"
  label="${labels[$index]}"
  checkpoint="$repo_root/outputs/$checkpoint_dir/checkpoints/best.pt"
  output_dir="$repo_root/evaluation/div2k_rgb_sat_${label}_uieb"

  if [[ "$dry_run" != "1" && ! -f "$checkpoint" ]]; then
    echo "ERROR: checkpoint not found: $checkpoint" >&2
    exit 1
  fi

  echo "[$((index + 1))/${#checkpoint_dirs[@]}] saturation=$label"
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
done

echo "DIV2K UIEB EVALUATION COMPLETE: ${labels[*]}"
