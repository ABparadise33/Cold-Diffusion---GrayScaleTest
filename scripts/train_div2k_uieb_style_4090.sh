#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
selection="${1:-1.0}"

case "$selection" in
  1|1.0)
    configs=("div2k_uieb_style_lab_sat1_50k.yaml")
    labels=("1.00x")
    ;;
  1.25)
    configs=("div2k_uieb_style_lab_sat1_25_50k.yaml")
    labels=("1.25x")
    ;;
  1.5)
    configs=("div2k_uieb_style_lab_sat1_5_50k.yaml")
    labels=("1.50x")
    ;;
  2|2.0)
    configs=("div2k_uieb_style_lab_sat2_50k.yaml")
    labels=("2.00x")
    ;;
  all)
    configs=(
      "div2k_uieb_style_lab_sat1_50k.yaml"
      "div2k_uieb_style_lab_sat1_25_50k.yaml"
      "div2k_uieb_style_lab_sat1_5_50k.yaml"
      "div2k_uieb_style_lab_sat2_50k.yaml"
    )
    labels=("1.00x" "1.25x" "1.50x" "2.00x")
    ;;
  *)
    echo "Usage: bash scripts/train_div2k_uieb_style_4090.sh {1.0|1.25|1.5|2.0|all}" >&2
    exit 2
    ;;
esac
shift

data_root="${DIV2K_DATA_ROOT:-$repo_root/data/DIV2K}"
split_file="${DIV2K_UIEB_STYLE_SPLIT:-$repo_root/splits/div2k_uieb_style_seed42.json}"
batch_size="${DIV2K_BATCH_SIZE:-16}"
grad_accum="${DIV2K_GRAD_ACCUM:-1}"
num_workers="${DIV2K_NUM_WORKERS:-4}"
dry_run="${DIV2K_UIEB_STYLE_DRY_RUN:-0}"

if [[ "$dry_run" != "1" && ! -x "$python_bin" ]]; then
  echo "ERROR: run bash scripts/setup_cuda_4090.sh first." >&2
  exit 1
fi
if [[ "$dry_run" != "1" && ( ! -d "$data_root/DIV2K_train_HR" || ! -d "$data_root/DIV2K_valid_HR" || ! -f "$split_file" ) ]]; then
  echo "ERROR: DIV2K train/validation data is missing under $data_root" >&2
  echo "Run .venv/bin/python tools/prepare_div2k.py first." >&2
  exit 1
fi

cd "$repo_root"
echo "UIEB-style paired Lab training on DIV2K"
echo "T=8, validation preview=0803 every 1000 steps"

for index in "${!configs[@]}"; do
  echo "[$((index + 1))/${#configs[@]}] target saturation=${labels[$index]}"
  command=(
    "$python_bin" -u train.py
    --config "configs/${configs[$index]}"
    --raw-dir "$data_root"
    --reference-dir "$data_root"
    --split-file "$split_file"
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
done

echo "DIV2K UIEB-STYLE TRAINING COMPLETE"
