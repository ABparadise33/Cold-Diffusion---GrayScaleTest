#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
saturation="${1:-}"

case "$saturation" in
  1|1.0)
    saturations=("1.0")
    configs=("$repo_root/configs/div2k_rgb_sat1_50k.yaml")
    ;;
  1.25)
    saturations=("1.25")
    configs=("$repo_root/configs/div2k_rgb_sat1_25_50k.yaml")
    ;;
  1.5)
    saturations=("1.5")
    configs=("$repo_root/configs/div2k_rgb_sat1_5_50k.yaml")
    ;;
  2|2.0)
    saturations=("2.0")
    configs=("$repo_root/configs/div2k_rgb_sat2_50k.yaml")
    ;;
  all)
    saturations=("1.0" "1.25" "1.5" "2.0")
    configs=(
      "$repo_root/configs/div2k_rgb_sat1_50k.yaml"
      "$repo_root/configs/div2k_rgb_sat1_25_50k.yaml"
      "$repo_root/configs/div2k_rgb_sat1_5_50k.yaml"
      "$repo_root/configs/div2k_rgb_sat2_50k.yaml"
    )
    ;;
  *)
    echo "Usage: bash scripts/train_div2k_4090.sh {1.0|1.25|1.5|2.0|all} [train.py options]" >&2
    exit 2
    ;;
esac
shift

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
echo "Starting ${#configs[@]} DIV2K RGB Cold colorization run(s) on CUDA"
echo "batch=$batch_size grad_accum=$grad_accum effective_batch=$((batch_size * grad_accum)) workers=$num_workers"

if [[ ${#configs[@]} -gt 1 && -n "${DIV2K_OUTPUT_DIR:-}" ]]; then
  echo "ERROR: DIV2K_OUTPUT_DIR can only be used with a single saturation run." >&2
  exit 2
fi

for index in "${!configs[@]}"; do
  config="${configs[$index]}"
  factor="${saturations[$index]}"
  echo "[$((index + 1))/${#configs[@]}] saturation=$factor config=$config"

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
done

echo "DIV2K SATURATION TRAINING COMPLETE: ${saturations[*]}"
