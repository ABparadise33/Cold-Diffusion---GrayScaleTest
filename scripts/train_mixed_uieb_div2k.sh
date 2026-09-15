#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${MIXED_PYTHON:-$repo_root/.venv/bin/python}"
div2k_root="${DIV2K_DATA_ROOT:-$repo_root/data/DIV2K}"
uieb_root="${UIEB_DATA_ROOT:-$repo_root/data/UIEB}"
uieb_default="$uieb_root/reference-890"
# Retain the previous workstation layout when no new-instance dataset is present.
if [[ ! -d "$uieb_default" && -z "${UIEB_DATA_ROOT:-}" ]]; then
  uieb_default="$repo_root/../Underwater_Dataset/UIEB/reference-890"
fi
uieb_reference="${UIEB_REFERENCE_DIR:-$uieb_default}"
mixed_root="${MIXED_DATA_ROOT:-$repo_root/data/UIEB_DIV2K}"
if [[ "${MIXED_FULL_SCENE_PREVIEWS:-1}" != 0 && "${MIXED_FULL_SCENE_PREVIEWS:-1}" != 1 ]]; then
  echo 'MIXED_FULL_SCENE_PREVIEWS must be 0 or 1' >&2
  exit 1
fi
"$python_bin" tools/check_environment.py --require-cuda --min-vram-gb 20
"$python_bin" tools/prepare_mixed_colorization.py \
  --uieb-reference "$uieb_reference" --split-file splits/uieb_seed42.json \
  --div2k-train "$div2k_root/DIV2K_train_HR" --div2k-val "$div2k_root/DIV2K_valid_HR" \
  --output "$mixed_root"
mkdir -p logs
"$python_bin" -u train.py --config configs/uieb_div2k_rgb_fullgray_pilot.yaml \
  --train-dir "$mixed_root/train" --val-dir "$mixed_root/val" --device cuda \
  --batch-size "${MIXED_BATCH_SIZE:-4}" --grad-accum "${MIXED_GRAD_ACCUM:-8}" \
  --num-workers "${MIXED_NUM_WORKERS:-4}" "$@" \
  2>&1 | tee "logs/mixed_$(date +%Y%m%d_%H%M%S).log"
