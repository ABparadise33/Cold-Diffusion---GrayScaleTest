#!/usr/bin/env bash
# Run after cloning the project on a new Linux NVIDIA GPU instance.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if ! command -v git >/dev/null 2>&1 || ! git lfs version >/dev/null 2>&1; then
  echo 'ERROR: git and git-lfs are required. On Ubuntu: apt-get update && apt-get install -y git git-lfs python3-venv' >&2
  exit 1
fi
bash scripts/setup_cuda_4090.sh
python_bin="$repo_root/.venv/bin/python"
uieb_root="${UIEB_DATA_ROOT:-$repo_root/data/UIEB}"
div2k_root="${DIV2K_DATA_ROOT:-$repo_root/data/DIV2K}"
uieb_reference="${UIEB_REFERENCE_DIR:-$uieb_root/reference-890}"
mixed_root="${MIXED_DATA_ROOT:-$repo_root/data/UIEB_DIV2K}"
if [[ -z "${UIEB_REFERENCE_DIR:-}" ]]; then
  # The downloaded split is an audit artifact; never rewrite the committed split.
  "$python_bin" tools/prepare_uieb.py --data-root "$uieb_root" --output "$uieb_root/download_split.json"
fi
"$python_bin" tools/prepare_div2k.py --data-root "$div2k_root"
"$python_bin" tools/prepare_mixed_colorization.py \
  --uieb-reference "$uieb_reference" --split-file splits/uieb_seed42.json \
  --div2k-train "$div2k_root/DIV2K_train_HR" --div2k-val "$div2k_root/DIV2K_valid_HR" \
  --output "$mixed_root"
echo 'Mixed training environment and datasets are ready.'
echo 'Next: bash scripts/train_mixed_uieb_div2k_4090.sh'
echo 'Keep the same exported data-path variables when starting training.'
