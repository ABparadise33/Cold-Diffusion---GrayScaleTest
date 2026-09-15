#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${MIXED_PYTHON:-$repo_root/.venv/bin/python}"
div2k_root="${DIV2K_DATA_ROOT:-$repo_root/data/DIV2K}"
mkdir -p evaluation
output_dir=$(mktemp -d evaluation/natural_fullgray_XXXXXX)
"$python_bin" -u tools/diagnose_natural_fullgray.py \
  --checkpoint "${MIXED_CHECKPOINT:-outputs/uieb_div2k_rgb_fullgray_pilot/checkpoints/best.pt}" \
  --train-dir "$div2k_root/DIV2K_train_HR" --val-dir "$div2k_root/DIV2K_valid_HR" \
  --output-dir "$output_dir" "$@"
echo "Diagnostic results: $output_dir"
