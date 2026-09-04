#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${OFFICIAL_PYTHON:-$repo_root/.venv/bin/python}"
data_root="${DIV2K_DATA_ROOT:-$repo_root/data/DIV2K}"
checkpoint="${OFFICIAL_CHECKPOINT:-outputs/div2k_official_rgb_sat1.00x_50k/checkpoints/step_050000.pt}"
output_root="${OFFICIAL_EVAL_OUTPUT:-evaluation/div2k_official_rgb_sat1.00x_step050000}"
# Both labels use identical weights and full-gray inputs; never select best.pt silently.
for sampler in paper_algorithm2 official_code; do
  command=("$python_bin" -u evaluate.py --checkpoint "$checkpoint"
    --raw-dir "$data_root/DIV2K_valid_HR" --reference-dir "$data_root/DIV2K_valid_HR"
    --split-file splits/div2k_valid_all.json --split test --device cuda
    --original-size --batch-size 1 --tile-size "${OFFICIAL_TILE_SIZE:-256}" --tile-overlap 32
    --sampler "$sampler" --preview-count 4 --preview-max-side 512
    --output-dir "$output_root/$sampler")
  if [[ "${OFFICIAL_DRY_RUN:-0}" == 1 ]]; then
    printf 'DRY RUN:'
    printf ' %q' "${command[@]}" "$@"
    printf '\n'
  else
    "${command[@]}" "$@"
  fi
done
