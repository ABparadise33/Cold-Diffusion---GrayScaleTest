#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${OFFICIAL_PYTHON:-$repo_root/.venv/bin/python}"
checkpoint="${OFFICIAL_CHECKPOINT:-outputs/div2k_official_rgb_sat1.00x_50k/checkpoints/step_050000.pt}"
output_root="${OFFICIAL_EVAL_OUTPUT:-evaluation/official_rgb_partial_uieb_test90_step050000}"
sampler="${OFFICIAL_SAMPLER:-paper_algorithm2}"
for arg in "$@"; do
  case "$arg" in
    --retain-color-percent|--retain-color-percent=*|--output-dir|--output-dir=*|--sampler|--sampler=*)
      echo "Do not override condition/output labels. Use OFFICIAL_SAMPLER or OFFICIAL_EVAL_OUTPUT." >&2
      exit 2 ;;
  esac
done
for retained in 5 25; do
  command=("$python_bin" -u evaluate.py --checkpoint "$checkpoint" --expected-checkpoint-step 50000
    --raw-dir "${UIEB_RAW_DIR:-data/UIEB/raw-890}"
    --reference-dir "${UIEB_REFERENCE_DIR:-data/UIEB/reference-890}"
    --split-file "${UIEB_SPLIT_FILE:-splits/uieb_seed42.json}" --split test --device cuda
    --original-size --batch-size 1 --tile-size "${OFFICIAL_TILE_SIZE:-256}" --tile-overlap 32
    --sampler "$sampler" --retain-color-percent "$retained" --preview-count 4 --preview-max-side 512
    --output-layout compact --output-dir "$output_root/$sampler/retain_${retained}pct")
  if [[ "${OFFICIAL_DRY_RUN:-0}" == 1 ]]; then
    printf 'DRY RUN:'
    printf ' %q' "${command[@]}" "$@"
    printf '\n'
  else
    "${command[@]}" "$@"
  fi
done
