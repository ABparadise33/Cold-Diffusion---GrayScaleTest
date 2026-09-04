#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
selection="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "$selection" in
  all) factors=(1.0 1.25 1.5 2.0) ;;
  1|1.0) factors=(1.0) ;;
  1.25) factors=(1.25) ;;
  1.5) factors=(1.5) ;;
  2|2.0) factors=(2.0) ;;
  *) echo "Usage: bash scripts/evaluate_rgb_partial_uieb_4090.sh {all|1|1.25|1.5|2} [options]" >&2; exit 2 ;;
esac
dry_run="${EVAL_DRY_RUN:-0}"
if [[ "$dry_run" != "1" && ! -x "$python_bin" ]]; then
  echo "ERROR: run bash scripts/setup_cuda_4090.sh first." >&2
  exit 1
fi
command=(
  "$python_bin" -u "$repo_root/evaluate_rgb_partial_uieb.py"
  --factors "${factors[@]}"
  --checkpoint-root "${RGB_CHECKPOINT_ROOT:-$repo_root/outputs}"
  --raw-dir "${UIEB_RAW_DIR:-$repo_root/data/UIEB/raw-890}"
  --reference-dir "${UIEB_REFERENCE_DIR:-$repo_root/data/UIEB/reference-890}"
  --split-file "${UIEB_SPLIT_FILE:-$repo_root/splits/uieb_seed42.json}"
  --output-dir "${RGB_PARTIAL_OUTPUT_DIR:-$repo_root/evaluation/rgb_partial_uieb_test90_step050000}"
  --device cuda --limit 0 --preview-count 4 --start-steps 15
)
if [[ "$dry_run" == "1" ]]; then
  printf 'DRY RUN:'
  printf ' %q' "${command[@]}" "$@"
  printf '\n'
else
  cd "$repo_root"
  "${command[@]}" "$@"
fi
