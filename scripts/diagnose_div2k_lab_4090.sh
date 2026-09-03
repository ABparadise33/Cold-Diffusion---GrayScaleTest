#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
checkpoint="$repo_root/outputs/div2k_uieb_style_lab_sat_1.00x_50k/checkpoints/step_050000.pt"
dry_run="${EVAL_DRY_RUN:-0}"

if [[ "$dry_run" != "1" && ! -x "$python_bin" ]]; then
  echo "ERROR: run bash scripts/setup_cuda_4090.sh first." >&2
  exit 1
fi
if [[ "$dry_run" != "1" && ! -f "$checkpoint" ]]; then
  echo "ERROR: missing the required T=8 DIV2K 50k checkpoint: $checkpoint" >&2
  echo "If you have not trained this control, run: bash scripts/train_div2k_uieb_style_4090.sh 1.0" >&2
  echo "Do not substitute a UIEB checkpoint, best.pt, or the older T=20 DIV2K model." >&2
  exit 1
fi

cd "$repo_root"
command=(
  "$python_bin" -u diagnose_lab_chroma.py
  --checkpoint "$checkpoint"
  --expected-checkpoint-step 50000
  --device cuda
  --limit 4
  --start-steps 4 6 7
  --output-dir "$repo_root/evaluation/div2k_lab_partial_t8_step050000"
)
if [[ "$dry_run" == "1" ]]; then
  printf 'DRY RUN:'
  printf ' %q' "${command[@]}" "$@"
  printf '\n'
else
  "${command[@]}" "$@"
fi
