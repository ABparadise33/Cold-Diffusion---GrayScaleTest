#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
checkpoint="${RGB_DIAGNOSTIC_CHECKPOINT:-$repo_root/outputs/div2k_rgb_sat1_50k/checkpoints/step_050000.pt}"
dry_run="${EVAL_DRY_RUN:-0}"

if [[ "$dry_run" != "1" && ! -x "$python_bin" ]]; then
  echo "ERROR: run bash scripts/setup_cuda_4090.sh first." >&2
  exit 1
fi
if [[ "$dry_run" != "1" && ! -f "$checkpoint" ]]; then
  echo "ERROR: existing RGB factor-1 checkpoint not found: $checkpoint" >&2
  echo "Set RGB_DIAGNOSTIC_CHECKPOINT to the existing RGB checkpoint if stored elsewhere." >&2
  echo "The embedded step must be 50000. Do not use Lab weights or start new training." >&2
  exit 1
fi

cd "$repo_root"
command=(
  "$python_bin" -u diagnose_rgb_chroma.py
  --checkpoint "$checkpoint"
  --baseline-config "$repo_root/configs/div2k_rgb_sat1_50k.yaml"
  --expected-checkpoint-step 50000
  --device cuda
  --limit 4
  --start-steps 10 15 18
  --include-gray-control
  --output-dir "$repo_root/evaluation/div2k_rgb_sat_1.00x_partial_t20_step050000"
)
if [[ "$dry_run" == "1" ]]; then
  printf 'DRY RUN:'
  printf ' %q' "${command[@]}" "$@"
  printf '\n'
else
  "${command[@]}" "$@"
fi
