#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_root/.venv/bin/python"

if [[ ! -x "$python_bin" ]]; then
  echo "ERROR: run bash scripts/setup_gputa_4090.sh first." >&2
  exit 1
fi

smoke_root="$(mktemp -d /tmp/gray_cold_cuda_smoke.XXXXXX)"
trap 'rm -rf "$smoke_root"' EXIT

cd "$repo_root"
"$python_bin" tools/check_environment.py --require-cuda --min-vram-gb 20
"$python_bin" tools/make_smoke_dataset.py --output "$smoke_root/data"

common_args=(
  --config configs/smoke.yaml
  --raw-dir "$smoke_root/data/raw"
  --reference-dir "$smoke_root/data/reference"
  --split-file "$smoke_root/data/split.json"
  --output-dir "$smoke_root/output"
  --device cuda
)

"$python_bin" train.py "${common_args[@]}"
"$python_bin" train.py "${common_args[@]}" --resume auto --max-steps 4

natural_args=(
  --config configs/smoke_div2k.yaml
  --train-dir "$smoke_root/data/raw"
  --val-dir "$smoke_root/data/reference"
  --output-dir "$smoke_root/output_div2k"
  --device cuda
)

"$python_bin" train.py "${natural_args[@]}"
"$python_bin" train.py "${natural_args[@]}" --resume auto --max-steps 4

echo "CUDA SMOKE TEST PASSED: UIEB Lab and DIV2K RGB training/resume both work."
