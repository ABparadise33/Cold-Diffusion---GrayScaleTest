#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${SPATIAL_PYTHON:-$repo_root/.venv/bin/python}"
data_root="${DIV2K_DATA_ROOT:-$repo_root/data/DIV2K}"
auto_batch=0
training_args=()
for argument in "$@"; do
  if [[ "$argument" == '--auto-batch' ]]; then
    auto_batch=1
  else
    training_args+=("$argument")
  fi
done
if [[ "$auto_batch" == 1 ]]; then
  if [[ -n "${SPATIAL_BATCH_SIZE:-}" || -n "${SPATIAL_GRAD_ACCUM:-}" ]]; then
    echo 'ERROR: --auto-batch cannot be combined with manual batch/accumulation settings.' >&2
    exit 2
  fi
  entrypoint=tools/autotune_official_batch.py
else
  entrypoint=train.py
fi
command=("$python_bin" -u "$entrypoint"
  --config configs/div2k_spatial_chroma_sat1_t20_pilot.yaml
  --train-dir "$data_root/DIV2K_train_HR"
  --val-dir "$data_root/DIV2K_valid_HR"
  --device cuda
  --num-workers "${SPATIAL_NUM_WORKERS:-4}")
if [[ "$auto_batch" == 0 ]]; then
  command+=(--batch-size "${SPATIAL_BATCH_SIZE:-4}" --grad-accum "${SPATIAL_GRAD_ACCUM:-8}")
fi
if (( ${#training_args[@]} )); then
  command+=("${training_args[@]}")
fi
if [[ "${SPATIAL_DRY_RUN:-0}" == 1 ]]; then
  printf 'DRY RUN:'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi
if [[ ! -x "$python_bin" ]]; then
  echo 'Missing Python environment. Run bash scripts/setup_cuda_4090.sh first.' >&2
  exit 1
fi
"$python_bin" tools/check_environment.py --require-cuda --min-vram-gb 20
"$python_bin" tools/prepare_div2k.py --data-root "$data_root" --skip-download
echo 'Pilot: DIV2K800, saturation1, T20 nested spatial chroma masks, upstream ConvNeXt56.6M.'
echo 'Default stops at 10k; validation and five random full-scene previews every 1k.'
echo 'Fresh by default; explicit --resume --max-steps 50000 continues the same run.'
"${command[@]}"
