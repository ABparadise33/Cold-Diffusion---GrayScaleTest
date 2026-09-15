#!/usr/bin/env bash
# Continue the existing official CIFAR10 pilot to 100,000 total updates.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CIFAR_PYTHON="${CIFAR_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
OFFICIAL_DIR="${OFFICIAL_CIFAR_DIR:-$PROJECT_ROOT/data/official_cold_diffusion/decolor-diffusion}"
PIN=f8b1379151ff0cccba49112cf61d439bd4dd4ad9
[[ -x "$CIFAR_PYTHON" ]] || { echo "Missing Python: $CIFAR_PYTHON" >&2; exit 1; }
[[ -f "$OFFICIAL_DIR/train.py" ]] || { echo "Missing existing official checkout: $OFFICIAL_DIR" >&2; exit 1; }
[[ "$(git -C "$OFFICIAL_DIR" rev-parse HEAD)" == "$PIN" ]] || { echo "Official checkout must be pinned to $PIN" >&2; exit 1; }
cd "$OFFICIAL_DIR"
EXP=cifar10_rgb_fullgray_100k
CHECKPOINT="$OFFICIAL_DIR/results/$EXP/model.pt"
export CIFAR_ALLOW_LEGACY_RESUME=0
export CIFAR_INFERENCE_ONLY=0
if [[ ! -f "$CHECKPOINT" ]]; then
    CHECKPOINT="$OFFICIAL_DIR/results/cifar10_rgb_fullgray_10k/model_10000.pt"
    export CIFAR_ALLOW_LEGACY_RESUME=1
fi
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
"$CIFAR_PYTHON" - "$CHECKPOINT" <<'PY'
import sys
import torch
state = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
step = int(state['step'])
if step >= 100000:
    raise SystemExit(f'Already at {step} updates; nothing to train.')
print(f'Resume {sys.argv[1]}: {step} -> 100000 total updates', flush=True)
if 'optimizer' not in state:
    print('Legacy checkpoint: weights and EMA restored; Adam moments restart.', flush=True)
PY
"$CIFAR_PYTHON" "$PROJECT_ROOT/tools/patch_official_cifar_resume.py" --official-dir "$OFFICIAL_DIR"
mkdir -p logs
LOG="logs/${EXP}_resume_$(date +%Y%m%d_%H%M%S).log"
"$CIFAR_PYTHON" -u train.py \
    --dataset cifar10 --dataset_folder ./data --model UnetConvNext \
    --forward_process_type Decolorization --decolor_routine Linear \
    --decolor_total_remove --time_steps 20 --train_steps 100000 \
    --train_routine Final --sampling_routine x0_step_down --loss_type l1 \
    --save_folder ./results --exp_name "$EXP" --load_path "$CHECKPOINT" \
    2>&1 | tee "$LOG"
