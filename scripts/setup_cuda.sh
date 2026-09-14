#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Use a Linux host with NVIDIA driver/CUDA support." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Python 3.10+ is required." >&2
  exit 1
fi

echo "[1/5] NVIDIA GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "[2/5] Python virtual environment"
if [[ ! -x .venv/bin/python ]]; then
  if command -v conda >/dev/null 2>&1; then
    conda create --prefix "$repo_root/.venv" python=3.10 pip -y
  else
    python3 -m venv .venv
  fi
fi
python_bin="$repo_root/.venv/bin/python"

"$python_bin" -c 'import sys; assert (3, 10) <= sys.version_info < (3, 13), "Python 3.10-3.12 is required for this pinned PyTorch build"'
"$python_bin" -m pip install --upgrade pip "setuptools<82" wheel

echo "[3/5] Select PyTorch/CUDA for detected GPU"
stack_spec=$("$python_bin" tools/select_gpu_stack.py --record .venv/gpu_environment.json)
read -r torch_version vision_version wheel_index cuda_runtime <<< "$stack_spec"
echo "Selected torch=$torch_version torchvision=$vision_version CUDA=$cuda_runtime"
if ! "$python_bin" -c 'import sys, torch, torchvision; assert torch.__version__.split("+")[0] == sys.argv[1]; assert torchvision.__version__.split("+")[0] == sys.argv[2]; assert torch.version.cuda == sys.argv[3]; assert torch.cuda.is_available()' "$torch_version" "$vision_version" "$cuda_runtime" 2>/dev/null; then
  "$python_bin" -m pip install --upgrade --force-reinstall \
    "torch==$torch_version" "torchvision==$vision_version" \
    --index-url "https://download.pytorch.org/whl/$wheel_index"
fi
# Keep dependency resolution from replacing the selected CUDA build.
printf 'torch==%s+%s\ntorchvision==%s+%s\n' "$torch_version" "$wheel_index" "$vision_version" "$wheel_index" > .venv/gpu_constraints.txt

echo "[4/5] Project dependencies"
"$python_bin" -m pip install -c .venv/gpu_constraints.txt -e . pytest
"$python_bin" -m pip check

echo "[5/5] CUDA and unit tests"
"$python_bin" tools/check_environment.py --require-cuda --min-vram-gb 20
"$python_bin" -m pytest -q

echo "ALL CHECKS PASSED"
echo "Next: prepare UIEB or run .venv/bin/python tools/prepare_div2k.py --delete-archives"
