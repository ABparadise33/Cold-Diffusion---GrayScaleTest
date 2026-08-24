#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Use a Linux instance with NVIDIA driver/CUDA support." >&2
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

echo "[3/5] PyTorch 2.5.1 + CUDA 12.1"
if ! "$python_bin" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  "$python_bin" -m pip install --upgrade --force-reinstall \
    "torch==2.5.1" "torchvision==0.20.1" \
    --index-url https://download.pytorch.org/whl/cu121
else
  echo "Reusing the CUDA-enabled PyTorch already installed in .venv."
fi

if ! "$python_bin" -c 'import torchvision; assert torchvision.__version__.startswith("0.20.1")' 2>/dev/null; then
  "$python_bin" -m pip install --upgrade --force-reinstall --no-deps \
    "torchvision==0.20.1" \
    --index-url https://download.pytorch.org/whl/cu121
fi

echo "[4/5] Project dependencies"
"$python_bin" -m pip install -e . pytest

echo "[5/5] CUDA and unit tests"
"$python_bin" tools/check_environment.py --require-cuda --min-vram-gb 20
"$python_bin" -m pytest -q

echo "ALL CHECKS PASSED"
echo "Next: .venv/bin/python tools/prepare_uieb.py"
