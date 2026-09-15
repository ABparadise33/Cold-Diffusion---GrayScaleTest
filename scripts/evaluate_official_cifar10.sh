#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CIFAR_PYTHON="${CIFAR_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
mkdir -p "$PROJECT_ROOT/logs"
"$CIFAR_PYTHON" -u "$PROJECT_ROOT/tools/evaluate_official_cifar10.py" "$@" \
    2>&1 | tee "$PROJECT_ROOT/logs/cifar10_evaluate_$(date +%Y%m%d_%H%M%S).log"
