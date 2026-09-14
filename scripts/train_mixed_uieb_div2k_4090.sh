#!/usr/bin/env bash
# Compatibility entrypoint; GPU selection is automatic.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/train_mixed_uieb_div2k.sh" "$@"
