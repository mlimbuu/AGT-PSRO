#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements-wandb.txt"
echo "W&B installed. Use 'wandb login' for online runs, or --wandb-mode offline."
