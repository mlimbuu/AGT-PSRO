#!/usr/bin/env bash
set -euo pipefail
# Launch from ANY working directory. All experimental outputs stay in results/.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" "$PROJECT_ROOT/experiments/long/run_ppo_ablations.py" "$@"
