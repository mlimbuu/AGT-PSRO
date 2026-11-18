#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEED="${SEED:-11}"
STEPS="${STEPS:-1000}"
ITERATIONS="${ITERATIONS:-5}"
for AGENTS in 2 3; do
  "$PYTHON_BIN" "$ROOT/experiments/quick/run_svi.py" --scenario five_node_two_risk --agents "$AGENTS" --ammo 1
  "$PYTHON_BIN" "$ROOT/experiments/quick/run_psro.py" \
    --scenario five_node_two_risk --agents "$AGENTS" --init scratch \
    --seed "$SEED" --steps "$STEPS" --iterations "$ITERATIONS" \
    --threads 1 --skip-svi "$@"
done
"$PYTHON_BIN" "$ROOT/results/code/compare_agents.py"
