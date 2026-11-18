#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Whitespace-delimited budgets are intentional; set PSRO_BUDGETS="1000 3000" to customize.
read -r -a budgets <<< "${PSRO_BUDGETS:-1000 3000 6000 10000}"
if [ "${#budgets[@]}" -eq 0 ]; then
  echo "PSRO_BUDGETS must contain at least one step count" >&2
  exit 2
fi
for budget in "${budgets[@]}"; do
  if [[ ! "$budget" =~ ^[0-9]+$ ]] || [ "$budget" -eq 0 ]; then
    echo "Invalid PPO step budget: $budget" >&2
    exit 2
  fi
  echo "=== PSRO budget: ${budget} PPO steps per BR ==="
  bash "$PROJECT_ROOT/scripts/run_long_psro.sh" --steps "$budget" "$@"
done
