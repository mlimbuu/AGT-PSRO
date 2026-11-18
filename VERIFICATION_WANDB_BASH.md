# W&B/Bash revision verification

## Changes

- Preserved the standard Python entry points and added Bash launchers under `scripts/`.
- Kept all training-data writes inside `results/` when commands are launched from elsewhere.
- Added online/offline/disabled W&B options to quick and long PSRO and PPO ablation runs.
- Added W&B run grouping, unique descriptive names, tags, experiment configuration, source/config fingerprint, optional SVI reference, and run ID in local JSON.
- Captured PSRO metrics, Blue/Red PPO training metrics, population sizes, mixture weights and timing with separate x-axis step metrics.
- Routed W&B run files, cache, and data directories into `results/wandb/` / `results/cache/wandb/`.
- Preserved Docker as optional; `--build-arg WITH_WANDB=1` includes W&B in the CPU image.
- Added `WANDB_SETUP.md`, native Bash examples, offline syncing steps, and a Bash budget sweep.

## Verified in this environment

| Test | Result |
|---|---|
| Six-node/three-risk Scratch PSRO, 1k PPO steps, seed 11, one iteration | PASS: Blue BR 787.2386, Red BR 104.9384, new restricted V 104.9384 |
| Five-node/two-risk SVI using native Bash from `/tmp` | PASS: exact V=51.9999999302 |
| Native Bash budget sweep (`PSRO_BUDGETS='8 16'`) | PASS: both budgets completed |
| Simulated W&B SDK: W&B init, tags/group, callbacks, metrics, summary, finish | PASS |
| Simulated W&B SDK: offline mode and W&B paths under `results/` | PASS |
| Disabled W&B does not require SDK | PASS |
| Missing SDK produces explicit setup instructions | PASS |
| 2 SVI + 4 W&B + 5 output-safety + 1 portability automated checks | 12 checks passed (two test runs) |

## Not verified

- Actual `wandb` SDK online or offline-file operations; the package could not be downloaded in this execution environment.
- Authentication to a W&B hosted/self-managed endpoint.
- `wandb sync`, Docker build, GPU execution, or full 15-iteration multi-seed PSRO reruns.

The existing scientific implementation and prior numerical regression fixtures were not altered. The new W&B code is opt-in and does not change PPO updates or the PSRO equilibrium solver.
