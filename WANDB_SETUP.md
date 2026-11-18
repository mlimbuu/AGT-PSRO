# W&B tracking (optional) — Bash/Python first, Docker optional

This repository supports **three execution modes** with the *same* PSRO implementation:

1. Plain Python/Bash with no tracking (default; W&B not required).
2. W&B **online** tracking (public or self-hosted W&B, internet + credentials required).
3. W&B **offline** tracking (no credentials or network while training; sync later).

Run from any working directory: the Bash scripts locate the project themselves.
All experiment JSON, metrics, figures, and W&B *run files* are stored below this repository's `results/` directory. The W&B CLI login may save your account credentials in your home directory; that is external authentication configuration, **not an experiment output**.

## 1. Install on any Python-capable Linux server

```bash
cd /path/to/AGT_PSRO_cloud_portable
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Only install W&B if tracking is needed:
python -m pip install -r requirements-wandb.txt
```

The Bash scripts use `python3` by default. If your environment uses a different executable, set `PYTHON_BIN`, e.g. `export PYTHON_BIN=python` after activating the virtual environment.

## 2. Option A: regular Bash/Python run (NO W&B)

```bash
bash scripts/run_quick_svi.sh --scenario five_node_two_risk
bash scripts/run_quick_psro.sh \
  --scenario five_node_two_risk --init scratch --seed 11 \
  --steps 1000 --iterations 1 --skip-svi
```

W&B is not imported or required unless you explicitly pass `--wandb` with an enabled mode.

## 3. Option B: W&B online tracking

First, configure **your own W&B service**. Use the default hosted W&B service, or set `WANDB_BASE_URL` for an alternative W&B server. Do not commit keys or endpoints containing credentials.

```bash
# Only for a self-hosted / alternative W&B service:
export WANDB_BASE_URL="https://your-wandb-server.example"

# For hosted W&B, WANDB_BASE_URL can be omitted.
# One-time interactive login:
wandb login
# Alternative in noninteractive jobs: provide WANDB_API_KEY securely via
# the cloud provider's secrets manager / job environment, never in source.

export WANDB_PROJECT="agt5-psro"
# Optional: export WANDB_ENTITY="your-team"

bash scripts/run_long_psro.sh \
  --scenario six_node_three_risk \
  --init all --seeds 11 12 13 \
  --steps 1000 --iterations 15 \
  --wandb --wandb-mode online --skip-svi
```

Each (scenario, initialization, seed, architecture, budget) creates its own W&B run. Runs in the same (scenario, budget, Blue actor, Blue critic) are grouped. The command accepts `--wandb-project`, `--wandb-entity`, and `--wandb-group` overrides. The W&B run config includes the scenario, PPO configuration, seed, Red ammo, source/config fingerprint, and optional SVI value.

To attach SVI to the W&B run, **omit** `--skip-svi` (computationally more expensive). `--skip-svi` does not change the PSRO training objective; it only skips recomputing the exact reference.

### Tracked metrics

- `psro/meta_value`, `psro/meta_before`, `psro/blue_best_response`, `psro/red_best_response`
- `psro/blue_gain`, `psro/red_gain`, `psro/learned_br_candidate_gap` (candidate gap **is not exploitability**)
- PSRO population sizes, mixture weights, restricted Nash LP gap, elapsed time
- `psro/abs_gap_to_svi` **only if SVI was computed**
- Blue/Red PPO metrics and cumulative training steps, whenever emitted by the trainer

## 4. Option C: W&B offline on a network-restricted cluster

```bash
bash scripts/run_quick_psro.sh \
  --scenario five_node_two_risk --init bc --seed 11 \
  --steps 1000 --iterations 1 --skip-svi \
  --wandb --wandb-mode offline
```

W&B run data are stored under `results/wandb/` (commonly `results/wandb/wandb/offline-run-*`). On a machine with network access and the same project files available:

```bash
# Authenticate to your target host first (WANDB_BASE_URL if applicable).
wandb login
wandb sync results/wandb/wandb/offline-run-*
```

Keep the `results/` directory when moving the project between machines; it contains both the JSON results and offline tracking files. Syncing is a separate action; offline runs are **not** automatically uploaded.

## 5. Run all four PPO budgets from Bash

```bash
bash scripts/run_budget_sweep.sh \
  --scenario five_node_two_risk --init all --seeds 11 12 13 \
  --iterations 15 --wandb --wandb-mode online --skip-svi
```

The sweep iterates over 1000, 3000, 6000 and 10000 steps sequentially using the same `experiments/long/run_psro.py`. Override the list with `PSRO_BUDGETS="1000 3000"` if desired. Completed configurations are skipped unless `--force` is provided (skips do **not** create new W&B runs).

To capture shell output under `results/`:

```bash
mkdir -p results/logs
bash scripts/run_long_psro.sh --scenario five_node_two_risk \
  --init scratch --seed 11 --steps 1000 --iterations 15 \
  --wandb --wandb-mode online --skip-svi \
  2>&1 | tee results/logs/psro_5node_1k_seed11.log
```

## 6. Optional Docker (CPU-only image)

```bash
docker build --build-arg WITH_WANDB=1 -t agt-psro:cpu-wandb .
# Offline W&B in Docker; results persist on host:
docker run --rm -v "$(pwd)/results:/app/results" agt-psro:cpu-wandb \
  python experiments/quick/run_psro.py --scenario five_node_two_risk \
  --init scratch --seed 11 --steps 1000 --iterations 1 \
  --skip-svi --wandb --wandb-mode offline

# Online W&B: pass environment variables, never bake secrets into image.
docker run --rm -v "$(pwd)/results:/app/results" \
  --env WANDB_API_KEY --env WANDB_BASE_URL --env WANDB_PROJECT \
  agt-psro:cpu-wandb python experiments/quick/run_psro.py \
  --scenario five_node_two_risk --init scratch --seed 11 \
  --steps 1000 --iterations 1 --skip-svi --wandb --wandb-mode online
```

Docker is **not required**. The image and actual W&B upload could not be built/tested in this environment; use the native Bash workflow if that is what your server already supports.

## 7. Troubleshooting

- `W&B requested but not installed`: `python -m pip install -r requirements-wandb.txt`.
- Authentication errors: run `wandb login` in the Python environment, or set `WANDB_API_KEY` from a secrets manager.
- Wrong W&B server: check `WANDB_BASE_URL`, or use `wandb login --host https://your-wandb-server.example`.
- No internet: use `--wandb-mode offline`, then `wandb sync` on a connected machine.
- Tracking undesired: remove `--wandb`, or use `--wandb-mode disabled`.
- Not seeing an expected W&B run: if JSON already shows a completed matching run, the experiment is **skipped** unless you use `--force`.
- The framework currently does **not** save full PPO checkpoint weights between PSRO iterations, so interrupted training restarts rather than resuming exactly.

## Tested in this revision

The W&B integration was verified with a **simulated SDK** exercising `init`, `define_metric`, `log`, `summary`, and `finish`, including Blue and Red PPO metrics. Native Bash/Python runs were executed from outside the repository. The real W&B SDK could not be installed in this runtime (package download unavailable), so actual offline-file creation, online upload, Docker build, and server authentication remain **unverified**. The legacy numerical SVI/PSRO checks remain the baseline for scientific reproducibility.
