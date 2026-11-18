# AGT PSRO — refactored, validated small-graph research implementation

This repository is organized around the two scientific algorithms, **SVI** and **PSRO**.
The PPO BR implementation lives at `psro/ppo_oracle/`. No experiment data are written
into the repository root. All generated JSON, figures, and W&B files belong in `results/`.

## Cloud portability

See **[WANDB_SETUP.md](WANDB_SETUP.md)** for optional W&B setup with native Bash scripts.
See **[CLOUD_SETUP.md](CLOUD_SETUP.md)** for a Python virtual-environment
installation on Linux cloud servers and an optional CPU-only Dockerfile. The
project uses no cluster-specific paths or scheduler modules. Experiment output
is confined to `results/` even when launched outside the project directory.
Docker build and GPU execution were not tested in this environment.

## Bash launchers (recommended on servers)

```bash
bash scripts/run_quick_svi.sh --scenario five_node_two_risk
bash scripts/run_quick_psro.sh --scenario five_node_two_risk --init scratch --seed 11 --steps 1000 --iterations 1 --skip-svi
bash scripts/run_long_psro.sh --scenario six_node_three_risk --init all --seeds 11 12 13 --steps 10000 --iterations 15 --skip-svi
# Optional W&B: install requirements-wandb.txt, authenticate if online, then add:
bash scripts/run_long_psro.sh --scenario six_node_three_risk --init all --seeds 11 12 13 --steps 1000 --iterations 15 --skip-svi --wandb --wandb-mode offline
```

All scripts work from a different current directory; the Python programs place
all experiment outputs under `results/`.

## Primary commands (from project root)

```bash
python -m pip install -r requirements.txt
python experiments/quick/run_svi.py --scenario five_node_two_risk
python experiments/quick/run_svi.py --scenario six_node_three_risk
python experiments/quick/run_psro.py --scenario five_node_two_risk --init scratch --seed 11 --steps 1000 --iterations 1 --skip-svi
python experiments/quick/run_psro.py --scenario five_node_two_risk --init bc --seed 11 --steps 1000 --iterations 1 --skip-svi
python experiments/quick/run_psro.py --scenario five_node_two_risk --init heuristic --seed 11 --steps 1000 --iterations 1 --skip-svi
python experiments/long/run_psro.py --scenario six_node_three_risk --init all --seeds 11 12 13 --steps 10000 --iterations 15
python results/code/plot_psro.py --scenario five_node_two_risk --steps 1000
python experiments/long/run_ppo_ablations.py --scenario five_node_two_risk --steps 1000 --iterations 1 --seeds 11
python results/code/plot_ppo_ablations.py --scenario five_node_two_risk --steps 1000
python -m pytest tests/test_game_and_svi.py tests/test_results_safety.py tests/test_cloud_portability.py -q
python -m pytest tests/test_six_node_ppo_regression.py -q -m slow
```

**Quick and long call the same `psro/psro.py` implementation**; only defaults differ.

## File navigation

- `scenarios/`: five-node and six-node fixed configurations + scenario builder.
- `game/`: scientific state, legal actions, transitions, risk and support costs.
- `svi/`: exact ammo-layer SVI, unpruned/full SVI, zero-sum LP, pruning.
- `psro/psro.py`: policy evaluation, exact BR utilities and PPO PSRO iteration.
- `psro/meta_game_matrix.py`: policy-pair cost matrix, restricted equilibrium.
- `psro/initialization.py`: random-proper, BC, snapshot heuristic initial populations.
- `psro/ppo_oracle/`: Blue PPO, Red PPO, MLP and relational-GNN architectures.
- `experiments/quick`, `experiments/long`: thin entry points, no duplicated mathematics.
- `results/code/`: figures and analysis only; `results/data/`: JSON/CSV; `results/wandb/`: opt-in W&B.
- `legacy/`: untouched original archived project for baseline comparison.

## Scope and caveats

- `PPOConfig` has the original training defaults used in the 2026 reference experiments.
- BC seeds the **PSRO population**, not all future PPO network weights.
- The old full-state `shapley.py` and ammo-layer `layers.py` are retained.
- The Red PPO implementation supports MLP. Blue supports MLP/static-relGNN/action-relGNN/virtual.
- Large-scale PPO checkpoint resumability is **not yet implemented**; JSON records and
  iteration metrics are saved as runs proceed, but not full policy/network snapshots.
- SVI exact value and PSRO restricted value are not interchangeable.

## Research regression references (five-node, two-risk, 1k budget, seed 11)

| Initialization | Initial restricted value | Blue candidate BR | Red candidate BR | Updated restricted value |
|---|---:|---:|---:|---:|
| Scratch | 440.0000 | 963.1574 | 286.3253 | 440.0000 |
| BC | 52.0000 | 963.1574 | 52.3858 | 52.3858 |
| Heuristic | 52.0000 | 963.1574 | 52.3858 | 52.3858 |

These are the first-iteration historical values, not guaranteed exact BRs or exploitability certificates.

Code-level refactoring intentionally preserves the optimizer update and seeding behavior for these configurations. Every output includes a configuration+source fingerprint to avoid silently reusing old run data after changing the game or model.

## Additional six-node PSRO regression (cloud-portable revision)

The 1k PPO, one-iteration tests for Scratch, BC and Heuristic on seeds 11 and 12
match the original six-node terminal logs (Blue BR, Red BR, and updated value).
Details: `results/data/verification/six_node_1k_first_iter_vs_archive.csv`.
The full 15-iteration/multi-seed run, seed 13 and optional W&B/Docker/GPU execution
were not tested in this revision.

## W&B / Bash / Docker (latest revision)

W&B integration remains opt-in via `--wandb`; it now provides stable run names,
scenario/budget/architecture grouping, configuration hashes, and optional offline
logging. Short/long Bash launchers and a 1k/3k/6k/10k budget sweep are available
under `scripts/`. Both native Python and optional Docker share the same code.
See [WANDB_SETUP.md](WANDB_SETUP.md) for setup, login, offline syncing and limitations.
