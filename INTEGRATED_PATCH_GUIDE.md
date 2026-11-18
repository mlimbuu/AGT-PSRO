# Fully integrated agent-count + SVI bounds codebase

This folder is a **complete project**. Do not copy a few patch files into an older checkout: replace the previous project folder with this one, or clone it to a new directory. Its `scenarios/__init__.py`, `experiment_config.py`, `psro/psro.py`, and SVI bounds runner are compatible.

## Run from the project directory

```bash
python -m pip install -r requirements.txt
# two Blue agents, five-node two-risk game
python experiments/quick/run_svi.py --scenario five_node_two_risk --agents 2
python experiments/quick/run_svi_with_bounds.py --scenario five_node_two_risk --agents 2
# three Blue agents, same game
python experiments/quick/run_svi.py --scenario five_node_two_risk --agents 3
python experiments/quick/run_svi_with_bounds.py --scenario five_node_two_risk --agents 3
# quick PPO-based PSRO regression (1000 steps, one iteration)
python experiments/quick/run_psro.py --scenario five_node_two_risk --agents 3 \
  --init scratch --seed 11 --steps 1000 --iterations 1 --skip-svi
# your requested longer experiment (5 iterations)
python experiments/quick/run_psro.py --scenario five_node_two_risk --agents 3 \
  --init scratch --seed 11 --steps 1000 --iterations 5 --skip-svi
```

`--agents 2` or `--agents 3` works with the 5-node and 6-node named scenarios. The scenario key is suffixed `_3agents` for 3-agent output, so files don't collide.

**Known references**: exact SVI 5-node 2 agents = 52; 5-node 3 agents = 73; 6-node 2 agents = 48.9442717723. Five-node 3-agent Scratch PSRO, seed 11, 1k PPO steps, first iteration: initial 660, Blue BR 5654.7457, Red BR 405.5028, updated value 660.

**Output**: Experiment data, saved runs and bounds are written exclusively inside `results/`. In particular the bounds runner writes to `results/data/svi/<scenario>/ammo1/with_support_bounds.json`.

**W&B**: optional (`python -m pip install -r requirements-wandb.txt`, `wandb login`, then `--wandb --wandb-project agt5-psro`). See `WANDB_SETUP.md`. Native Bash scripts remain supported; Docker is optional.

**Bounds**: `svi/value_bounds.py` exposes initial-state certified value intervals using the existing lower/upper pure-security bound propagation. `experiments/quick/run_svi_with_bounds.py` provides the CLI. Bounds are on the game value, not an exploitability certificate for PPO.

**Tests**:

```bash
python -m pytest -q tests/test_agent_count_config.py tests/test_svi_bounds_agents.py
```
