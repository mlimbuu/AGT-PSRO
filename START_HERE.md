# Start here — the six files to read first

You do **not** need to understand the entire repository to run PSRO experiments.

1. **`scenarios/five_node_two_risk.py`** — What graph, attacks, support relations and Red transitions are we studying?
2. **`game/environment.py`** — What actions can Blue and Red take, and how does one step change the state?
3. **`svi/layers.py`** — How does the exact SVI reference solve the game?
4. **`psro/psro.py`** — How are Blue/Red policies evaluated, and how do PSRO iterations expand the populations? Search for `run_ppo_psro_experiment` to find the experiment loop at the end; the preceding functions are mathematical utilities.
5. **`psro/meta_game_matrix.py`** — How do existing Blue/Red policies form the restricted zero-sum matrix?
6. **`psro/initialization.py`** — Where do Scratch, BC and Heuristic initial populations come from?

Read neural-network internals **later**:

- `psro/ppo_oracle/blue_ppo.py`: Blue approximate BR training, actor–critic ablations.
- `psro/ppo_oracle/red_ppo.py`: Red MLP PPO training + shared PPO state/action codecs.
- `psro/ppo_oracle/mlp.py`: MLP actor/critic networks.
- `psro/ppo_oracle/relational_gnn.py`: Relational graph representation and networks.

`experiments/quick/` and `experiments/long/` are **thin launchers**, not competing PSRO implementations. `results/code/` contains plotting/analysis scripts, not training. All experiment JSON and generated figures are kept within `results/`.

## First two commands to run

```bash
python experiments/quick/run_svi.py --scenario five_node_two_risk
python experiments/quick/run_psro.py --scenario five_node_two_risk --init scratch --seed 11 --steps 1000 --iterations 1 --skip-svi
```

Expected outputs:

- SVI: `V* = 51.99999993`, about 52.
- PSRO Scratch: `START ... 440`, then `ITER ... B 963.1574 R 286.3253 new 440`.

The new project includes an untouched source ZIP in `legacy/` for comparison.

For W&B monitoring and Bash job scripts, start with **WANDB_SETUP.md**. Docker is optional.
