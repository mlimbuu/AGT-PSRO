# Five-node, two-risk game — 2 vs 3 agents

Parameters: support enabled, Red ammunition=1, Blue initial positions=(0,)*N, goal node=4, seed=11, Scratch initialization, 1,000 PPO steps per best-response oracle, original environment costs.

| Agents | Exact SVI | PSRO restricted meta after 3 iterations | PSRO restricted meta after 5 iterations |
|---:|---:|---:|---:|
| 2 | 52.0000 | 440.0000 | 440.0000 |
| 3 | 73.0000 | 660.0000 | not completed (iter 4 exceeded execution limit) |

Three-agent PSRO was independently completed with 3 iterations; an attempt to complete 5 timed out during iteration 4. Thus this is a *matched 3-iteration comparison*, NOT a completed matched 5-iteration experiment. Restricted values are not exploitability certificates.

Commands (from any directory):

```bash
bash /path/to/project/scripts/run_agent_comparison.sh
```

To rerun manually:

```bash
python experiments/quick/run_svi.py --scenario five_node_two_risk --agents 3 --ammo 1
python experiments/quick/run_psro.py --scenario five_node_two_risk --agents 3 --init scratch --seed 11 --steps 1000 --iterations 5 --skip-svi
```

All generated files are inside results/. `results/code/compare_agents.py` writes a CSV under `results/data/agent_comparison/`.
