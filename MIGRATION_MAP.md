# Code migration map and invariants

This is a **behavior-preserving refactor** of the supplied
`AGT5_PSRO_long_initialization_comparison.zip` snapshot, not a from-scratch rewrite
of the stochastic game or neural networks. `legacy/original_verified_source.zip`
contains the original source and historical diagnostic files.

| Original module or function | New location | Reason |
|---|---|---|
| `code_src/runs/main_run_SB.py` scenario dataclasses and `build_scenario()` | `scenarios/config.py` | One explicit scenario registry; no hidden `SMALL_SCENARIO` global |
| `main_run_SB.py` `SMALL_SCENARIO` | `scenarios/five_node_two_risk.py` | Named, independently selectable five-node game |
| User-specified six-node configuration | `scenarios/six_node_three_risk.py` | Independently selectable six-node game |
| `adversarial_graph_game/{models,costs,environment,graph_factory,symmetry,state_space,...}.py` | `game/` | State, actions, transitions and support-cost semantics |
| `adversarial_graph_game/{layers,shapley,pruning,matrix_game,...}.py` | `svi/` | Exact SVI and shared zero-sum LP calculation |
| `adversarial_graph_game/psro.py` | `psro/psro.py` + `psro/meta_game_matrix.py` | Exact policy evaluation/BRs + restricted-game payoffs |
| `run_full_psro_long.py` | PSRO orchestration in `psro/psro.py`; launchers in `experiments/quick/` and `experiments/long/` | One shared scientific loop for both experiment modes |
| `main_run_SB_PSRO_BC_seed.py` | `psro/initialization.py` | BC population initializer, accepting an explicit scenario |
| `main_run_SB_heuristic_security_bounds.py` | `psro/initialization.py` | Current-snapshot shortest-path initializer |
| `adversarial_graph_game/psro_ppo.py` | `psro/ppo_oracle/red_ppo.py` + `mlp.py` | Red PPO optimizer and shared MLP architecture |
| `adversarial_graph_game/psro_ppo_relgnn_ablation.py` | `psro/ppo_oracle/blue_ppo.py`, `relational_gnn.py` and `mlp.py` | Blue PPO optimizer + actor/critic network options |
| Original short diagnostic and benchmark scripts | `legacy/original_verified_source.zip` | Historical research retained, not imported by active code |

## Scientific invariants retained

1. Red is the cost maximizer; Blue is the cost minimizer.
2. The Red graph state is a complete risk snapshot; changing graphs consumes one unit of ammo.
3. Support activates only when the supporting teammate waits as required by the game cost model.
4. PPO's seed-offset conventions, reward scaling, action masks, and policy snapshots remain unchanged.
5. BC/Heuristic/Scratch seed the **initial PSRO population**; later PPO oracle parameters start fresh.
6. The restricted payoff matrix uses Red rows and Blue columns.
7. The pair-cost evaluation cache is keyed by policy identity, and PSRO policies are never modified after being added. Therefore it reuses exact values without changing any optimization update.

## Known limitations

- Only the five-node PSRO smoke/regression and the two exact SVI values were independently verified here. This is **not** a verification of full 15-iteration or larger-graph training.
- W&B integration is optional and was not tested because `wandb` is not installed in this runtime. Its local directory is explicitly under `results/wandb/`.
- The long-run JSON files do not serialize the complete learned policy population or optimizer state, so interruption recovery still requires additional checkpointing work.
- Source-equivalence of other historical benchmark scripts in the legacy archive was not checked. They were deliberately preserved rather than silently ported.

## Outputs

All generated outputs use paths rooted in `results/` and are rejected if an
`--output-root` escapes that directory. The scenario, PPO budget, initializer,
seed, and source/config fingerprint identify each JSON run.
