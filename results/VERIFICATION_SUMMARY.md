# Verification of the refactored research code

**Source:** archived AGT5 PSRO November 2026 implementation.

## Exact game solver regression

| Scenario | Refactored exact SVI | Reference | Result |
|---|---:|---:|---|
| 5 nodes, 2 risky edges, 2 Blue agents, Red ammo 1, support | 51.9999999302 | 52.0000 | PASS |
| 6 nodes, 3 risky edges, 2 Blue agents, Red ammo 1, support | 48.9442717723 | 48.94427177 | PASS |

## PSRO + Blue/Red PPO regression

**Five-node, two-risk game**; seed 11, 1000 PPO steps per BR, 3 PSRO iterations,
Blue actor static Rel-GNN, Blue critic MLP, Red actor/critic MLP.

| Initializer | Iteration 1 (Blue BR, Red BR, new V) | Iteration 2 | Iteration 3 |
|---|---|---|---|
| Scratch | 963.1574, 286.3253, 440.0000 | 1046.6999, 319.3398, 440.0000 | 1030.3609, 286.2113, 440.0000 |
| BC | 963.1574, 52.3858, 52.3858 | 573.7650, 52.1221, 52.3858 | 952.3229, 52.2905, 52.3858 |
| Heuristic | 963.1574, 52.3858, 52.3858 | 573.7650, 52.1221, 52.3858 | 952.3229, 52.2905, 52.3858 |

All 36 matched numerical fields (including `meta_before`) agree with the archived
log to within **4.3 × 10⁻⁵**, below the 2 × 10⁻⁴ regression tolerance.
See `results/data/verification/five_node_1k_seed11_3iter_vs_archive.json` for
field-by-field comparisons.

## Actor/critic ablation smoke tests

All four combinations ran for one 128-step PSRO iteration (seed 11) without errors:
MLP/MLP, static Rel-GNN/MLP, action Rel-GNN/MLP, static Rel-GNN/static Rel-GNN.
These are implementation smoke tests, not meaningful performance comparisons.

## Automated tests and output routing

The fast regression suite contains 8 passing tests for SVI reference values,
legal Red transitions and ammo decrement, policy-pair evaluation, meta-game
orientation, support semantics, and output-path containment. Three additional
slow regression tests reproduce the 1k/3-iteration PSRO runs (Scratch, BC,
Heuristic) against archived values. The slow tests were run successfully as
individual cases; the values were also checked directly from completed runs.

All generated JSON, CSV, PNG, tracker files, and pytest caches are configured to
remain within `results/`. No experiment script writes numerical outputs to the
project root.

## Not covered

Full 15-iteration PSRO training, large graph scalability, GPU differences,
W&B synchronization, and resumable training checkpoints have not been verified.

## Cloud portability and six-node retest (November 2026)

* Recreated the project in a different directory and ran SVI/PSRO from an
  unrelated working directory; all outputs remained under `results/`.
* Six-node exact SVI: **48.9442717723**, matching the known reference.
* Six 1k PPO, one-iteration PSRO checks passed (seeds 11,12; Scratch, BC, Heuristic).
  Maximum absolute numerical difference from four-decimal terminal-log fields:
  **0.00011447** (tolerance 0.00025).
* Eleven non-PPO tests passed, including script launching outside the source
  directory, output containment and atomic JSON writes.
* Added CPU-only Dockerfile and `CLOUD_SETUP.md`. Docker **not built** here; no
  guarantee of bitwise identity across cloud runtimes or GPU platforms.
* Not retested here: seed 13, 15-iteration long runs, W&B, actual Docker build,
  and GPU acceleration.

See `results/data/verification/six_node_1k_first_iter_vs_archive.csv`.
