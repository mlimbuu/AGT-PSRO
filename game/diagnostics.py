from __future__ import annotations

from game.compat import dataclass, strict_zip

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from game.costs import EdgeRiskCost, EdgeSupportCost, NodeSupportCost
from game.environment import AdversarialGraphGame
from game.models import BlueAction, GameState
from game.state_space import all_position_states

LOGGER_NAME = "adversarial_graph_game.diagnostics"


@dataclass(frozen=True, slots=True)
class PairDiagnosticSummary:
    states_checked: int
    actions_checked: int
    red_transitions_checked: int
    action_set_mismatches: int
    red_action_set_mismatches: int
    transition_mismatches: int
    support_cost_increases: int
    support_cost_decreases: int
    equal_stage_costs: int
    maximum_cost_increase: float
    maximum_cost_decrease: float
    records_truncated: bool

    @property
    def passed(self) -> bool:
        return (
            self.action_set_mismatches == 0
            and self.red_action_set_mismatches == 0
            and self.transition_mismatches == 0
            and self.support_cost_increases == 0
        )


def configure_diagnostic_logger(output_dir: str | Path) -> logging.Logger:
    """Create a console + file logger without duplicating handlers."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(
        directory / "diagnostics.log", mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_jsonable(value), sort_keys=True)
                    if isinstance(value, (dict, list, tuple, set, frozenset))
                    else value
                    for key, value in row.items()
                }
            )


def _edge_support_sets(
    game: AdversarialGraphGame,
    state: GameState,
    blue_action: BlueAction,
) -> tuple[set[int], set[int]]:
    model = game.cost_model
    if not isinstance(model, EdgeSupportCost):
        return set(), set()

    penalties = model.support.risk_penalty_by_graph.get(state.graph_id, {})
    supported_agents: set[int] = set()
    active_supporters: set[int] = set()

    for traverser_i, (current, successor) in enumerate(
        strict_zip(state.blue_positions, blue_action)
    ):
        risky_edge = (current, successor)
        if risky_edge not in penalties:
            continue
        valid_support_nodes = model.support.support_nodes_by_risky_edge.get(
            risky_edge, frozenset()
        )
        for supporter_j, supporter_position in enumerate(state.blue_positions):
            if supporter_j == traverser_i:
                continue
            if supporter_position not in valid_support_nodes:
                continue
            if (
                model.support.supporter_must_wait
                and blue_action[supporter_j] != supporter_position
            ):
                continue
            supported_agents.add(traverser_i)
            active_supporters.add(supporter_j)
            break

    return supported_agents, active_supporters


def cost_breakdown(
    game: AdversarialGraphGame,
    state: GameState,
    blue_action: BlueAction,
) -> dict[str, Any]:
    """Return an explicit per-agent explanation of a stage-cost calculation."""

    model = game.cost_model
    base_costs = game.spec.graph_costs[state.graph_id]
    total = game.stage_cost(state, blue_action)
    rows: list[dict[str, Any]] = []

    if isinstance(model, EdgeRiskCost):
        penalties = model.risk.risk_penalty_by_graph.get(state.graph_id, {})
        for agent_i, (current, successor) in enumerate(
            strict_zip(state.blue_positions, blue_action)
        ):
            edge = (current, successor)
            rows.append(
                {
                    "agent": agent_i,
                    "current": current,
                    "successor": successor,
                    "edge": edge,
                    "base_cost": float(base_costs[edge]),
                    "wait_cost_source": (
                        "goal_self_loop"
                        if current == successor == game.spec.goal_node
                        else "graph_base_cost"
                    ),
                    "risk_penalty_before_support": float(
                        penalties.get(edge, 0.0)
                    ),
                    "risk_penalty_after_support": float(
                        penalties.get(edge, 0.0)
                    ),
                    "supported": False,
                    "active_supporter": False,
                }
            )

    elif isinstance(model, EdgeSupportCost):
        penalties = model.support.risk_penalty_by_graph.get(state.graph_id, {})
        supported_agents, active_supporters = _edge_support_sets(
            game, state, blue_action
        )
        for agent_i, (current, successor) in enumerate(
            strict_zip(state.blue_positions, blue_action)
        ):
            edge = (current, successor)
            if current == successor == game.spec.goal_node:
                charged_base = float(base_costs[edge])
                source = "goal_self_loop"
            elif current == successor and agent_i in active_supporters and (
                model.support.active_support_wait_cost is not None
            ):
                charged_base = float(model.support.active_support_wait_cost)
                source = "active_support_wait_cost"
            elif current == successor and model.support.idle_wait_cost is not None:
                charged_base = float(model.support.idle_wait_cost)
                source = "idle_wait_cost"
            else:
                charged_base = float(base_costs[edge])
                source = "graph_base_cost"

            penalty_before = float(penalties.get(edge, 0.0))
            penalty_after = penalty_before
            if agent_i in supported_agents:
                penalty_after *= model.support.supported_penalty_multiplier

            rows.append(
                {
                    "agent": agent_i,
                    "current": current,
                    "successor": successor,
                    "edge": edge,
                    "base_cost": charged_base,
                    "wait_cost_source": source,
                    "risk_penalty_before_support": penalty_before,
                    "risk_penalty_after_support": penalty_after,
                    "supported": agent_i in supported_agents,
                    "active_supporter": agent_i in active_supporters,
                    "valid_support_nodes": sorted(
                        model.support.support_nodes_by_risky_edge.get(
                            edge, frozenset()
                        )
                    ),
                }
            )

    elif isinstance(model, NodeSupportCost):
        rows.append(
            {
                "note": (
                    "NodeSupportCost is active. The aggregate stage cost is "
                    "reported; use the same state/action in costs.py for its "
                    "node-level support trace."
                )
            }
        )
    else:
        for agent_i, (current, successor) in enumerate(
            strict_zip(state.blue_positions, blue_action)
        ):
            edge = (current, successor)
            rows.append(
                {
                    "agent": agent_i,
                    "current": current,
                    "successor": successor,
                    "edge": edge,
                    "base_cost": float(base_costs[edge]),
                }
            )

    return {
        "model": type(model).__name__,
        "state": state,
        "blue_action": blue_action,
        "total": float(total),
        "agents": rows,
    }


def compare_game_pair(
    no_support_game: AdversarialGraphGame,
    support_game: AdversarialGraphGame,
    *,
    output_dir: str | Path,
    tolerance: float = 1e-10,
    max_records: int = 20_000,
    fail_on_violation: bool = False,
) -> PairDiagnosticSummary:
    """Exhaustively compare matched no-support and support games.

    The check verifies that support does not change action sets, Red actions,
    transitions, or increase the cost of any matching state/action pair.
    Detailed records are written before an optional exception is raised.
    """

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    logger = configure_diagnostic_logger(directory)

    if no_support_game.spec != support_game.spec:
        logger.error("GameSpec mismatch between comparison conditions.")
        _write_json(
            directory / "spec_mismatch.json",
            {
                "no_support": no_support_game.spec,
                "support": support_game.spec,
            },
        )

    config_snapshot = {
        "no_support_model": type(no_support_game.cost_model).__name__,
        "support_model": type(support_game.cost_model).__name__,
        "spec": no_support_game.spec,
    }
    if isinstance(support_game.cost_model, EdgeSupportCost):
        config_snapshot["support_parameters"] = support_game.cost_model.support
    _write_json(directory / "configuration.json", config_snapshot)

    action_mismatches: list[dict[str, Any]] = []
    red_action_mismatches: list[dict[str, Any]] = []
    transition_mismatches: list[dict[str, Any]] = []
    cost_increases: list[dict[str, Any]] = []
    cost_decreases: list[dict[str, Any]] = []
    goal_wait_rows: list[dict[str, Any]] = []

    states_checked = 0
    actions_checked = 0
    transitions_checked = 0
    increase_count = 0
    decrease_count = 0
    equal_count = 0
    max_increase = 0.0
    max_decrease = 0.0
    truncated = False

    for positions in all_position_states(no_support_game.spec):
        for graph_id in no_support_game.spec.graph_costs:
            for ammo in range(no_support_game.spec.max_ammo + 1):
                state = GameState(positions, graph_id, ammo)
                states_checked += 1

                no_actions = no_support_game.blue_actions(state)
                support_actions = support_game.blue_actions(state)
                no_action_set = set(no_actions)
                support_action_set = set(support_actions)
                if no_action_set != support_action_set:
                    record = {
                        "state": state,
                        "missing_from_support": sorted(
                            no_action_set - support_action_set
                        ),
                        "extra_in_support": sorted(
                            support_action_set - no_action_set
                        ),
                    }
                    if len(action_mismatches) < max_records:
                        action_mismatches.append(record)
                    else:
                        truncated = True

                no_red_actions = set(no_support_game.red_actions(state))
                support_red_actions = set(support_game.red_actions(state))
                if no_red_actions != support_red_actions:
                    record = {
                        "state": state,
                        "no_support_red_actions": sorted(no_red_actions),
                        "support_red_actions": sorted(support_red_actions),
                    }
                    if len(red_action_mismatches) < max_records:
                        red_action_mismatches.append(record)
                    else:
                        truncated = True

                for action in sorted(no_action_set & support_action_set):
                    actions_checked += 1
                    no_cost = no_support_game.stage_cost(state, action)
                    support_cost = support_game.stage_cost(state, action)
                    delta = support_cost - no_cost

                    if any(node == no_support_game.spec.goal_node for node in positions):
                        if len(goal_wait_rows) < max_records:
                            goal_wait_rows.append(
                                {
                                    "positions": positions,
                                    "graph_id": graph_id,
                                    "ammo": ammo,
                                    "action": action,
                                    "no_support_cost": no_cost,
                                    "support_cost": support_cost,
                                    "support_minus_no_support": delta,
                                }
                            )
                        else:
                            truncated = True

                    if delta > tolerance:
                        increase_count += 1
                        max_increase = max(max_increase, delta)
                        record = {
                            "positions": positions,
                            "graph_id": graph_id,
                            "ammo": ammo,
                            "action": action,
                            "no_support_cost": no_cost,
                            "support_cost": support_cost,
                            "support_minus_no_support": delta,
                            "no_support_breakdown": cost_breakdown(
                                no_support_game, state, action
                            ),
                            "support_breakdown": cost_breakdown(
                                support_game, state, action
                            ),
                        }
                        if len(cost_increases) < max_records:
                            cost_increases.append(record)
                        else:
                            truncated = True
                    elif delta < -tolerance:
                        decrease_count += 1
                        max_decrease = max(max_decrease, -delta)
                        if len(cost_decreases) < max_records:
                            cost_decreases.append(
                                {
                                    "positions": positions,
                                    "graph_id": graph_id,
                                    "ammo": ammo,
                                    "action": action,
                                    "no_support_cost": no_cost,
                                    "support_cost": support_cost,
                                    "support_saving": -delta,
                                }
                            )
                        else:
                            truncated = True
                    else:
                        equal_count += 1

                    for red_action in sorted(no_red_actions & support_red_actions):
                        transitions_checked += 1
                        no_next = no_support_game.transition(
                            state, action, red_action
                        )
                        support_next = support_game.transition(
                            state, action, red_action
                        )
                        if no_next != support_next:
                            record = {
                                "state": state,
                                "blue_action": action,
                                "red_action": red_action,
                                "no_support_next": no_next,
                                "support_next": support_next,
                            }
                            if len(transition_mismatches) < max_records:
                                transition_mismatches.append(record)
                            else:
                                truncated = True

    summary = PairDiagnosticSummary(
        states_checked=states_checked,
        actions_checked=actions_checked,
        red_transitions_checked=transitions_checked,
        action_set_mismatches=len(action_mismatches),
        red_action_set_mismatches=len(red_action_mismatches),
        transition_mismatches=len(transition_mismatches),
        support_cost_increases=increase_count,
        support_cost_decreases=decrease_count,
        equal_stage_costs=equal_count,
        maximum_cost_increase=max_increase,
        maximum_cost_decrease=max_decrease,
        records_truncated=truncated,
    )

    _write_json(directory / "summary.json", summary)
    _write_jsonl(directory / "action_set_mismatches.jsonl", action_mismatches)
    _write_jsonl(
        directory / "red_action_set_mismatches.jsonl",
        red_action_mismatches,
    )
    _write_jsonl(
        directory / "transition_mismatches.jsonl", transition_mismatches
    )
    _write_jsonl(directory / "cost_increases.jsonl", cost_increases)
    _write_csv(directory / "cost_decreases.csv", cost_decreases)
    _write_csv(directory / "goal_wait_comparison.csv", goal_wait_rows)

    logger.info("Checked %d states and %d matching Blue actions.", states_checked, actions_checked)
    logger.info("Support cost decreases: %d.", decrease_count)
    logger.info("Support cost increases: %d.", increase_count)
    logger.info("Action-set mismatches: %d.", len(action_mismatches))
    logger.info("Red-action mismatches: %d.", len(red_action_mismatches))
    logger.info("Transition mismatches: %d.", len(transition_mismatches))
    logger.info("Diagnostic result: %s.", "PASS" if summary.passed else "FAIL")

    if increase_count:
        first = cost_increases[0] if cost_increases else None
        logger.error(
            "Support increased a matching immediate cost. First recorded "
            "violation: %s",
            first,
        )

    if fail_on_violation and not summary.passed:
        raise RuntimeError(
            "Support/no-support preflight failed. See "
            f"{directory / 'diagnostics.log'} and summary.json."
        )

    return summary


def write_solver_diagnostics(
    *,
    result: Any,
    game: AdversarialGraphGame,
    initial_graph: int,
    ammo: int,
    initial_positions: tuple[int, ...],
    output_dir: str | Path,
    label: str,
) -> None:
    """Write layer convergence and initial-equilibrium-policy diagnostics."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    layer_rows = []
    for key in sorted(result.values):
        graph_id, layer_ammo = key
        layer_rows.append(
            {
                "graph_id": graph_id,
                "ammo": layer_ammo,
                "converged": result.converged.get(key),
                "iterations": result.iterations.get(key),
                "bellman_residual": result.residuals.get(key),
                "bound_sweeps": result.bound_sweeps.get(key),
                "bound_gap": result.bound_gaps.get(key),
            }
        )
    _write_csv(directory / f"{label}_layers.csv", layer_rows)

    key = (initial_graph, ammo)
    positions = game.canonical_positions(initial_positions)
    policy = result.policies.get(key, {}).get(positions)
    value = result.values.get(key, {}).get(positions)
    policy_record: dict[str, Any] = {
        "layer": key,
        "positions": positions,
        "value": value,
    }
    if policy is not None:
        policy_record.update(
            {
                "blue_actions": policy.blue_actions,
                "blue_probabilities": policy.blue_probabilities.tolist(),
                "red_actions": policy.red_actions,
                "red_probabilities": policy.red_probabilities.tolist(),
            }
        )
    _write_json(directory / f"{label}_initial_policy.json", policy_record)
