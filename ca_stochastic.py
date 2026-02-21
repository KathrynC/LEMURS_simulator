"""Stochastic variant of the LEMURS semantic cellular automaton.

Provides a stochastic rule application engine and an ensemble runner for
Monte Carlo analysis of CA dynamics. While the deterministic CA in ca_rules.py
resolves conflicts by highest-confidence-wins, this module samples proportional
to confidence, enabling probabilistic outcome estimation.

Three public functions:

1. apply_rules_stochastic() — stochastic rule application with confidence-
   weighted sampling. Low-confidence rules may not fire; multi-rule conflicts
   are resolved by confidence-proportional random selection.

2. run_single_cell_stochastic() — Monte Carlo ensemble of single-cell CA
   trajectories. Each trial uses a distinct RNG stream derived from
   (seed + trial_index).

3. compute_ensemble_analytics() — aggregate statistics over ensemble trials:
   attractor distributions, burnout probability, anxiety crossing fraction,
   per-variable terminal bin distributions.
"""
from __future__ import annotations

import numpy as np

from constants import N_STEPS, DEFAULT_INTERVENTION, DEFAULT_PATIENT
from simulator import initial_state
from ca_schema import discretize_state, BIN_SCHEMA, _VAR_ORDER
from ca_rules import get_applicable_rules, _apply_direction, RULE_TABLE
from ca_simulator import _build_context
from ca_analytics import _classify_attractor


def apply_rules_stochastic(
    discrete_state: dict[str, str],
    context: dict,
    rng: np.random.Generator,
    rules: list[dict] | None = None,
) -> tuple[dict[str, str], list[dict]]:
    """Apply matching rules stochastically to produce the next discrete state.

    Rules are selected from those whose input and context conditions match.
    The burnout cascade (absorbing state) is still deterministic — if it
    matches, the state is frozen. For other rules:

    - Every rule fires with probability equal to its confidence. A rule
      with confidence 0.85 fires 85% of the time and is skipped 15% of
      the time. This generates genuine distributional spread across
      Monte Carlo trials.
    - When multiple surviving rules propose updates to the same variable,
      one is sampled proportional to its confidence weight.
    - When only one surviving rule proposes an update, it is applied
      directly (it already passed the probabilistic gate).

    Parameters
    ----------
    discrete_state : dict[str, str]
        Current discretized state {var_name: bin_label}.
    context : dict
        Calendar and patient context.
    rng : np.random.Generator
        Random number generator for stochastic decisions.
    rules : list[dict] or None
        Rule table. Defaults to RULE_TABLE.

    Returns
    -------
    tuple[dict[str, str], list[dict]]
        (new_state, applicable_rules) — same signature as apply_rules.
    """
    applicable = get_applicable_rules(discrete_state, context, rules)

    # Burnout cascade is an absorbing state — deterministic freeze
    for rule in applicable:
        if rule["name"] == "burnout_cascade":
            return dict(discrete_state), applicable

    # Probabilistic firing: each rule fires with probability = confidence
    surviving_rules = []
    for rule in applicable:
        if rng.random() > rule["confidence"]:
            continue  # rule did not fire this step
        surviving_rules.append(rule)

    # Collect proposals per variable: {var_name: [(direction, confidence, rule)]}
    proposals: dict[str, list[tuple[str, float, dict]]] = {}
    for rule in surviving_rules:
        for var_name, direction in rule["outputs"].items():
            if var_name not in proposals:
                proposals[var_name] = []
            proposals[var_name].append((direction, rule["confidence"], rule))

    # Resolve conflicts and apply updates
    new_state = dict(discrete_state)
    for var_name, candidates in proposals.items():
        if len(candidates) == 1:
            # Single proposal — apply directly
            direction, _, _ = candidates[0]
        else:
            # Multiple proposals — sample proportional to confidence
            confidences = np.array([c[1] for c in candidates], dtype=np.float64)
            probs = confidences / confidences.sum()
            idx = rng.choice(len(candidates), p=probs)
            direction, _, _ = candidates[idx]

        new_state[var_name] = _apply_direction(
            new_state[var_name], direction, var_name
        )

    return new_state, applicable


def run_single_cell_stochastic(
    patient: dict | None = None,
    intervention: dict | None = None,
    sim_days: int = N_STEPS,
    n_trials: int = 100,
    seed: int = 42,
    rules: list[dict] | None = None,
) -> dict:
    """Run a Monte Carlo ensemble of single-cell CA simulations.

    Each trial uses independent RNG seeded with (seed + trial_index),
    producing a distribution of trajectories from identical initial
    conditions. The stochastic variation comes from apply_rules_stochastic.

    Parameters
    ----------
    patient : dict or None
        Student characteristics (merged with DEFAULT_PATIENT).
    intervention : dict or None
        Intervention parameters (merged with DEFAULT_INTERVENTION).
    sim_days : int
        Number of daily steps (default 105 = 15 weeks).
    n_trials : int
        Number of Monte Carlo trials (default 100).
    seed : int
        Base random seed. Trial i uses seed + i.
    rules : list[dict] or None
        Rule table. Defaults to RULE_TABLE.

    Returns
    -------
    dict
        {
            "trajectories": list[list[dict]],
            "final_states": list[dict],
            "rule_logs": list[list[list[str]]],
            "n_trials": int,
            "initial_state": dict,
            "patient": dict,
            "intervention": dict,
        }
    """
    pat = {**DEFAULT_PATIENT, **(patient or {})}
    intv = {**DEFAULT_INTERVENTION, **(intervention or {})}

    # Initialize from ODE initial state (same as run_single_cell)
    pat_with_rx = {**pat, "nature_rx": intv.get("nature_rx", 0.0)}
    continuous_init = initial_state(pat_with_rx)
    initial_discrete = discretize_state(continuous_init)

    trajectories = []
    final_states = []
    rule_logs = []

    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial)

        state = dict(initial_discrete)
        trajectory = [dict(state)]
        trial_rule_log = []
        prev_state = None

        for day in range(sim_days):
            ctx = _build_context(day, pat, intv, prev_state, state)
            new_state, fired = apply_rules_stochastic(state, ctx, rng, rules)
            trial_rule_log.append([r["name"] for r in fired])
            prev_state = state
            state = new_state
            trajectory.append(dict(state))

        trajectories.append(trajectory)
        final_states.append(dict(state))
        rule_logs.append(trial_rule_log)

    return {
        "trajectories": trajectories,
        "final_states": final_states,
        "rule_logs": rule_logs,
        "n_trials": n_trials,
        "initial_state": dict(initial_discrete),
        "patient": pat,
        "intervention": intv,
    }


def compute_ensemble_analytics(ensemble_result: dict) -> dict:
    """Compute aggregate statistics over a stochastic CA ensemble.

    Takes the output of run_single_cell_stochastic and computes attractor
    distributions, burnout probability, anxiety crossing probability, and
    per-variable terminal bin distributions.

    Parameters
    ----------
    ensemble_result : dict
        Output from run_single_cell_stochastic().

    Returns
    -------
    dict
        {
            "attractor_counts": dict[str, int],
            "attractor_probabilities": dict[str, float],
            "burnout_probability": float,
            "anxiety_crossing_probability": float,
            "variable_distributions": dict[str, dict[str, float]],
        }
    """
    n_trials = ensemble_result["n_trials"]
    final_states = ensemble_result["final_states"]
    trajectories = ensemble_result["trajectories"]

    # ── Attractor counts and probabilities ──
    attractor_labels = ["healthy", "struggling", "stressed", "burnout"]
    attractor_counts: dict[str, int] = {a: 0 for a in attractor_labels}

    for state in final_states:
        attractor = _classify_attractor(state)
        if attractor in attractor_counts:
            attractor_counts[attractor] += 1
        else:
            attractor_counts[attractor] = 1

    attractor_probabilities: dict[str, float] = {
        a: count / n_trials for a, count in attractor_counts.items()
    }

    burnout_probability = attractor_probabilities.get("burnout", 0.0)

    # ── Anxiety crossing probability ──
    # Fraction of trials where GAD7 ever reaches "clinical" in any step
    anxiety_crossings = 0
    for trajectory in trajectories:
        for state in trajectory:
            if state.get("GAD7") == "clinical":
                anxiety_crossings += 1
                break  # count each trial at most once

    anxiety_crossing_probability = anxiety_crossings / n_trials

    # ── Variable distributions: fraction of trials ending in each bin ──
    variable_distributions: dict[str, dict[str, float]] = {}
    for var_name in _VAR_ORDER:
        bin_labels = BIN_SCHEMA[var_name]["labels"]
        counts: dict[str, int] = {label: 0 for label in bin_labels}
        for state in final_states:
            label = state.get(var_name, bin_labels[0])
            if label in counts:
                counts[label] += 1
            else:
                counts[label] = 1
        variable_distributions[var_name] = {
            label: count / n_trials for label, count in counts.items()
        }

    return {
        "attractor_counts": attractor_counts,
        "attractor_probabilities": attractor_probabilities,
        "burnout_probability": burnout_probability,
        "anxiety_crossing_probability": anxiety_crossing_probability,
        "variable_distributions": variable_distributions,
    }
