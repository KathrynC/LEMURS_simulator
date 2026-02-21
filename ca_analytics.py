"""Analytics for the LEMURS semantic cellular automaton.

Computes four sections of metrics from a CA simulation result:

1. **Rule stats**: Which rules fired most often, tier-level breakdown.
2. **Cascade stats**: Multi-tier chain reactions where one rule's output
   triggers another tier's rule on the next step.
3. **Attractor stats**: Terminal states the CA converges to (healthy,
   stressed, burnout), identified from the final days of the trajectory.
4. **Fidelity stats**: How well the CA trajectory agrees with the ODE
   trajectory at the bin level (requires an ODE result for comparison).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from constants import N_STEPS, STATE_NAMES, SPRING_BREAK_WEEK
from ca_schema import discretize_state, BIN_SCHEMA, _VAR_ORDER


def _rule_stats(rule_log: list[list[str]]) -> dict:
    """Compute rule firing frequency and tier breakdown.

    Parameters
    ----------
    rule_log : list[list[str]]
        For each day, the list of rule names that fired.

    Returns
    -------
    dict
        {
            "total_firings": int,
            "unique_rules": int,
            "rule_counts": dict[str, int],
            "top_10": list[tuple[str, int]],
            "days_with_rules": int,
            "mean_rules_per_day": float,
        }
    """
    all_rules = []
    days_with = 0
    for day_rules in rule_log:
        all_rules.extend(day_rules)
        if day_rules:
            days_with += 1

    counts = Counter(all_rules)
    n_days = len(rule_log) if rule_log else 1

    return {
        "total_firings": len(all_rules),
        "unique_rules": len(counts),
        "rule_counts": dict(counts),
        "top_10": counts.most_common(10),
        "days_with_rules": days_with,
        "mean_rules_per_day": len(all_rules) / n_days,
    }


def _cascade_stats(
    trajectory: list[dict[str, str]],
    rule_log: list[list[str]],
) -> dict:
    """Detect cascade sequences where one rule's output triggers another.

    A cascade occurs when a rule fires on day N, changes a variable's bin,
    and that change satisfies the input condition of a different-tier rule
    that fires on day N+1.

    Parameters
    ----------
    trajectory : list[dict[str, str]]
        Discrete state at each timestep (length sim_days+1).
    rule_log : list[list[str]]
        Rules fired per day (length sim_days).

    Returns
    -------
    dict
        {
            "cascade_count": int,
            "cascade_sequences": list[dict],
            "max_cascade_length": int,
            "cascade_days": list[int],
        }
    """
    cascades = []
    cascade_days = []

    for day in range(len(rule_log) - 1):
        if not rule_log[day]:
            continue

        # Check if the state changed between day and day+1
        state_before = trajectory[day]
        state_after = trajectory[day + 1]

        changed_vars = []
        for var_name in _VAR_ORDER:
            if state_before.get(var_name) != state_after.get(var_name):
                changed_vars.append(var_name)

        if not changed_vars:
            continue

        # Check if next day has new rules that weren't firing before
        if not rule_log[day + 1]:
            continue

        new_rules = set(rule_log[day + 1]) - set(rule_log[day])
        if new_rules:
            cascades.append({
                "trigger_day": day,
                "trigger_rules": rule_log[day],
                "changed_vars": changed_vars,
                "cascade_rules": list(new_rules),
            })
            cascade_days.append(day)

    # Measure cascade chain lengths
    max_length = 0
    if cascade_days:
        current_length = 1
        for i in range(1, len(cascade_days)):
            if cascade_days[i] == cascade_days[i - 1] + 1:
                current_length += 1
            else:
                max_length = max(max_length, current_length)
                current_length = 1
        max_length = max(max_length, current_length)

    return {
        "cascade_count": len(cascades),
        "cascade_sequences": cascades[:20],  # cap at 20 for readability
        "max_cascade_length": max_length,
        "cascade_days": cascade_days,
    }


def _classify_attractor(state: dict[str, str]) -> str:
    """Classify a terminal discrete state into an attractor category."""
    pss = state.get("PSS", "moderate")
    gad7 = state.get("GAD7", "sub_threshold")
    tst = state.get("TST", "adequate")
    dac = state.get("DAC", "available")

    # Burnout: worst case across all dimensions
    if (tst == "deprived" and pss == "high"
            and dac == "depleted" and gad7 == "clinical"):
        return "burnout"

    # Stressed: elevated stress and/or clinical anxiety
    if pss == "high" or gad7 == "clinical":
        return "stressed"

    # Struggling: moderate stress or sleep issues
    if pss == "moderate" or tst == "deprived":
        return "struggling"

    # Healthy: everything in acceptable range
    return "healthy"


def _attractor_stats(trajectory: list[dict[str, str]]) -> dict:
    """Identify the terminal attractor state from the last week of trajectory.

    Parameters
    ----------
    trajectory : list[dict[str, str]]
        Discrete state at each timestep.

    Returns
    -------
    dict
        {
            "final_attractor": str,
            "attractor_stable": bool,
            "last_week_attractors": list[str],
            "transition_count": int,
        }
    """
    # Look at last 7 days (last week of semester)
    last_week = trajectory[-8:] if len(trajectory) >= 8 else trajectory

    attractors = [_classify_attractor(s) for s in last_week]
    final = attractors[-1] if attractors else "unknown"

    # Count transitions in the last week
    transitions = 0
    for var_name in _VAR_ORDER:
        bins = [s.get(var_name) for s in last_week]
        for i in range(1, len(bins)):
            if bins[i] != bins[i - 1]:
                transitions += 1

    # Stable if no transitions in last week
    stable = transitions == 0

    return {
        "final_attractor": final,
        "attractor_stable": stable,
        "last_week_attractors": attractors,
        "transition_count": transitions,
    }


def _fidelity_stats(
    trajectory: list[dict[str, str]],
    ode_result: dict,
) -> dict:
    """Compare CA trajectory to ODE trajectory at the bin level.

    Discretizes each ODE state into bins and checks agreement with the
    CA trajectory at each timestep.

    Parameters
    ----------
    trajectory : list[dict[str, str]]
        CA discrete states (length sim_days+1).
    ode_result : dict
        Output from simulator.simulate() with 'states' array.

    Returns
    -------
    dict
        {
            "overall_agreement": float,
            "per_variable_agreement": dict[str, float],
            "worst_variable": str,
            "best_variable": str,
            "disagreement_days": list[int],
        }
    """
    ode_states = ode_result["states"]
    n_steps = min(len(trajectory), len(ode_states))

    per_var_agree: dict[str, list[bool]] = {v: [] for v in _VAR_ORDER}
    all_agreements = []
    disagreement_days = set()

    for t in range(n_steps):
        ode_discrete = discretize_state(ode_states[t])
        ca_state = trajectory[t]

        for var_name in _VAR_ORDER:
            agree = ode_discrete.get(var_name) == ca_state.get(var_name)
            per_var_agree[var_name].append(agree)
            all_agreements.append(agree)
            if not agree:
                disagreement_days.add(t)

    overall = sum(all_agreements) / len(all_agreements) if all_agreements else 0.0

    per_var_rates = {}
    for var_name, agrees in per_var_agree.items():
        per_var_rates[var_name] = sum(agrees) / len(agrees) if agrees else 0.0

    worst = min(per_var_rates, key=per_var_rates.get) if per_var_rates else ""
    best = max(per_var_rates, key=per_var_rates.get) if per_var_rates else ""

    return {
        "overall_agreement": overall,
        "per_variable_agreement": per_var_rates,
        "worst_variable": worst,
        "best_variable": best,
        "disagreement_days": sorted(disagreement_days),
    }


def _spring_break_diagnostic(trajectory: list[dict[str, str]]) -> dict:
    """Compare state before and after spring break week.

    Spring break is week 8 (1-indexed), covering days 49-55 (0-indexed).
    We compare the state on day 48 (last day before break) to day 56
    (first day after break).
    """
    break_start_day = (SPRING_BREAK_WEEK - 1) * 7  # day 49
    pre_break_idx = break_start_day  # state at start of break
    post_break_idx = break_start_day + 7  # state at end of break

    if post_break_idx >= len(trajectory) or pre_break_idx < 0:
        return {"available": False}

    pre = trajectory[pre_break_idx]
    post = trajectory[post_break_idx]

    changes = {}
    for var_name in _VAR_ORDER:
        pre_bin = pre.get(var_name, "")
        post_bin = post.get(var_name, "")
        if pre_bin != post_bin:
            changes[var_name] = {"before": pre_bin, "after": post_bin}

    return {
        "available": True,
        "pre_break_state": dict(pre),
        "post_break_state": dict(post),
        "changed_variables": changes,
        "n_changed": len(changes),
    }


def compute_ca_analytics(
    ca_result: dict,
    ode_result: dict | None = None,
) -> dict:
    """Compute all CA analytics from a single-cell CA result.

    Parameters
    ----------
    ca_result : dict
        Output from run_single_cell() with 'trajectory' and 'rule_log'.
    ode_result : dict or None
        Output from simulator.simulate() for fidelity comparison.

    Returns
    -------
    dict
        {
            "rule_stats": dict,
            "cascade_stats": dict,
            "attractor_stats": dict,
            "fidelity_stats": dict or None,
            "spring_break": dict,
        }
    """
    trajectory = ca_result["trajectory"]
    rule_log = ca_result["rule_log"]

    result = {
        "rule_stats": _rule_stats(rule_log),
        "cascade_stats": _cascade_stats(trajectory, rule_log),
        "attractor_stats": _attractor_stats(trajectory),
        "spring_break": _spring_break_diagnostic(trajectory),
    }

    if ode_result is not None:
        result["fidelity_stats"] = _fidelity_stats(trajectory, ode_result)
    else:
        result["fidelity_stats"] = None

    return result
