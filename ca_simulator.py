"""Cellular automaton simulator for the LEMURS student well-being model.

Provides two simulation modes:

1. **Single-cell mode**: One student, 105 daily steps (one semester).
   The 14D continuous ODE state is discretized into clinical bins, and
   tiered rules determine daily state transitions. Produces a trajectory
   of discrete states and a log of which rules fired each day.

2. **Population grid mode**: An NxN grid of students with shared
   institutional forcing (calendar, academic schedule) and optional
   social coupling between neighboring cells.

Both modes use the calendar functions from constants.py to drive
weekday/weekend and spring break dynamics.
"""
from __future__ import annotations

import numpy as np

from constants import (
    N_STEPS, N_STATES, STATE_NAMES,
    DEFAULT_INTERVENTION, DEFAULT_PATIENT, STUDENT_ARCHETYPES,
    is_weekday, is_school_day, week_of_semester,
    SPRING_BREAK_WEEK,
    _TST, _SQ, _PSS, _GAD7, _DEP, _ACT, _NAT,
    _RHR, _HRV, _ARR, _SJL, _SHAPE, _WB, _DAC,
    _LOWER, _UPPER,
)
from simulator import initial_state
from ca_schema import discretize_state, continuous_exemplar, BIN_SCHEMA, _VAR_ORDER
from ca_rules import apply_rules, get_applicable_rules, RULE_TABLE


def _build_context(
    day: int,
    patient: dict,
    intervention: dict,
    prev_state: dict[str, str] | None = None,
    curr_state: dict[str, str] | None = None,
) -> dict:
    """Build the context dict for rule evaluation on a given day.

    Combines calendar information, patient parameters, intervention
    parameters, and derived flags into a single context dict that
    rules can query.
    """
    week = week_of_semester(day)
    wd = is_weekday(day)
    school = is_school_day(day)
    spring_break = (week == SPRING_BREAK_WEEK - 1)

    ctx = {
        "day": day,
        "week": week,
        "is_weekday": wd,
        "is_school_day": school,
        "is_spring_break": spring_break,
    }

    # Patient parameters
    ctx["gender"] = patient.get("gender", 1.0)
    ctx["emotional_stability"] = patient.get("emotional_stability", 4.5)
    ctx["trauma_load"] = patient.get("trauma_load", 0.0)
    ctx["mh_diagnosis"] = patient.get("mh_diagnosis", 0.0)
    ctx["baseline_chronotype"] = patient.get("baseline_chronotype", 4.5)

    # Intervention parameters
    ctx["nature_rx"] = intervention.get("nature_rx", 0.0)
    ctx["exercise_rx"] = intervention.get("exercise_rx", 0.2)
    ctx["therapy_rx"] = intervention.get("therapy_rx", 0.0)
    ctx["sleep_hygiene"] = intervention.get("sleep_hygiene", 0.3)
    ctx["caffeine_reduction"] = intervention.get("caffeine_reduction", 0.0)
    ctx["academic_load"] = intervention.get("academic_load", 0.5)

    # Derived flags for rule evaluation
    ctx["female"] = ctx["gender"] < 0.5
    ctx["trauma_high"] = ctx["trauma_load"] >= 2.0
    ctx["late_chronotype"] = ctx["baseline_chronotype"] >= 5.0
    ctx["academic_load_high"] = ctx["academic_load"] > 0.6
    ctx["therapy_active"] = ctx["therapy_rx"] >= 0.2
    ctx["emotional_stability_low"] = ctx["emotional_stability"] < 4.0
    ctx["emotional_stability_high"] = ctx["emotional_stability"] >= 5.0

    # Within-person TST bin drop detection
    if prev_state is not None and curr_state is not None:
        prev_tst = prev_state.get("TST", "adequate")
        curr_tst = curr_state.get("TST", "adequate")
        tst_labels = BIN_SCHEMA["TST"]["labels"]
        ctx["tst_bin_dropped"] = (
            tst_labels.index(curr_tst) < tst_labels.index(prev_tst)
        )
    else:
        ctx["tst_bin_dropped"] = False

    return ctx


def step_cell(
    state: dict[str, str],
    context: dict,
    rules: list[dict] | None = None,
) -> tuple[dict[str, str], list[dict]]:
    """Apply tiered rules to one student for one day.

    Parameters
    ----------
    state : dict[str, str]
        Current discretized state {var_name: bin_label}.
    context : dict
        Calendar and patient/intervention context.
    rules : list[dict] or None
        Rule table. Defaults to RULE_TABLE.

    Returns
    -------
    tuple[dict[str, str], list[dict]]
        (new_state, fired_rules).
    """
    return apply_rules(state, context, rules)


def run_single_cell(
    patient: dict | None = None,
    intervention: dict | None = None,
    sim_days: int = N_STEPS,
    rules: list[dict] | None = None,
) -> dict:
    """Run a full semester CA simulation for one student.

    Parameters
    ----------
    patient : dict or None
        Student characteristics (merged with DEFAULT_PATIENT).
    intervention : dict or None
        Intervention parameters (merged with DEFAULT_INTERVENTION).
    sim_days : int
        Number of daily steps (default 105 = 15 weeks).
    rules : list[dict] or None
        Rule table. Defaults to RULE_TABLE.

    Returns
    -------
    dict
        {
            "trajectory": list[dict],   # discrete state at each day
            "rule_log": list[list],     # fired rules per day
            "final_state": dict,        # final discrete state
            "initial_continuous": np.ndarray,  # ODE initial state used
            "patient": dict,
            "intervention": dict,
        }
    """
    pat = {**DEFAULT_PATIENT, **(patient or {})}
    intv = {**DEFAULT_INTERVENTION, **(intervention or {})}

    # Initialize from ODE initial state
    pat_with_rx = {**pat, "nature_rx": intv.get("nature_rx", 0.0)}
    continuous_init = initial_state(pat_with_rx)
    state = discretize_state(continuous_init)

    trajectory = [dict(state)]
    rule_log = []
    prev_state = None

    for day in range(sim_days):
        ctx = _build_context(day, pat, intv, prev_state, state)
        new_state, fired = step_cell(state, ctx, rules)
        rule_log.append([r["name"] for r in fired])
        prev_state = state
        state = new_state
        trajectory.append(dict(state))

    return {
        "trajectory": trajectory,
        "rule_log": rule_log,
        "final_state": dict(state),
        "initial_continuous": continuous_init,
        "patient": pat,
        "intervention": intv,
    }


def _sample_patient(patient_distribution: dict, rng: np.random.Generator) -> dict:
    """Sample a patient from a distribution specification.

    The distribution dict maps patient parameter names to either:
    - A scalar value (used as-is)
    - A tuple (mean, std) for Gaussian sampling
    - A list of possible values (uniform discrete choice)
    """
    patient = {}
    for key, spec in patient_distribution.items():
        if isinstance(spec, (int, float)):
            patient[key] = float(spec)
        elif isinstance(spec, (list, tuple)) and len(spec) == 2:
            mean, std = spec
            patient[key] = float(rng.normal(mean, std))
        else:
            patient[key] = float(rng.choice(spec))
    return patient


def run_population_grid(
    grid_size: int = 5,
    patient_distribution: dict | None = None,
    intervention: dict | None = None,
    sim_days: int = N_STEPS,
    social_coupling: float | dict = 0.0,
    seed: int = 42,
    rules: list[dict] | None = None,
) -> dict:
    """Run a 2D population grid CA simulation.

    Each cell in the NxN grid represents a student. All students share
    institutional forcing (calendar, academic schedule). Optional social
    coupling allows neighboring students to influence each other's
    NatureEngagement, Activity, PSS, TST, and GAD7 levels.

    Parameters
    ----------
    grid_size : int
        Side length of the square grid (default 5 = 25 students).
    patient_distribution : dict or None
        Distribution specification for sampling patient parameters.
        If None, all students get DEFAULT_PATIENT.
    intervention : dict or None
        Shared intervention applied to all students.
    sim_days : int
        Number of daily steps.
    social_coupling : float or dict
        Strength of neighbor influence [0, 1]. 0 = independent cells.
        If a float, all 5 channels (nature, activity, stress, sleep,
        anxiety) use the same strength. If a dict, keys are channel
        names mapping to per-channel coupling strengths.
    seed : int
        Random seed for patient sampling.
    rules : list[dict] or None
        Rule table. Defaults to RULE_TABLE.

    Returns
    -------
    dict
        {
            "grid_states": list[list[list[dict]]],  # [day][row][col] discrete states
            "final_grid": list[list[dict]],          # final discrete states
            "population_summary": dict,              # aggregate statistics
            "grid_size": int,
            "sim_days": int,
            "social_coupling": dict,                 # per-channel coupling config
        }
    """
    intv = {**DEFAULT_INTERVENTION, **(intervention or {})}
    rng = np.random.default_rng(seed)

    # Parse social coupling config
    if isinstance(social_coupling, (int, float)):
        coupling_config = {
            "nature": float(social_coupling),
            "activity": float(social_coupling),
            "stress": float(social_coupling),
            "sleep": float(social_coupling),
            "anxiety": float(social_coupling),
        }
        coupling_scalar = float(social_coupling)
    else:
        coupling_config = {
            "nature": float(social_coupling.get("nature", 0.0)),
            "activity": float(social_coupling.get("activity", 0.0)),
            "stress": float(social_coupling.get("stress", 0.0)),
            "sleep": float(social_coupling.get("sleep", 0.0)),
            "anxiety": float(social_coupling.get("anxiety", 0.0)),
        }
        coupling_scalar = max(coupling_config.values()) if coupling_config else 0.0

    # Initialize grid of students
    grid: list[list[dict]] = []  # grid[row][col] = discrete_state
    patients: list[list[dict]] = []
    prev_grid: list[list[dict | None]] = [
        [None for _ in range(grid_size)] for _ in range(grid_size)
    ]

    for r in range(grid_size):
        row_states = []
        row_patients = []
        for c in range(grid_size):
            if patient_distribution:
                pat = _sample_patient(patient_distribution, rng)
                # Fill in any missing defaults
                for k, v in DEFAULT_PATIENT.items():
                    pat.setdefault(k, v)
            else:
                pat = dict(DEFAULT_PATIENT)
            row_patients.append(pat)

            pat_with_rx = {**pat, "nature_rx": intv.get("nature_rx", 0.0)}
            continuous_init = initial_state(pat_with_rx)
            state = discretize_state(continuous_init)
            row_states.append(state)
        grid.append(row_states)
        patients.append(row_patients)

    grid_states = [[
        [dict(grid[r][c]) for c in range(grid_size)]
        for r in range(grid_size)
    ]]

    # Simulate
    for day in range(sim_days):
        new_grid: list[list[dict]] = []
        for r in range(grid_size):
            new_row = []
            for c in range(grid_size):
                ctx = _build_context(
                    day, patients[r][c], intv,
                    prev_grid[r][c], grid[r][c],
                )

                new_state, _ = step_cell(grid[r][c], ctx, rules)

                # Social coupling: neighbor influence on Nature, Activity,
                # PSS (stress contagion), TST (sleep norms), GAD7 (anxiety diffusion)
                if coupling_scalar > 0:
                    neighbors = []
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid_size and 0 <= nc < grid_size:
                            neighbors.append(grid[nr][nc])

                    if neighbors:
                        n_neighbors = len(neighbors)

                        # --- Nature engagement channel ---
                        if coupling_config["nature"] > 0:
                            nature_engaged = sum(
                                1 for n in neighbors
                                if n.get("NatureEngagement") == "engaged"
                            )
                            if (nature_engaged / n_neighbors > 0.5
                                    and rng.random() < coupling_config["nature"]):
                                nat_labels = BIN_SCHEMA["NatureEngagement"]["labels"]
                                curr_idx = nat_labels.index(
                                    new_state.get("NatureEngagement", "low")
                                )
                                new_state["NatureEngagement"] = nat_labels[
                                    min(curr_idx + 1, len(nat_labels) - 1)
                                ]

                        # --- Activity channel ---
                        if coupling_config["activity"] > 0:
                            active_count = sum(
                                1 for n in neighbors
                                if n.get("Activity") == "active"
                            )
                            if (active_count / n_neighbors > 0.5
                                    and rng.random() < coupling_config["activity"]):
                                act_labels = BIN_SCHEMA["Activity"]["labels"]
                                curr_idx = act_labels.index(
                                    new_state.get("Activity", "moderate")
                                )
                                new_state["Activity"] = act_labels[
                                    min(curr_idx + 1, len(act_labels) - 1)
                                ]

                        # --- Stress contagion (PSS) ---
                        # Asymmetric: high-stress majority pushes PSS +1;
                        # low-stress majority helps reduce at half strength.
                        if coupling_config["stress"] > 0:
                            high_stress_count = sum(
                                1 for n in neighbors
                                if n.get("PSS") == "high"
                            )
                            low_stress_count = sum(
                                1 for n in neighbors
                                if n.get("PSS") == "low"
                            )
                            pss_labels = BIN_SCHEMA["PSS"]["labels"]
                            curr_pss_idx = pss_labels.index(
                                new_state.get("PSS", "low")
                            )
                            if (high_stress_count / n_neighbors > 0.5
                                    and rng.random() < coupling_config["stress"]):
                                new_state["PSS"] = pss_labels[
                                    min(curr_pss_idx + 1, len(pss_labels) - 1)
                                ]
                            elif (low_stress_count / n_neighbors > 0.5
                                    and rng.random() < coupling_config["stress"] * 0.5):
                                new_state["PSS"] = pss_labels[
                                    max(curr_pss_idx - 1, 0)
                                ]

                        # --- Sleep norm influence (TST) ---
                        # Only on school days: if majority of neighbors are
                        # deprived, push TST -1 (social norm toward less sleep).
                        if coupling_config["sleep"] > 0 and is_school_day(day):
                            deprived_count = sum(
                                1 for n in neighbors
                                if n.get("TST") == "deprived"
                            )
                            if (deprived_count / n_neighbors > 0.5
                                    and rng.random() < coupling_config["sleep"]):
                                tst_labels = BIN_SCHEMA["TST"]["labels"]
                                curr_tst_idx = tst_labels.index(
                                    new_state.get("TST", "adequate")
                                )
                                new_state["TST"] = tst_labels[
                                    max(curr_tst_idx - 1, 0)
                                ]

                        # --- Anxiety diffusion (GAD7) ---
                        # If >30% of neighbors are clinical AND student is
                        # sub_threshold AND student's PSS is moderate or high,
                        # flip to clinical.
                        if coupling_config["anxiety"] > 0:
                            clinical_count = sum(
                                1 for n in neighbors
                                if n.get("GAD7") == "clinical"
                            )
                            if (clinical_count / n_neighbors > 0.3
                                    and new_state.get("GAD7") == "sub_threshold"
                                    and new_state.get("PSS") in ("moderate", "high")
                                    and rng.random() < coupling_config["anxiety"]):
                                new_state["GAD7"] = "clinical"

                new_row.append(new_state)
            new_grid.append(new_row)

        prev_grid = grid
        grid = new_grid
        grid_states.append([
            [dict(grid[r][c]) for c in range(grid_size)]
            for r in range(grid_size)
        ])

    # Compute population summary
    summary = _compute_population_summary(grid, grid_size)

    return {
        "grid_states": grid_states,
        "final_grid": [
            [dict(grid[r][c]) for c in range(grid_size)]
            for r in range(grid_size)
        ],
        "population_summary": summary,
        "grid_size": grid_size,
        "sim_days": sim_days,
        "social_coupling": coupling_config,
    }


def _compute_population_summary(
    grid: list[list[dict]], grid_size: int
) -> dict:
    """Compute aggregate statistics over the final population grid."""
    total = grid_size * grid_size

    # Count bin distributions for each variable
    var_distributions: dict[str, dict[str, int]] = {}
    for var_name in _VAR_ORDER:
        counts: dict[str, int] = {}
        for r in range(grid_size):
            for c in range(grid_size):
                label = grid[r][c].get(var_name, "unknown")
                counts[label] = counts.get(label, 0) + 1
        var_distributions[var_name] = counts

    # Key clinical counts
    n_deprived = var_distributions["TST"].get("deprived", 0)
    n_clinical_anxiety = var_distributions["GAD7"].get("clinical", 0)
    n_high_stress = var_distributions["PSS"].get("high", 0)
    n_depleted_dac = var_distributions["DAC"].get("depleted", 0)
    n_disrupted_sleep = var_distributions["SleepShape"].get("disrupted", 0)

    # Burnout count: all four burnout conditions met
    n_burnout = 0
    for r in range(grid_size):
        for c in range(grid_size):
            s = grid[r][c]
            if (s.get("TST") == "deprived"
                    and s.get("PSS") == "high"
                    and s.get("DAC") == "depleted"
                    and s.get("GAD7") == "clinical"):
                n_burnout += 1

    return {
        "total_students": total,
        "sleep_deprived_count": n_deprived,
        "sleep_deprived_fraction": n_deprived / total,
        "clinical_anxiety_count": n_clinical_anxiety,
        "clinical_anxiety_fraction": n_clinical_anxiety / total,
        "high_stress_count": n_high_stress,
        "high_stress_fraction": n_high_stress / total,
        "depleted_attention_count": n_depleted_dac,
        "depleted_attention_fraction": n_depleted_dac / total,
        "disrupted_sleep_count": n_disrupted_sleep,
        "disrupted_sleep_fraction": n_disrupted_sleep / total,
        "burnout_count": n_burnout,
        "burnout_fraction": n_burnout / total,
        "variable_distributions": var_distributions,
    }
