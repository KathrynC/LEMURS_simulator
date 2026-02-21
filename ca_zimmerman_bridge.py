"""Zimmerman protocol adapters for the LEMURS CA simulator.

Two adapter classes expose CA simulation modes through the standard
Zimmerman protocol (run() + param_spec()), making them compatible with
zimmerman-toolkit's 14 interrogation modules and cramer-toolkit's
scenario-based resilience analysis.

LEMURSCASimulator: Single-cell CA with same 12D param_spec as the ODE.
LEMURSPopulationSimulator: Population grid with extended params.
"""
from __future__ import annotations

import math

from constants import (
    INTERVENTION_BOUNDS, PATIENT_BOUNDS,
    DEFAULT_INTERVENTION, DEFAULT_PATIENT,
)
from ca_simulator import run_single_cell, run_population_grid
from ca_analytics import compute_ca_analytics
from ca_schema import BIN_SCHEMA, _VAR_ORDER


class LEMURSCASimulator:
    """Zimmerman-protocol-compatible single-cell CA simulator.

    Uses the same 12D parameter space as LEMURSSimulator (6 intervention
    + 6 patient), but runs the CA rule engine instead of the ODE integrator.
    Output metrics come from CA analytics: rule firing patterns, attractor
    classification, and trajectory statistics.

    Example
    -------
        sim = LEMURSCASimulator()
        spec = sim.param_spec()
        result = sim.run({"nature_rx": 0.8, "emotional_stability": 6.0})
    """

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Return the 12D parameter bounds (same as ODE simulator)."""
        return {**INTERVENTION_BOUNDS, **PATIENT_BOUNDS}

    def run(self, params: dict[str, float]) -> dict[str, float]:
        """Run a single-cell CA simulation and return flattened metrics.

        Parameters
        ----------
        params : dict[str, float]
            Any subset of the 12 input parameters.

        Returns
        -------
        dict[str, float]
            Flat dict of scalar metrics suitable for Zimmerman analysis.
        """
        intervention = dict(DEFAULT_INTERVENTION)
        patient = dict(DEFAULT_PATIENT)

        for k, v in params.items():
            if k in INTERVENTION_BOUNDS:
                intervention[k] = float(v)
            elif k in PATIENT_BOUNDS:
                patient[k] = float(v)

        ca_result = run_single_cell(patient=patient, intervention=intervention)
        analytics = compute_ca_analytics(ca_result)

        flat = _flatten_ca_analytics(analytics, ca_result)
        return flat


class LEMURSPopulationSimulator:
    """Zimmerman-protocol-compatible population grid CA simulator.

    Extended parameter space: the 12 base parameters plus grid_size
    and social_coupling. Patient parameters define the population
    distribution center (all students sampled around these values).

    Example
    -------
        sim = LEMURSPopulationSimulator()
        spec = sim.param_spec()
        result = sim.run({"nature_rx": 0.5, "grid_size": 5, "social_coupling": 0.3})
    """

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Return parameter bounds including population-level controls."""
        spec = {**INTERVENTION_BOUNDS, **PATIENT_BOUNDS}
        spec["grid_size"] = (3.0, 10.0)
        spec["social_coupling"] = (0.0, 1.0)
        return spec

    def run(self, params: dict[str, float]) -> dict[str, float]:
        """Run a population grid CA and return summary statistics.

        Parameters
        ----------
        params : dict[str, float]
            Intervention, patient, and population parameters.

        Returns
        -------
        dict[str, float]
            Population-level summary metrics.
        """
        intervention = dict(DEFAULT_INTERVENTION)
        patient_center = dict(DEFAULT_PATIENT)

        grid_size = int(params.get("grid_size", 5))
        grid_size = max(3, min(10, grid_size))
        social_coupling = float(params.get("social_coupling", 0.0))

        for k, v in params.items():
            if k in INTERVENTION_BOUNDS:
                intervention[k] = float(v)
            elif k in PATIENT_BOUNDS:
                patient_center[k] = float(v)

        # Build distribution: each patient param is (center, small_std)
        patient_dist = {}
        for k, (lo, hi) in PATIENT_BOUNDS.items():
            center = patient_center[k]
            spread = (hi - lo) * 0.1  # 10% of range as std
            patient_dist[k] = (center, spread)

        pop_result = run_population_grid(
            grid_size=grid_size,
            patient_distribution=patient_dist,
            intervention=intervention,
            social_coupling=social_coupling,
        )

        flat = _flatten_population_summary(pop_result["population_summary"])
        return flat


def _flatten_ca_analytics(analytics: dict, ca_result: dict) -> dict[str, float]:
    """Convert nested CA analytics to a flat dict of floats."""
    flat: dict[str, float] = {}

    # Rule stats
    rs = analytics["rule_stats"]
    flat["ca_total_rule_firings"] = float(rs["total_firings"])
    flat["ca_unique_rules"] = float(rs["unique_rules"])
    flat["ca_mean_rules_per_day"] = float(rs["mean_rules_per_day"])
    flat["ca_days_with_rules"] = float(rs["days_with_rules"])

    # Cascade stats
    cs = analytics["cascade_stats"]
    flat["ca_cascade_count"] = float(cs["cascade_count"])
    flat["ca_max_cascade_length"] = float(cs["max_cascade_length"])

    # Attractor stats
    att = analytics["attractor_stats"]
    attractor_map = {"healthy": 0.0, "struggling": 1.0, "stressed": 2.0, "burnout": 3.0}
    flat["ca_final_attractor"] = attractor_map.get(att["final_attractor"], -1.0)
    flat["ca_attractor_stable"] = 1.0 if att["attractor_stable"] else 0.0
    flat["ca_last_week_transitions"] = float(att["transition_count"])

    # Spring break
    sb = analytics["spring_break"]
    flat["ca_spring_break_changes"] = float(sb.get("n_changed", 0))

    # Final state bin indices (encode as ordinal)
    final = ca_result["final_state"]
    for var_name in _VAR_ORDER:
        label = final.get(var_name, "")
        labels = BIN_SCHEMA[var_name]["labels"]
        flat[f"ca_final_{var_name}_bin"] = float(labels.index(label)) if label in labels else -1.0

    # Trajectory transition counts per variable
    trajectory = ca_result["trajectory"]
    for var_name in _VAR_ORDER:
        bins = [s.get(var_name) for s in trajectory]
        transitions = sum(1 for i in range(1, len(bins)) if bins[i] != bins[i - 1])
        flat[f"ca_transitions_{var_name}"] = float(transitions)

    # Fidelity (if available)
    fs = analytics.get("fidelity_stats")
    if fs is not None:
        flat["ca_ode_agreement"] = float(fs["overall_agreement"])

    # Guard against NaN/Inf
    for k, v in flat.items():
        if math.isnan(v):
            flat[k] = 0.0
        elif math.isinf(v):
            flat[k] = 999.0

    return flat


def _flatten_population_summary(summary: dict) -> dict[str, float]:
    """Convert population summary to a flat dict of floats."""
    flat: dict[str, float] = {}

    flat["pop_total_students"] = float(summary["total_students"])
    flat["pop_sleep_deprived_frac"] = float(summary["sleep_deprived_fraction"])
    flat["pop_clinical_anxiety_frac"] = float(summary["clinical_anxiety_fraction"])
    flat["pop_high_stress_frac"] = float(summary["high_stress_fraction"])
    flat["pop_depleted_attention_frac"] = float(summary["depleted_attention_fraction"])
    flat["pop_disrupted_sleep_frac"] = float(summary["disrupted_sleep_fraction"])
    flat["pop_burnout_frac"] = float(summary["burnout_fraction"])
    flat["pop_burnout_count"] = float(summary["burnout_count"])

    # Per-variable distribution: encode as fraction in each bin
    var_dists = summary.get("variable_distributions", {})
    total = summary["total_students"]
    for var_name, counts in var_dists.items():
        for label, count in counts.items():
            flat[f"pop_{var_name}_{label}_frac"] = float(count) / total if total > 0 else 0.0

    # Guard against NaN/Inf
    for k, v in flat.items():
        if math.isnan(v):
            flat[k] = 0.0
        elif math.isinf(v):
            flat[k] = 999.0

    return flat
