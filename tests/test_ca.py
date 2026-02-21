"""Tests for the LEMURS semantic cellular automaton.

Covers state discretization, rule table, single-cell simulation,
population grid, CA analytics, and Zimmerman bridge adapters.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from constants import (
    N_STATES, N_STEPS, STATE_NAMES,
    _TST, _SQ, _PSS, _GAD7, _DEP, _ACT, _NAT,
    _RHR, _HRV, _ARR, _SJL, _SHAPE, _WB, _DAC,
    _LOWER, _UPPER,
    DEFAULT_INTERVENTION, DEFAULT_PATIENT,
    STUDENT_ARCHETYPES,
)
from ca_schema import (
    BIN_SCHEMA, discretize_state, continuous_exemplar,
    bin_index, bin_count, _VAR_ORDER, _classify,
)
from ca_rules import (
    RULE_TABLE, apply_rules, get_applicable_rules,
    save_rules, load_rules,
    _evaluate_context, _evaluate_inputs, _apply_direction,
)
from ca_simulator import (
    step_cell, run_single_cell, run_population_grid,
    _build_context,
)
from ca_analytics import (
    compute_ca_analytics,
    compute_population_analytics,
    _rule_stats, _cascade_stats, _attractor_stats,
    _fidelity_stats, _spring_break_diagnostic,
    _classify_attractor,
)
from ca_zimmerman_bridge import (
    LEMURSCASimulator, LEMURSPopulationSimulator,
)
from ca_stochastic import (
    apply_rules_stochastic, run_single_cell_stochastic,
    compute_ensemble_analytics,
)
from ca_zimmerman_bridge import LEMURSCAEnsembleSimulator
from ca_visualize import (
    plot_ca_trajectory, plot_rule_timeline, plot_ca_fidelity,
    plot_population_grid,
)
from simulator import simulate


# ══════════════════════════════════════════════════════════════════════════════
# TestSchema
# ══════════════════════════════════════════════════════════════════════════════

class TestSchema:
    """State discretization, bin schema, and round-trip tests."""

    def test_all_14_variables_covered(self):
        """BIN_SCHEMA must have entries for all 14 state variables."""
        assert len(BIN_SCHEMA) == 14
        for name in STATE_NAMES:
            assert name in BIN_SCHEMA, f"Missing schema for {name}"

    def test_var_order_matches_state_names(self):
        """_VAR_ORDER must match STATE_NAMES ordering."""
        assert _VAR_ORDER == STATE_NAMES

    def test_schema_has_required_keys(self):
        """Each schema entry needs index, thresholds, labels, centers."""
        required = {"index", "thresholds", "labels", "centers", "unit", "source"}
        for name, schema in BIN_SCHEMA.items():
            assert required.issubset(schema.keys()), f"{name} missing keys"

    def test_labels_count_matches_thresholds(self):
        """Number of labels should be len(thresholds) + 1."""
        for name, schema in BIN_SCHEMA.items():
            assert len(schema["labels"]) == len(schema["thresholds"]) + 1, (
                f"{name}: {len(schema['labels'])} labels, "
                f"{len(schema['thresholds'])} thresholds"
            )

    def test_centers_count_matches_labels(self):
        """One center per bin."""
        for name, schema in BIN_SCHEMA.items():
            assert len(schema["centers"]) == len(schema["labels"]), (
                f"{name}: {len(schema['centers'])} centers, "
                f"{len(schema['labels'])} labels"
            )

    def test_thresholds_monotonically_increasing(self):
        """Thresholds must be in ascending order."""
        for name, schema in BIN_SCHEMA.items():
            thresholds = schema["thresholds"]
            for i in range(1, len(thresholds)):
                assert thresholds[i] > thresholds[i - 1], (
                    f"{name}: thresholds not monotonic: {thresholds}"
                )

    def test_discretize_default_initial_state(self):
        """Discretizing the default initial state should produce valid bins."""
        from simulator import initial_state
        state = initial_state()
        discrete = discretize_state(state)
        assert len(discrete) == 14
        for name, label in discrete.items():
            assert label in BIN_SCHEMA[name]["labels"], (
                f"{name}: got '{label}', expected one of {BIN_SCHEMA[name]['labels']}"
            )

    def test_discretize_lower_bounds(self):
        """All lower-bound values should land in the first bin."""
        discrete = discretize_state(_LOWER)
        for name in _VAR_ORDER:
            expected = BIN_SCHEMA[name]["labels"][0]
            assert discrete[name] == expected, (
                f"{name}: lower bound -> {discrete[name]}, expected {expected}"
            )

    def test_discretize_upper_bounds(self):
        """All upper-bound values should land in the last bin."""
        discrete = discretize_state(_UPPER)
        for name in _VAR_ORDER:
            expected = BIN_SCHEMA[name]["labels"][-1]
            assert discrete[name] == expected, (
                f"{name}: upper bound -> {discrete[name]}, expected {expected}"
            )

    def test_continuous_exemplar_produces_14d(self):
        """continuous_exemplar should return a 14-element float64 array."""
        discrete = discretize_state(_LOWER)
        exemplar = continuous_exemplar(discrete)
        assert exemplar.shape == (14,)
        assert exemplar.dtype == np.float64

    def test_round_trip_preserves_bins(self):
        """discretize(exemplar(discrete)) should equal the original discrete."""
        # Start from a known discrete state
        discrete_original = {name: BIN_SCHEMA[name]["labels"][0] for name in _VAR_ORDER}
        exemplar = continuous_exemplar(discrete_original)
        discrete_back = discretize_state(exemplar)
        assert discrete_back == discrete_original

    def test_classify_boundary_value_gad7(self):
        """GAD7 at exactly 10.0 should be 'clinical' (>= threshold)."""
        result = _classify(10.0, [10.0], ["sub_threshold", "clinical"])
        assert result == "clinical"

    def test_classify_just_below_boundary(self):
        """GAD7 at 9.99 should be 'sub_threshold'."""
        result = _classify(9.99, [10.0], ["sub_threshold", "clinical"])
        assert result == "sub_threshold"

    def test_bin_index(self):
        assert bin_index("GAD7", "sub_threshold") == 0
        assert bin_index("GAD7", "clinical") == 1

    def test_bin_count(self):
        assert bin_count("GAD7") == 2
        assert bin_count("TST") == 3
        assert bin_count("PSS") == 3


# ══════════════════════════════════════════════════════════════════════════════
# TestRules
# ══════════════════════════════════════════════════════════════════════════════

class TestRules:
    """Rule table structure, tier firing, and JSON serialization."""

    def test_rule_table_nonempty(self):
        assert len(RULE_TABLE) > 0

    def test_all_rules_have_required_keys(self):
        required = {"tier", "name", "inputs", "context", "outputs", "confidence", "citation"}
        for i, rule in enumerate(RULE_TABLE):
            assert required.issubset(rule.keys()), (
                f"Rule {i} ({rule.get('name', '?')}): missing keys"
            )

    def test_rule_names_unique(self):
        names = [r["name"] for r in RULE_TABLE]
        assert len(names) == len(set(names)), "Duplicate rule names"

    def test_all_tiers_represented(self):
        """Tiers 0-6 should all have at least one rule."""
        tiers = {r["tier"] for r in RULE_TABLE}
        for t in range(7):
            assert t in tiers, f"Tier {t} has no rules"

    def test_confidence_range(self):
        for rule in RULE_TABLE:
            assert 0.0 <= rule["confidence"] <= 1.0, (
                f"{rule['name']}: confidence={rule['confidence']}"
            )

    def test_tier1_sleep_debt_fires(self):
        """Tier 1: deprived sleep + poor quality should fire stress_up."""
        state = {name: BIN_SCHEMA[name]["labels"][0] for name in _VAR_ORDER}
        state["TST"] = "deprived"
        state["SleepQuality"] = "poor"
        applicable = get_applicable_rules(state, {})
        names = [r["name"] for r in applicable]
        assert "sleep_debt_stress_up" in names

    def test_tier2_anxiety_development(self):
        """Tier 2: sub-threshold GAD7 + high stress + low stability -> development."""
        state = {name: BIN_SCHEMA[name]["labels"][0] for name in _VAR_ORDER}
        state["GAD7"] = "sub_threshold"
        state["PSS"] = "high"
        ctx = {"emotional_stability": 2.0}
        applicable = get_applicable_rules(state, ctx)
        names = [r["name"] for r in applicable]
        assert "anxiety_development_risk" in names

    def test_tier3_nature_restoration(self):
        """Tier 3: engaged nature + available DAC -> stress down, HRV up."""
        state = {name: BIN_SCHEMA[name]["labels"][0] for name in _VAR_ORDER}
        state["NatureEngagement"] = "engaged"
        state["DAC"] = "available"
        applicable = get_applicable_rules(state, {})
        names = [r["name"] for r in applicable]
        assert "nature_restoration" in names

    def test_tier4_school_day_sleep_debt(self):
        """Tier 4: school day -> TST decreases."""
        state = {name: BIN_SCHEMA[name]["labels"][0] for name in _VAR_ORDER}
        ctx = {"is_school_day": True}
        applicable = get_applicable_rules(state, ctx)
        names = [r["name"] for r in applicable]
        assert "school_day_sleep_debt" in names

    def test_tier5_late_chronotype_jetlag(self):
        """Tier 5: late chronotype on school day -> misaligned."""
        state = {name: BIN_SCHEMA[name]["labels"][0] for name in _VAR_ORDER}
        ctx = {"baseline_chronotype": 6.0, "is_school_day": True}
        applicable = get_applicable_rules(state, ctx)
        names = [r["name"] for r in applicable]
        assert "late_chronotype_jetlag" in names

    def test_tier6_academic_depletion(self):
        """Tier 6: high academic load -> DAC depletes."""
        state = {name: BIN_SCHEMA[name]["labels"][0] for name in _VAR_ORDER}
        ctx = {"academic_load": 0.9}
        applicable = get_applicable_rules(state, ctx)
        names = [r["name"] for r in applicable]
        assert "academic_attention_depletion" in names

    def test_burnout_cascade_fires(self):
        """Cross-tier: all four burnout conditions -> absorbing state."""
        state = {name: BIN_SCHEMA[name]["labels"][0] for name in _VAR_ORDER}
        state["TST"] = "deprived"
        state["PSS"] = "high"
        state["DAC"] = "depleted"
        state["GAD7"] = "clinical"
        applicable = get_applicable_rules(state, {})
        names = [r["name"] for r in applicable]
        assert "burnout_cascade" in names

    def test_burnout_cascade_freezes_state(self):
        """When burnout cascade fires, state should not change."""
        state = {name: BIN_SCHEMA[name]["labels"][0] for name in _VAR_ORDER}
        state["TST"] = "deprived"
        state["PSS"] = "high"
        state["DAC"] = "depleted"
        state["GAD7"] = "clinical"
        new_state, fired = apply_rules(state, {})
        assert new_state == state

    def test_spring_break_fires(self):
        """Spring break context should trigger reset rule."""
        state = {name: BIN_SCHEMA[name]["labels"][0] for name in _VAR_ORDER}
        ctx = {"is_spring_break": True}
        applicable = get_applicable_rules(state, ctx)
        names = [r["name"] for r in applicable]
        assert "spring_break_reset" in names

    def test_apply_direction_up(self):
        result = _apply_direction("deprived", "+1", "TST")
        assert result == "adequate"

    def test_apply_direction_down(self):
        result = _apply_direction("adequate", "-1", "TST")
        assert result == "deprived"

    def test_apply_direction_clamp_top(self):
        result = _apply_direction("excess", "+1", "TST")
        assert result == "excess"

    def test_apply_direction_clamp_bottom(self):
        result = _apply_direction("deprived", "-1", "TST")
        assert result == "deprived"

    def test_apply_direction_absolute(self):
        result = _apply_direction("deprived", "adequate", "TST")
        assert result == "adequate"

    def test_apply_direction_zero(self):
        result = _apply_direction("adequate", "0", "TST")
        assert result == "adequate"

    def test_json_round_trip(self):
        """Rules should survive JSON serialization and deserialization."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_rules(path)
            loaded = load_rules(path)
            assert len(loaded) == len(RULE_TABLE)
            for orig, load in zip(RULE_TABLE, loaded):
                assert orig["name"] == load["name"]
                assert orig["tier"] == load["tier"]
                assert orig["inputs"] == load["inputs"]
                assert orig["outputs"] == load["outputs"]
        finally:
            os.unlink(path)

    def test_conflict_resolution_by_confidence(self):
        """When two rules update the same variable, higher confidence wins."""
        rules = [
            {"tier": 1, "name": "low_conf", "inputs": {}, "context": {},
             "outputs": {"PSS": "+1"}, "confidence": 0.3, "citation": "test"},
            {"tier": 1, "name": "high_conf", "inputs": {}, "context": {},
             "outputs": {"PSS": "-1"}, "confidence": 0.9, "citation": "test"},
        ]
        state = {"PSS": "moderate"}
        # Both match (empty inputs) — high_conf should win
        new_state, fired = apply_rules(state, {}, rules)
        assert new_state["PSS"] == "low"  # -1 from moderate = low


# ══════════════════════════════════════════════════════════════════════════════
# TestSingleCell
# ══════════════════════════════════════════════════════════════════════════════

class TestSingleCell:
    """Single-cell CA simulation end-to-end tests."""

    def test_default_runs_without_error(self):
        """Default student should complete a full semester."""
        result = run_single_cell()
        assert "trajectory" in result
        assert "rule_log" in result
        assert "final_state" in result

    def test_trajectory_length(self):
        """Trajectory should have sim_days+1 entries (initial + 105 steps)."""
        result = run_single_cell()
        assert len(result["trajectory"]) == N_STEPS + 1

    def test_rule_log_length(self):
        """Rule log should have sim_days entries (one per step)."""
        result = run_single_cell()
        assert len(result["rule_log"]) == N_STEPS

    def test_final_state_has_all_vars(self):
        """Final state should have all 14 variables."""
        result = run_single_cell()
        for name in _VAR_ORDER:
            assert name in result["final_state"]

    def test_all_bins_valid(self):
        """Every bin in every trajectory step should be a valid label."""
        result = run_single_cell()
        for step, state in enumerate(result["trajectory"]):
            for name in _VAR_ORDER:
                label = state[name]
                assert label in BIN_SCHEMA[name]["labels"], (
                    f"Step {step}, {name}: invalid bin '{label}'"
                )

    def test_custom_sim_days(self):
        """sim_days parameter should control trajectory length."""
        result = run_single_cell(sim_days=21)
        assert len(result["trajectory"]) == 22  # 21 steps + initial
        assert len(result["rule_log"]) == 21

    def test_vulnerable_student_runs(self):
        """Vulnerable archetype should run without error."""
        arch = STUDENT_ARCHETYPES[2]  # vulnerable_female
        result = run_single_cell(
            patient=arch["patient"],
            intervention=arch.get("intervention"),
        )
        assert len(result["trajectory"]) == N_STEPS + 1

    def test_all_archetypes_run(self):
        """All 8 student archetypes should complete without error."""
        for arch in STUDENT_ARCHETYPES:
            result = run_single_cell(
                patient=arch["patient"],
                intervention=arch.get("intervention"),
            )
            assert len(result["trajectory"]) == N_STEPS + 1, (
                f"Archetype {arch['name']} failed"
            )

    def test_build_context_school_day(self):
        """Day 0 (Monday) should be a school day."""
        ctx = _build_context(0, DEFAULT_PATIENT, DEFAULT_INTERVENTION)
        assert ctx["is_school_day"] is True
        assert ctx["is_weekday"] is True

    def test_build_context_weekend(self):
        """Day 5 (Saturday) should not be a school day."""
        ctx = _build_context(5, DEFAULT_PATIENT, DEFAULT_INTERVENTION)
        assert ctx["is_weekday"] is False
        assert ctx["is_school_day"] is False

    def test_build_context_spring_break(self):
        """Day 49 (start of week 8) should be spring break."""
        ctx = _build_context(49, DEFAULT_PATIENT, DEFAULT_INTERVENTION)
        assert ctx["is_spring_break"] is True


# ══════════════════════════════════════════════════════════════════════════════
# TestPopulationGrid
# ══════════════════════════════════════════════════════════════════════════════

class TestPopulationGrid:
    """Population grid CA simulation tests."""

    def test_grid_initializes(self):
        """3x3 grid with 21 days should produce valid output."""
        result = run_population_grid(grid_size=3, sim_days=21)
        assert result["grid_size"] == 3
        assert result["sim_days"] == 21
        assert len(result["final_grid"]) == 3
        assert len(result["final_grid"][0]) == 3

    def test_grid_states_length(self):
        """grid_states should have sim_days+1 entries."""
        result = run_population_grid(grid_size=3, sim_days=7)
        assert len(result["grid_states"]) == 8  # 7 + initial

    def test_shared_forcing(self):
        """All students should respond to the same calendar forcing."""
        result = run_population_grid(grid_size=3, sim_days=7)
        # All cells should exist and have valid bins
        for r in range(3):
            for c in range(3):
                state = result["final_grid"][r][c]
                for name in _VAR_ORDER:
                    assert name in state

    def test_social_coupling_optional(self):
        """Grid should work with social_coupling=0 (independent cells)."""
        result = run_population_grid(grid_size=3, sim_days=7, social_coupling=0.0)
        assert isinstance(result["social_coupling"], dict)
        for ch in ("nature", "activity", "stress", "sleep", "anxiety"):
            assert result["social_coupling"][ch] == 0.0

    def test_social_coupling_nonzero(self):
        """Grid should work with social_coupling > 0."""
        result = run_population_grid(grid_size=3, sim_days=7, social_coupling=0.5)
        assert isinstance(result["social_coupling"], dict)
        for ch in ("nature", "activity", "stress", "sleep", "anxiety"):
            assert result["social_coupling"][ch] == 0.5

    def test_population_summary_has_keys(self):
        """Population summary should contain expected metrics."""
        result = run_population_grid(grid_size=3, sim_days=7)
        summary = result["population_summary"]
        assert "total_students" in summary
        assert summary["total_students"] == 9
        assert "sleep_deprived_fraction" in summary
        assert "clinical_anxiety_fraction" in summary
        assert "burnout_fraction" in summary
        assert "variable_distributions" in summary

    def test_patient_distribution(self):
        """Custom patient distribution should be used for initialization."""
        dist = {
            "gender": 0.0,  # all male
            "emotional_stability": (4.5, 0.5),  # normal around 4.5
        }
        result = run_population_grid(
            grid_size=3, patient_distribution=dist, sim_days=7
        )
        assert result["population_summary"]["total_students"] == 9

    def test_deterministic_with_seed(self):
        """Same seed should produce identical results."""
        r1 = run_population_grid(grid_size=3, sim_days=7, seed=123)
        r2 = run_population_grid(grid_size=3, sim_days=7, seed=123)
        assert r1["final_grid"] == r2["final_grid"]


# ══════════════════════════════════════════════════════════════════════════════
# TestAnalytics
# ══════════════════════════════════════════════════════════════════════════════

class TestAnalytics:
    """CA analytics: rule stats, cascades, attractors, fidelity."""

    @pytest.fixture
    def ca_result(self):
        return run_single_cell()

    @pytest.fixture
    def ode_result(self):
        from simulator import simulate
        return simulate()

    def test_compute_all_sections(self, ca_result):
        """compute_ca_analytics should return all 4 sections."""
        analytics = compute_ca_analytics(ca_result)
        assert "rule_stats" in analytics
        assert "cascade_stats" in analytics
        assert "attractor_stats" in analytics
        assert "spring_break" in analytics

    def test_rule_stats_structure(self, ca_result):
        rs = compute_ca_analytics(ca_result)["rule_stats"]
        assert "total_firings" in rs
        assert "unique_rules" in rs
        assert "rule_counts" in rs
        assert "mean_rules_per_day" in rs
        assert rs["total_firings"] > 0

    def test_cascade_stats_structure(self, ca_result):
        cs = compute_ca_analytics(ca_result)["cascade_stats"]
        assert "cascade_count" in cs
        assert "max_cascade_length" in cs
        assert isinstance(cs["cascade_sequences"], list)

    def test_attractor_stats_structure(self, ca_result):
        att = compute_ca_analytics(ca_result)["attractor_stats"]
        assert "final_attractor" in att
        assert att["final_attractor"] in {"healthy", "struggling", "stressed", "burnout", "unknown"}
        assert "attractor_stable" in att

    def test_fidelity_without_ode(self, ca_result):
        """Fidelity should be None when no ODE result provided."""
        analytics = compute_ca_analytics(ca_result)
        assert analytics["fidelity_stats"] is None

    def test_fidelity_with_ode(self, ca_result, ode_result):
        """Fidelity should be computed when ODE result provided."""
        analytics = compute_ca_analytics(ca_result, ode_result=ode_result)
        fs = analytics["fidelity_stats"]
        assert fs is not None
        assert "overall_agreement" in fs
        assert 0.0 <= fs["overall_agreement"] <= 1.0
        assert "per_variable_agreement" in fs
        assert len(fs["per_variable_agreement"]) == 14

    def test_spring_break_diagnostic(self, ca_result):
        sb = compute_ca_analytics(ca_result)["spring_break"]
        assert sb["available"] is True
        assert "pre_break_state" in sb
        assert "post_break_state" in sb

    def test_classify_attractor_healthy(self):
        state = {"PSS": "low", "GAD7": "sub_threshold", "TST": "adequate", "DAC": "available"}
        assert _classify_attractor(state) == "healthy"

    def test_classify_attractor_burnout(self):
        state = {"PSS": "high", "GAD7": "clinical", "TST": "deprived", "DAC": "depleted"}
        assert _classify_attractor(state) == "burnout"

    def test_classify_attractor_stressed(self):
        state = {"PSS": "high", "GAD7": "sub_threshold", "TST": "adequate", "DAC": "available"}
        assert _classify_attractor(state) == "stressed"

    def test_classify_attractor_struggling(self):
        state = {"PSS": "moderate", "GAD7": "sub_threshold", "TST": "adequate", "DAC": "available"}
        assert _classify_attractor(state) == "struggling"


# ══════════════════════════════════════════════════════════════════════════════
# TestZimmermanBridge
# ══════════════════════════════════════════════════════════════════════════════

class TestZimmermanBridge:
    """Zimmerman protocol adapter tests."""

    def test_ca_simulator_param_spec(self):
        """LEMURSCASimulator should return valid 12D param_spec."""
        sim = LEMURSCASimulator()
        spec = sim.param_spec()
        assert isinstance(spec, dict)
        assert len(spec) == 12
        for name, bounds in spec.items():
            assert isinstance(bounds, tuple)
            assert len(bounds) == 2
            assert bounds[0] <= bounds[1]

    def test_ca_simulator_run_default(self):
        """LEMURSCASimulator.run({}) should return dict of floats."""
        sim = LEMURSCASimulator()
        result = sim.run({})
        assert isinstance(result, dict)
        assert len(result) > 0
        for key, value in result.items():
            assert isinstance(key, str)
            assert isinstance(value, float), f"{key}: {type(value)}"

    def test_ca_simulator_run_custom_params(self):
        """Custom parameters should produce valid output."""
        sim = LEMURSCASimulator()
        result = sim.run({"nature_rx": 0.8, "emotional_stability": 6.0})
        assert isinstance(result, dict)
        assert "ca_final_attractor" in result
        assert "ca_total_rule_firings" in result

    def test_ca_simulator_no_nan(self):
        """Output should have no NaN values."""
        sim = LEMURSCASimulator()
        result = sim.run({})
        for key, value in result.items():
            assert not np.isnan(value), f"{key} is NaN"
            assert not np.isinf(value) or value == 999.0, f"{key} is Inf"

    def test_ca_simulator_has_final_bins(self):
        """Output should include final bin indices for each variable."""
        sim = LEMURSCASimulator()
        result = sim.run({})
        for var_name in _VAR_ORDER:
            key = f"ca_final_{var_name}_bin"
            assert key in result, f"Missing {key}"
            assert result[key] >= 0, f"{key} = {result[key]}"

    def test_ca_simulator_has_transitions(self):
        """Output should include transition counts per variable."""
        sim = LEMURSCASimulator()
        result = sim.run({})
        for var_name in _VAR_ORDER:
            key = f"ca_transitions_{var_name}"
            assert key in result, f"Missing {key}"

    def test_population_simulator_param_spec(self):
        """LEMURSPopulationSimulator should include grid_size and coupling."""
        sim = LEMURSPopulationSimulator()
        spec = sim.param_spec()
        assert "grid_size" in spec
        assert "social_coupling" in spec
        assert len(spec) == 14  # 12 base + 2 population

    def test_population_simulator_run_default(self):
        """Population simulator should return dict of floats."""
        sim = LEMURSPopulationSimulator()
        result = sim.run({})
        assert isinstance(result, dict)
        assert "pop_total_students" in result
        assert "pop_burnout_frac" in result
        for key, value in result.items():
            assert isinstance(value, float), f"{key}: {type(value)}"

    def test_population_simulator_custom_grid(self):
        """Custom grid_size should be respected."""
        sim = LEMURSPopulationSimulator()
        result = sim.run({"grid_size": 4})
        assert result["pop_total_students"] == 16.0

    def test_population_simulator_no_nan(self):
        """Population output should have no NaN values."""
        sim = LEMURSPopulationSimulator()
        result = sim.run({})
        for key, value in result.items():
            assert not np.isnan(value), f"{key} is NaN"


# ══════════════════════════════════════════════════════════════════════════════
# TestCAVisualization
# ══════════════════════════════════════════════════════════════════════════════

class TestCAVisualization:
    """CA visualization: trajectory heatmap, rule timeline, fidelity, population grid."""

    def test_trajectory_plot_creates_file(self, tmp_path):
        """plot_ca_trajectory should produce a non-empty PNG file."""
        ca_result = run_single_cell()
        out = str(tmp_path / "trajectory.png")
        plot_ca_trajectory(ca_result, "Test Trajectory", out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_rule_timeline_creates_file(self, tmp_path):
        """plot_rule_timeline should produce a non-empty PNG file."""
        ca_result = run_single_cell()
        out = str(tmp_path / "rules.png")
        plot_rule_timeline(ca_result, "Test Rules", out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_fidelity_plot_creates_file(self, tmp_path):
        """plot_ca_fidelity should produce a non-empty PNG file."""
        ca_result = run_single_cell()
        ode_result = simulate()
        out = str(tmp_path / "fidelity.png")
        plot_ca_fidelity(ca_result, ode_result, "Test Fidelity", out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_population_grid_creates_file(self, tmp_path):
        """plot_population_grid should produce a non-empty PNG file."""
        pop_result = run_population_grid(grid_size=3, social_coupling=0.2)
        out = str(tmp_path / "population.png")
        plot_population_grid(pop_result, output_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0


# ══════════════════════════════════════════════════════════════════════════════
# TestSocialCoupling
# ══════════════════════════════════════════════════════════════════════════════

class TestSocialCoupling:
    """Tests for expanded social coupling channels."""

    def test_stress_contagion_runs(self):
        result = run_population_grid(
            grid_size=3,
            social_coupling={"nature": 0.0, "activity": 0.0, "stress": 0.8,
                             "sleep": 0.0, "anxiety": 0.0},
            seed=42,
        )
        assert "population_summary" in result

    def test_sleep_norm_influence_runs(self):
        result = run_population_grid(
            grid_size=3,
            social_coupling={"nature": 0.0, "activity": 0.0, "stress": 0.0,
                             "sleep": 0.8, "anxiety": 0.0},
            seed=42,
        )
        assert "population_summary" in result

    def test_anxiety_diffusion_runs(self):
        result = run_population_grid(
            grid_size=3,
            social_coupling={"nature": 0.0, "activity": 0.0, "stress": 0.0,
                             "sleep": 0.0, "anxiety": 0.8},
            seed=42,
        )
        assert "population_summary" in result

    def test_dict_coupling_config_in_result(self):
        result = run_population_grid(grid_size=3, social_coupling=0.3, seed=42)
        assert isinstance(result["social_coupling"], dict)
        assert "stress" in result["social_coupling"]
        assert "sleep" in result["social_coupling"]
        assert "anxiety" in result["social_coupling"]

    def test_all_channels_active(self):
        result = run_population_grid(
            grid_size=3,
            social_coupling={"nature": 0.5, "activity": 0.5, "stress": 0.5,
                             "sleep": 0.5, "anxiety": 0.5},
            seed=42,
        )
        assert result["population_summary"]["total_students"] == 9

    def test_population_analytics(self):
        pop_result = run_population_grid(grid_size=3, social_coupling=0.3, seed=42)
        analytics = compute_population_analytics(pop_result)
        assert "attractor_distribution" in analytics
        assert "attractor_fractions" in analytics
        assert "largest_stressed_cluster" in analytics
        total_fracs = sum(analytics["attractor_fractions"].values())
        assert abs(total_fracs - 1.0) < 1e-6

    def test_largest_cluster_bounded(self):
        pop_result = run_population_grid(grid_size=3, social_coupling=0.3, seed=42)
        analytics = compute_population_analytics(pop_result)
        assert 0 <= analytics["largest_stressed_cluster"] <= 9


# ══════════════════════════════════════════════════════════════════════════════
# TestStochastic
# ══════════════════════════════════════════════════════════════════════════════

class TestStochastic:
    """Stochastic rule engine and ensemble tests."""

    def test_stochastic_rules_produce_valid_state(self):
        ca_result = run_single_cell()
        state = ca_result["trajectory"][50]
        ctx = _build_context(50, DEFAULT_PATIENT, DEFAULT_INTERVENTION)
        rng = np.random.default_rng(42)
        new_state, fired = apply_rules_stochastic(state, ctx, rng)
        for var_name in _VAR_ORDER:
            assert new_state[var_name] in BIN_SCHEMA[var_name]["labels"]

    def test_ensemble_runs(self):
        result = run_single_cell_stochastic(n_trials=10, seed=42)
        assert len(result["trajectories"]) == 10
        assert len(result["final_states"]) == 10

    def test_ensemble_analytics(self):
        result = run_single_cell_stochastic(n_trials=20, seed=42)
        analytics = compute_ensemble_analytics(result)
        assert 0.0 <= analytics["burnout_probability"] <= 1.0
        assert 0.0 <= analytics["anxiety_crossing_probability"] <= 1.0
        total_prob = sum(analytics["attractor_probabilities"].values())
        assert abs(total_prob - 1.0) < 1e-6

    def test_ensemble_deterministic_same_seed(self):
        r1 = run_single_cell_stochastic(n_trials=5, seed=42)
        r2 = run_single_cell_stochastic(n_trials=5, seed=42)
        for i in range(5):
            assert r1["final_states"][i] == r2["final_states"][i]

    def test_burnout_cascade_still_absorbing(self):
        burnout_state = {
            "TST": "deprived", "SleepQuality": "poor", "PSS": "high",
            "GAD7": "clinical", "Depression": "moderate_plus", "Activity": "sedentary",
            "NatureEngagement": "low", "RHR": "elevated", "HRV": "low",
            "ARR": "elevated", "SocialJetlag": "misaligned", "SleepShape": "disrupted",
            "WEMWBS": "low", "DAC": "depleted",
        }
        ctx = _build_context(50, DEFAULT_PATIENT, DEFAULT_INTERVENTION)
        rng = np.random.default_rng(42)
        new_state, _ = apply_rules_stochastic(burnout_state, ctx, rng)
        assert new_state == burnout_state


# ══════════════════════════════════════════════════════════════════════════════
# TestEnsembleBridge
# ══════════════════════════════════════════════════════════════════════════════

class TestEnsembleBridge:
    """Zimmerman adapter for stochastic ensemble."""

    def test_ensemble_simulator_runs(self):
        sim = LEMURSCAEnsembleSimulator(n_trials=5)
        result = sim.run({})
        assert isinstance(result, dict)
        assert "ens_burnout_probability" in result
        assert "ens_anxiety_crossing_prob" in result

    def test_ensemble_param_spec(self):
        sim = LEMURSCAEnsembleSimulator()
        spec = sim.param_spec()
        assert len(spec) == 12

    def test_ensemble_all_values_finite(self):
        import math as _math
        sim = LEMURSCAEnsembleSimulator(n_trials=5)
        result = sim.run({"nature_rx": 0.8})
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not float"
            assert not _math.isnan(v), f"{k} is NaN"
            assert not _math.isinf(v), f"{k} is Inf"
