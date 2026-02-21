"""Tests for the K-Cramer Toolkit bridge.

Verifies that domain-specific stress scenarios are valid cramer-toolkit
objects, the scenario bank is complete, and convenience analysis functions
are wired up correctly for the LEMURS semester simulator.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure cramer-toolkit is importable
PROJECT = Path(__file__).resolve().parent.parent
KCRAMER_TOOLKIT_PATH = PROJECT.parent / "cramer-toolkit"
if str(KCRAMER_TOOLKIT_PATH) not in sys.path:
    sys.path.insert(0, str(KCRAMER_TOOLKIT_PATH))

from kcramer.base import Scenario, ScenarioSet, Modification

from kcramer_bridge import (
    HAS_CRAMER,
    ACADEMIC_SCENARIOS,
    SLEEP_SCENARIOS,
    SOCIAL_SCENARIOS,
    SEASONAL_SCENARIOS,
    DIGITAL_SCENARIOS,
    HEALTH_SCENARIOS,
    COMBINED_SCENARIOS,
    ALL_STRESS_SCENARIOS,
    ALL_SCENARIOS,
    PROTOCOLS,
    REFERENCE_PROTOCOLS,
    apply_scenario,
    run_scenario_sweep,
    run_resilience_analysis,
    run_vulnerability_analysis,
    run_scenario_comparison,
)

from constants import (
    INTERVENTION_NAMES,
    DEFAULT_INTERVENTION,
    DEFAULT_PATIENT,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

ALL_BANKS = [
    ACADEMIC_SCENARIOS,
    SLEEP_SCENARIOS,
    SOCIAL_SCENARIOS,
    SEASONAL_SCENARIOS,
    DIGITAL_SCENARIOS,
    HEALTH_SCENARIOS,
    COMBINED_SCENARIOS,
]


@pytest.fixture(scope="module")
def default_params():
    """Default 12D parameter dict."""
    return {**DEFAULT_INTERVENTION, **DEFAULT_PATIENT}


# ── Import and flag ─────────────────────────────────────────────────────────

class TestImport:
    """Verify cramer-toolkit imports succeeded."""

    def test_has_cramer_is_true(self):
        assert HAS_CRAMER is True

    def test_scenario_class_available(self):
        assert Scenario is not None

    def test_scenario_set_class_available(self):
        assert ScenarioSet is not None


# ── Scenario bank structure ─────────────────────────────────────────────────

class TestScenarioBankStructure:
    """Verify all scenario banks are properly constructed ScenarioSet objects."""

    def test_academic_is_scenario_set(self):
        assert isinstance(ACADEMIC_SCENARIOS, ScenarioSet)

    def test_sleep_is_scenario_set(self):
        assert isinstance(SLEEP_SCENARIOS, ScenarioSet)

    def test_social_is_scenario_set(self):
        assert isinstance(SOCIAL_SCENARIOS, ScenarioSet)

    def test_seasonal_is_scenario_set(self):
        assert isinstance(SEASONAL_SCENARIOS, ScenarioSet)

    def test_digital_is_scenario_set(self):
        assert isinstance(DIGITAL_SCENARIOS, ScenarioSet)

    def test_health_is_scenario_set(self):
        assert isinstance(HEALTH_SCENARIOS, ScenarioSet)

    def test_combined_is_scenario_set(self):
        assert isinstance(COMBINED_SCENARIOS, ScenarioSet)

    def test_all_stress_is_scenario_set(self):
        assert isinstance(ALL_STRESS_SCENARIOS, ScenarioSet)

    def test_all_scenarios_alias(self):
        """ALL_SCENARIOS is a backward compat alias for ALL_STRESS_SCENARIOS."""
        assert ALL_SCENARIOS is ALL_STRESS_SCENARIOS

    def test_per_bank_counts(self):
        assert len(ACADEMIC_SCENARIOS) == 3
        assert len(SLEEP_SCENARIOS) == 3
        assert len(SOCIAL_SCENARIOS) == 3
        assert len(SEASONAL_SCENARIOS) == 2
        assert len(DIGITAL_SCENARIOS) == 2
        assert len(HEALTH_SCENARIOS) == 3
        assert len(COMBINED_SCENARIOS) == 3

    def test_total_scenario_count(self):
        """19 total scenarios across 7 banks."""
        expected = sum(len(bank) for bank in ALL_BANKS)
        assert expected == 19
        assert len(ALL_STRESS_SCENARIOS) == 19

    def test_all_scenarios_are_scenario_type(self):
        for s in ALL_STRESS_SCENARIOS:
            assert isinstance(s, Scenario), f"{s!r} is not a Scenario"

    def test_all_scenarios_have_names(self):
        for s in ALL_STRESS_SCENARIOS:
            assert s.name, f"Scenario missing name: {s!r}"
            assert len(s.name) > 3, f"Name too short: {s.name}"

    def test_all_scenarios_have_descriptions(self):
        for s in ALL_STRESS_SCENARIOS:
            assert s.description, f"Scenario {s.name} missing description"

    def test_all_scenarios_have_modifications(self):
        for s in ALL_STRESS_SCENARIOS:
            assert len(s.modifications) > 0, \
                f"Scenario {s.name} has no modifications"
            for mod in s.modifications:
                assert isinstance(mod, Modification), \
                    f"Scenario {s.name}: {mod!r} is not a Modification"

    def test_unique_names(self):
        names = [s.name for s in ALL_STRESS_SCENARIOS]
        assert len(names) == len(set(names)), \
            f"Duplicate names: {[n for n in names if names.count(n) > 1]}"


# ── Expected scenario names ─────────────────────────────────────────────────

class TestExpectedScenarios:
    """Verify all 19 expected scenarios exist with correct names."""

    @pytest.mark.parametrize("name", [
        "mild_academic_stress", "exam_week", "academic_crisis",
    ])
    def test_academic_names(self, name):
        assert name in ACADEMIC_SCENARIOS

    @pytest.mark.parametrize("name", [
        "mild_insomnia", "chronic_insomnia", "severe_deprivation",
    ])
    def test_sleep_names(self, name):
        assert name in SLEEP_SCENARIOS

    @pytest.mark.parametrize("name", [
        "mild_isolation", "moderate_isolation", "full_isolation",
    ])
    def test_social_names(self, name):
        assert name in SOCIAL_SCENARIOS

    @pytest.mark.parametrize("name", [
        "winter_darkness", "summer_break",
    ])
    def test_seasonal_names(self, name):
        assert name in SEASONAL_SCENARIOS

    @pytest.mark.parametrize("name", [
        "moderate_screen", "digital_addiction",
    ])
    def test_digital_names(self, name):
        assert name in DIGITAL_SCENARIOS

    @pytest.mark.parametrize("name", [
        "prior_anxiety", "prior_depression", "trauma_exposure",
    ])
    def test_health_names(self, name):
        assert name in HEALTH_SCENARIOS

    @pytest.mark.parametrize("name", [
        "finals_week_vulnerable", "pandemic_isolation", "burnout_cascade",
    ])
    def test_combined_names(self, name):
        assert name in COMBINED_SCENARIOS


# ── Scenario application ────────────────────────────────────────────────────

class TestScenarioApplication:
    """Verify scenarios modify parameters correctly."""

    def test_set_param_works(self, default_params):
        """set_param: exam_week sets academic_load to 1.0."""
        s = ACADEMIC_SCENARIOS["exam_week"]
        modified = s.apply(default_params)
        assert modified["academic_load"] == 1.0
        # Other params should be unchanged
        for k in default_params:
            if k != "academic_load":
                assert modified[k] == default_params[k], \
                    f"{k} was unexpectedly modified"

    def test_scale_param_works(self, default_params):
        """scale_param: winter_darkness scales nature_rx by 0.3."""
        s = SEASONAL_SCENARIOS["winter_darkness"]
        modified = s.apply(default_params)
        expected = default_params["nature_rx"] * 0.3
        assert modified["nature_rx"] == pytest.approx(expected, abs=1e-10)
        # Other params unchanged
        for k in default_params:
            if k != "nature_rx":
                assert modified[k] == default_params[k], \
                    f"{k} was unexpectedly modified"

    def test_compose_modifies_multiple_params(self, default_params):
        """compose: academic_crisis sets both academic_load and sleep_hygiene."""
        s = ACADEMIC_SCENARIOS["academic_crisis"]
        modified = s.apply(default_params)
        assert modified["academic_load"] == 1.0
        assert modified["sleep_hygiene"] == 0.0

    def test_compose_three_params(self, default_params):
        """compose: full_isolation sets therapy_rx, nature_rx, exercise_rx."""
        s = SOCIAL_SCENARIOS["full_isolation"]
        modified = s.apply(default_params)
        assert modified["therapy_rx"] == 0.0
        assert modified["nature_rx"] == 0.0
        assert modified["exercise_rx"] == 0.0

    def test_compose_four_params(self, default_params):
        """compose: burnout_cascade modifies 4 params."""
        s = COMBINED_SCENARIOS["burnout_cascade"]
        modified = s.apply(default_params)
        assert modified["academic_load"] == 1.0
        assert modified["sleep_hygiene"] == 0.0
        assert modified["nature_rx"] == 0.0
        assert modified["emotional_stability"] == 2.0

    def test_apply_does_not_mutate_original(self, default_params):
        """Scenario.apply() returns a new dict, does not modify the original."""
        original_copy = dict(default_params)
        s = COMBINED_SCENARIOS["burnout_cascade"]
        _ = s.apply(default_params)
        assert default_params == original_copy

    def test_prior_anxiety_modifies_both_params(self, default_params):
        """prior_anxiety sets mh_diagnosis=1.0 and emotional_stability=3.0."""
        s = HEALTH_SCENARIOS["prior_anxiety"]
        modified = s.apply(default_params)
        assert modified["mh_diagnosis"] == 1.0
        assert modified["emotional_stability"] == 3.0


# ── Protocol bank ───────────────────────────────────────────────────────────

class TestProtocols:
    """Verify protocol definitions are valid."""

    def test_protocol_count(self):
        assert len(PROTOCOLS) == 5

    def test_protocol_names(self):
        expected = {"no_treatment", "nature_only", "exercise_only",
                    "therapy_only", "full_protocol"}
        assert set(PROTOCOLS.keys()) == expected

    def test_all_protocols_have_6_intervention_keys(self):
        for name, protocol in PROTOCOLS.items():
            for k in INTERVENTION_NAMES:
                assert k in protocol, f"Protocol {name} missing key {k}"

    def test_protocol_values_in_range(self):
        for name, protocol in PROTOCOLS.items():
            for k, v in protocol.items():
                assert 0.0 <= v <= 1.0, \
                    f"Protocol {name}: {k}={v} out of [0,1]"

    def test_no_treatment_has_zero_interventions(self):
        p = PROTOCOLS["no_treatment"]
        assert p["nature_rx"] == 0.0
        assert p["exercise_rx"] == 0.0
        assert p["therapy_rx"] == 0.0
        assert p["sleep_hygiene"] == 0.0
        assert p["caffeine_reduction"] == 0.0

    def test_full_protocol_has_all_nonzero(self):
        p = PROTOCOLS["full_protocol"]
        for k in INTERVENTION_NAMES:
            if k != "academic_load":
                assert p[k] > 0.0, \
                    f"full_protocol should have nonzero {k}"

    def test_backward_compat_reference_protocols(self):
        """REFERENCE_PROTOCOLS is an alias for PROTOCOLS."""
        assert REFERENCE_PROTOCOLS is PROTOCOLS


# ── Backward compatibility ──────────────────────────────────────────────────

class TestBackwardCompatibility:
    """Verify backward-compatible functions exist and work."""

    def test_apply_scenario_with_scenario_object(self, default_params):
        s = ACADEMIC_SCENARIOS["exam_week"]
        result = apply_scenario(default_params, s)
        assert result["academic_load"] == 1.0

    def test_apply_scenario_with_legacy_dict(self, default_params):
        """apply_scenario should accept old-style plain dicts."""
        legacy = {
            "name": "test_legacy",
            "modifications": [
                {"operation": "set", "param": "academic_load", "value": 0.9}
            ],
        }
        result = apply_scenario(default_params, legacy)
        assert result["academic_load"] == 0.9

    def test_run_scenario_sweep_exists(self):
        assert callable(run_scenario_sweep)

    def test_run_scenario_sweep_signature(self):
        """run_scenario_sweep should accept (sim, protocol, scenarios, output_key)."""
        import inspect
        sig = inspect.signature(run_scenario_sweep)
        params = list(sig.parameters.keys())
        assert "sim" in params
        assert "protocol" in params
        assert "scenarios" in params
        assert "output_key" in params


# ── New convenience functions ───────────────────────────────────────────────

class TestConvenienceFunctions:
    """Verify new cramer-toolkit convenience functions exist."""

    def test_run_resilience_analysis_exists(self):
        assert callable(run_resilience_analysis)

    def test_run_vulnerability_analysis_exists(self):
        assert callable(run_vulnerability_analysis)

    def test_run_scenario_comparison_exists(self):
        assert callable(run_scenario_comparison)

    def test_run_resilience_analysis_signature(self):
        import inspect
        sig = inspect.signature(run_resilience_analysis)
        params = list(sig.parameters.keys())
        assert "sim" in params
        assert "protocols" in params
        assert "scenarios" in params
        assert "output_key" in params
        assert "higher_is_better" in params

    def test_run_vulnerability_analysis_signature(self):
        import inspect
        sig = inspect.signature(run_vulnerability_analysis)
        params = list(sig.parameters.keys())
        assert "sim" in params
        assert "protocol" in params
        assert "scenarios" in params
        assert "output_key" in params
        assert "higher_is_better" in params

    def test_run_scenario_comparison_signature(self):
        import inspect
        sig = inspect.signature(run_scenario_comparison)
        params = list(sig.parameters.keys())
        assert "analysis_fn" in params
        assert "sim" in params
        assert "scenarios" in params
        assert "extract" in params
