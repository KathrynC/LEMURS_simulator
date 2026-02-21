"""Tests for Zimmerman protocol compatibility of the LEMURS simulator.

Verifies that LEMURSSimulator conforms to the Zimmerman protocol -- the
universal plug that makes any simulator interchangeable with the analysis
tools in zimmerman-toolkit and cramer-toolkit.

The protocol requires two things:
    1. param_spec() returning {name: (lo, hi)} for all input parameters
    2. run(params) accepting a flat dict and returning a flat dict of
       scalar numeric outputs

These tests ensure the adapter works correctly without needing to know
the internal structure of the LEMURS ODE model.
"""
from __future__ import annotations

import math
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lemurs_simulator import LEMURSSimulator
from constants import (
    INTERVENTION_NAMES, INTERVENTION_BOUNDS,
    PATIENT_NAMES, PATIENT_BOUNDS,
)


# ══════════════════════════════════════════════════════════════════════════════
# TestProtocol
# ══════════════════════════════════════════════════════════════════════════════

class TestProtocol:
    """LEMURSSimulator has the two required Zimmerman protocol methods."""

    def test_has_param_spec(self):
        sim = LEMURSSimulator()
        assert hasattr(sim, "param_spec")

    def test_has_run(self):
        sim = LEMURSSimulator()
        assert hasattr(sim, "run")

    def test_both_callable(self):
        sim = LEMURSSimulator()
        assert callable(sim.param_spec)
        assert callable(sim.run)


# ══════════════════════════════════════════════════════════════════════════════
# TestParamSpec
# ══════════════════════════════════════════════════════════════════════════════

class TestParamSpec:
    """param_spec() returns a well-formed 12D parameter space."""

    def test_returns_dict(self):
        sim = LEMURSSimulator()
        spec = sim.param_spec()
        assert isinstance(spec, dict)

    def test_has_12_entries(self):
        """6 intervention + 6 patient = 12 total parameters."""
        sim = LEMURSSimulator()
        spec = sim.param_spec()
        assert len(spec) == 12, f"Expected 12 params, got {len(spec)}: {list(spec.keys())}"

    def test_all_values_are_tuples_of_length_2(self):
        sim = LEMURSSimulator()
        spec = sim.param_spec()
        for name, bounds in spec.items():
            assert isinstance(bounds, tuple), f"{name} bounds is {type(bounds)}, expected tuple"
            assert len(bounds) == 2, f"{name} has {len(bounds)} bounds, expected 2"

    def test_all_lower_less_than_upper(self):
        sim = LEMURSSimulator()
        spec = sim.param_spec()
        for name, (lo, hi) in spec.items():
            assert lo < hi, f"{name}: lower ({lo}) >= upper ({hi})"

    def test_contains_expected_names(self):
        """Spot-check that key parameter names are present."""
        sim = LEMURSSimulator()
        spec = sim.param_spec()
        expected = [
            "nature_rx", "exercise_rx", "therapy_rx",
            "sleep_hygiene", "caffeine_reduction", "academic_load",
            "age", "gender", "emotional_stability",
            "trauma_load", "mh_diagnosis", "baseline_chronotype",
        ]
        for name in expected:
            assert name in spec, f"Missing expected param: {name}"


# ══════════════════════════════════════════════════════════════════════════════
# TestRunOutput
# ══════════════════════════════════════════════════════════════════════════════

class TestRunOutput:
    """run() returns a well-formed flat dict of scalar metrics."""

    @pytest.fixture(scope="class")
    def default_result(self):
        """Cache a single default run for all tests in this class."""
        sim = LEMURSSimulator()
        return sim.run({})

    def test_returns_dict(self, default_result):
        assert isinstance(default_result, dict)

    def test_all_values_are_floats(self, default_result):
        for k, v in default_result.items():
            assert isinstance(v, float), f"{k} is {type(v)}, expected float"

    def test_no_nan_values(self, default_result):
        for k, v in default_result.items():
            assert not math.isnan(v), f"{k} is NaN"

    def test_no_inf_values(self, default_result):
        for k, v in default_result.items():
            assert not math.isinf(v), f"{k} is inf"

    def test_keys_follow_pillar_metric_pattern(self, default_result):
        """All keys should contain at least one underscore (pillar_metric)."""
        for key in default_result:
            assert "_" in key, f"Key '{key}' does not follow pillar_metric pattern"

    def test_has_at_least_20_metrics(self, default_result):
        """The 4 pillars should produce at least 20 total metrics."""
        assert len(default_result) >= 20, (
            f"Expected at least 20 metrics, got {len(default_result)}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TestDeterminism
# ══════════════════════════════════════════════════════════════════════════════

class TestDeterminism:
    """Identical inputs must produce identical outputs."""

    def test_same_params_same_output(self):
        """Explicit params produce identical results across two runs."""
        sim = LEMURSSimulator()
        params = {"nature_rx": 0.5, "age": 20.0, "emotional_stability": 5.0}
        r1 = sim.run(params)
        r2 = sim.run(params)
        assert r1 == r2, "Two runs with same params produced different results"

    def test_default_params_same_output(self):
        """Empty dict (all defaults) produces identical results across two runs."""
        sim = LEMURSSimulator()
        r1 = sim.run({})
        r2 = sim.run({})
        assert r1 == r2, "Two default runs produced different results"


# ══════════════════════════════════════════════════════════════════════════════
# TestEdgeCases
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Extreme and partial inputs must not crash the adapter."""

    def test_empty_dict_no_crash(self):
        """Empty dict uses all defaults -- must not crash."""
        sim = LEMURSSimulator()
        result = sim.run({})
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_max_intervention_params_no_crash(self):
        """All intervention params at maximum -- must not crash."""
        sim = LEMURSSimulator()
        params = {name: hi for name, (_, hi) in INTERVENTION_BOUNDS.items()}
        result = sim.run(params)
        assert isinstance(result, dict)
        for k, v in result.items():
            assert not math.isnan(v), f"{k} is NaN with max interventions"
            assert not math.isinf(v), f"{k} is inf with max interventions"

    def test_max_patient_params_no_crash(self):
        """All patient params at maximum -- must not crash."""
        sim = LEMURSSimulator()
        params = {name: hi for name, (_, hi) in PATIENT_BOUNDS.items()}
        result = sim.run(params)
        assert isinstance(result, dict)
        for k, v in result.items():
            assert not math.isnan(v), f"{k} is NaN with max patient params"
            assert not math.isinf(v), f"{k} is inf with max patient params"

    def test_partial_params_no_crash(self):
        """Only some params provided -- rest should use defaults."""
        sim = LEMURSSimulator()
        result = sim.run({"nature_rx": 0.6, "age": 21.0})
        assert isinstance(result, dict)
        assert len(result) > 0
        for k, v in result.items():
            assert not math.isnan(v), f"{k} is NaN with partial params"
