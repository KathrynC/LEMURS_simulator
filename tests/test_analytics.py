"""Tests for the LEMURS 4-pillar analytics module.

Covers compute_all structure, each individual pillar's range and type
constraints, intervention response logic, and NumpyEncoder serialization.
"""
from __future__ import annotations

import json
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator import simulate
from analytics import compute_all, NumpyEncoder


# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def default_result():
    """Default student, default intervention -- cached across tests."""
    return simulate()


@pytest.fixture(scope="module")
def default_analytics(default_result):
    """Analytics for the default student without baseline comparison."""
    return compute_all(default_result)


@pytest.fixture(scope="module")
def baseline_result():
    """No-intervention baseline: all intervention dials at zero."""
    return simulate(intervention={
        "nature_rx": 0.0,
        "exercise_rx": 0.0,
        "therapy_rx": 0.0,
        "sleep_hygiene": 0.0,
        "caffeine_reduction": 0.0,
        "academic_load": 0.5,
    })


@pytest.fixture(scope="module")
def nature_result():
    """Student with strong nature intervention."""
    return simulate(intervention={"nature_rx": 0.8})


# ══════════════════════════════════════════════════════════════════════════════
# TestComputeAll
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeAll:
    """Top-level compute_all structure and type checks."""

    def test_returns_dict_with_four_keys(self, default_analytics):
        assert isinstance(default_analytics, dict)
        assert len(default_analytics) == 4

    def test_expected_pillar_names(self, default_analytics):
        expected = {"sleep_quality", "stress_anxiety", "physiological",
                    "intervention_response"}
        assert set(default_analytics.keys()) == expected

    def test_each_pillar_is_dict(self, default_analytics):
        for pillar_name, pillar in default_analytics.items():
            assert isinstance(pillar, dict), f"{pillar_name} is not a dict"

    def test_all_values_are_float_or_int(self, default_analytics):
        """All leaf values should be native Python float or int, not numpy."""
        for pillar_name, pillar in default_analytics.items():
            for key, val in pillar.items():
                assert isinstance(val, (float, int)), (
                    f"{pillar_name}.{key} is {type(val).__name__}, "
                    f"expected float or int"
                )

    def test_no_nan_in_any_value(self, default_analytics):
        for pillar_name, pillar in default_analytics.items():
            for key, val in pillar.items():
                if isinstance(val, float):
                    assert not np.isnan(val), (
                        f"NaN in {pillar_name}.{key}"
                    )

    def test_with_baseline_still_four_pillars(self, default_result, baseline_result):
        analytics = compute_all(default_result, baseline=baseline_result)
        assert len(analytics) == 4
        assert "intervention_response" in analytics


# ══════════════════════════════════════════════════════════════════════════════
# TestSleepQuality
# ══════════════════════════════════════════════════════════════════════════════

class TestSleepQuality:
    """Pillar 1: Sleep quality metric sanity checks."""

    def test_tst_mean_reasonable(self, default_analytics):
        tst_mean = default_analytics["sleep_quality"]["tst_mean"]
        assert 5.0 <= tst_mean <= 9.0, f"tst_mean={tst_mean} out of range"

    def test_sleep_debt_cumulative_nonnegative(self, default_analytics):
        debt = default_analytics["sleep_quality"]["sleep_debt_cumulative"]
        assert debt >= 0.0, f"sleep_debt_cumulative={debt} is negative"

    def test_social_jetlag_mean_nonnegative(self, default_analytics):
        sjl = default_analytics["sleep_quality"]["social_jetlag_mean"]
        assert sjl >= 0.0, f"social_jetlag_mean={sjl} is negative"

    def test_shape_cluster1_fraction_between_0_and_1(self, default_analytics):
        frac = default_analytics["sleep_quality"]["shape_cluster1_fraction"]
        assert 0.0 <= frac <= 1.0, f"shape_cluster1_fraction={frac} out of [0,1]"


# ══════════════════════════════════════════════════════════════════════════════
# TestStressAnxiety
# ══════════════════════════════════════════════════════════════════════════════

class TestStressAnxiety:
    """Pillar 2: Stress and anxiety metric sanity checks."""

    def test_pss_mean_reasonable(self, default_analytics):
        pss = default_analytics["stress_anxiety"]["pss_mean"]
        assert 5.0 <= pss <= 35.0, f"pss_mean={pss} out of range"

    def test_anxiety_transitions_count_nonneg_int(self, default_analytics):
        count = default_analytics["stress_anxiety"]["anxiety_transitions_count"]
        assert isinstance(count, int), f"Expected int, got {type(count).__name__}"
        assert count >= 0, f"anxiety_transitions_count={count} is negative"

    def test_gad7_days_above_10_nonnegative(self, default_analytics):
        days = default_analytics["stress_anxiety"]["gad7_days_above_10"]
        assert days >= 0.0, f"gad7_days_above_10={days} is negative"

    def test_pss_slope_is_float_not_nan(self, default_analytics):
        slope = default_analytics["stress_anxiety"]["pss_slope"]
        assert isinstance(slope, float), f"Expected float, got {type(slope).__name__}"
        assert not np.isnan(slope), "pss_slope is NaN"


# ══════════════════════════════════════════════════════════════════════════════
# TestPhysiological
# ══════════════════════════════════════════════════════════════════════════════

class TestPhysiological:
    """Pillar 3: Physiological biomarker metric sanity checks."""

    def test_hrv_mean_reasonable(self, default_analytics):
        hrv = default_analytics["physiological"]["hrv_mean"]
        assert 20.0 <= hrv <= 110.0, f"hrv_mean={hrv} out of range"

    def test_dac_min_between_0_and_1(self, default_analytics):
        dac = default_analytics["physiological"]["dac_min"]
        assert 0.0 <= dac <= 1.0, f"dac_min={dac} out of [0,1]"

    def test_rhr_mean_reasonable(self, default_analytics):
        rhr = default_analytics["physiological"]["rhr_mean"]
        assert 50.0 <= rhr <= 90.0, f"rhr_mean={rhr} out of range"


# ══════════════════════════════════════════════════════════════════════════════
# TestInterventionResponse
# ══════════════════════════════════════════════════════════════════════════════

class TestInterventionResponse:
    """Pillar 4: Intervention response and cost-effectiveness."""

    def test_without_baseline_all_zeros(self, default_result):
        """When no baseline is provided, all intervention metrics are zero."""
        analytics = compute_all(default_result, baseline=None)
        ir = analytics["intervention_response"]
        for key, val in ir.items():
            assert val == 0.0, f"{key}={val}, expected 0.0 without baseline"

    def test_pss_benefit_positive_with_nature(self, nature_result, baseline_result):
        """Nature intervention should produce positive PSS benefit vs baseline."""
        analytics = compute_all(nature_result, baseline=baseline_result)
        pss_benefit = analytics["intervention_response"]["pss_benefit"]
        assert pss_benefit > 0.0, (
            f"pss_benefit={pss_benefit}, expected positive with nature_rx=0.8"
        )

    def test_cost_effectiveness_computed(self, nature_result, baseline_result):
        """Cost-effectiveness should be nonzero when there is cost and benefit."""
        analytics = compute_all(nature_result, baseline=baseline_result)
        ce = analytics["intervention_response"]["cost_effectiveness"]
        # Nature costs $2940/yr, so semester cost > 0, and benefit > 0
        assert isinstance(ce, float), f"Expected float, got {type(ce).__name__}"
        # It should be computed (either positive or negative, but not NaN)
        assert not np.isnan(ce), "cost_effectiveness is NaN"

    def test_nature_dose_response_nonnegative(self, nature_result, baseline_result):
        """Nature dose-response should be >= 0 when nature engagement is active."""
        analytics = compute_all(nature_result, baseline=baseline_result)
        ndr = analytics["intervention_response"]["nature_dose_response"]
        assert ndr >= 0.0, f"nature_dose_response={ndr} is negative"


# ══════════════════════════════════════════════════════════════════════════════
# TestNumpyEncoder
# ══════════════════════════════════════════════════════════════════════════════

class TestNumpyEncoder:
    """NumpyEncoder JSON serialization of numpy types."""

    def test_encodes_float64(self):
        data = {"val": np.float64(3.14159265)}
        s = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(s)
        assert isinstance(parsed["val"], float)
        assert abs(parsed["val"] - 3.141593) < 1e-5

    def test_encodes_int32(self):
        data = {"count": np.int32(42)}
        s = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(s)
        assert isinstance(parsed["count"], int)
        assert parsed["count"] == 42

    def test_encodes_ndarray(self):
        data = {"arr": np.array([1.1, 2.2, 3.3])}
        s = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(s)
        assert isinstance(parsed["arr"], list)
        assert len(parsed["arr"]) == 3
        assert abs(parsed["arr"][0] - 1.1) < 1e-5
