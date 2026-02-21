"""Tests for the LEMURS college-student biopsychosocial ODE simulator.

Covers semester context, initial state, derivatives, full simulation,
biological sanity checks, and boundary conditions.
"""
from __future__ import annotations

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator import (
    _semester_context,
    initial_state,
    derivatives,
    _rk4_step,
    simulate,
)
from constants import (
    N_STATES, N_STEPS, DT,
    _TST, _SQ, _PSS, _GAD7, _DEP, _ACT, _NAT,
    _RHR, _HRV, _ARR, _SJL, _SHAPE, _WB, _DAC,
    _LOWER, _UPPER,
    DEFAULT_INTERVENTION, DEFAULT_PATIENT,
    STUDENT_ARCHETYPES,
    SPRING_BREAK_WEEK,
)


# ══════════════════════════════════════════════════════════════════════════════
# TestSemesterContext
# ══════════════════════════════════════════════════════════════════════════════

class TestSemesterContext:
    """Semester calendar logic: day-of-week, weekday, school-day, break."""

    def test_t_zero_is_day_zero_monday(self):
        ctx = _semester_context(0.0)
        assert ctx["day"] == 0
        assert ctx["day_of_week"] == 0  # Monday
        assert ctx["is_weekday"] is True
        assert ctx["is_school"] is True
        assert ctx["week"] == 0

    def test_t_one_is_day_seven_monday(self):
        ctx = _semester_context(1.0)
        assert ctx["day"] == 7
        assert ctx["day_of_week"] == 0  # Monday again

    def test_weekend_not_weekday(self):
        # Day 5 = Saturday (0-indexed: Mon=0, Sat=5)
        ctx = _semester_context(5.0 / 7.0)
        assert ctx["day"] == 5
        assert ctx["day_of_week"] == 5
        assert ctx["is_weekday"] is False

    def test_spring_break_not_school(self):
        # Spring break is week 8 (1-indexed), so 0-indexed week 7.
        # Day 52 = 7*7 + 3 = Wednesday of week 7 (spring break).
        spring_break_day = 7 * (SPRING_BREAK_WEEK - 1) + 3  # day 52
        t = spring_break_day / 7.0
        ctx = _semester_context(t)
        assert ctx["day"] == spring_break_day
        assert ctx["is_school"] is False

    def test_spring_break_monday_not_school(self):
        # Monday of spring break week (day 49).
        spring_break_monday = 7 * (SPRING_BREAK_WEEK - 1)
        t = spring_break_monday / 7.0
        ctx = _semester_context(t)
        assert ctx["is_weekday"] is True  # it IS a weekday
        assert ctx["is_school"] is False  # but NOT a school day

    def test_week_after_break_is_school(self):
        # Week 9 should be back to school.
        day = 7 * SPRING_BREAK_WEEK  # first day of week 9 (0-indexed week 8)
        t = day / 7.0
        ctx = _semester_context(t)
        assert ctx["is_school"] is True


# ══════════════════════════════════════════════════════════════════════════════
# TestInitialState
# ══════════════════════════════════════════════════════════════════════════════

class TestInitialState:
    """Initial state vector at semester start."""

    def test_shape(self):
        s = initial_state()
        assert s.shape == (N_STATES,)
        assert s.dtype == np.float64

    def test_shape_is_14(self):
        s = initial_state()
        assert len(s) == 14

    def test_tst_near_baseline(self):
        s = initial_state()
        assert 7.0 <= s[_TST] <= 7.5, f"TST={s[_TST]}"

    def test_pss_near_baseline(self):
        s = initial_state()
        assert abs(s[_PSS] - 16.84) < 0.01

    def test_gad7_reasonable(self):
        s = initial_state()
        assert 0.0 <= s[_GAD7] <= 21.0

    def test_hrv_reasonable(self):
        s = initial_state()
        assert 15.0 <= s[_HRV] <= 120.0

    def test_rhr_reasonable(self):
        s = initial_state()
        assert 45.0 <= s[_RHR] <= 100.0

    def test_dac_at_baseline(self):
        s = initial_state()
        assert abs(s[_DAC] - 0.8) < 0.01

    def test_mh_diagnosis_elevates_gad7(self):
        s_healthy = initial_state({"mh_diagnosis": 0.0})
        s_mh = initial_state({"mh_diagnosis": 1.0})
        assert s_mh[_GAD7] > s_healthy[_GAD7]

    def test_mh_diagnosis_elevates_depression(self):
        s_healthy = initial_state({"mh_diagnosis": 0.0})
        s_mh = initial_state({"mh_diagnosis": 1.0})
        assert s_mh[_DEP] > s_healthy[_DEP]

    def test_mh_diagnosis_raises_rhr(self):
        s_healthy = initial_state({"mh_diagnosis": 0.0})
        s_mh = initial_state({"mh_diagnosis": 1.0})
        assert s_mh[_RHR] > s_healthy[_RHR]

    def test_mh_diagnosis_lowers_hrv(self):
        s_healthy = initial_state({"mh_diagnosis": 0.0})
        s_mh = initial_state({"mh_diagnosis": 1.0})
        assert s_mh[_HRV] < s_healthy[_HRV]

    def test_male_higher_activity(self):
        s_female = initial_state({"gender": 1.0})
        s_male = initial_state({"gender": 0.0})
        assert s_male[_ACT] > s_female[_ACT]

    def test_late_chronotype_more_sjl(self):
        s_early = initial_state({"baseline_chronotype": 3.0})
        s_late = initial_state({"baseline_chronotype": 6.0})
        assert s_late[_SJL] > s_early[_SJL]

    def test_all_within_bounds(self):
        s = initial_state()
        for i in range(N_STATES):
            assert s[i] >= _LOWER[i] - 1e-9, f"State {i} below lower bound"
            assert s[i] <= _UPPER[i] + 1e-9, f"State {i} above upper bound"


# ══════════════════════════════════════════════════════════════════════════════
# TestDerivatives
# ══════════════════════════════════════════════════════════════════════════════

class TestDerivatives:
    """Derivative computation -- the biological coupling core."""

    def _default_state(self):
        return initial_state()

    def test_shape_and_no_nan(self):
        s = self._default_state()
        d = derivatives(s, 0.0, DEFAULT_INTERVENTION, DEFAULT_PATIENT)
        assert d.shape == (N_STATES,)
        assert not np.any(np.isnan(d)), "NaN in derivatives"

    def test_pss_increases_when_sleep_deprived(self):
        """When TST is very low (5.0 hrs), PSS should increase."""
        s = self._default_state()
        s[_TST] = 5.0  # severe sleep deprivation
        d = derivatives(s, 0.0, DEFAULT_INTERVENTION, DEFAULT_PATIENT)
        # PSS derivative should be positive (stress increasing)
        # because BETA_TST_PSS is negative and (TST - TST_REFERENCE) is negative
        # so BETA_TST_PSS * (5.0 - 7.0) = -0.877 * (-2.0) = +1.754 > 0
        assert d[_PSS] > 0, f"Expected positive PSS derivative with low TST, got {d[_PSS]}"

    def test_hrv_increases_with_nature(self):
        """Nature intervention should push HRV upward."""
        s = self._default_state()
        s[_NAT] = 4.0  # substantial nature engagement
        s[_DAC] = 0.8  # good attention capacity for engagement quality
        intv_nature = {**DEFAULT_INTERVENTION, "nature_rx": 0.8}
        d = derivatives(s, 2.0, intv_nature, DEFAULT_PATIENT)
        # HRV derivative should be positive (HRV increasing)
        # due to NATURE_HRV_DIRECT contribution
        assert d[_HRV] > 0, f"Expected positive HRV derivative with nature, got {d[_HRV]}"

    def test_gad7_development_below_threshold(self):
        """Below GAD7 threshold, with low stability, development rate should be positive."""
        s = self._default_state()
        s[_GAD7] = 5.0  # below threshold of 10
        patient = {**DEFAULT_PATIENT, "emotional_stability": 2.0}
        d = derivatives(s, 0.0, DEFAULT_INTERVENTION, patient)
        # GAD7 should be pulled toward the anxious attractor (15.0)
        assert d[_GAD7] > 0, f"Expected positive GAD7 derivative below threshold, got {d[_GAD7]}"

    def test_gad7_recovery_above_threshold(self):
        """Above GAD7 threshold, recovery should pull score downward."""
        s = self._default_state()
        s[_GAD7] = 15.0  # above threshold of 10
        patient = {**DEFAULT_PATIENT, "emotional_stability": 6.0}
        intv = {**DEFAULT_INTERVENTION, "academic_load": 0.3}  # low stressor
        d = derivatives(s, 0.0, intv, patient)
        # GAD7 should be pulled toward recovery attractor (5.0)
        assert d[_GAD7] < 0, f"Expected negative GAD7 derivative above threshold, got {d[_GAD7]}"

    def test_sleep_debt_on_school_day(self):
        """On a school weekday, TST should be pushed downward."""
        s = self._default_state()
        s[_TST] = 7.5  # at baseline
        # t=0 is Monday week 0 (school day)
        d = derivatives(s, 0.0, DEFAULT_INTERVENTION, DEFAULT_PATIENT)
        # The sleep debt term should make d[_TST] negative or at least
        # lower than on a non-school day
        d_school = d[_TST]

        # Now compute on a weekend (t = 5/7 = Saturday)
        d_weekend = derivatives(s, 5.0 / 7.0, DEFAULT_INTERVENTION, DEFAULT_PATIENT)
        assert d_school < d_weekend[_TST], "School day should have lower TST derivative than weekend"

    def test_dac_depleted_by_academic_load(self):
        """High academic load should deplete DAC."""
        s = self._default_state()
        s[_DAC] = 0.8
        intv_high = {**DEFAULT_INTERVENTION, "academic_load": 1.0, "nature_rx": 0.0}
        d = derivatives(s, 0.0, intv_high, DEFAULT_PATIENT)
        # DAC depletion from high academic load should dominate
        assert d[_DAC] < 0, f"Expected negative DAC derivative with high load, got {d[_DAC]}"

    def test_no_nan_at_extremes(self):
        """Derivatives should not produce NaN at extreme state values."""
        s = _LOWER.copy()
        d = derivatives(s, 0.0, DEFAULT_INTERVENTION, DEFAULT_PATIENT)
        assert not np.any(np.isnan(d)), "NaN at lower bounds"

        s = _UPPER.copy()
        d = derivatives(s, 7.0, DEFAULT_INTERVENTION, DEFAULT_PATIENT)
        assert not np.any(np.isnan(d)), "NaN at upper bounds"


# ══════════════════════════════════════════════════════════════════════════════
# TestSimulate
# ══════════════════════════════════════════════════════════════════════════════

class TestSimulate:
    """Full simulation integration."""

    def test_returns_expected_keys(self):
        result = simulate()
        assert "states" in result
        assert "times" in result
        assert "intervention" in result
        assert "patient" in result

    def test_states_shape(self):
        result = simulate()
        assert result["states"].shape == (N_STEPS + 1, N_STATES)

    def test_times_shape(self):
        result = simulate()
        assert result["times"].shape == (N_STEPS + 1,)

    def test_deterministic(self):
        """Two runs with identical inputs produce identical outputs."""
        r1 = simulate()
        r2 = simulate()
        np.testing.assert_array_equal(r1["states"], r2["states"])
        np.testing.assert_array_equal(r1["times"], r2["times"])

    def test_no_nan_in_trajectory(self):
        result = simulate()
        assert not np.any(np.isnan(result["states"])), "NaN in trajectory"
        assert not np.any(np.isnan(result["times"])), "NaN in times"

    def test_no_inf_in_trajectory(self):
        result = simulate()
        assert not np.any(np.isinf(result["states"])), "Inf in trajectory"

    def test_all_states_within_bounds(self):
        """Every state at every timestep must be within [_LOWER, _UPPER]."""
        result = simulate()
        eps = 1e-6
        for i in range(N_STATES):
            col = result["states"][:, i]
            assert np.all(col >= _LOWER[i] - eps), (
                f"State {i} below lower bound: min={col.min()}, bound={_LOWER[i]}"
            )
            assert np.all(col <= _UPPER[i] + eps), (
                f"State {i} above upper bound: max={col.max()}, bound={_UPPER[i]}"
            )

    def test_custom_sim_weeks(self):
        result = simulate(sim_weeks=5)
        expected_steps = 5 * 7
        assert result["states"].shape == (expected_steps + 1, N_STATES)

    def test_times_monotonically_increasing(self):
        result = simulate()
        diffs = np.diff(result["times"])
        assert np.all(diffs > 0), "Times not monotonically increasing"


# ══════════════════════════════════════════════════════════════════════════════
# TestBiologicalSanity
# ══════════════════════════════════════════════════════════════════════════════

class TestBiologicalSanity:
    """High-level biological plausibility checks."""

    def test_resilient_student_pss_stays_moderate(self):
        """Resilient male student: PSS should stay below 25 over semester."""
        result = simulate(
            patient={"gender": 0.0, "emotional_stability": 6.0,
                     "trauma_load": 0.0, "mh_diagnosis": 0.0,
                     "baseline_chronotype": 4.0},
        )
        pss_max = np.max(result["states"][:, _PSS])
        assert pss_max < 25.0, f"Resilient student PSS too high: {pss_max}"

    def test_nature_intervention_reduces_mean_pss(self):
        """Nature intervention should produce lower mean PSS than no intervention."""
        r_none = simulate(intervention={"nature_rx": 0.0})
        r_nature = simulate(intervention={"nature_rx": 0.8})
        pss_none = np.mean(r_none["states"][:, _PSS])
        pss_nature = np.mean(r_nature["states"][:, _PSS])
        assert pss_nature < pss_none, (
            f"Nature PSS ({pss_nature:.2f}) not lower than no-nature ({pss_none:.2f})"
        )

    def test_vulnerable_female_higher_gad7_than_resilient_male(self):
        """Vulnerable female should have higher mean GAD7 than resilient male."""
        r_vuln = simulate(
            patient={"gender": 1.0, "emotional_stability": 3.0,
                     "trauma_load": 3.0, "mh_diagnosis": 1.0,
                     "baseline_chronotype": 5.5},
        )
        r_resilient = simulate(
            patient={"gender": 0.0, "emotional_stability": 6.0,
                     "trauma_load": 0.0, "mh_diagnosis": 0.0,
                     "baseline_chronotype": 4.0},
        )
        gad7_vuln = np.mean(r_vuln["states"][:, _GAD7])
        gad7_resilient = np.mean(r_resilient["states"][:, _GAD7])
        assert gad7_vuln > gad7_resilient, (
            f"Vulnerable GAD7 ({gad7_vuln:.2f}) not higher than resilient ({gad7_resilient:.2f})"
        )

    def test_spring_break_tst_increases(self):
        """During spring break, TST should be higher than the preceding school week."""
        result = simulate()
        states = result["states"]

        # Spring break is week 8 (1-indexed), so 0-indexed week 7.
        # Days 49-55 are spring break.
        break_start = 7 * (SPRING_BREAK_WEEK - 1)  # day 49
        break_end = break_start + 7                  # day 56

        # School week before break: days 42-48
        school_week_before = states[42:49, _TST]
        break_week = states[break_start:break_end, _TST]

        assert np.mean(break_week) > np.mean(school_week_before), (
            f"Break TST ({np.mean(break_week):.3f}) not higher than "
            f"school week ({np.mean(school_week_before):.3f})"
        )

    def test_therapy_reduces_arr(self):
        """Therapy intervention should reduce ARR over the semester."""
        r_no_therapy = simulate(intervention={"therapy_rx": 0.0})
        r_therapy = simulate(intervention={"therapy_rx": 0.8})
        arr_no = np.mean(r_no_therapy["states"][-14:, _ARR])  # last 2 weeks
        arr_therapy = np.mean(r_therapy["states"][-14:, _ARR])
        assert arr_therapy < arr_no, (
            f"Therapy ARR ({arr_therapy:.3f}) not lower than no-therapy ({arr_no:.3f})"
        )

    def test_exercise_boosts_wellbeing(self):
        """Exercise intervention should produce higher mean well-being."""
        r_sedentary = simulate(intervention={"exercise_rx": 0.0})
        r_active = simulate(intervention={"exercise_rx": 0.8})
        wb_sedentary = np.mean(r_sedentary["states"][:, _WB])
        wb_active = np.mean(r_active["states"][:, _WB])
        assert wb_active > wb_sedentary, (
            f"Active WEMWBS ({wb_active:.2f}) not higher than sedentary ({wb_sedentary:.2f})"
        )

    def test_all_archetypes_run_without_crash(self):
        """All 8 student archetypes must complete without error."""
        for arch in STUDENT_ARCHETYPES:
            intv = arch.get("intervention", {})
            pat = arch.get("patient", {})
            result = simulate(intervention=intv, patient=pat)
            assert not np.any(np.isnan(result["states"])), (
                f"NaN in {arch['name']} trajectory"
            )


# ══════════════════════════════════════════════════════════════════════════════
# TestBoundaryConditions
# ══════════════════════════════════════════════════════════════════════════════

class TestBoundaryConditions:
    """Extreme inputs must not crash the simulator."""

    def test_max_stress_no_crash(self):
        """Max academic load + max trauma + low stability: no crash."""
        result = simulate(
            intervention={"academic_load": 1.0, "nature_rx": 0.0,
                          "exercise_rx": 0.0, "therapy_rx": 0.0,
                          "sleep_hygiene": 0.0, "caffeine_reduction": 0.0},
            patient={"emotional_stability": 1.0, "trauma_load": 5.0,
                     "mh_diagnosis": 1.0, "baseline_chronotype": 7.0,
                     "gender": 1.0},
        )
        assert not np.any(np.isnan(result["states"]))
        assert not np.any(np.isinf(result["states"]))

    def test_zero_everything_no_crash(self):
        """All intervention and patient params at zero: no crash."""
        result = simulate(
            intervention={"academic_load": 0.0, "nature_rx": 0.0,
                          "exercise_rx": 0.0, "therapy_rx": 0.0,
                          "sleep_hygiene": 0.0, "caffeine_reduction": 0.0},
            patient={"emotional_stability": 1.0, "trauma_load": 0.0,
                     "mh_diagnosis": 0.0, "baseline_chronotype": 2.0,
                     "gender": 0.0},
        )
        assert not np.any(np.isnan(result["states"]))
        assert not np.any(np.isinf(result["states"]))

    def test_max_everything_no_crash(self):
        """All intervention params at max: no crash."""
        result = simulate(
            intervention={"academic_load": 1.0, "nature_rx": 1.0,
                          "exercise_rx": 1.0, "therapy_rx": 1.0,
                          "sleep_hygiene": 1.0, "caffeine_reduction": 1.0},
            patient={"emotional_stability": 7.0, "trauma_load": 5.0,
                     "mh_diagnosis": 1.0, "baseline_chronotype": 7.0,
                     "gender": 2.0},
        )
        assert not np.any(np.isnan(result["states"]))
        assert not np.any(np.isinf(result["states"]))

    def test_all_states_bounded_under_extreme_stress(self):
        """Under maximum stress, all states still stay within bounds."""
        result = simulate(
            intervention={"academic_load": 1.0, "nature_rx": 0.0},
            patient={"emotional_stability": 1.0, "trauma_load": 5.0,
                     "mh_diagnosis": 1.0},
        )
        eps = 1e-6
        for i in range(N_STATES):
            col = result["states"][:, i]
            assert np.all(col >= _LOWER[i] - eps), (
                f"State {i} below lower bound under stress: min={col.min()}"
            )
            assert np.all(col <= _UPPER[i] + eps), (
                f"State {i} above upper bound under stress: max={col.max()}"
            )

    def test_nonbinary_gender_no_crash(self):
        """Nonbinary gender (2.0) must not crash or produce NaN."""
        result = simulate(
            patient={"gender": 2.0, "emotional_stability": 4.0,
                     "trauma_load": 2.0},
        )
        assert not np.any(np.isnan(result["states"]))
