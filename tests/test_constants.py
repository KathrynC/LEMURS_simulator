"""Tests for LEMURS simulator constants module.

Validates that the configuration is internally consistent: dimensions match,
bounds are well-ordered, defaults are within bounds, grids are sorted, student
archetypes are properly defined, snap functions work, and the semester calendar
produces correct day/week classifications.
"""
from __future__ import annotations

import numpy as np
import pytest

from constants import (
    # Simulation config
    SIM_WEEKS, DT, N_STEPS,
    # State variables
    N_STATES, STATE_NAMES, _LOWER, _UPPER,
    _TST, _SQ, _PSS, _GAD7, _DEP, _ACT, _NAT,
    _RHR, _HRV, _ARR, _SJL, _SHAPE, _WB, _DAC,
    # Intervention params
    INTERVENTION_NAMES, INTERVENTION_BOUNDS, INTERVENTION_GRIDS,
    DEFAULT_INTERVENTION,
    # Patient params
    PATIENT_NAMES, PATIENT_BOUNDS, PATIENT_GRIDS, DEFAULT_PATIENT,
    # Student archetypes
    STUDENT_ARCHETYPES,
    # Snap functions
    snap_param, snap_all, _ALL_GRIDS,
    # Semester calendar
    SPRING_BREAK_WEEK, day_of_week, is_weekday, is_school_day, week_of_semester,
)


class TestSimConfig:
    """Test simulation configuration constants."""

    def test_sim_weeks(self):
        assert SIM_WEEKS == 15

    def test_dt_is_one_day_in_weeks(self):
        assert abs(DT - 1.0 / 7.0) < 1e-12

    def test_n_steps(self):
        assert N_STEPS == 105
        assert N_STEPS == SIM_WEEKS * 7


class TestStateConfig:
    """Test state variable configuration."""

    def test_n_states_is_14(self):
        assert N_STATES == 14

    def test_state_names_length(self):
        assert len(STATE_NAMES) == N_STATES

    def test_state_names_are_unique(self):
        assert len(set(STATE_NAMES)) == len(STATE_NAMES)

    def test_lower_shape(self):
        assert _LOWER.shape == (N_STATES,)

    def test_upper_shape(self):
        assert _UPPER.shape == (N_STATES,)

    def test_lower_less_than_upper(self):
        """Every lower bound must be strictly less than the corresponding upper bound."""
        for i in range(N_STATES):
            assert _LOWER[i] < _UPPER[i], (
                f"State {STATE_NAMES[i]}: lower={_LOWER[i]} >= upper={_UPPER[i]}"
            )

    def test_state_indices_match_names(self):
        """State index constants should correspond to STATE_NAMES positions."""
        assert STATE_NAMES[_TST] == "TST"
        assert STATE_NAMES[_SQ] == "SleepQuality"
        assert STATE_NAMES[_PSS] == "PSS"
        assert STATE_NAMES[_GAD7] == "GAD7"
        assert STATE_NAMES[_DEP] == "Depression"
        assert STATE_NAMES[_ACT] == "Activity"
        assert STATE_NAMES[_NAT] == "NatureEngagement"
        assert STATE_NAMES[_RHR] == "RHR"
        assert STATE_NAMES[_HRV] == "HRV"
        assert STATE_NAMES[_ARR] == "ARR"
        assert STATE_NAMES[_SJL] == "SocialJetlag"
        assert STATE_NAMES[_SHAPE] == "SleepShape"
        assert STATE_NAMES[_WB] == "WEMWBS"
        assert STATE_NAMES[_DAC] == "DAC"

    def test_index_constants_are_sequential(self):
        indices = [_TST, _SQ, _PSS, _GAD7, _DEP, _ACT, _NAT,
                   _RHR, _HRV, _ARR, _SJL, _SHAPE, _WB, _DAC]
        assert indices == list(range(14))


class TestInterventionParams:
    """Test intervention parameter configuration."""

    def test_count(self):
        assert len(INTERVENTION_NAMES) == 6

    def test_names_are_unique(self):
        assert len(set(INTERVENTION_NAMES)) == len(INTERVENTION_NAMES)

    def test_all_bounds_are_zero_to_one(self):
        for name in INTERVENTION_NAMES:
            lo, hi = INTERVENTION_BOUNDS[name]
            assert lo == 0.0, f"{name} lower bound is {lo}, expected 0.0"
            assert hi == 1.0, f"{name} upper bound is {hi}, expected 1.0"

    def test_bounds_keys_match_names(self):
        assert set(INTERVENTION_BOUNDS.keys()) == set(INTERVENTION_NAMES)

    def test_grids_keys_match_names(self):
        assert set(INTERVENTION_GRIDS.keys()) == set(INTERVENTION_NAMES)

    def test_grids_are_sorted(self):
        for name, grid in INTERVENTION_GRIDS.items():
            assert grid == sorted(grid), f"{name} grid is not sorted: {grid}"

    def test_grids_within_bounds(self):
        for name, grid in INTERVENTION_GRIDS.items():
            lo, hi = INTERVENTION_BOUNDS[name]
            for v in grid:
                assert lo <= v <= hi, f"{name} grid value {v} out of bounds [{lo}, {hi}]"

    def test_defaults_within_bounds(self):
        for name in INTERVENTION_NAMES:
            lo, hi = INTERVENTION_BOUNDS[name]
            v = DEFAULT_INTERVENTION[name]
            assert lo <= v <= hi, (
                f"Default {name}={v} out of bounds [{lo}, {hi}]"
            )

    def test_defaults_keys_match_names(self):
        assert set(DEFAULT_INTERVENTION.keys()) == set(INTERVENTION_NAMES)


class TestPatientParams:
    """Test patient/student parameter configuration."""

    def test_count(self):
        assert len(PATIENT_NAMES) == 6

    def test_names_are_unique(self):
        assert len(set(PATIENT_NAMES)) == len(PATIENT_NAMES)

    def test_bounds_keys_match_names(self):
        assert set(PATIENT_BOUNDS.keys()) == set(PATIENT_NAMES)

    def test_grids_keys_match_names(self):
        assert set(PATIENT_GRIDS.keys()) == set(PATIENT_NAMES)

    def test_grids_are_sorted(self):
        for name, grid in PATIENT_GRIDS.items():
            assert grid == sorted(grid), f"{name} grid is not sorted: {grid}"

    def test_grids_within_bounds(self):
        for name, grid in PATIENT_GRIDS.items():
            lo, hi = PATIENT_BOUNDS[name]
            for v in grid:
                assert lo <= v <= hi, f"{name} grid value {v} out of bounds [{lo}, {hi}]"

    def test_defaults_within_bounds(self):
        for name in PATIENT_NAMES:
            lo, hi = PATIENT_BOUNDS[name]
            v = DEFAULT_PATIENT[name]
            assert lo <= v <= hi, (
                f"Default {name}={v} out of bounds [{lo}, {hi}]"
            )

    def test_defaults_keys_match_names(self):
        assert set(DEFAULT_PATIENT.keys()) == set(PATIENT_NAMES)

    def test_bounds_are_well_ordered(self):
        for name in PATIENT_NAMES:
            lo, hi = PATIENT_BOUNDS[name]
            assert lo < hi, f"{name}: lower={lo} >= upper={hi}"


class TestStudentArchetypes:
    """Test student archetype seed definitions."""

    def test_count(self):
        assert len(STUDENT_ARCHETYPES) == 8

    def test_all_have_required_fields(self):
        for arch in STUDENT_ARCHETYPES:
            assert "name" in arch, f"Archetype missing 'name': {arch}"
            assert "description" in arch, f"Archetype missing 'description': {arch}"
            assert "patient" in arch, f"Archetype missing 'patient': {arch}"

    def test_names_are_unique(self):
        names = [a["name"] for a in STUDENT_ARCHETYPES]
        assert len(set(names)) == len(names)

    def test_patient_values_within_bounds(self):
        for arch in STUDENT_ARCHETYPES:
            patient = arch["patient"]
            for name, value in patient.items():
                assert name in PATIENT_BOUNDS, (
                    f"Archetype '{arch['name']}' has unknown patient param '{name}'"
                )
                lo, hi = PATIENT_BOUNDS[name]
                assert lo <= value <= hi, (
                    f"Archetype '{arch['name']}' param {name}={value} "
                    f"out of bounds [{lo}, {hi}]"
                )

    def test_intervention_overrides_within_bounds(self):
        for arch in STUDENT_ARCHETYPES:
            if "intervention" not in arch:
                continue
            for name, value in arch["intervention"].items():
                assert name in INTERVENTION_BOUNDS, (
                    f"Archetype '{arch['name']}' has unknown intervention param '{name}'"
                )
                lo, hi = INTERVENTION_BOUNDS[name]
                assert lo <= value <= hi, (
                    f"Archetype '{arch['name']}' intervention {name}={value} "
                    f"out of bounds [{lo}, {hi}]"
                )

    def test_patient_keys_are_subset_of_patient_names(self):
        for arch in STUDENT_ARCHETYPES:
            patient_keys = set(arch["patient"].keys())
            assert patient_keys <= set(PATIENT_NAMES), (
                f"Archetype '{arch['name']}' has unexpected patient keys: "
                f"{patient_keys - set(PATIENT_NAMES)}"
            )

    def test_expected_archetype_names(self):
        names = {a["name"] for a in STUDENT_ARCHETYPES}
        expected = {
            "resilient_male", "resilient_female", "vulnerable_female",
            "anxious_male", "sleep_deprived", "nature_seeker",
            "digital_immersed", "recovery_trajectory",
        }
        assert names == expected


class TestSnap:
    """Test grid snapping functions."""

    def test_snap_param_exact_grid_value(self):
        """Snapping a value that is already on the grid returns it unchanged."""
        assert snap_param("nature_rx", 0.4) == 0.4

    def test_snap_param_returns_nearest(self):
        """Snapping 0.37 to the [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] grid gives 0.4."""
        assert snap_param("nature_rx", 0.37) == 0.4

    def test_snap_param_returns_lower_neighbor(self):
        """Snapping 0.29 to the grid gives 0.2 (closer than 0.4)."""
        assert snap_param("nature_rx", 0.29) == 0.2

    def test_snap_param_boundary_low(self):
        """Snapping below the grid minimum returns the minimum."""
        assert snap_param("nature_rx", -0.5) == 0.0

    def test_snap_param_boundary_high(self):
        """Snapping above the grid maximum returns the maximum."""
        assert snap_param("nature_rx", 1.5) == 1.0

    def test_snap_param_patient(self):
        """Snapping works for patient parameters too."""
        assert snap_param("age", 19.3) == 19.0
        assert snap_param("age", 19.8) == 20.0

    def test_snap_all_preserves_non_grid_params(self):
        """Parameters not in any grid are left unchanged."""
        params = {"nature_rx": 0.37, "unknown_param": 42.0}
        result = snap_all(params)
        assert result["nature_rx"] == 0.4
        assert result["unknown_param"] == 42.0

    def test_snap_all_does_not_modify_input(self):
        """snap_all returns a new dict, not a mutation of the input."""
        params = {"nature_rx": 0.37, "exercise_rx": 0.55}
        original_nr = params["nature_rx"]
        _ = snap_all(params)
        assert params["nature_rx"] == original_nr

    def test_all_grids_contains_all_params(self):
        """_ALL_GRIDS should contain all intervention and patient grids."""
        for name in INTERVENTION_NAMES:
            assert name in _ALL_GRIDS
        for name in PATIENT_NAMES:
            assert name in _ALL_GRIDS


class TestSemesterCalendar:
    """Test semester calendar helper functions."""

    def test_day_of_week_monday(self):
        """Day 0 is Monday (semester starts on a Monday)."""
        assert day_of_week(0) == 0

    def test_day_of_week_sunday(self):
        """Day 6 is Sunday."""
        assert day_of_week(6) == 6

    def test_day_of_week_wraps(self):
        """Day 7 is Monday again."""
        assert day_of_week(7) == 0
        assert day_of_week(14) == 0

    def test_is_weekday_monday_through_friday(self):
        """Days 0-4 (Mon-Fri) are weekdays."""
        for d in range(5):
            assert is_weekday(d) is True, f"Day {d} should be a weekday"

    def test_is_weekday_saturday(self):
        """Day 5 (Saturday) is not a weekday."""
        assert is_weekday(5) is False

    def test_is_weekday_sunday(self):
        """Day 6 (Sunday) is not a weekday."""
        assert is_weekday(6) is False

    def test_is_school_day_normal_weekday(self):
        """A normal Monday (day 0) is a school day."""
        assert is_school_day(0) is True

    def test_is_school_day_weekend(self):
        """Saturday (day 5) is not a school day."""
        assert is_school_day(5) is False

    def test_is_school_day_spring_break(self):
        """All days during spring break week (week 8, 0-indexed week 7) are NOT school days."""
        break_start = (SPRING_BREAK_WEEK - 1) * 7  # day 49
        for d in range(break_start, break_start + 7):
            assert is_school_day(d) is False, (
                f"Day {d} (spring break) should not be a school day"
            )

    def test_is_school_day_after_break(self):
        """The Monday after spring break IS a school day."""
        post_break_monday = SPRING_BREAK_WEEK * 7  # day 56
        assert is_school_day(post_break_monday) is True

    def test_week_of_semester(self):
        """week_of_semester returns 0-indexed week number."""
        assert week_of_semester(0) == 0
        assert week_of_semester(6) == 0
        assert week_of_semester(7) == 1
        assert week_of_semester(104) == 14  # last day of semester

    def test_spring_break_week_value(self):
        assert SPRING_BREAK_WEEK == 8

    def test_full_semester_school_days(self):
        """Count total school days across the semester (should be 14 weeks * 5 = 70)."""
        count = sum(1 for d in range(N_STEPS) if is_school_day(d))
        expected = 14 * 5  # 15 weeks minus 1 break week, 5 school days per week
        assert count == expected
