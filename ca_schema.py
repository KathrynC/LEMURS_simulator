"""State discretization schema for the LEMURS semantic cellular automaton.

Discretizes the 14D continuous ODE state vector into clinically meaningful
bins. Each variable gets 2-4 bins with thresholds drawn from published
clinical norms, LEMURS cohort statistics, and the ODE coupling structure.

The bin labels are human-readable clinical categories (e.g., "deprived",
"adequate", "excess" for sleep) rather than generic "low/medium/high",
making rule tables self-documenting and interpretable by clinicians.
"""
from __future__ import annotations

import numpy as np

from constants import (
    STATE_NAMES, N_STATES,
    _TST, _SQ, _PSS, _GAD7, _DEP, _ACT, _NAT,
    _RHR, _HRV, _ARR, _SJL, _SHAPE, _WB, _DAC,
    _LOWER, _UPPER,
)

# ── Bin schema ────────────────────────────────────────────────────────────
# Each entry: variable name, state index, list of (threshold, label) pairs.
# Bins are defined by upper thresholds: value < threshold[0] -> bin[0],
# threshold[0] <= value < threshold[1] -> bin[1], etc.
# The last label is the bin for values >= the final threshold.

BIN_SCHEMA: dict[str, dict] = {
    "TST": {
        "index": _TST,
        "thresholds": [6.0, 8.0],
        "labels": ["deprived", "adequate", "excess"],
        "centers": [5.0, 7.0, 10.0],
        "unit": "hours/night",
        "source": "Paper 3 sleep debt",
    },
    "SleepQuality": {
        "index": _SQ,
        "thresholds": [60.0, 80.0],
        "labels": ["poor", "fair", "good"],
        "centers": [40.0, 70.0, 90.0],
        "unit": "Oura score",
        "source": "Oura score ranges",
    },
    "PSS": {
        "index": _PSS,
        "thresholds": [14.0, 27.0],
        "labels": ["low", "moderate", "high"],
        "centers": [7.0, 20.0, 34.0],
        "unit": "PSS scale",
        "source": "PSS clinical norms",
    },
    "GAD7": {
        "index": _GAD7,
        "thresholds": [10.0],
        "labels": ["sub_threshold", "clinical"],
        "centers": [5.0, 15.0],
        "unit": "GAD-7 scale",
        "source": "Paper 4 bistability threshold",
    },
    "Depression": {
        "index": _DEP,
        "thresholds": [5.0, 14.0],
        "labels": ["normal", "mild", "moderate_plus"],
        "centers": [2.5, 9.5, 17.5],
        "unit": "DASS-21",
        "source": "DASS-21 cutoffs",
    },
    "Activity": {
        "index": _ACT,
        "thresholds": [100.0, 300.0],
        "labels": ["sedentary", "moderate", "active"],
        "centers": [50.0, 200.0, 400.0],
        "unit": "kcal",
        "source": "Active calorie ranges",
    },
    "NatureEngagement": {
        "index": _NAT,
        "thresholds": [3.0],
        "labels": ["low", "engaged"],
        "centers": [1.5, 9.0],
        "unit": "hours/week",
        "source": "Paper 7 dose threshold",
    },
    "RHR": {
        "index": _RHR,
        "thresholds": [60.0, 75.0],
        "labels": ["low", "normal", "elevated"],
        "centers": [52.5, 67.5, 87.5],
        "unit": "bpm",
        "source": "Clinical ranges",
    },
    "HRV": {
        "index": _HRV,
        "thresholds": [30.0, 60.0],
        "labels": ["low", "moderate", "high"],
        "centers": [22.5, 45.0, 90.0],
        "unit": "RMSSD ms",
        "source": "RMSSD ranges",
    },
    "ARR": {
        "index": _ARR,
        "thresholds": [18.0],
        "labels": ["normal", "elevated"],
        "centers": [14.0, 21.5],
        "unit": "breaths/min",
        "source": "Paper 7 respiratory",
    },
    "SocialJetlag": {
        "index": _SJL,
        "thresholds": [1.0],
        "labels": ["aligned", "misaligned"],
        "centers": [0.5, 2.0],
        "unit": "hours",
        "source": "Paper 6",
    },
    "SleepShape": {
        "index": _SHAPE,
        "thresholds": [0.5],
        "labels": ["stable", "disrupted"],
        "centers": [0.25, 0.75],
        "unit": "Cluster 1 fraction",
        "source": "Paper 2",
    },
    "WEMWBS": {
        "index": _WB,
        "thresholds": [40.0, 55.0],
        "labels": ["low", "moderate", "high"],
        "centers": [27.0, 47.5, 62.5],
        "unit": "WEMWBS scale",
        "source": "WEMWBS norms",
    },
    "DAC": {
        "index": _DAC,
        "thresholds": [0.3],
        "labels": ["depleted", "available"],
        "centers": [0.15, 0.65],
        "unit": "fraction",
        "source": "Paper 8 attention trap",
    },
}

# Ordered variable names matching state vector indices
_VAR_ORDER: list[str] = [
    "TST", "SleepQuality", "PSS", "GAD7", "Depression", "Activity",
    "NatureEngagement", "RHR", "HRV", "ARR", "SocialJetlag", "SleepShape",
    "WEMWBS", "DAC",
]


def _classify(value: float, thresholds: list[float], labels: list[str]) -> str:
    """Assign a continuous value to a named bin."""
    for i, thresh in enumerate(thresholds):
        if value < thresh:
            return labels[i]
    return labels[-1]


def discretize_state(continuous_state: np.ndarray) -> dict[str, str]:
    """Convert a 14D continuous state vector to named clinical bins.

    Parameters
    ----------
    continuous_state : np.ndarray
        14-element array of continuous state values (same ordering as
        constants.STATE_NAMES).

    Returns
    -------
    dict[str, str]
        Mapping from variable name to bin label, e.g.
        {"TST": "deprived", "GAD7": "clinical", ...}.
    """
    result = {}
    for var_name in _VAR_ORDER:
        schema = BIN_SCHEMA[var_name]
        val = float(continuous_state[schema["index"]])
        result[var_name] = _classify(val, schema["thresholds"], schema["labels"])
    return result


def continuous_exemplar(discrete_state: dict[str, str]) -> np.ndarray:
    """Convert a discrete bin assignment back to a 14D continuous exemplar.

    Uses the center value of each bin. This is a lossy inverse of
    discretize_state() -- it produces a representative point within each
    bin, not the original continuous value.

    Parameters
    ----------
    discrete_state : dict[str, str]
        Mapping from variable name to bin label.

    Returns
    -------
    np.ndarray
        14-element float64 array of bin center values.
    """
    state = np.zeros(N_STATES, dtype=np.float64)
    for var_name, label in discrete_state.items():
        schema = BIN_SCHEMA[var_name]
        idx = schema["labels"].index(label)
        state[schema["index"]] = schema["centers"][idx]
    return state


def bin_index(var_name: str, label: str) -> int:
    """Return the integer index of a bin label within its variable's bins."""
    return BIN_SCHEMA[var_name]["labels"].index(label)


def bin_count(var_name: str) -> int:
    """Return the number of bins for a given variable."""
    return len(BIN_SCHEMA[var_name]["labels"])
