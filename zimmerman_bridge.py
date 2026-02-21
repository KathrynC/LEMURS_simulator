"""Full Zimmerman bridge for the LEMURS semester simulator.

Extends LEMURSSimulator with dual-mode operation:
  - Full mode (default): 12D parameter space (6 intervention + 6 patient)
  - Intervention-only mode: 6D, patient characteristics fixed

Compatible with zimmerman-toolkit's 14 interrogation modules.
"""
from __future__ import annotations

import math

from constants import (
    INTERVENTION_NAMES, INTERVENTION_BOUNDS, DEFAULT_INTERVENTION,
    PATIENT_NAMES, PATIENT_BOUNDS, DEFAULT_PATIENT,
)
from simulator import simulate
from analytics import compute_all

INF_CAP = 999.0


class LEMURSBridge:
    """Zimmerman-compatible LEMURS simulator with dual-mode operation.

    Args:
        intervention_only: If True, only expose intervention params (6D).
        patient_override: Fixed patient params for intervention-only mode.
    """

    def __init__(self, intervention_only=False, patient_override=None):
        self.intervention_only = intervention_only
        self.patient_override = patient_override or dict(DEFAULT_PATIENT)
        self._baseline_cache = {}

    def param_spec(self):
        if self.intervention_only:
            return dict(INTERVENTION_BOUNDS)
        return {**INTERVENTION_BOUNDS, **PATIENT_BOUNDS}

    def _get_baseline(self, patient):
        key = tuple(sorted(patient.items()))
        if key not in self._baseline_cache:
            self._baseline_cache[key] = simulate(
                intervention=dict(DEFAULT_INTERVENTION), patient=patient
            )
        return self._baseline_cache[key]

    def run(self, params):
        intervention = dict(DEFAULT_INTERVENTION)
        if self.intervention_only:
            patient = dict(self.patient_override)
        else:
            patient = dict(DEFAULT_PATIENT)

        for k, v in params.items():
            if k in INTERVENTION_BOUNDS:
                intervention[k] = float(v)
            elif k in PATIENT_BOUNDS and not self.intervention_only:
                patient[k] = float(v)

        result = simulate(intervention=intervention, patient=patient)
        baseline = self._get_baseline(patient)
        analytics = compute_all(result, baseline=baseline)

        flat = {}
        for pillar_name, pillar_metrics in analytics.items():
            for metric_name, value in pillar_metrics.items():
                key = f"{pillar_name}_{metric_name}"
                v = float(value)
                if math.isnan(v):
                    continue
                if math.isinf(v):
                    v = INF_CAP if v > 0 else -INF_CAP
                flat[key] = v

        return flat
