"""Zimmerman protocol adapter for the LEMURS semester simulator.

┌─────────────────────────────────────────────────────────────────────────────┐
│                        THE UNIVERSAL SIMULATOR PLUG                        │
│                                                                            │
│  Imagine you have a library of scientific instruments, each measuring      │
│  something different -- one measures ocean currents, another measures      │
│  stock markets, another measures robot gaits. They all produce different   │
│  data, but you want to be able to ask the same questions of any of them:   │
│                                                                            │
│    "What happens if I change this input?"                                  │
│    "Which inputs matter the most?"                                         │
│    "Where does the system break?"                                          │
│                                                                            │
│  The Zimmerman protocol is a universal plug that makes this possible.      │
│  Any simulator that speaks this protocol -- LEMURS, grief, mitochondrial   │
│  aging, financial markets -- can be analyzed by the same suite of tools.   │
│                                                                            │
│  The protocol requires just two things:                                    │
│    1. param_spec(): "What knobs does this simulator have, and how far      │
│                      can each one turn?"                                   │
│    2. run(params):   "Turn the knobs to these settings and tell me what    │
│                      happened."                                            │
│                                                                            │
│  That's it. Two methods. But they unlock a powerful ecosystem of           │
│  analysis tools: sensitivity analysis (which inputs matter?), falsifier    │
│  (can we find scenarios that break the model?), contrastive analysis       │
│  (what makes a resilient student different from a vulnerable one?),        │
│  and more.                                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Compatible with:
  - zimmerman-toolkit: 14 interrogation modules for scientific simulators
  - cramer-toolkit: scenario-based resilience analysis under stress
"""
from __future__ import annotations

import math

from constants import (
    INTERVENTION_NAMES, INTERVENTION_BOUNDS, DEFAULT_INTERVENTION,
    PATIENT_NAMES, PATIENT_BOUNDS, DEFAULT_PATIENT,
)
from simulator import simulate
from analytics import compute_all


class LEMURSSimulator:
    """Zimmerman-protocol-compatible LEMURS semester simulator.

    This class wraps the full LEMURS simulation pipeline -- the 14-state
    ODE integrator and the 4-pillar analytics -- into a simple interface
    that external tools can call without knowing anything about sleep debt,
    anxiety Markov chains, or attention restoration theory.

    From the outside, it looks like this:

        sim = LEMURSSimulator()
        spec = sim.param_spec()   # -> {"nature_rx": (0.0, 1.0), "age": (18.0, 25.0), ...}
        result = sim.run({"nature_rx": 0.8, "emotional_stability": 6.0})
        # result -> {"sleep_quality_tst_mean": 7.12, "stress_anxiety_pss_mean": 14.3, ...}

    The external tools don't need to know about PSS secular trends, GAD-7
    clinical thresholds, or social jetlag forcing functions. They just see
    numbers going in and numbers coming out. The biopsychosocial dynamics
    are encapsulated inside.
    """

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Describe all 12 input knobs and their valid ranges.

        Returns a dictionary mapping parameter names to (min, max) tuples.
        The 12 parameters divide into two groups:

          Intervention factors (6): These are the behavioral levers that can
          be applied during the semester. Nature prescription, exercise,
          therapy, sleep hygiene, caffeine reduction, and academic load.

          Student/patient factors (6): These describe who the student is.
          Age, gender, emotional stability, trauma load, prior MH diagnosis,
          and baseline chronotype.

        External tools use this to know what inputs they can vary and
        what ranges are meaningful.
        """
        return {**INTERVENTION_BOUNDS, **PATIENT_BOUNDS}

    def run(self, params: dict[str, float]) -> dict[str, float]:
        """Run a full 15-week semester simulation and return all metrics.

        Takes a dictionary of parameter values (any subset of the 12 inputs --
        missing parameters use clinically reasonable defaults) and returns a
        flat dictionary of ~30+ output metrics covering all four pillars:

          sleep_quality_*:           How did sleep hold up?
          stress_anxiety_*:          What was the psychological burden?
          physiological_*:           What do the biomarkers show?
          intervention_response_*:   How effective were the interventions?

        The naming convention is pillar_metric, so external tools can easily
        group related outputs without knowing the internal structure.
        """
        # -- Step 1: Sort the incoming parameters --
        # External tools send all parameters in one flat dictionary.
        # We need to separate them into the two groups the simulator expects:
        # student characteristics and intervention choices.
        intervention = dict(DEFAULT_INTERVENTION)
        patient = dict(DEFAULT_PATIENT)

        for k, v in params.items():
            if k in INTERVENTION_BOUNDS:
                intervention[k] = v
            elif k in PATIENT_BOUNDS:
                patient[k] = v

        # -- Step 2: Run the simulation --
        # This calls the ODE integrator (105 daily timesteps over 15 weeks)
        # and produces the full 14-state trajectory.
        result = simulate(intervention=intervention, patient=patient)

        # -- Step 3: Compute the no-intervention baseline --
        # To measure intervention effectiveness (Pillar 4), we need to know
        # what would have happened to the SAME student with no intervention.
        # This means running the same patient profile but with default
        # (no-intervention) settings.
        baseline_result = simulate(
            intervention=dict(DEFAULT_INTERVENTION),
            patient=patient,
        )

        # -- Step 4: Compute all 4 pillars of analytics --
        analytics = compute_all(result, baseline=baseline_result)

        # -- Step 5: Flatten the nested results --
        # The analytics come back as a nested dictionary:
        #   {"sleep_quality": {"tst_mean": 7.12, ...}, ...}
        #
        # External tools expect a flat dictionary:
        #   {"sleep_quality_tst_mean": 7.12, ...}
        #
        # We also guard against infinity and NaN values, which can occur
        # in edge cases and would confuse downstream analysis tools.
        flat = {}
        for pillar_name, pillar_metrics in analytics.items():
            for metric_name, value in pillar_metrics.items():
                key = f"{pillar_name}_{metric_name}"
                v = float(value)
                if math.isinf(v):
                    v = 999.0
                if math.isnan(v):
                    v = 0.0
                flat[key] = v

        return flat
