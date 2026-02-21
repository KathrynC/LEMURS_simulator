"""Scenario bank for the LEMURS semester simulator.

Defines environmental stress scenarios for resilience analysis using
cramer-toolkit's scenario-based framework. Each scenario modifies
student or intervention parameters to simulate a stressful event.

Seven scenario banks:
  1. Academic stress (exam periods, course overload)
  2. Sleep disruption (insomnia, schedule chaos)
  3. Social isolation (loneliness, loss of community)
  4. Seasonal effects (winter darkness, summer break)
  5. Digital overload (screen addiction, attention depletion)
  6. Pre-existing conditions (prior anxiety, depression, trauma)
  7. Combined crises (multiple simultaneous stressors)

Usage:
    from kcramer_bridge import (
        ALL_SCENARIOS, REFERENCE_PROTOCOLS,
        apply_scenario, run_scenario_sweep,
    )

    from lemurs_simulator import LEMURSSimulator
    sim = LEMURSSimulator()
    results = run_scenario_sweep(sim, REFERENCE_PROTOCOLS["full_protocol"])
"""
from __future__ import annotations

# Try cramer-toolkit import, fall back to standalone mode
try:
    import sys
    sys.path.insert(0, "../cramer-toolkit")
    from cramer_toolkit.scenario import Scenario, ScenarioSet
    HAS_CRAMER = True
except ImportError:
    HAS_CRAMER = False

from constants import (
    INTERVENTION_NAMES,
    PATIENT_NAMES,
    INTERVENTION_BOUNDS,
    PATIENT_BOUNDS,
    DEFAULT_INTERVENTION,
    DEFAULT_PATIENT,
)


# ── Scenario helper functions ────────────────────────────────────────────────


def scale_param(name, factor):
    """Scale a parameter by a multiplicative factor."""
    return {"operation": "scale", "param": name, "factor": factor}


def set_param(name, value):
    """Set a parameter to a fixed value."""
    return {"operation": "set", "param": name, "value": value}


def compose(*modifications):
    """Combine multiple parameter modifications."""
    return list(modifications)


# ── 1. Academic stress scenarios ─────────────────────────────────────────────
# Paper 4: academic stressors are a significant anxiety trigger (AOR=1.68).
# Paper 8: academic load depletes directed attention capacity.
# academic_load range is [0.0, 1.0].

ACADEMIC_SCENARIOS = [
    {"name": "mild_academic_stress",
     "description": "Moderate coursework pressure",
     "modifications": [set_param("academic_load", 0.7)]},

    {"name": "exam_week",
     "description": "Finals week: maximum academic pressure",
     "modifications": [set_param("academic_load", 1.0)]},

    {"name": "academic_crisis",
     "description": "Course overload with failing grades — maximum academic "
                    "pressure plus sleep disruption from all-night studying",
     "modifications": [set_param("academic_load", 1.0),
                       set_param("sleep_hygiene", 0.0)]},
]


# ── 2. Sleep disruption scenarios ────────────────────────────────────────────
# Paper 3: TST is the strongest wearable predictor of perceived stress
#   (beta = -0.877 PSS/hr). Within-person deviation 2.2x amplified.
# Paper 6: social jetlag and sleep debt drive compensatory activity.
# sleep_hygiene range is [0.0, 1.0]; baseline_chronotype range is [2.0, 7.0].

SLEEP_SCENARIOS = [
    {"name": "mild_insomnia",
     "description": "Occasional difficulty sleeping — sleep hygiene abandoned",
     "modifications": [set_param("sleep_hygiene", 0.0)]},

    {"name": "chronic_insomnia",
     "description": "Persistent sleep disruption with stimulant dependence",
     "modifications": [set_param("sleep_hygiene", 0.0),
                       set_param("caffeine_reduction", 0.0)]},

    {"name": "severe_deprivation",
     "description": "Extreme sleep loss from late chronotype and no hygiene — "
                    "maximum social jetlag on school days",
     "modifications": [set_param("sleep_hygiene", 0.0),
                       set_param("baseline_chronotype", 6.5)]},
]


# ── 3. Social isolation scenarios ────────────────────────────────────────────
# Paper 7: therapy engagement reduces DASS stress and anxiety directly.
# Paper 7: nature engagement is the primary WEMWBS driver.
# Isolation removes these protective pathways.

SOCIAL_SCENARIOS = [
    {"name": "mild_isolation",
     "description": "Loss of counseling access — no therapy support",
     "modifications": [set_param("therapy_rx", 0.0)]},

    {"name": "moderate_isolation",
     "description": "No therapy and no nature contact — indoor confinement",
     "modifications": [set_param("therapy_rx", 0.0),
                       set_param("nature_rx", 0.0)]},

    {"name": "full_isolation",
     "description": "Complete social/nature/therapy withdrawal — sedentary "
                    "indoor existence with no protective factors",
     "modifications": [set_param("therapy_rx", 0.0),
                       set_param("nature_rx", 0.0),
                       set_param("exercise_rx", 0.0)]},
]


# ── 4. Seasonal effect scenarios ─────────────────────────────────────────────
# Paper 7: nature engagement is the primary well-being pathway.
# Paper 8: perceived nature (not GPS-measured) drives restoration.
# Seasonal changes alter nature availability and institutional structure.

SEASONAL_SCENARIOS = [
    {"name": "winter_darkness",
     "description": "Reduced daylight, indoor confinement — nature engagement "
                    "reduced by 70%",
     "modifications": [scale_param("nature_rx", 0.3)]},

    {"name": "summer_break",
     "description": "No institutional structure — no academic pressure but "
                    "reduced exercise from loss of campus routine",
     "modifications": [set_param("academic_load", 0.0),
                       set_param("exercise_rx", 0.5)]},
]


# ── 5. Digital overload scenarios ────────────────────────────────────────────
# Paper 8: directed attention is a finite resource depleted by academic work
#   and restored by nature engagement. Digital overload depletes attention
#   while simultaneously removing nature's restorative pathway.

DIGITAL_SCENARIOS = [
    {"name": "moderate_screen",
     "description": "Excessive screen time — high academic load displaces "
                    "nature engagement entirely",
     "modifications": [set_param("academic_load", 0.8),
                       set_param("nature_rx", 0.0)]},

    {"name": "digital_addiction",
     "description": "Screen addiction — near-maximum academic load, no nature "
                    "contact, and disrupted sleep from blue light exposure",
     "modifications": [set_param("academic_load", 0.95),
                       set_param("nature_rx", 0.0),
                       set_param("sleep_hygiene", 0.0)]},
]


# ── 6. Pre-existing health condition scenarios ───────────────────────────────
# Paper 4: prior MH diagnosis is the strongest predictor of anxiety
#   occurrence (AOR=2.10). Trauma load >= 2 is a risk factor (AOR=1.80).
#   Emotional stability is the strongest protective factor (AOR=0.58/pt).

HEALTH_SCENARIOS = [
    {"name": "prior_anxiety",
     "description": "Student with prior anxiety diagnosis and low emotional "
                    "stability — high vulnerability to anxiety occurrence",
     "modifications": [set_param("mh_diagnosis", 1.0),
                       set_param("emotional_stability", 3.0)]},

    {"name": "prior_depression",
     "description": "Student with prior depression diagnosis — elevated "
                    "baseline stress and reduced recovery capacity",
     "modifications": [set_param("mh_diagnosis", 1.0)]},

    {"name": "trauma_exposure",
     "description": "Student with significant trauma history — lowered "
                    "anxiety threshold and disrupted sleep architecture",
     "modifications": [set_param("trauma_load", 3.0)]},
]


# ── 7. Combined crisis scenarios ─────────────────────────────────────────────
# Real students often face multiple simultaneous stressors. These scenarios
# test the model's response to compounded adversity.

COMBINED_SCENARIOS = [
    {"name": "finals_week_vulnerable",
     "description": "Exam stress + prior MH + insomnia — the convergence "
                    "of academic pressure, psychological vulnerability, and "
                    "sleep disruption during finals week",
     "modifications": compose(
         set_param("academic_load", 1.0),
         set_param("mh_diagnosis", 1.0),
         set_param("sleep_hygiene", 0.0),
     )},

    {"name": "pandemic_isolation",
     "description": "No school structure, no social contact, no nature — "
                    "complete removal of institutional and environmental "
                    "protective factors (COVID-era scenario)",
     "modifications": compose(
         set_param("academic_load", 0.0),
         set_param("therapy_rx", 0.0),
         set_param("nature_rx", 0.0),
         set_param("exercise_rx", 0.0),
     )},

    {"name": "burnout_cascade",
     "description": "Maximum academic load, no sleep hygiene, no nature, "
                    "low emotional stability — the burnout spiral where "
                    "every protective factor is absent",
     "modifications": compose(
         set_param("academic_load", 1.0),
         set_param("sleep_hygiene", 0.0),
         set_param("nature_rx", 0.0),
         set_param("emotional_stability", 2.0),
     )},
]


# ── Aggregated scenario bank ─────────────────────────────────────────────────

ALL_SCENARIOS = (
    ACADEMIC_SCENARIOS
    + SLEEP_SCENARIOS
    + SOCIAL_SCENARIOS
    + SEASONAL_SCENARIOS
    + DIGITAL_SCENARIOS
    + HEALTH_SCENARIOS
    + COMBINED_SCENARIOS
)


# ── Reference intervention protocols ─────────────────────────────────────────
# Standard intervention packages for comparison in resilience analysis.
# Each protocol represents a different level of support a university
# might provide to students.

REFERENCE_PROTOCOLS = {
    "no_treatment": {
        "nature_rx":          0.0,
        "exercise_rx":        0.0,
        "therapy_rx":         0.0,
        "sleep_hygiene":      0.0,
        "caffeine_reduction": 0.0,
        "academic_load":      0.5,
    },
    "nature_only": {
        "nature_rx":          0.8,
        "exercise_rx":        0.0,
        "therapy_rx":         0.0,
        "sleep_hygiene":      0.3,
        "caffeine_reduction": 0.0,
        "academic_load":      0.5,
    },
    "exercise_only": {
        "nature_rx":          0.0,
        "exercise_rx":        0.6,
        "therapy_rx":         0.0,
        "sleep_hygiene":      0.3,
        "caffeine_reduction": 0.0,
        "academic_load":      0.5,
    },
    "therapy_only": {
        "nature_rx":          0.0,
        "exercise_rx":        0.0,
        "therapy_rx":         0.4,
        "sleep_hygiene":      0.3,
        "caffeine_reduction": 0.0,
        "academic_load":      0.5,
    },
    "full_protocol": {
        "nature_rx":          0.8,
        "exercise_rx":        0.6,
        "therapy_rx":         0.4,
        "sleep_hygiene":      0.8,
        "caffeine_reduction": 0.5,
        "academic_load":      0.5,
    },
}


# ── Convenience functions ────────────────────────────────────────────────────


def apply_scenario(base_params, scenario):
    """Apply a scenario's modifications to base parameters.

    Args:
        base_params: Dict of parameter name -> value (intervention + patient).
        scenario: Dict with "modifications" key containing a list of
            modification dicts (from set_param/scale_param/compose).

    Returns:
        New dict with modifications applied (original unchanged).
    """
    params = dict(base_params)
    mods = scenario["modifications"]
    if not isinstance(mods, list):
        mods = [mods]
    for mod in mods:
        if mod["operation"] == "set":
            params[mod["param"]] = mod["value"]
        elif mod["operation"] == "scale":
            if mod["param"] in params:
                params[mod["param"]] *= mod["factor"]
    return params


def run_scenario_sweep(sim, protocol, scenarios=None,
                       output_key="stress_anxiety_pss_mean"):
    """Run a protocol under multiple stress scenarios, return {name: output_value}.

    Args:
        sim: Simulator instance with a .run(params) method (Zimmerman protocol).
        protocol: Base parameter dict (intervention + patient).
        scenarios: List of scenario dicts. Defaults to ALL_SCENARIOS.
        output_key: Which output metric to extract from simulation results.

    Returns:
        Dict of {scenario_name: output_value} for each scenario.
    """
    if scenarios is None:
        scenarios = ALL_SCENARIOS
    results = {}
    for scenario in scenarios:
        params = apply_scenario(protocol, scenario)
        output = sim.run(params)
        results[scenario["name"]] = output.get(output_key, 0.0)
    return results
