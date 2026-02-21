"""Scenario bank for the LEMURS semester simulator.

Defines environmental stress scenarios for resilience analysis using
cramer-toolkit's scenario-based framework. Each scenario modifies
student or intervention parameters to simulate a stressful event.

Seven scenario banks (19 scenarios total):
  1. Academic stress (exam periods, course overload)
  2. Sleep disruption (insomnia, schedule chaos)
  3. Social isolation (loneliness, loss of community)
  4. Seasonal effects (winter darkness, summer break)
  5. Digital overload (screen addiction, attention depletion)
  6. Pre-existing conditions (prior anxiety, depression, trauma)
  7. Combined crises (multiple simultaneous stressors)

Usage:
    from kcramer_bridge import (
        ALL_SCENARIOS, PROTOCOLS, REFERENCE_PROTOCOLS,
        apply_scenario, run_scenario_sweep,
        run_resilience_analysis, run_vulnerability_analysis,
        run_scenario_comparison,
    )

    from lemurs_simulator import LEMURSSimulator
    sim = LEMURSSimulator()
    report = run_resilience_analysis(sim)

Requires:
    cramer-toolkit repo (kcramer namespace) at ~/cramer-toolkit
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────

_CRAMER_PATH = str(Path(__file__).resolve().parent.parent / "cramer-toolkit")
if _CRAMER_PATH not in sys.path:
    sys.path.insert(0, _CRAMER_PATH)

from kcramer import (  # noqa: E402
    Scenario,
    ScenarioSet,
    Modification,
    scale_param,
    set_param,
    shift_param,
    compose,
    run_scenarios,
    run_protocol_suite,
    resilience_summary,
    vulnerability_profile,
    robustness_score,
    scenario_regret,
    scenario_compare,
)

HAS_CRAMER = True

from constants import (  # noqa: E402
    INTERVENTION_NAMES,
    PATIENT_NAMES,
    INTERVENTION_BOUNDS,
    PATIENT_BOUNDS,
    DEFAULT_INTERVENTION,
    DEFAULT_PATIENT,
)


# ── 1. Academic stress scenarios ─────────────────────────────────────────────
# Paper 4: academic stressors are a significant anxiety trigger (AOR=1.68).
# Paper 8: academic load depletes directed attention capacity.
# academic_load range is [0.0, 1.0].

ACADEMIC_SCENARIOS = ScenarioSet(
    "academic_stress",
    "Academic pressure scenarios",
    scenarios=[
        set_param("academic_load", 0.7,
                  name="mild_academic_stress",
                  description="Moderate coursework pressure"),

        set_param("academic_load", 1.0,
                  name="exam_week",
                  description="Finals week: maximum academic pressure"),

        compose(
            set_param("academic_load", 1.0),
            set_param("sleep_hygiene", 0.0),
            name="academic_crisis",
            description="Course overload with failing grades — maximum academic "
                        "pressure plus sleep disruption from all-night studying",
        ),
    ],
)


# ── 2. Sleep disruption scenarios ────────────────────────────────────────────
# Paper 3: TST is the strongest wearable predictor of perceived stress
#   (beta = -0.877 PSS/hr). Within-person deviation 2.2x amplified.
# Paper 6: social jetlag and sleep debt drive compensatory activity.
# sleep_hygiene range is [0.0, 1.0]; baseline_chronotype range is [2.0, 7.0].

SLEEP_SCENARIOS = ScenarioSet(
    "sleep_disruption",
    "Sleep disruption scenarios",
    scenarios=[
        set_param("sleep_hygiene", 0.0,
                  name="mild_insomnia",
                  description="Occasional difficulty sleeping — sleep hygiene "
                              "abandoned"),

        compose(
            set_param("sleep_hygiene", 0.0),
            set_param("caffeine_reduction", 0.0),
            name="chronic_insomnia",
            description="Persistent sleep disruption with stimulant dependence",
        ),

        compose(
            set_param("sleep_hygiene", 0.0),
            set_param("baseline_chronotype", 6.5),
            name="severe_deprivation",
            description="Extreme sleep loss from late chronotype and no "
                        "hygiene — maximum social jetlag on school days",
        ),
    ],
)


# ── 3. Social isolation scenarios ────────────────────────────────────────────
# Paper 7: therapy engagement reduces DASS stress and anxiety directly.
# Paper 7: nature engagement is the primary WEMWBS driver.
# Isolation removes these protective pathways.

SOCIAL_SCENARIOS = ScenarioSet(
    "social_isolation",
    "Social isolation scenarios",
    scenarios=[
        set_param("therapy_rx", 0.0,
                  name="mild_isolation",
                  description="Loss of counseling access — no therapy support"),

        compose(
            set_param("therapy_rx", 0.0),
            set_param("nature_rx", 0.0),
            name="moderate_isolation",
            description="No therapy and no nature contact — indoor confinement",
        ),

        compose(
            set_param("therapy_rx", 0.0),
            set_param("nature_rx", 0.0),
            set_param("exercise_rx", 0.0),
            name="full_isolation",
            description="Complete social/nature/therapy withdrawal — sedentary "
                        "indoor existence with no protective factors",
        ),
    ],
)


# ── 4. Seasonal effect scenarios ─────────────────────────────────────────────
# Paper 7: nature engagement is the primary well-being pathway.
# Paper 8: perceived nature (not GPS-measured) drives restoration.
# Seasonal changes alter nature availability and institutional structure.

SEASONAL_SCENARIOS = ScenarioSet(
    "seasonal_effects",
    "Seasonal effect scenarios",
    scenarios=[
        scale_param("nature_rx", 0.3,
                    name="winter_darkness",
                    description="Reduced daylight, indoor confinement — nature "
                                "engagement reduced by 70%"),

        compose(
            set_param("academic_load", 0.0),
            set_param("exercise_rx", 0.5),
            name="summer_break",
            description="No institutional structure — no academic pressure but "
                        "reduced exercise from loss of campus routine",
        ),
    ],
)


# ── 5. Digital overload scenarios ────────────────────────────────────────────
# Paper 8: directed attention is a finite resource depleted by academic work
#   and restored by nature engagement. Digital overload depletes attention
#   while simultaneously removing nature's restorative pathway.

DIGITAL_SCENARIOS = ScenarioSet(
    "digital_overload",
    "Digital overload scenarios",
    scenarios=[
        compose(
            set_param("academic_load", 0.8),
            set_param("nature_rx", 0.0),
            name="moderate_screen",
            description="Excessive screen time — high academic load displaces "
                        "nature engagement entirely",
        ),

        compose(
            set_param("academic_load", 0.95),
            set_param("nature_rx", 0.0),
            set_param("sleep_hygiene", 0.0),
            name="digital_addiction",
            description="Screen addiction — near-maximum academic load, no "
                        "nature contact, and disrupted sleep from blue light "
                        "exposure",
        ),
    ],
)


# ── 6. Pre-existing health condition scenarios ───────────────────────────────
# Paper 4: prior MH diagnosis is the strongest predictor of anxiety
#   occurrence (AOR=2.10). Trauma load >= 2 is a risk factor (AOR=1.80).
#   Emotional stability is the strongest protective factor (AOR=0.58/pt).

HEALTH_SCENARIOS = ScenarioSet(
    "preexisting_health",
    "Pre-existing health condition scenarios",
    scenarios=[
        compose(
            set_param("mh_diagnosis", 1.0),
            set_param("emotional_stability", 3.0),
            name="prior_anxiety",
            description="Student with prior anxiety diagnosis and low "
                        "emotional stability — high vulnerability to anxiety "
                        "occurrence",
        ),

        set_param("mh_diagnosis", 1.0,
                  name="prior_depression",
                  description="Student with prior depression diagnosis — "
                              "elevated baseline stress and reduced recovery "
                              "capacity"),

        set_param("trauma_load", 3.0,
                  name="trauma_exposure",
                  description="Student with significant trauma history — "
                              "lowered anxiety threshold and disrupted sleep "
                              "architecture"),
    ],
)


# ── 7. Combined crisis scenarios ─────────────────────────────────────────────
# Real students often face multiple simultaneous stressors. These scenarios
# test the model's response to compounded adversity.

COMBINED_SCENARIOS = ScenarioSet(
    "combined_crisis",
    "Combined crisis scenarios",
    scenarios=[
        compose(
            set_param("academic_load", 1.0),
            set_param("mh_diagnosis", 1.0),
            set_param("sleep_hygiene", 0.0),
            name="finals_week_vulnerable",
            description="Exam stress + prior MH + insomnia — the convergence "
                        "of academic pressure, psychological vulnerability, "
                        "and sleep disruption during finals week",
        ),

        compose(
            set_param("academic_load", 0.0),
            set_param("therapy_rx", 0.0),
            set_param("nature_rx", 0.0),
            set_param("exercise_rx", 0.0),
            name="pandemic_isolation",
            description="No school structure, no social contact, no nature — "
                        "complete removal of institutional and environmental "
                        "protective factors (COVID-era scenario)",
        ),

        compose(
            set_param("academic_load", 1.0),
            set_param("sleep_hygiene", 0.0),
            set_param("nature_rx", 0.0),
            set_param("emotional_stability", 2.0),
            name="burnout_cascade",
            description="Maximum academic load, no sleep hygiene, no nature, "
                        "low emotional stability — the burnout spiral where "
                        "every protective factor is absent",
        ),
    ],
)


# ── Aggregated scenario bank ─────────────────────────────────────────────────

ALL_STRESS_SCENARIOS = (
    ACADEMIC_SCENARIOS
    + SLEEP_SCENARIOS
    + SOCIAL_SCENARIOS
    + SEASONAL_SCENARIOS
    + DIGITAL_SCENARIOS
    + HEALTH_SCENARIOS
    + COMBINED_SCENARIOS
)

ALL_SCENARIOS = ALL_STRESS_SCENARIOS  # backward compat alias


# ── Reference intervention protocols ─────────────────────────────────────────
# Standard intervention packages for comparison in resilience analysis.
# Each protocol represents a different level of support a university
# might provide to students.

PROTOCOLS = {
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

REFERENCE_PROTOCOLS = PROTOCOLS  # backward compat alias


# ── Convenience functions ────────────────────────────────────────────────────


def apply_scenario(base_params, scenario):
    """Apply a scenario's modifications to base parameters.

    Accepts both Scenario objects (cramer-toolkit) and legacy plain dicts.

    Args:
        base_params: Dict of parameter name -> value (intervention + patient).
        scenario: Either a Scenario object or a legacy dict with
            "modifications" key.

    Returns:
        New dict with modifications applied (original unchanged).
    """
    if isinstance(scenario, Scenario):
        return scenario.apply(base_params)

    # Legacy dict support for backward compatibility
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

    Backward-compatible convenience function. For new code, prefer
    run_scenarios() from cramer-toolkit.

    Args:
        sim: Simulator instance with a .run(params) method (Zimmerman protocol).
        protocol: Base parameter dict (intervention + patient).
        scenarios: List/ScenarioSet of Scenario objects, or list of legacy dicts.
            Defaults to ALL_SCENARIOS.
        output_key: Which output metric to extract from simulation results.

    Returns:
        Dict of {scenario_name: output_value} for each scenario.
    """
    if scenarios is None:
        scenarios = ALL_SCENARIOS
    results = {}
    for scenario in scenarios:
        if isinstance(scenario, Scenario):
            params = scenario.apply(protocol)
            name = scenario.name
        else:
            params = apply_scenario(protocol, scenario)
            name = scenario["name"]
        output = sim.run(params)
        results[name] = output.get(output_key, 0.0)
    return results


def run_resilience_analysis(
    sim=None,
    protocols=None,
    scenarios=None,
    output_key="stress_anxiety_pss_mean",
    higher_is_better=False,
):
    """Run a full resilience analysis on the LEMURS simulator.

    Evaluates multiple intervention protocols across the full scenario bank,
    computing robustness scores, regret analysis, vulnerability profiles,
    and protocol rankings.

    Args:
        sim: LEMURSSimulator instance. If None, imports and creates one.
        protocols: Dict of protocol_name -> param_dict.
            Defaults to PROTOCOLS.
        scenarios: Scenarios to test. Defaults to ALL_STRESS_SCENARIOS.
        output_key: Output metric for scoring.
            Defaults to "stress_anxiety_pss_mean" (lower is better).
        higher_is_better: Whether higher values of output_key are better.
            Defaults to False (lower PSS = less stress = better).

    Returns:
        Full resilience summary from kcramer.resilience_summary().
    """
    if sim is None:
        from lemurs_simulator import LEMURSSimulator
        sim = LEMURSSimulator()
    if protocols is None:
        protocols = PROTOCOLS
    if scenarios is None:
        scenarios = ALL_STRESS_SCENARIOS

    results = run_protocol_suite(sim, protocols, scenarios)
    return resilience_summary(
        results, output_key=output_key, higher_is_better=higher_is_better
    )


def run_vulnerability_analysis(
    sim=None,
    protocol=None,
    scenarios=None,
    output_key="stress_anxiety_pss_mean",
    higher_is_better=False,
):
    """Identify which stress scenarios most damage a protocol.

    Args:
        sim: LEMURSSimulator instance.
        protocol: Intervention params. Defaults to "full_protocol".
        scenarios: Scenarios to test. Defaults to ALL_STRESS_SCENARIOS.
        output_key: Output metric for comparison.
        higher_is_better: Whether higher values of output_key are better.

    Returns:
        Sorted list of {scenario, impact, ...}, worst first.
    """
    if sim is None:
        from lemurs_simulator import LEMURSSimulator
        sim = LEMURSSimulator()
    if protocol is None:
        protocol = PROTOCOLS["full_protocol"]
    if scenarios is None:
        scenarios = ALL_STRESS_SCENARIOS

    results = run_scenarios(sim, protocol, scenarios)
    return vulnerability_profile(
        results, output_key=output_key, higher_is_better=higher_is_better
    )


def run_scenario_comparison(
    analysis_fn,
    sim=None,
    scenarios=None,
    extract=None,
    **kwargs,
):
    """Run any analysis function under multiple stress scenarios.

    Wraps the simulator in ScenarioSimulator for each scenario, so
    any Zimmerman or other analysis tool becomes scenario-conditioned.

    Args:
        analysis_fn: Function with signature fn(sim, **kwargs).
        sim: LEMURSSimulator instance.
        scenarios: Scenarios to apply.
        extract: Optional scalar extractor for delta computation.
        **kwargs: Passed to analysis_fn.

    Returns:
        Dict of {scenario_name: {result, value, baseline_value, delta}}.
    """
    if sim is None:
        from lemurs_simulator import LEMURSSimulator
        sim = LEMURSSimulator()
    if scenarios is None:
        scenarios = ALL_STRESS_SCENARIOS

    return scenario_compare(
        analysis_fn, sim, scenarios, extract=extract, **kwargs
    )
