# Scenario Bank Harmonization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the LEMURS kcramer_bridge.py from plain-dict standalone scenarios to proper cramer-toolkit `Scenario`/`ScenarioSet` objects, matching the pattern already used by the mito and stock simulators — and add the three cramer-toolkit convenience functions (run_resilience_analysis, run_vulnerability_analysis, run_scenario_comparison).

**Architecture:** Replace local stub functions (`set_param`, `scale_param`, `compose`) with cramer-toolkit imports (`from kcramer import Scenario, ScenarioSet, scale_param, set_param, shift_param, compose`). Convert 7 bank lists to `ScenarioSet` instances. Add `shift_param` capability. Add convenience functions. Keep backward compatibility for `apply_scenario` and `run_scenario_sweep` by delegating to cramer-toolkit internals.

**Tech Stack:** Python 3.11+, cramer-toolkit (kcramer), numpy-only

---

## Context

The LEMURS simulator's `kcramer_bridge.py` was written with a standalone fallback pattern — it tries to import cramer-toolkit and, when that fails, uses local stub functions. However:

1. **The import path is wrong.** Line 32 does `from cramer_toolkit.scenario import Scenario, ScenarioSet` — the actual package is `cramer` (aliased as `kcramer`), not `cramer_toolkit`, and there is no `scenario` submodule. This import **always fails silently**, leaving `HAS_CRAMER = False` permanently.

2. **Scenarios are plain dicts, not `Scenario` objects.** The mito bridge (`~/how-to-live-much-longer/kcramer_bridge.py`) and stock bridge (`~/stock-simulator/stock_simulator/stock_kcramer_bridge.py`) both use proper `Scenario`/`ScenarioSet` objects from cramer-toolkit. LEMURS uses plain Python dicts.

3. **Missing capabilities.** LEMURS has no `shift_param` (additive delta), no `run_resilience_analysis()`, no `run_vulnerability_analysis()`, and no `run_scenario_comparison()`. The mito and stock bridges have all three.

4. **Naming conventions diverge.** LEMURS uses `ALL_SCENARIOS` and `REFERENCE_PROTOCOLS`; mito and stock use `ALL_STRESS_SCENARIOS` and `PROTOCOLS`.

After this upgrade, all three simulator bridges will have identical structure and can interchangeably participate in cramer-toolkit's resilience, vulnerability, and regret analysis pipelines.

---

## Reference Files

**Pattern to follow (canonical):**
- `~/how-to-live-much-longer/kcramer_bridge.py` — mito bridge (25 scenarios, 6 banks, 3 convenience functions)
- `~/stock-simulator/stock_simulator/stock_kcramer_bridge.py` — stock bridge (16 scenarios, 5 banks, 3 convenience functions)

**What to rewrite:**
- `~/lemurs-simulator/kcramer_bridge.py` — current LEMURS bridge (19 scenarios, 7 banks, plain dicts)

**cramer-toolkit internals:**
- `~/cramer-toolkit/cramer/base.py` — `Modification`, `Scenario`, `ScenarioSet` classes
- `~/cramer-toolkit/cramer/bank.py` — `scale_param`, `set_param`, `shift_param`, `compose` factory functions
- `~/cramer-toolkit/cramer/runner.py` — `run_scenarios`, `run_protocol_suite`
- `~/cramer-toolkit/cramer/analysis.py` — `resilience_summary`, `vulnerability_profile`, `scenario_compare`, `robustness_score`, `scenario_regret`

---

## Tasks

### Task 1: Fix imports and replace local stubs

**Files:**
- Modify: `~/lemurs-simulator/kcramer_bridge.py`
- Test: `~/lemurs-simulator/tests/test_kcramer_bridge.py` (create)

**Step 1: Write tests for proper cramer-toolkit integration**

Create `~/lemurs-simulator/tests/test_kcramer_bridge.py`:

```python
"""Tests for kcramer_bridge scenario bank — cramer-toolkit integration."""
import sys
import pytest
from pathlib import Path

# Ensure cramer-toolkit is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "cramer-toolkit"))

from kcramer_bridge import (
    ALL_STRESS_SCENARIOS,
    ACADEMIC_SCENARIOS,
    SLEEP_SCENARIOS,
    SOCIAL_SCENARIOS,
    SEASONAL_SCENARIOS,
    DIGITAL_SCENARIOS,
    HEALTH_SCENARIOS,
    COMBINED_SCENARIOS,
    PROTOCOLS,
    HAS_CRAMER,
)


class TestCramerIntegration:
    """Verify cramer-toolkit is properly imported."""

    def test_has_cramer_is_true(self):
        assert HAS_CRAMER is True, "cramer-toolkit import failed"

    def test_scenarios_are_scenario_objects(self):
        from kcramer import Scenario
        for scenario in ALL_STRESS_SCENARIOS:
            assert isinstance(scenario, Scenario), (
                f"{scenario.name} is {type(scenario)}, not Scenario"
            )

    def test_all_stress_scenarios_is_scenario_set(self):
        from kcramer import ScenarioSet
        assert isinstance(ALL_STRESS_SCENARIOS, ScenarioSet)

    def test_bank_types(self):
        from kcramer import ScenarioSet
        for bank in [ACADEMIC_SCENARIOS, SLEEP_SCENARIOS, SOCIAL_SCENARIOS,
                     SEASONAL_SCENARIOS, DIGITAL_SCENARIOS, HEALTH_SCENARIOS,
                     COMBINED_SCENARIOS]:
            assert isinstance(bank, ScenarioSet), f"{bank.name} is not ScenarioSet"


class TestScenarioCounts:
    """Verify all 19 scenarios are present."""

    def test_total_count(self):
        assert len(list(ALL_STRESS_SCENARIOS)) == 19

    def test_bank_counts(self):
        assert len(list(ACADEMIC_SCENARIOS)) == 3
        assert len(list(SLEEP_SCENARIOS)) == 3
        assert len(list(SOCIAL_SCENARIOS)) == 3
        assert len(list(SEASONAL_SCENARIOS)) == 2
        assert len(list(DIGITAL_SCENARIOS)) == 2
        assert len(list(HEALTH_SCENARIOS)) == 3
        assert len(list(COMBINED_SCENARIOS)) == 3


class TestScenarioFields:
    """Verify each scenario has required fields."""

    def test_all_have_names(self):
        for scenario in ALL_STRESS_SCENARIOS:
            assert isinstance(scenario.name, str)
            assert len(scenario.name) > 0

    def test_all_have_descriptions(self):
        for scenario in ALL_STRESS_SCENARIOS:
            assert isinstance(scenario.description, str)
            assert len(scenario.description) > 0

    def test_all_have_modifications(self):
        from kcramer import Modification
        for scenario in ALL_STRESS_SCENARIOS:
            assert len(scenario.modifications) > 0
            for mod in scenario.modifications:
                assert isinstance(mod, Modification)

    def test_unique_names(self):
        names = [s.name for s in ALL_STRESS_SCENARIOS]
        assert len(names) == len(set(names)), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"


class TestScenarioApplication:
    """Verify scenarios apply correctly to parameters."""

    def test_set_param_applies(self):
        from kcramer_bridge import ACADEMIC_SCENARIOS
        scenario = list(ACADEMIC_SCENARIOS)[0]  # mild_academic_stress
        base = {"academic_load": 0.5, "nature_rx": 0.8}
        modified = scenario.apply(base)
        assert modified["academic_load"] == 0.7
        assert modified["nature_rx"] == 0.8  # unchanged

    def test_scale_param_applies(self):
        from kcramer_bridge import SEASONAL_SCENARIOS
        scenario = list(SEASONAL_SCENARIOS)[0]  # winter_darkness
        base = {"nature_rx": 1.0, "exercise_rx": 0.5}
        modified = scenario.apply(base)
        assert abs(modified["nature_rx"] - 0.3) < 1e-10  # scaled by 0.3
        assert modified["exercise_rx"] == 0.5  # unchanged

    def test_combined_scenario_applies_multiple(self):
        from kcramer_bridge import COMBINED_SCENARIOS
        scenario = list(COMBINED_SCENARIOS)[0]  # finals_week_vulnerable
        base = {
            "academic_load": 0.5,
            "mh_diagnosis": 0.0,
            "sleep_hygiene": 0.5,
        }
        modified = scenario.apply(base)
        assert modified["academic_load"] == 1.0
        assert modified["mh_diagnosis"] == 1.0
        assert modified["sleep_hygiene"] == 0.0


class TestProtocols:
    """Verify PROTOCOLS dict structure."""

    def test_protocol_count(self):
        assert len(PROTOCOLS) == 5

    def test_protocol_names(self):
        expected = {"no_treatment", "nature_only", "exercise_only",
                    "therapy_only", "full_protocol"}
        assert set(PROTOCOLS.keys()) == expected

    def test_protocol_values_are_dicts(self):
        for name, params in PROTOCOLS.items():
            assert isinstance(params, dict), f"{name} is not a dict"
            for k, v in params.items():
                assert isinstance(k, str)
                assert isinstance(v, (int, float))


class TestBackwardCompatibility:
    """Verify old apply_scenario and run_scenario_sweep still work."""

    def test_apply_scenario_exists(self):
        from kcramer_bridge import apply_scenario
        assert callable(apply_scenario)

    def test_run_scenario_sweep_exists(self):
        from kcramer_bridge import run_scenario_sweep
        assert callable(run_scenario_sweep)


class TestConvenienceFunctions:
    """Verify new cramer-toolkit convenience functions exist."""

    def test_run_resilience_analysis_exists(self):
        from kcramer_bridge import run_resilience_analysis
        assert callable(run_resilience_analysis)

    def test_run_vulnerability_analysis_exists(self):
        from kcramer_bridge import run_vulnerability_analysis
        assert callable(run_vulnerability_analysis)

    def test_run_scenario_comparison_exists(self):
        from kcramer_bridge import run_scenario_comparison
        assert callable(run_scenario_comparison)
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/lemurs-simulator && python -m pytest tests/test_kcramer_bridge.py -v`
Expected: Multiple failures (HAS_CRAMER is False, ALL_STRESS_SCENARIOS doesn't exist, etc.)

**Step 3: Rewrite kcramer_bridge.py**

Rewrite `~/lemurs-simulator/kcramer_bridge.py` to use proper cramer-toolkit imports:

```python
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
        ALL_STRESS_SCENARIOS, PROTOCOLS,
        run_resilience_analysis, run_vulnerability_analysis,
    )

    from lemurs_simulator import LEMURSSimulator
    sim = LEMURSSimulator()
    report = run_resilience_analysis(sim, PROTOCOLS, ALL_STRESS_SCENARIOS)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure cramer-toolkit is importable
_CRAMER_PATH = str(Path(__file__).resolve().parent.parent / "cramer-toolkit")
if _CRAMER_PATH not in sys.path:
    sys.path.insert(0, _CRAMER_PATH)

try:
    from kcramer import (
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
except ImportError:
    HAS_CRAMER = False
    raise ImportError(
        "cramer-toolkit not found. Install it or ensure ../cramer-toolkit/ exists."
    )


# ── 1. Academic stress scenarios ─────────────────────────────────────────────
# Paper 4: academic stressors are a significant anxiety trigger (AOR=1.68).
# Paper 8: academic load depletes directed attention capacity.
# academic_load range is [0.0, 1.0].

ACADEMIC_SCENARIOS = ScenarioSet("academic_stress", "Academic pressure scenarios", [
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
])


# ── 2. Sleep disruption scenarios ────────────────────────────────────────────
# Paper 3: TST is the strongest wearable predictor of perceived stress
#   (beta = -0.877 PSS/hr). Within-person deviation 2.2x amplified.
# Paper 6: social jetlag and sleep debt drive compensatory activity.
# sleep_hygiene range is [0.0, 1.0]; baseline_chronotype range is [2.0, 7.0].

SLEEP_SCENARIOS = ScenarioSet("sleep_disruption", "Sleep disruption scenarios", [
    set_param("sleep_hygiene", 0.0,
              name="mild_insomnia",
              description="Occasional difficulty sleeping — sleep hygiene abandoned"),

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
        description="Extreme sleep loss from late chronotype and no hygiene — "
                    "maximum social jetlag on school days",
    ),
])


# ── 3. Social isolation scenarios ────────────────────────────────────────────
# Paper 7: therapy engagement reduces DASS stress and anxiety directly.
# Paper 7: nature engagement is the primary WEMWBS driver.
# Isolation removes these protective pathways.

SOCIAL_SCENARIOS = ScenarioSet("social_isolation", "Social isolation scenarios", [
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
])


# ── 4. Seasonal effect scenarios ─────────────────────────────────────────────
# Paper 7: nature engagement is the primary well-being pathway.
# Paper 8: perceived nature (not GPS-measured) drives restoration.
# Seasonal changes alter nature availability and institutional structure.

SEASONAL_SCENARIOS = ScenarioSet("seasonal_effects", "Seasonal effect scenarios", [
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
])


# ── 5. Digital overload scenarios ────────────────────────────────────────────
# Paper 8: directed attention is a finite resource depleted by academic work
#   and restored by nature engagement. Digital overload depletes attention
#   while simultaneously removing nature's restorative pathway.

DIGITAL_SCENARIOS = ScenarioSet("digital_overload", "Digital overload scenarios", [
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
        description="Screen addiction — near-maximum academic load, no nature "
                    "contact, and disrupted sleep from blue light exposure",
    ),
])


# ── 6. Pre-existing health condition scenarios ───────────────────────────────
# Paper 4: prior MH diagnosis is the strongest predictor of anxiety
#   occurrence (AOR=2.10). Trauma load >= 2 is a risk factor (AOR=1.80).
#   Emotional stability is the strongest protective factor (AOR=0.58/pt).

HEALTH_SCENARIOS = ScenarioSet("pre_existing_health", "Pre-existing condition scenarios", [
    compose(
        set_param("mh_diagnosis", 1.0),
        set_param("emotional_stability", 3.0),
        name="prior_anxiety",
        description="Student with prior anxiety diagnosis and low emotional "
                    "stability — high vulnerability to anxiety occurrence",
    ),

    set_param("mh_diagnosis", 1.0,
              name="prior_depression",
              description="Student with prior depression diagnosis — elevated "
                          "baseline stress and reduced recovery capacity"),

    set_param("trauma_load", 3.0,
              name="trauma_exposure",
              description="Student with significant trauma history — lowered "
                          "anxiety threshold and disrupted sleep architecture"),
])


# ── 7. Combined crisis scenarios ─────────────────────────────────────────────
# Real students often face multiple simultaneous stressors. These scenarios
# test the model's response to compounded adversity.

COMBINED_SCENARIOS = ScenarioSet("combined_crises", "Combined crisis scenarios", [
    compose(
        set_param("academic_load", 1.0),
        set_param("mh_diagnosis", 1.0),
        set_param("sleep_hygiene", 0.0),
        name="finals_week_vulnerable",
        description="Exam stress + prior MH + insomnia — the convergence "
                    "of academic pressure, psychological vulnerability, and "
                    "sleep disruption during finals week",
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
])


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

# Backward compatibility alias
ALL_SCENARIOS = ALL_STRESS_SCENARIOS


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

# Backward compatibility alias
REFERENCE_PROTOCOLS = PROTOCOLS


# ── Convenience functions ────────────────────────────────────────────────────


def apply_scenario(base_params, scenario):
    """Apply a scenario's modifications to base parameters.

    Accepts both cramer-toolkit Scenario objects and legacy plain dicts.

    Args:
        base_params: Dict of parameter name -> value.
        scenario: Scenario object or legacy dict with "modifications" key.

    Returns:
        New dict with modifications applied.
    """
    if isinstance(scenario, Scenario):
        return scenario.apply(base_params)
    # Legacy dict fallback
    params = dict(base_params)
    mods = scenario.get("modifications", [])
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
    """Run a protocol under multiple stress scenarios.

    Args:
        sim: Simulator with .run(params) method.
        protocol: Base parameter dict.
        scenarios: ScenarioSet, list of Scenarios, or None (defaults to ALL_STRESS_SCENARIOS).
        output_key: Output metric to extract.

    Returns:
        Dict of {scenario_name: output_value}.
    """
    if scenarios is None:
        scenarios = ALL_STRESS_SCENARIOS
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


def run_resilience_analysis(sim, protocols=None, scenarios=None,
                            output_key="stress_anxiety_pss_mean",
                            higher_is_better=False):
    """Full resilience analysis: all protocols × all scenarios.

    Args:
        sim: Simulator with .run(params) method.
        protocols: Dict of {name: params}. Defaults to PROTOCOLS.
        scenarios: ScenarioSet. Defaults to ALL_STRESS_SCENARIOS.
        output_key: Output metric for scoring.
        higher_is_better: Whether higher output_key is desirable.

    Returns:
        Resilience summary dict from cramer-toolkit.
    """
    if protocols is None:
        protocols = PROTOCOLS
    if scenarios is None:
        scenarios = ALL_STRESS_SCENARIOS
    all_results = run_protocol_suite(sim, protocols, scenarios)
    return resilience_summary(all_results, output_key, higher_is_better)


def run_vulnerability_analysis(sim, protocol=None, scenarios=None,
                               output_key="stress_anxiety_pss_mean",
                               higher_is_better=False):
    """Vulnerability profiling: one protocol × all scenarios.

    Args:
        sim: Simulator with .run(params) method.
        protocol: Base parameter dict. Defaults to PROTOCOLS["full_protocol"].
        scenarios: ScenarioSet. Defaults to ALL_STRESS_SCENARIOS.
        output_key: Output metric for profiling.
        higher_is_better: Whether higher output_key is desirable.

    Returns:
        Vulnerability profile dict from cramer-toolkit.
    """
    if protocol is None:
        protocol = PROTOCOLS["full_protocol"]
    if scenarios is None:
        scenarios = ALL_STRESS_SCENARIOS
    results = run_scenarios(sim, protocol, scenarios)
    return vulnerability_profile(results, output_key, higher_is_better)


def run_scenario_comparison(analysis_fn, sim, scenarios=None, extract=None,
                            **kwargs):
    """Compare scenario outcomes using cramer-toolkit's scenario_compare.

    Args:
        analysis_fn: Function(sim, **kwargs) -> dict to run per scenario.
        sim: Simulator.
        scenarios: ScenarioSet. Defaults to ALL_STRESS_SCENARIOS.
        extract: Function(result) -> float to extract comparison value.
        **kwargs: Passed to analysis_fn.

    Returns:
        Comparison dict from cramer-toolkit.
    """
    if scenarios is None:
        scenarios = ALL_STRESS_SCENARIOS
    return scenario_compare(analysis_fn, sim, scenarios, extract, **kwargs)
```

**Step 4: Run tests**

Run: `cd ~/lemurs-simulator && python -m pytest tests/test_kcramer_bridge.py -v`
Expected: All tests pass.

**Step 5: Run full test suite for regressions**

Run: `cd ~/lemurs-simulator && python -m pytest tests/ -v`
Expected: All existing tests still pass.

**Step 6: Commit**

```bash
cd ~/lemurs-simulator
git add kcramer_bridge.py tests/test_kcramer_bridge.py
git commit -m "feat: upgrade kcramer_bridge to proper cramer-toolkit Scenario/ScenarioSet objects

Replace plain-dict scenarios with cramer-toolkit Scenario objects.
Fix broken import path (cramer_toolkit -> kcramer).
Add run_resilience_analysis, run_vulnerability_analysis,
run_scenario_comparison convenience functions.
Rename ALL_SCENARIOS -> ALL_STRESS_SCENARIOS, REFERENCE_PROTOCOLS -> PROTOCOLS.
Keep backward-compatible aliases."
```

---

### Task 2: Update CLAUDE.md and verify cross-project integration

**Files:**
- Modify: `~/lemurs-simulator/CLAUDE.md`

**Step 1: Update CLAUDE.md**

In the Commands section, add cramer-toolkit commands:

```bash
# Cramer-toolkit resilience analysis (requires ~/cramer-toolkit)
python -c "
from lemurs_simulator import LEMURSSimulator
from kcramer_bridge import run_resilience_analysis, PROTOCOLS, ALL_STRESS_SCENARIOS
sim = LEMURSSimulator()
report = run_resilience_analysis(sim, PROTOCOLS, ALL_STRESS_SCENARIOS)
print(f'Protocols tested: {len(PROTOCOLS)}')
print(f'Scenarios tested: {len(list(ALL_STRESS_SCENARIOS))}')
print(f'Most resilient: {report[\"rankings\"][0]}')
"
```

In the Architecture section, update the kcramer_bridge line:

```
kcramer_bridge.py         <- 19 stress scenarios in 7 ScenarioSet banks (cramer-toolkit integrated)
```

**Step 2: Run the integration one-liner**

Run: `cd ~/lemurs-simulator && PYTHONPATH=../cramer-toolkit python -c "from kcramer_bridge import ALL_STRESS_SCENARIOS, HAS_CRAMER; print(f'HAS_CRAMER={HAS_CRAMER}, scenarios={len(list(ALL_STRESS_SCENARIOS))}')"`
Expected: `HAS_CRAMER=True, scenarios=19`

**Step 3: Commit**

```bash
cd ~/lemurs-simulator
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for cramer-toolkit integration"
```

---

### Task 3: Add grief-simulator scenario bank (optional extension)

**Files:**
- Create: `~/grief-simulator/kcramer_bridge.py`
- Test: `~/grief-simulator/tests/test_kcramer_bridge.py`

This task adds a scenario bank to the grief-simulator, which currently has none. The grief simulator's 8 clinical seeds can be expressed as patient-parameter scenarios.

**Step 1: Write tests**

Create `~/grief-simulator/tests/test_kcramer_bridge.py`:

```python
"""Tests for grief-simulator cramer-toolkit scenario bank."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "cramer-toolkit"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kcramer_bridge import (
    ALL_STRESS_SCENARIOS,
    BEREAVEMENT_SCENARIOS,
    SUPPORT_SCENARIOS,
    COMBINED_SCENARIOS,
    PROTOCOLS,
    HAS_CRAMER,
)
from kcramer import Scenario, ScenarioSet


class TestGriefScenarioBank:
    def test_has_cramer(self):
        assert HAS_CRAMER is True

    def test_total_scenarios(self):
        assert len(list(ALL_STRESS_SCENARIOS)) >= 9

    def test_all_are_scenarios(self):
        for s in ALL_STRESS_SCENARIOS:
            assert isinstance(s, Scenario)

    def test_banks_are_scenario_sets(self):
        for bank in [BEREAVEMENT_SCENARIOS, SUPPORT_SCENARIOS, COMBINED_SCENARIOS]:
            assert isinstance(bank, ScenarioSet)

    def test_protocols_exist(self):
        assert "no_support" in PROTOCOLS
        assert "full_support" in PROTOCOLS
```

**Step 2: Implement grief kcramer_bridge.py**

Create `~/grief-simulator/kcramer_bridge.py` following the LEMURS/mito pattern. Use the 8 clinical seeds from `constants.py:CLINICAL_SEEDS` as the basis for bereavement severity scenarios, and the intervention parameters as support scenarios.

The grief-simulator has 7 patient params (relationship_closeness, grief_personality, prior_loss_count, support_network, age_at_loss, health_baseline, attachment_style) and 7 intervention params (therapy_start_month, therapy_intensity, medication_intensity, social_support_enhancement, exercise_program, mindfulness_program, grief_group_participation).

Three banks:
- BEREAVEMENT_SCENARIOS (4): mild, moderate, severe, complicated — varying relationship_closeness, grief_personality, prior_loss_count
- SUPPORT_SCENARIOS (3): no support, partial support, full support — varying intervention params
- COMBINED_SCENARIOS (3): complicated + no support, severe + partial, mild + full

= 10 scenarios total, matching the mito/stock/LEMURS pattern.

**Step 3: Run tests**

Run: `cd ~/grief-simulator && python -m pytest tests/test_kcramer_bridge.py -v`
Expected: All tests pass.

**Step 4: Commit**

```bash
cd ~/grief-simulator
git add kcramer_bridge.py tests/test_kcramer_bridge.py
git commit -m "feat: add cramer-toolkit scenario bank for grief resilience analysis"
```

---

## Verification

After all tasks:
```bash
# LEMURS — all tests pass, cramer integration works
cd ~/lemurs-simulator
PYTHONPATH=../cramer-toolkit python -m pytest tests/ -v
PYTHONPATH=../cramer-toolkit python -m pytest tests/test_kcramer_bridge.py -v

# Verify cramer-toolkit types
PYTHONPATH=../cramer-toolkit python -c "
from kcramer_bridge import ALL_STRESS_SCENARIOS, PROTOCOLS, HAS_CRAMER
from kcramer import ScenarioSet, Scenario
assert HAS_CRAMER
assert isinstance(ALL_STRESS_SCENARIOS, ScenarioSet)
assert all(isinstance(s, Scenario) for s in ALL_STRESS_SCENARIOS)
print(f'OK: {len(list(ALL_STRESS_SCENARIOS))} scenarios in {len(PROTOCOLS)} protocols')
"

# Grief (if Task 3 completed)
cd ~/grief-simulator
PYTHONPATH=../cramer-toolkit python -m pytest tests/test_kcramer_bridge.py -v
```

## Design Rationale

**Why full rewrite rather than patching the import:**
The existing code has dict-based scenarios throughout — every scenario bank, every modification, the compose() function, apply_scenario(), run_scenario_sweep(). Patching just the import would leave the dict structure in place, requiring a separate dict→Scenario conversion layer. A full rewrite is cleaner and ensures the LEMURS bridge is structurally identical to the mito and stock bridges.

**Why keep backward-compatible aliases:**
`ALL_SCENARIOS` and `REFERENCE_PROTOCOLS` may be referenced by external code or notebooks. The aliases cost nothing and prevent breakage.

**Why keep apply_scenario() with legacy dict support:**
Some users may have scenario dicts in notebooks or scripts. The function now accepts both Scenario objects and legacy dicts, providing a graceful migration path.

**Why add grief-simulator scenario bank:**
The grief-simulator is the only ODE simulator without cramer-toolkit integration. Adding it completes the harmonization across all 4 ODE simulators that have the Zimmerman protocol. (ER is excluded because its parameter space — synapse weights — doesn't map to environmental stress scenarios in the same way.)
