# LEMURS Semester Simulator

A 14-state ODE simulator modeling the coupled sleep-stress-anxiety-nature dynamical system of college students through a 15-week semester. Based on 9 published papers from the **LEMURS** research program (Lived Experiences Measured Using Rings Study) at the University of Vermont, 2023-2025.

The simulator tracks how sleep disruption, academic stress, nature engagement, and circadian misalignment interact to shape psychological and physiological outcomes -- day by day, from move-in to finals.

## What This Models

A college semester is 105 days of coupled biopsychosocial dynamics. Sleep debt accumulates on weeknights. Stress ratchets up toward finals. Anxiety crosses and recrosses clinical thresholds. Directed attention drains under academic load and restores in green spaces. Spring break briefly removes all institutional forcing, revealing which students bounce back and which have accumulated too much damage.

The simulator captures six empirically grounded tiers of coupling:

| Tier | Coupling | Source |
|------|----------|--------|
| 1 | **Sleep -> Stress** | Wearable biomarkers (TST, RHR, HRV, ARR) predict perceived stress; within-person deviations 2.2x stronger than between-person differences | Paper 3 |
| 2 | **Anxiety Markov dynamics** | GAD-7 threshold at >= 10 creates bistable regime; development risk modified by emotional stability, MH history, trauma, academic load; recovery decays with hysteresis | Paper 4 |
| 3 | **Nature -> Stress -> HRV mediation** | Nature engagement reduces PSS (-1.507/unit), PSS reduction improves HRV; therapy independently reduces DASS stress and respiratory rate | Paper 7 |
| 4 | **Sleep debt & activity** | Weekday sleep debt (37-55 min/school night), weekend partial recovery; paradoxical activity compensation (less sleep = more active) | Paper 6 |
| 5 | **Chronotype & sleep shape** | Chronotype determines social jetlag via weekday forcing; sleep phenotype (cluster membership) gender-modulated | Papers 2, 6 |
| 6 | **Attention restoration** | Academic load depletes directed attention capacity; perceived nature engagement restores it (not GPS-measured exposure) | Paper 8 |

## Key Dynamics

**GAD-7 bistability.** The clinical anxiety threshold at GAD-7 >= 10 creates two dynamical regimes. Below threshold, students face stochastic development risk. Above threshold, recovery probability decays each week (hysteresis factor = 0.92). Approximately 30% of students cross this threshold during a semester.

**Sleep-stress vicious cycle.** Within-person deviations from a student's own sleep baseline are 2.2x stronger predictors of stress than between-person differences. A student who normally sleeps 8 hours and drops to 6 is hit harder than a student who always sleeps 6.

**Attention depletion trap.** Academic load depletes directed attention capacity (Kaplan & Kaplan). When DAC approaches zero, engagement quality degrades, reducing the effectiveness of nature interventions. The students who most need restoration are least able to benefit from it.

**Spring break phase transition.** Week 8 removes all institutional constraints -- no school-day forcing, no sleep debt, no academic stressors. The week-long perturbation reveals which students are resilient (they bounce back) and which have accumulated irreversible damage.

**Perceived vs quantified nature paradox.** Perceived nature engagement predicts well-being improvement; GPS-measured green space exposure without subjective engagement does not -- and paradoxically predicts slightly higher depression. The engagement quality gate (DAC * threshold function) operationalizes this finding.

## State Variables (14D)

| Index | Variable | Range | Units | Source |
|-------|----------|-------|-------|--------|
| 0 | TST | 4-12 | hours/night | Paper 3 |
| 1 | SleepQuality | 0-100 | Oura sleep score | Paper 2 |
| 2 | PSS | 0-40 | Perceived Stress Scale | Papers 3, 7, 8 |
| 3 | GAD7 | 0-21 | Generalized Anxiety Disorder-7 | Paper 4 |
| 4 | Depression | 0-21 | DASS-21 depression subscale | Papers 7, 8 |
| 5 | Activity | 0-500 | active calories (kcal) | Paper 6 |
| 6 | NatureEngagement | 0-15 | perceived hours in nature/week | Papers 7, 8 |
| 7 | RHR | 45-100 | resting heart rate (bpm) | Paper 3 |
| 8 | HRV | 15-120 | RMSSD heart rate variability (ms) | Papers 3, 7 |
| 9 | ARR | 10-25 | average respiratory rate (breaths/min) | Papers 3, 7 |
| 10 | SocialJetlag | 0-3 | MSF_free - MSF_school (hours) | Paper 6 |
| 11 | SleepShape | 0-1 | fraction of nights in Cluster 1 | Paper 2 |
| 12 | WEMWBS | 14-70 | Warwick-Edinburgh Well-Being Scale | Paper 7 |
| 13 | DAC | 0-1 | Directed Attention Capacity | Paper 8 |

## Input Parameters (12D)

**Student characteristics (6D):**

| Parameter | Range | Description |
|-----------|-------|-------------|
| `age` | 18-25 | Student age |
| `gender` | 0/1/2 | 0=male, 1=female, 2=nonbinary |
| `emotional_stability` | 1-7 | Likert scale; strongest protective factor (AOR=0.58) |
| `trauma_load` | 0-5 | ACE-like adverse experience count |
| `mh_diagnosis` | 0/1 | Prior mental health diagnosis (AOR=2.10 for anxiety) |
| `baseline_chronotype` | 2-7 | MSF_free hours; late = more social jetlag |

**Intervention dials (6D):**

| Parameter | Range | Description |
|-----------|-------|-------------|
| `nature_rx` | 0-1 | Nature engagement prescription intensity |
| `exercise_rx` | 0-1 | Exercise prescription intensity |
| `therapy_rx` | 0-1 | Counseling/therapy engagement |
| `sleep_hygiene` | 0-1 | Sleep routine quality |
| `caffeine_reduction` | 0-1 | Stimulant reduction |
| `academic_load` | 0-1 | Course pressure (0=light, 0.5=typical, 1.0=overloaded) |

## Quickstart

```bash
# Dependencies: Python 3.10+, numpy, matplotlib
pip install numpy matplotlib

# Run the simulator with 8 student archetypes
python simulator.py

# Generate 4-panel trajectory plots
python visualize.py

# Run the test suite (229 tests)
python -m pytest tests/ -v
```

### From Python

```python
from simulator import simulate
from analytics import compute_all

# Default student, no intervention
result = simulate()
states = result["states"]  # shape (106, 14) -- daily values for all 14 variables
print(f"Final PSS: {states[-1, 2]:.1f}")
print(f"Mean TST: {states[:, 0].mean():.2f} hours")

# Vulnerable student with full intervention package
result = simulate(
    patient={"emotional_stability": 3.0, "mh_diagnosis": 1.0, "trauma_load": 3.0},
    intervention={"nature_rx": 0.8, "exercise_rx": 0.6, "therapy_rx": 0.4,
                  "sleep_hygiene": 0.8, "caffeine_reduction": 0.5},
)

# Compare against no-intervention baseline
baseline = simulate(
    patient={"emotional_stability": 3.0, "mh_diagnosis": 1.0, "trauma_load": 3.0},
)
analytics = compute_all(result, baseline=baseline)
print(f"PSS benefit: {analytics['intervention_response']['pss_benefit']:.2f} points")
print(f"HRV benefit: {analytics['intervention_response']['hrv_benefit']:.2f} ms")
```

## Student Archetypes

Eight representative student profiles are included in `constants.STUDENT_ARCHETYPES`:

| Archetype | Description | Key dynamics |
|-----------|-------------|--------------|
| `resilient_male` | High stability, no MH, moderate chronotype | Absorbs semester stress without crossing clinical thresholds |
| `resilient_female` | High stability, early chronotype, active | Stays healthy despite +2.956 PSS gender baseline |
| `vulnerable_female` | Low stability, MH history, trauma, late chronotype | Highest anxiety risk; tests worst-case compound vulnerability |
| `anxious_male` | Prior anxiety, moderate stability | Tests male anxiety trajectory with MH history |
| `sleep_deprived` | Extreme late chronotype (MSF=6.5h) | Massive social jetlag drives sleep debt cascade |
| `nature_seeker` | High nature engagement, reduced academic load | Tests attention restoration pathway |
| `digital_immersed` | No nature, high academic load | Tests attention depletion and burnout trap |
| `recovery_trajectory` | Starts anxious, full intervention package | Tests whether combined intervention can overcome high vulnerability |

## Architecture

```
constants.py              14D state, 12D params, coupling constants, 8 archetypes, semester calendar
simulator.py              initial_state(), derivatives() (6-tier coupling), RK4, simulate()
analytics.py              4-pillar compute_all(), NumpyEncoder
lemurs_simulator.py       LEMURSSimulator (Zimmerman protocol adapter)
zimmerman_bridge.py       Dual-mode bridge (12D full or 6D intervention-only)
zimmerman_analysis.py     14-tool Zimmerman analysis runner + CLI
kcramer_bridge.py         19 stress scenarios in 7 banks, 5 reference protocols
visualize.py              4-panel trajectory plots, spring break highlighting

ca_schema.py              Semantic CA: 14-variable bin schema, discretize/exemplar round-trip
ca_rules.py               Semantic CA: 32 tiered rules (6 tiers + cross-tier compounds), JSON-serializable
ca_simulator.py           Semantic CA: single-cell stepper + NxN population grid with social coupling
ca_analytics.py           Semantic CA: rule firing stats, cascade detection, attractor ID, ODE fidelity
ca_zimmerman_bridge.py    Semantic CA: LEMURSCASimulator + LEMURSPopulationSimulator (Zimmerman adapters)
```

**Dependency graph:** `constants <- simulator <- analytics <- lemurs_simulator`, `zimmerman_analysis <- zimmerman_bridge + zimmerman-toolkit`, `visualize <- simulator + constants`, `ca_schema <- ca_rules <- ca_simulator <- ca_analytics`, `ca_zimmerman_bridge <- ca_simulator + ca_analytics`.

### 4-Pillar Analytics

| Pillar | Metrics |
|--------|---------|
| **Sleep Quality** | TST mean/final/min, sleep quality mean/final, cumulative sleep debt, social jetlag mean, Cluster 1 fraction |
| **Stress & Anxiety** | PSS mean/final/slope/peak/time above threshold, GAD-7 mean/peak/days above 10, anxiety transition count, depression mean/final |
| **Physiological** | RHR mean/slope, HRV mean/final/slope, ARR mean/slope, DAC min |
| **Intervention Response** | PSS benefit, HRV benefit, well-being gain, nature dose-response, cost-effectiveness (PSS benefit per $1,000) |

### Simulator Protocol

The simulator implements the [Zimmerman protocol](https://github.com/KathrynC/zimmerman-toolkit), making it compatible with the full ecosystem of black-box simulator analysis tools:

```python
from lemurs_simulator import LEMURSSimulator

sim = LEMURSSimulator()
spec = sim.param_spec()   # -> {"nature_rx": (0.0, 1.0), "age": (18.0, 25.0), ...}
result = sim.run({"nature_rx": 0.8, "emotional_stability": 6.0})
# result -> {"sleep_quality_tst_mean": 7.12, "stress_anxiety_pss_mean": 14.3, ...}
```

## Scenario-Based Resilience Analysis

The `kcramer_bridge.py` module provides 19 environmental stress scenarios in 7 banks for use with the [Cramer toolkit](https://github.com/KathrynC/cramer-toolkit):

| Bank | Scenarios | What it tests |
|------|-----------|---------------|
| Academic stress | mild, exam_week, academic_crisis | Course pressure escalation |
| Sleep disruption | mild_insomnia, chronic_insomnia, severe_deprivation | Sleep hygiene collapse + late chronotype |
| Social isolation | mild, moderate, full | Progressive removal of protective factors |
| Seasonal effects | winter_darkness, summer_break | Nature availability and institutional structure |
| Digital overload | moderate_screen, digital_addiction | Attention depletion with no restoration |
| Pre-existing conditions | prior_anxiety, prior_depression, trauma_exposure | Student vulnerability amplification |
| Combined crises | finals_week_vulnerable, pandemic_isolation, burnout_cascade | Compound stressor convergence |

Five reference intervention protocols (no_treatment, nature_only, exercise_only, therapy_only, full_protocol) provide standardized comparison points.

```python
from kcramer_bridge import ALL_SCENARIOS, REFERENCE_PROTOCOLS, run_scenario_sweep
from lemurs_simulator import LEMURSSimulator

sim = LEMURSSimulator()
results = run_scenario_sweep(sim, REFERENCE_PROTOCOLS["full_protocol"])
# results -> {"mild_academic_stress": 14.2, "burnout_cascade": 22.7, ...}
```

## Zimmerman Toolkit Integration

The `zimmerman_analysis.py` module runs all 14 Zimmerman interrogation tools against the simulator:

```bash
# All 14 tools (~1 min at n_base=32)
python zimmerman_analysis.py --n-base 32

# Individual tools
python zimmerman_analysis.py --tools sobol --n-base 256    # global sensitivity (~7 min)
python zimmerman_analysis.py --tools falsifier             # systematic falsification
python zimmerman_analysis.py --tools contrastive           # minimal anxiety-flipping changes
python zimmerman_analysis.py --tools locality              # perturbation decay profiles

# Intervention-only mode (6D, fixed student profile)
python zimmerman_analysis.py --student vulnerable_female
```

**Available tools:** Sobol sensitivity, Falsifier, Contrastive, Contrast Sets, PDS Mapper, POSIWID Auditor, Prompt Builder, Locality Profiler, Relation Graph, Diegeticizer, Token Extispicy, Receptive Field, Supradiegetic Benchmark, Dashboard.

Reports are saved as JSON to `artifacts/zimmerman/` with a compiled markdown dashboard.

## Semantic Cellular Automaton

The CA layer discretizes the 14D continuous ODE state into clinically meaningful bins and simulates state transitions using tiered rules derived from the same published coupling structure. It provides an interpretable, rule-based complement to the continuous ODE -- local rules composing into global dynamics (bistability, burnout traps, spring break recovery).

### State Discretization

Each of the 14 state variables is mapped to 2-4 bins with clinically grounded thresholds:

| Variable | Bins | Thresholds | Basis |
|----------|------|------------|-------|
| TST | deprived / adequate / excess | 6h, 8h | Paper 3 sleep debt |
| GAD7 | sub_threshold / clinical | 10 | Paper 4 bistability |
| PSS | low / moderate / high | 14, 27 | PSS clinical norms |
| DAC | depleted / available | 0.3 | Paper 8 attention trap |
| NatureEngagement | low / engaged | 3h/wk | Paper 7 dose threshold |

### Rule Table

32 rules organized by ODE coupling tier, each with input bin conditions, output bin updates, confidence weight, and paper citation. Rules are JSON-serializable for inspection and editing.

Key dynamics captured:
- **Burnout cascade** (absorbing state): when TST=deprived, PSS=high, DAC=depleted, GAD7=clinical simultaneously, all restoration pathways are blocked and the state freezes
- **Spring break reset**: removes institutional forcing, tests recovery capacity
- **Within-person amplification**: TST bin drops from personal baseline trigger 2.2x stress rule strength
- **Confidence-based conflict resolution**: when multiple rules update the same variable, highest confidence wins

### Simulation Modes

**Single-cell** — one student, 105 daily steps:
```python
from ca_simulator import run_single_cell
result = run_single_cell(
    patient={"emotional_stability": 3.0, "mh_diagnosis": 1},
    intervention={"nature_rx": 0.8},
)
print(result["final_state"])    # {"TST": "adequate", "GAD7": "clinical", ...}
print(len(result["rule_log"]))  # 105 days of rule firing logs
```

**Population grid** — NxN students with shared institutional forcing and optional social coupling:
```python
from ca_simulator import run_population_grid
result = run_population_grid(
    grid_size=5, social_coupling=0.3,
    intervention={"nature_rx": 0.5},
)
print(result["population_summary"]["burnout_fraction"])
```

### CA Analytics

```python
from ca_analytics import compute_ca_analytics
analytics = compute_ca_analytics(ca_result, ode_result=ode_result)
# Returns: rule_stats, cascade_stats, attractor_stats, fidelity_stats, spring_break
```

| Section | Metrics |
|---------|---------|
| Rule stats | Firing frequency, unique rules, mean rules/day, top-10 rules |
| Cascade stats | Multi-tier chain reactions, max cascade length |
| Attractor stats | Terminal state classification (healthy/struggling/stressed/burnout), stability |
| Fidelity stats | Per-variable bin agreement rate between CA and ODE trajectories |
| Spring break | State before vs after break, variables that changed |

### CA Zimmerman Bridges

Both CA modes are Zimmerman-protocol compatible:

```python
from ca_zimmerman_bridge import LEMURSCASimulator, LEMURSPopulationSimulator

# Single-cell CA (same 12D param_spec as ODE simulator)
sim = LEMURSCASimulator()
result = sim.run({"nature_rx": 0.8})  # -> {"ca_final_attractor": 0.0, ...}

# Population grid CA (12D + grid_size + social_coupling)
pop = LEMURSPopulationSimulator()
result = pop.run({"grid_size": 5, "social_coupling": 0.3})
# -> {"pop_burnout_frac": 0.04, "pop_clinical_anxiety_frac": 0.12, ...}
```

## Simulation Details

**Integrator:** Runge-Kutta 4th order (RK4) with daily timesteps (dt = 1/7 week). 105 steps per semester.

**State clamping:** All 14 state variables are clamped to biological bounds after each RK4 step (e.g., TST cannot drop below 4 hours or exceed 12 hours; GAD-7 stays in [0, 21]).

**Semester calendar:** The simulation tracks the day of the week, weekday/weekend status, and school-day status. Spring break (week 8) removes all school-day forcing. This drives the weekday/weekend sleep-debt oscillation and the mid-semester perturbation.

**Deterministic:** Identical inputs always produce identical outputs. Float64 precision throughout.

## Source Papers

1. **Price et al. (2023)** -- LEMURS trial design. *Contemporary Clinical Trials*, 131, 107262.
2. **Fudolig et al. (2024)** -- Sleep heart rate shapes and phenotyping via Oura Ring clustering. *Digital Biomarkers*, 8(1), 80-90.
3. **Bloomfield et al. (2024)** -- Wearable biomarkers (TST, RHR, HRV, ARR) predict perceived stress; within-person deviations 2.2x stronger. *PLOS Digital Health*, 3(6), e0000530.
4. **Bloomfield et al. (2024)** -- Anxiety prevalence and persistence: 30% cross GAD-7 >= 10; Markov transition dynamics with emotional stability as strongest protective factor. *JAACAP Open*, 2(3), 200-210.
5. **Hidalgo et al. (2024)** -- Wellness practices and self-reported health in college students. *PLOS Digital Health*, 3(8), e0000581.
6. **Fudolig et al. (2025)** -- Collective sleep patterns: sleep debt (37-55 min/school night), social jetlag, paradoxical activity compensation, weekday/school modifiers. *npj Complexity*, 2(1), 1-12.
7. **Bloomfield et al. (2025)** -- Behavioral RCT: nature engagement reduces PSS (-1.507/unit), mediates HRV improvement (+9.13 ms over 14 weeks), increases WEMWBS. *Preprint*.
8. **Bloomfield et al. (2025)** -- Perceived vs GPS-measured nature: perceived engagement predicts depression reduction (-0.066/hr), GPS-only exposure does not. Attention Restoration Theory operationalized. *Preprint*.
9. **Bloomfield et al. (2025)** -- Spatial nature exposure and campus green space utilization patterns. *Preprint*.

## Related Projects

This simulator is part of a family of Zimmerman-protocol-compatible ODE simulators:

| Project | Domain | States | Cliff phenomenon |
|---------|--------|--------|------------------|
| **[LEMURS](https://github.com/KathrynC/LEMURS_simulator)** | College student well-being | 14D, 15 weeks | GAD-7 bistability at threshold 10 |
| **[how-to-live-much-longer](https://github.com/KathrynC/how-to-live-much-longer)** | Mitochondrial aging | 8D, 30 years | Heteroplasmy cliff at ~50% deletion het |
| **[grief-simulator](https://github.com/KathrynC/grief-simulator)** | Bereavement stress | 11D, 10 years | PGD bifurcation |
| **[stock-simulator](https://github.com/KathrynC/stock-simulator)** | Financial dynamics | 7D strategy | Margin-call cascade |

All share the same analysis ecosystem:
- **[zimmerman-toolkit](https://github.com/KathrynC/zimmerman-toolkit)** -- 14-module simulator interrogation (sensitivity, falsification, contrastive analysis, causal structure)
- **[cramer-toolkit](https://github.com/KathrynC/cramer-toolkit)** -- Scenario-based resilience analysis (robustness scoring, regret analysis, vulnerability profiling)

## License

Research software. See individual LEMURS papers for data usage terms.
