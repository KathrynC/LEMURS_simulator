# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

14-state ODE simulator modeling the sleep-stress-anxiety-nature coupled dynamical system of college students through a 15-week semester. Based on 9 LEMURS papers (UVM, 2023-2025) -- the Lived Experiences Measured Using Rings Study. Models how sleep disruption, academic stress, nature engagement, and circadian misalignment interact to shape psychological and physiological outcomes.

Implements the Zimmerman protocol (`run()` + `param_spec()`), compatible with zimmerman-toolkit and cramer-toolkit.

## Commands

```bash
python simulator.py                    # standalone: 8 student archetypes
python visualize.py                    # generate all plots to output/
python -m pytest tests/ -v             # full test suite (152 tests)
python -m pytest tests/test_simulator.py -v
python -c "from lemurs_simulator import LEMURSSimulator; s = LEMURSSimulator(); print(s.run({}))"

# Zimmerman toolkit analysis (requires ~/zimmerman-toolkit)
python zimmerman_analysis.py                           # all 14 tools (~1 min at n_base=32)
python zimmerman_analysis.py --tools sobol             # Sobol only
python zimmerman_analysis.py --tools sobol --n-base 256  # full Sobol (~7 min, 6656 sims)
python zimmerman_analysis.py --tools falsifier         # falsification (200 tests)
python zimmerman_analysis.py --tools contrastive       # contrastive pairs
python zimmerman_analysis.py --tools sobol,falsifier,locality  # multiple tools
python zimmerman_analysis.py --student vulnerable_female  # intervention-only mode (6D)
python zimmerman_analysis.py --intervention-only       # 6D mode, default student
```

## Architecture

```
constants.py           <- 14D state, 12D params, coupling constants, 8 archetypes, semester calendar
simulator.py           <- initial_state(), derivatives() (6-tier coupling), RK4, simulate()
analytics.py           <- 4-pillar compute_all(), NumpyEncoder
lemurs_simulator.py    <- LEMURSSimulator (Zimmerman protocol adapter)
zimmerman_bridge.py    <- Dual-mode bridge (12D or 6D intervention-only)
zimmerman_analysis.py  <- Full 14-tool Zimmerman analysis runner + CLI
kcramer_bridge.py      <- 19 stress scenarios in 7 banks
visualize.py           <- 4-panel trajectory plots, spring break highlighting
```

Dependency graph: `constants <- simulator <- analytics <- lemurs_simulator`, `zimmerman_analysis <- zimmerman_bridge + zimmerman-toolkit`, `visualize <- simulator + constants`.

## State Variables (14D)

| Index | Name | Range | Units | Source Paper |
|-------|------|-------|-------|--------------|
| 0 | TST | 4-12 | hours/night | Paper 3 (Bloomfield 2024, PLOS Digital Health) |
| 1 | SleepQuality | 0-100 | Oura sleep score | Paper 2 (Bloomfield 2024, sleep phenotyping) |
| 2 | PSS | 0-40 | Perceived Stress Scale | Papers 3, 7, 8 |
| 3 | GAD7 | 0-21 | Generalized Anxiety Disorder-7 | Paper 4 (Bloomfield 2024, JAACAP Open) |
| 4 | Depression | 0-21 | DASS-21 depression subscale | Papers 7, 8 |
| 5 | Activity | 0-500 | active calories (kcal) | Paper 6 (Fudolig 2025, npj Complexity) |
| 6 | NatureEngagement | 0-15 | perceived hours in nature/week | Papers 7, 8 |
| 7 | RHR | 45-100 | resting heart rate (bpm) | Paper 3 |
| 8 | HRV | 15-120 | RMSSD heart rate variability (ms) | Papers 3, 7 |
| 9 | ARR | 10-25 | average respiratory rate (breaths/min) | Papers 3, 7 |
| 10 | SocialJetlag | 0-3 | MSF_free - MSF_school (hours) | Paper 6 |
| 11 | SleepShape | 0-1 | fraction of nights in Cluster 1 | Paper 2 |
| 12 | WEMWBS | 14-70 | Warwick-Edinburgh Well-Being Scale | Paper 7 |
| 13 | DAC | 0-1 | Directed Attention Capacity | Paper 8 |

## Input Parameters (12D)

**Student/patient (6D):** age (18-25), gender (0=male, 1=female, 2=nonbinary), emotional_stability (1-7 Likert), trauma_load (0-5 ACE-like), mh_diagnosis (0/1), baseline_chronotype (2-7 MSF_free hours).

**Intervention (6D):** nature_rx (0-1), exercise_rx (0-1), therapy_rx (0-1), sleep_hygiene (0-1), caffeine_reduction (0-1), academic_load (0-1, where 0=light, 0.5=typical, 1.0=overloaded).

## ODE Coupling Structure

6-tier cascade computed in `derivatives()`:
1. **Sleep -> Stress** (Paper 3): Wearable biomarkers (TST, RHR, HRV, ARR) predict PSS via regression coefficients; within-person deviations 2.2x stronger than between-person; gender level shift (+2.956 PSS for nonmale)
2. **Anxiety Markov state** (Paper 4): GAD-7 threshold dynamics at >= 10; development rate modified by emotional stability (AOR=0.58), MH diagnosis (AOR=2.10), trauma (AOR=1.80), academic stressors (AOR=1.68); recovery with hysteresis (gets harder over semester)
3. **Nature -> Stress -> HRV mediation** (Paper 7): Nature engagement reduces PSS (-1.507/unit), PSS reduction improves HRV (-0.618 ms/PSS point), therapy reduces DASS stress and ARR independently
4. **Sleep debt & activity** (Paper 6): Weekday sleep debt (45 min/school night), weekend partial recovery; paradoxical activity compensation (less sleep = more active); school/weekday/gender/MH activity modifiers
5. **Chronotype & sleep shape** (Papers 2, 6): Chronotype determines social jetlag via weekday forcing (0.924h); sleep shape (cluster membership) gender-modulated -- female+MH and female+trauma shift toward Cluster 1
6. **Attention restoration** (Paper 8): DAC depleted by academic load (0.3/week), restored by perceived nature engagement (0.2/week); perceived nature (not GPS-measured) drives restoration

## Key Dynamics

- **GAD-7 bistability:** Threshold at 10 creates two dynamical regimes -- below-threshold (development risk) and above-threshold (recovery with hysteresis). Once anxious, recovery probability decays each week (RECOVERY_HYSTERESIS=0.92). ~30% of students cross threshold during a semester.
- **Sleep-stress vicious cycle:** Within-person deviations from a student's own baseline are 2.2x stronger than between-person differences (Paper 3). A student who normally sleeps 8h and drops to 6h is hit harder than a student who always sleeps 6h.
- **Attention depletion:** Academic load depletes DAC (Kaplan & Kaplan's directed attention). When DAC approaches 0, engagement quality degrades, reducing the effectiveness of nature interventions. This creates a burnout trap: the students who most need restoration are least able to benefit from it.
- **Spring break phase transition:** Week 8 removes all institutional constraints -- no school-day forcing, no sleep debt, no academic stressors. Sleep recovers, activity drops, stress temporarily abates. The week-long perturbation reveals which students bounce back and which have accumulated too much damage.
- **Gender-modulated sleep shape coupling:** Female students with MH diagnoses (SHAPE_FEMALE_MH_COEFF=0.3) or trauma history (SHAPE_FEMALE_TRAUMA_COEFF=0.2) shift toward the disrupted sleep phenotype (Cluster 1). Male students show weak/no coupling. Nonbinary students show trauma coupling only.
- **Perceived vs quantified nature paradox:** Perceived nature engagement predicts well-being improvement; GPS-measured green space exposure without subjective engagement does NOT -- and paradoxically predicts slightly higher depression (BETA_GPS_NATURE_DEP=+0.032). The engagement_quality gate (DAC * threshold function) operationalizes this.

## Conventions

- numpy-only -- no scipy, no sklearn
- Matplotlib Agg backend -- headless plotting to `output/`
- Float64 precision, deterministic simulations
- `NumpyEncoder` for JSON serialization of numpy types
- Grid snapping via `snap_param()` / `snap_all()`
- 8 student archetype seeds in `constants.STUDENT_ARCHETYPES`
- Real clinical units (hours, bpm, ms, PSS 0-40) not normalized
- RK4 integrator with daily timesteps (dt = 1/7 week), 105 steps per semester
- Semester calendar functions: `is_weekday()`, `is_school_day()`, `week_of_semester()`

## Sibling Projects

- **grief-simulator** -- bereavement ODE (structural analog: 11 state vars, 7-layer cascade, same Zimmerman protocol, same 4-pillar analytics)
- **how-to-live-much-longer** -- mitochondrial aging ODE (12D params, heteroplasmy cliff at ~70%, RK4 over 30 years)
- **stock-simulator** -- financial ODE (7D strategy params, margin-call cascade cliff at leverage 3.0)
- **zimmerman-toolkit** -- 14-module simulator interrogation (compatible via LEMURSSimulator and LEMURSBridge)
- **cramer-toolkit** -- scenario-based resilience analysis (compatible via LEMURSSimulator; kcramer_bridge.py provides 19 stress scenarios in 7 banks)

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
