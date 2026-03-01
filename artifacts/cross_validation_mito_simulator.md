# LEMURS Simulator vs. How-to-Live-Much-Longer: Cross-Validation Analysis

**Date:** 2026-02-21
**Scope:** Does the LEMURS 14D sleep-stress-anxiety ODE support, challenge, or falsify assumptions in the mitochondrial aging 8D ODE?

---

## Executive Summary

The LEMURS simulator **partially supports** the how-to-live-much-longer simulator's stress-sleep-inflammation pathway but **exposes five falsification-grade problems** in the mito model's assumptions. The strongest support comes from quantified sleep→stress coefficients that can replace the mito model's unvalidated placeholders. The strongest falsification comes from the LEMURS finding that within-person deviations are 2.2x stronger than between-person differences — a dynamical property the mito model entirely ignores.

| Verdict | Count | Severity |
|---------|-------|----------|
| **Supports** | 6 findings | Directional validation of stress-sleep-health cascade |
| **Falsifies** | 5 findings | Three are coefficient-level (fixable), two are structural (require model redesign) |
| **Gaps** | 4 findings | Missing bridge variables (inflammation, cortisol, cellular biomarkers) |

---

## Part I: What LEMURS Supports

### S1. Sleep Disruption Drives Stress (Quantified)

**Mito model assumption:** Poor sleep increases inflammation, which accelerates mitochondrial damage.
```
inflammation += (1.0 - sleep_quality) * 0.05    # parameter_resolver.py:169
```

**LEMURS provides:** Direct quantification of the sleep→stress pathway (Paper 3):
- TST → PSS: β = -0.877 per hour (p<0.001)
- Each hour of lost sleep = +0.877 PSS points = +38.3% odds of high stress
- This is the strongest wearable predictor of stress, stronger than RHR, HRV, or ARR

**Verdict:** SUPPORTS the direction. Sleep disruption unambiguously drives stress. The mito model's assumption that poor sleep → worse health outcomes is validated by real data from 525 students across 3,112 weekly observations.

### S2. HRV as Stress-Health Mediator

**Mito model assumption:** HRV reflects autonomic health; stress reduces HRV; low HRV indicates poor recovery capacity.

**LEMURS provides:** The full mediation chain (Paper 7):
```
Nature → PSS reduction (-1.507) → HRV improvement (-0.618 ms/PSS point) → +9.13 ms total
```
- PSS → HRV: β = -0.618 (p=0.024). Each PSS point reduction yields 0.618 ms RMSSD improvement
- HRV → PSS: β = -0.012 (p=0.035). Higher HRV independently predicts lower stress
- Nature intervention produces +9.13 ms HRV over 14 weeks (comparable to pharmacological effects)

**Verdict:** SUPPORTS. HRV mediates the stress→physiology pathway. The mito model's use of HRV as a proxy for autonomic and recovery health is validated. The bidirectional coupling (stress ↔ HRV) is real.

### S3. Behavioral Interventions Alter Physiological Trajectories

**Mito model assumption:** Non-pharmacological interventions (sleep hygiene, exercise) can slow mitochondrial damage by reducing inflammation and improving repair.

**LEMURS provides:** Nature intervention (cost: $2,940/year) produces:
- PSS reduction: -1.507 points (p=0.047)
- HRV improvement: +9.13 ms (p=0.037)
- WEMWBS well-being: +0.104/week (p<0.001)
- These effects are cumulative over 14 weeks (time-dependent accrual)

**Verdict:** SUPPORTS. Cheap behavioral interventions measurably change autonomic physiology over semester timescales. If the stress→inflammation→mitochondrial damage chain is real, then nature/exercise interventions should slow aging — which is exactly what the mito model predicts.

### S4. Gender Modulates Stress Vulnerability

**Mito model assumption:** Sex modifies mitochondrial vulnerability (genetics_module.py: female menopause affects NAD decline, APOE4 expression).

**LEMURS provides:** Gender modulates stress at multiple levels:
- PSS level shift: +2.956 for female/nonbinary (same biomarkers → more perceived stress)
- Sleep shape: Female + MH → Cluster 1 (disrupted phenotype, SHAPE_FEMALE_MH_COEFF = 0.3)
- Sleep shape: Female + trauma → additional shift (SHAPE_FEMALE_TRAUMA_COEFF = 0.2)
- Male sleep shape: No coupling (sex-asymmetric)
- Chronotype: Males +28.3 min later (BETA_MALE_CHRONOTYPE = +0.472 hrs)

**Verdict:** SUPPORTS directionally. Both models agree that sex/gender modulates health vulnerability through different mechanisms. The LEMURS finding that female students carry a +3 PSS baseline penalty suggests chronically elevated stress that, over decades, could compound into greater mitochondrial damage — aligning with the mito model's sex-dependent aging trajectories.

### S5. Anxiety Bistability Parallels the Heteroplasmy Cliff

**Mito model:** Heteroplasmy cliff at ~50% deletion het — a threshold crossing where ATP production collapses nonlinearly. Once past it, recovery is extremely difficult (bistability).

**LEMURS provides:** GAD-7 threshold at 10 with Markov dynamics (Paper 4):
- Below 10: development risk (12%/week base rate, modified by AORs)
- Above 10: recovery probability (65%/week BUT declining via hysteresis 0.92^week)
- 9% of students are chronically above threshold (trapped in anxious attractor)
- Risk factors compound multiplicatively (up to ~57x odds with all risk factors)

**Verdict:** SUPPORTS the structural analogy. Both systems exhibit threshold-crossing dynamics with hysteresis. The anxiety cliff (hard to recover once above GAD-7 ≥ 10 for extended periods) is a behavioral analog of the heteroplasmy cliff (hard to recover once deletion het > 50%). This validates the broader thesis that biological systems exhibit cliff phenomena across scales.

### S6. Cumulative Semester Effects (Irreversibility Signal)

**Mito model:** Mitochondrial damage accumulates irreversibly. Deletions have replication advantage; damage is a one-way ratchet.

**LEMURS provides:** Semester-long trends that resist recovery:
- PSS secular trend: +0.077/week (stress ramps up regardless of intervention)
- Recovery hysteresis: 0.92^week (anxiety recovery gets 8% harder each week)
- TST trend: -0.037/week (sleep declines across semester, p=0.037)
- ARR trend: +0.004/week (respiratory rate worsens, p=0.004)

**Verdict:** PARTIALLY SUPPORTS. The semester shows quasi-irreversible accumulation — stress and autonomic dysregulation worsen over 15 weeks despite homeostatic mechanisms. However, spring break fully resets many variables in 1 week (see F4 below), which challenges the irreversibility claim.

---

## Part II: What LEMURS Falsifies or Challenges

### F1. Within-Person Amplification Ignored (STRUCTURAL)

**Mito model assumption:** Sleep quality is a simple 0-1 parameter. A patient with sleep_quality=0.5 always gets the same mitophagy reduction regardless of their personal baseline.

**LEMURS falsifies:** Within-person deviations from a student's OWN baseline are **2.2x stronger** than between-person differences (Paper 3).
- A student who normally sleeps 8 hours and drops to 6 is hit MUCH harder than a student who always sleeps 6 hours
- TST deviation coefficient: -1.312 (within-person) vs -0.877 (between-person)
- This means the mito model's treatment of sleep as a static parameter fundamentally misrepresents how sleep disruption works

**Impact:** The mito model's `sleep_repair_factor = 1.0 - 0.7 * (1.0 - sleep_quality)` treats all patients the same at a given sleep level. A patient adapted to poor sleep should show less mitophagy impairment than a patient suddenly shifted to poor sleep. The model needs a **deviation-from-baseline** term, not an absolute-level term.

**Severity:** STRUCTURAL. Requires adding a personal baseline sleep parameter and computing disruption as deviation, not level.

### F2. Dose ≠ Bioavailability (STRUCTURAL)

**Mito model assumption:** Interventions have direct dose-response effects. Rapamycin dose → mitophagy rate. NAD supplement → NAD level. Sleep intervention → sleep quality. The relationship is linear and context-independent.

**LEMURS falsifies:** Paper 8 demonstrates that physical presence in nature (GPS-measured) without psychological engagement has **no benefit or is paradoxically harmful**:
- Perceived nature → depression: β = -0.066 (protective)
- GPS-measured nature → depression: β = +0.032 (**harmful**, paradoxical)
- The engagement quality gate (DAC) means burned-out students cannot benefit from nature even when prescribed

**Impact:** The mito model's `rapamycin_dose *= sleep_repair_factor` is a simple multiplicative gate. But the LEMURS finding suggests intervention efficacy should depend on the patient's **capacity to engage** with the intervention. A severely depleted patient (analogous to DAC→0) may not be able to follow an exercise regimen, maintain sleep hygiene, or properly metabolize supplements. The mito model has no "engagement capacity" state variable.

**Severity:** STRUCTURAL. Suggests the mito model needs an "adherence/engagement capacity" variable that gates all behavioral intervention efficacy. The DAC concept from LEMURS could be imported directly.

### F3. Three Unvalidated Sleep Coefficients (COEFFICIENT-LEVEL)

**Mito model uses:** Three sleep-related constants explicitly flagged as "literature-approximated, awaiting LEMURS data":

| Constant | Mito Value | What LEMURS Actually Says |
|----------|-----------|--------------------------|
| `SLEEP_DISRUPTION_IMPACT` | 0.5 → 0.7 (audit changed it) | LEMURS measures TST→PSS (β=-0.877/hr) and HRV→PSS (β=-0.012/ms), but NOT sleep→mitophagy. The coefficient cannot be derived from LEMURS data. |
| `ALCOHOL_SLEEP_DISRUPTION` | 0.4 | LEMURS surveys likely capture alcohol but **no published coefficient exists**. The value 0.4 is a guess. |
| `(1-sleep_quality) * 0.05` inflammation | 0.05 | LEMURS measures sleep→stress (PSS), NOT sleep→inflammation (IL-6, CRP). The 0.05 value has no empirical basis in LEMURS. |

**Impact:** The mito model's CLAUDE.md says "ASK DODDS & DANFORTH" for these values, implying they are derivable from LEMURS data. But LEMURS measures stress (PSS-10) and autonomic markers (HRV, RHR, ARR), not inflammation biomarkers or mitochondrial function. The leap from "sleep disruption → higher PSS" to "sleep disruption → higher inflammation → faster mitochondrial damage" requires a bridge variable (cortisol, IL-6, TNF-α) that **neither simulator measures**.

**Severity:** COEFFICIENT-LEVEL. The direction is correct (poor sleep → worse health), but the quantitative values cannot be derived from LEMURS data. They need either: (a) different literature sources with actual sleep→inflammation coefficients, or (b) honest acknowledgment that these are modeling assumptions, not LEMURS-derived.

### F4. Spring Break Recovery Contradicts Irreversibility (STRUCTURAL)

**Mito model:** Mitochondrial damage is irreversible. Deletions accumulate monotonically. The heteroplasmy cliff is a one-way trip.

**LEMURS shows:** Spring break (1 week, week 8) produces near-complete recovery of several state variables:
- TST recovers to unconstrained baseline (7.5 hrs) within days
- Activity drops 13.7% (removal of institutional forcing)
- PSS drops when academic stressors removed
- Social jetlag resolves when wake-time constraints lifted

The biological sanity test `test_spring_break_tst_increases` confirms: week-8 TST > surrounding weeks.

**Impact:** The mito model assumes stress-driven inflammation drives irreversible mitochondrial damage. But if stress, sleep, and autonomic markers fully recover within 1 week of removing stressors, then the "damage accumulation" pathway is NOT operating on the timescale the mito model assumes — at least for 18-25 year old college students. The reversibility of semester effects challenges the claim that chronic stress causes permanent cellular damage.

**Counterargument:** College students are young (18-25) with high repair capacity. The mito model targets 20-90 year olds, and older patients may NOT recover as quickly. The irreversibility may emerge only after decades of accumulated damage. But LEMURS provides no evidence for this — it only shows that in young adults, autonomic and psychological stress effects are rapidly reversible.

**Severity:** STRUCTURAL for the bridge between models, though not for the mito model's core dynamics (which model aging over 30 years, not semesters).

### F5. Grief Bridge Coupling Coefficients Are Simulation-Calibrated, Not Empirical

**Mito model's grief bridge uses:**

| Grief → Mito Coupling | Value | Source |
|------------------------|-------|--------|
| SNS → ROS | 0.05/step | "Simulation-calibrated" |
| Cortisol → metabolic demand | 0.2 additive | "Simulation-calibrated" |
| Inflammation → inflammation_level | 0.4 additive | "Simulation-calibrated" |
| Sleep disruption → inflammation | 0.15 additive | "Simulation-calibrated" |
| CVD rate → genetic vulnerability | 0.3 multiplicative | "Simulation-calibrated" |

**LEMURS reveals:** None of these coefficients can be validated by LEMURS data, because LEMURS measures behavioral/psychological/autonomic endpoints, NOT cellular/mitochondrial endpoints. The grief bridge has the same validation gap as the sleep coefficients — it assumes a causal chain (stress → inflammation → mitochondrial damage) where the intermediate variable (inflammation) is never measured.

**LEMURS does provide** the upstream part of the chain: stress → autonomic dysregulation (RHR, HRV, ARR changes) with quantified coefficients. But the downstream part (autonomic dysregulation → inflammation → cellular damage) remains empirically ungrounded.

**Severity:** COEFFICIENT-LEVEL. The grief bridge is structurally plausible but quantitatively arbitrary. The same critique applies to a potential LEMURS→mito bridge: the sleep→stress→autonomic path is validated, but the autonomic→cellular path requires different data sources entirely.

---

## Part III: Gaps (Neither Supports Nor Falsifies)

### G1. Missing Bridge Variable: Inflammation

The critical variable connecting LEMURS dynamics to mitochondrial damage is **systemic inflammation** (IL-6, TNF-α, CRP, NF-κB). Neither simulator measures it directly:
- LEMURS: measures stress (PSS), autonomic markers (RHR, HRV, ARR), no blood biomarkers
- Mito model: uses inflammation_level as a parameter, but values are assumed not measured

The causal chain is:
```
LEMURS domain                    Bridge (unmeasured)           Mito domain
Sleep loss → ↑PSS → ↑RHR/↓HRV → [↑cortisol → ↑IL-6/TNF-α] → ↑ROS → ↑deletions → ↓ATP
```

The brackets mark the gap. Both simulators model their respective sides accurately, but the connection between them is assumed, not measured.

### G2. Timescale Mismatch

- LEMURS: 15-week semester, daily timesteps, effects largely reversible
- Mito: 30-year lifespan, 3.65-day timesteps, effects largely irreversible

How do semester-scale stress effects translate to decade-scale mitochondrial damage? Possibilities:
1. **Cumulative micro-damage:** Each semester of poor sleep adds a small irreversible increment. Over 40 semesters (20 years), it compounds. But LEMURS shows semester effects are reversible in 1 week (spring break), arguing against this.
2. **Threshold-crossing:** Occasional extreme stress episodes (like grief, which the mito model already handles) cause lasting damage. Semester stress is below the damage threshold. This is plausible but means LEMURS-level stress is mostly irrelevant to mitochondrial aging.
3. **Allostatic load:** The cumulative biological cost of chronic stress adaptation. Not measured by either simulator. Would require a slow-accumulating variable in the LEMURS model that doesn't reset during breaks.

### G3. Demographic Mismatch

- LEMURS: 18-25 year old college students, 65% female, 97% aged 18-19
- Mito model: 20-90 year old patients, variable demographics

Sleep-stress coupling coefficients from college students may not generalize to elderly patients because:
- Young adults have high mitochondrial repair capacity
- Elderly patients have accumulated baseline damage (higher het) making them more vulnerable
- Sleep architecture changes with age (reduced slow-wave sleep, earlier chronotype)
- HRV declines with age (~1 ms/year after 30), changing the baseline

### G4. No Mitochondrial Biomarkers in LEMURS

LEMURS measures no variables that directly index mitochondrial function:
- No peripheral blood mtDNA heteroplasmy
- No ATP production capacity (could use exercise tolerance as proxy)
- No NAD+ levels
- No senescence markers (p16, SA-β-gal)
- No oxidative stress markers (8-OHdG, MDA)

This means the LEMURS simulator cannot directly validate or falsify the mito model's core dynamics. It can only validate the upstream behavioral/autonomic pathways that feed into the mito model's assumptions.

---

## Part IV: Actionable Recommendations

### R1. Replace Placeholder Sleep Coefficients (Priority: HIGH)

The mito model's three unvalidated sleep constants should be handled as follows:

| Constant | Current | Recommendation |
|----------|---------|----------------|
| `SLEEP_DISRUPTION_IMPACT` (0.7) | Falsely attributed to LEMURS | Derive from sleep→HRV literature (e.g., Irwin 2015 meta-analysis on sleep deprivation → inflammation). LEMURS provides TST→PSS (β=-0.877) but NOT TST→mitophagy. Rename source citation. |
| `ALCOHOL_SLEEP_DISRUPTION` (0.4) | Unvalidated guess | Derive from Ebrahim et al. 2013 (alcohol + polysomnography) or similar. Not derivable from LEMURS. |
| `inflammation += 0.05` | Unvalidated guess | Replace 0.05 with a value derived from Irwin 2016 (sleep disturbance → CRP/IL-6 meta-analysis). LEMURS provides no inflammation data. |

### R2. Add Within-Person Deviation Dynamics (Priority: HIGH)

The mito model should distinguish between **chronic baseline** and **acute deviation** for sleep quality:
```python
# Current (wrong):
sleep_repair_factor = 1.0 - 0.7 * (1.0 - sleep_quality)

# Proposed (incorporating LEMURS F1 finding):
deviation = sleep_quality - patient_sleep_baseline
within_person_effect = 2.2 * deviation  # LEMURS amplification factor
sleep_repair_factor = 1.0 - 0.7 * (1.0 - sleep_quality) - 0.3 * within_person_effect
```

This captures the LEMURS finding that sudden changes matter more than steady states.

### R3. Add Engagement Capacity Gate (Priority: MEDIUM)

Import the DAC (Directed Attention Capacity) concept from LEMURS into the mito model:
```python
# Before applying behavioral interventions:
engagement_capacity = max(0.1, 1.0 - fatigue_factor)  # Depleted patients can't engage
exercise_effective = exercise_level * engagement_capacity
sleep_hygiene_effective = sleep_intervention * engagement_capacity
```

This addresses the dose ≠ bioavailability finding (F2) and prevents the mito model from assuming perfect adherence.

### R4. Build the LEMURS→Mito Bridge (Priority: FUTURE)

Following the grief→mito bridge pattern, a LEMURS→mito bridge would:
1. Run the LEMURS simulator for 15 weeks (1 semester)
2. Extract terminal PSS, HRV, sleep debt, GAD-7 state
3. Map these to mito disturbance channels:
   - PSS → inflammation_level (via assumed cortisol mediation)
   - HRV deficit → genetic_vulnerability modifier (reduced recovery capacity)
   - Cumulative sleep debt → sleep_quality parameter
   - GAD-7 above threshold → chronic stress flag → sustained inflammation
4. Apply as a semester-resolution perturbation to the 30-year mito trajectory

This bridge would model the hypothesis: "each semester of college stress adds a small increment of mitochondrial wear." But the coupling coefficients would be modeling assumptions, not LEMURS-derived.

### R5. Acknowledge the Validation Gap Honestly (Priority: IMMEDIATE)

The mito model's CLAUDE.md should be updated:
- Change "ASK DODDS & DANFORTH" to acknowledge that LEMURS cannot provide mitochondrial coupling coefficients — only upstream stress/sleep coefficients
- Rename the three sleep constants from "LEMURS-attributed" to "literature-approximated; LEMURS validates upstream direction but not downstream magnitude"
- Add a note that the grief bridge coefficients are simulation-calibrated, not empirically derived

---

## Part V: The Structural Parallel

Both simulators share deep architectural similarities that transcend their specific domains:

| Feature | LEMURS | Mito |
|---------|--------|------|
| Cliff phenomenon | GAD-7 ≥ 10 (anxiety threshold) | Deletion het ≥ 0.50 (energy collapse) |
| Hysteresis | Recovery gets harder (0.92^week) | Damaged copies replicate faster (1.10x advantage) |
| Irreversibility signal | PSS secular trend (+0.077/week) | Deletion accumulation (monotonic) |
| Gender modulation | Female +2.956 PSS, sleep shape coupling | Sex-dependent NAD decline, APOE4 expression |
| Intervention mediation | Nature → PSS → HRV (indirect) | Rapamycin → mitophagy → heteroplasmy (indirect) |
| Engagement gate | DAC gates nature efficacy | ATP gates mitophagy (autophagy requires energy) |
| Environmental forcing | Semester calendar (weekday/weekend/break) | Age-dependent deletion rate (young/old transition) |
| Mean reversion | PSS, TST, HRV revert to baselines | Healthy copy homeostasis toward total ≈ 1.0 |
| Slow vs fast variables | Trait (RHR, HRV) vs state (PSS, TST) | Trait (deletions, senescence) vs state (ROS, ATP) |

The deepest parallel: both systems have a **fast observable surface** (PSS/GAD-7 for LEMURS; ATP/ROS for mito) driven by **slow hidden accumulators** (sleep shape/chronotype/semester trends for LEMURS; deletion heteroplasmy/senescent fraction for mito). The fast surface can recover quickly (spring break, acute intervention), but the slow accumulator determines long-term fate.

This structural parallel is the strongest form of support — not that LEMURS validates specific mito coefficients, but that both systems exhibit the same dynamical grammar: threshold-crossing, hysteresis, timescale separation, irreversible accumulation gated by repair capacity.

---

## Summary Table

| # | Finding | Type | Severity | Action |
|---|---------|------|----------|--------|
| S1 | Sleep→stress quantified (β=-0.877/hr) | Support | — | Use to anchor mito upstream |
| S2 | HRV mediates stress→physiology | Support | — | Validates mito bridge concept |
| S3 | Behavioral interventions alter autonomic trajectories | Support | — | Validates non-pharma intervention modeling |
| S4 | Gender modulates vulnerability | Support | — | Validates sex-dependent aging paths |
| S5 | Anxiety bistability parallels het cliff | Support | — | Validates cliff dynamical grammar |
| S6 | Semester trends show quasi-irreversibility | Support | Partial | Supports accumulation concept (but see F4) |
| F1 | Within-person 2.2x amplification ignored | **Falsification** | **Structural** | Add deviation-from-baseline to mito sleep model |
| F2 | Dose ≠ bioavailability (DAC gate) | **Falsification** | **Structural** | Add engagement capacity to mito model |
| F3 | Three sleep coefficients unvalidated | **Falsification** | Coefficient | Source from sleep-inflammation literature, not LEMURS |
| F4 | Spring break recovery contradicts irreversibility | **Falsification** | Structural (bridge) | Limits applicability of semester→lifetime extrapolation |
| F5 | Grief bridge coefficients are arbitrary | **Falsification** | Coefficient | Acknowledge calibration status |
| G1 | Inflammation never measured | Gap | Critical | Need blood biomarker data for bridge |
| G2 | 15-week vs 30-year timescale mismatch | Gap | Major | Allostatic load concept needed |
| G3 | College students vs aging population | Gap | Moderate | Age-dependent coefficient scaling needed |
| G4 | No mitochondrial biomarkers in LEMURS | Gap | Fundamental | Different study design needed |
