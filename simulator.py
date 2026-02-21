"""LEMURS college-student biopsychosocial ODE simulator.

This simulates a college student's biopsychosocial trajectory through a
15-week semester. The model tracks 14 coupled state variables -- sleep,
stress, anxiety, depression, physical activity, nature engagement,
cardiac physiology, chronotype alignment, sleep architecture, well-being,
and directed attention capacity -- and integrates them forward in time
using the Runge-Kutta 4th order method (RK4) with daily timesteps.

The dynamical system is organized into 6 tiers of coupling, each grounded
in empirical coefficients from the 9 published LEMURS papers:

    Tier 1: Sleep -> Stress (Paper 3, strongest empirical support)
    Tier 2: Anxiety Markov dynamics (Paper 4, threshold-crossing)
    Tier 3: Nature -> Stress -> HRV mediation (Paper 7, RCT)
    Tier 4: Sleep debt & activity coupling (Paper 6, circadian)
    Tier 5: Chronotype & sleep shape (Papers 2, 6)
    Tier 6: Attention restoration (Paper 8, Kaplan & Kaplan)

The simulator accepts 12 input parameters (6 student characteristics +
6 intervention dials) and produces a 106-step trajectory (day 0 through
day 105) of all 14 state variables.

References:
    Bloomfield et al. (2024). Sleep phenotyping via clustering. (Paper 2)
    Bloomfield et al. (2024). Wearable biomarkers predict stress. (Paper 3)
    Bloomfield et al. (2024). Anxiety prevalence & persistence. (Paper 4)
    Fudolig et al. (2025). Sleep debt, SJL, physical activity. (Paper 6)
    Bloomfield et al. (2025). Nature engagement as stress mediator. (Paper 7)
    Bloomfield et al. (2025). Attention restoration & perceived nature. (Paper 8)
"""
from __future__ import annotations

import numpy as np

from constants import (
    # Simulation config
    SIM_WEEKS, DT, N_STEPS, N_STATES,
    # Defaults
    DEFAULT_INTERVENTION, DEFAULT_PATIENT,
    # State indices
    _TST, _SQ, _PSS, _GAD7, _DEP, _ACT, _NAT,
    _RHR, _HRV, _ARR, _SJL, _SHAPE, _WB, _DAC,
    # State bounds
    _LOWER, _UPPER,
    # Tier 1: Sleep -> Stress (Paper 3)
    PSS_SECULAR_TREND,
    BETA_TST_PSS, BETA_RHR_PSS, BETA_HRV_PSS, BETA_ARR_PSS,
    BETA_GENDER_PSS,
    TST_REFERENCE, RHR_REFERENCE, HRV_REFERENCE, ARR_REFERENCE,
    PSS_REVERSION,
    # Tier 2: Anxiety Markov (Paper 4)
    GAD7_THRESHOLD,
    GAD7_DEVELOPMENT_RATE, GAD7_RECOVERY_RATE,
    AOR_EMOTIONAL_STABILITY, AOR_MH_DIAGNOSIS, AOR_TRAUMA, AOR_ACADEMIC_STRESSOR,
    RECOVERY_HYSTERESIS, RECOVERY_STABILITY_AOR, RECOVERY_STRESSOR_AOR,
    GAD7_ANXIOUS_ATTRACTOR, GAD7_RECOVERY_ATTRACTOR,
    # Tier 3: Nature mediation (Paper 7)
    BETA_NATURE_PSS, BETA_PSS_HRV,
    NATURE_HRV_DIRECT, BETA_NATURE_WEMWBS,
    BETA_THERAPY_STRESS, BETA_THERAPY_ARR,
    # Tier 4: Sleep debt & activity (Paper 6)
    SLEEP_DEBT_WEEKDAY,
    BETA_SJL_ACTIVITY, BETA_TST_ACTIVITY,
    BETA_WEEKDAY_ACTIVITY, BETA_SCHOOL_ACTIVITY,
    BETA_MALE_ACTIVITY, BETA_MH_ACTIVITY,
    ACTIVITY_BASELINE, TST_BASELINE, TST_REVERSION, ACTIVITY_REVERSION,
    # Tier 5: Chronotype & sleep shape (Papers 2, 6)
    SJL_WEEKDAY_FORCING, SJL_REVERSION, SHAPE_REVERSION,
    SHAPE_FEMALE_MH_COEFF, SHAPE_FEMALE_TRAUMA_COEFF, SHAPE_NB_TRAUMA_COEFF,
    BETA_MH_CHRONOTYPE,
    # Tier 6: Attention restoration (Paper 8)
    DAC_DEPLETION, DAC_RESTORATION, DAC_REVERSION, DAC_BASELINE,
    # Cross-tier coupling
    DEPRESSION_PSS_COUPLING, DEPRESSION_GAD7_COUPLING, DEPRESSION_SLEEP_COUPLING,
    DEPRESSION_REVERSION,
    WEMWBS_BASELINE, WEMWBS_REVERSION,
    RHR_REVERSION, HRV_REVERSION_TRAIT, ARR_REVERSION,
    SLEEP_QUALITY_TST_COUPLING, SLEEP_QUALITY_BASELINE, SLEEP_QUALITY_REVERSION,
    NATURE_ENGAGEMENT_REVERSION,
    # Calendar
    day_of_week, is_weekday, is_school_day, week_of_semester,
    # Archetypes
    STUDENT_ARCHETYPES,
)


# ── Semester context ─────────────────────────────────────────────────────────

def _semester_context(t: float) -> dict:
    """Convert simulation time *t* (in weeks) to semester calendar context.

    Returns a dict with day number, day-of-week, weekday flag, school-day
    flag, and week number.  These drive the weekday/weekend sleep-debt
    oscillation and the spring-break perturbation.
    """
    day = int(t * 7.0)  # convert weeks to days
    return {
        "day": day,
        "day_of_week": day_of_week(day),
        "is_weekday": is_weekday(day),
        "is_school": is_school_day(day),
        "week": week_of_semester(day),
    }


# ── Initial state ────────────────────────────────────────────────────────────

def initial_state(patient: dict[str, float] | None = None) -> np.ndarray:
    """Set up the 14-element state vector at the start of the semester.

    This is Week 0, Day 0 -- the student has just arrived on campus.  The
    initial conditions reflect a student transitioning from summer break
    into the academic term: sleep is close to baseline (with a small debt
    from new-semester adjustment), stress is at the cohort mean, and the
    biometric variables match the Paper 2/3/7 baselines.

    The strength of vulnerability depends on:
    - mh_diagnosis: prior mental health diagnosis elevates GAD7 and
      depression, shifts RHR up and HRV down (Paper 2 cluster effects).
    - gender: males have higher baseline activity (Paper 6).
    - baseline_chronotype: late chronotypes start with more social jetlag.
    - nature_rx intervention: sets initial nature engagement level.
    """
    p = {**DEFAULT_PATIENT, **(patient or {})}

    gender = p.get("gender", 1.0)
    is_male = 1.0 if gender < 0.5 else 0.0
    mh_diagnosis = p.get("mh_diagnosis", 0.0)
    baseline_chronotype = p.get("baseline_chronotype", 4.5)

    # Allow reading nature_rx from a merged patient+intervention dict,
    # or default to 0.0 (no nature prescription).
    nature_rx = p.get("nature_rx", 0.0)

    state = np.zeros(N_STATES, dtype=np.float64)

    # ── Sleep ──
    # School just started; slight debt from schedule adjustment.
    state[_TST] = TST_BASELINE - SLEEP_DEBT_WEEKDAY * 0.5  # ~7.125 hrs

    # ── Sleep quality ──
    state[_SQ] = SLEEP_QUALITY_BASELINE  # 75.0

    # ── Perceived stress ──
    # Paper 7 control group mean baseline.
    state[_PSS] = 16.84

    # ── Anxiety ──
    # Paper 4 cohort mean; elevated if prior MH diagnosis.
    state[_GAD7] = 7.53 * (1.0 + 0.3 * mh_diagnosis)

    # ── Depression ──
    # Paper 7 baseline; slightly elevated with MH history.
    state[_DEP] = 8.42 * (1.0 + 0.2 * mh_diagnosis)

    # ── Physical activity ──
    # Males are 10.5% more active (Paper 6).
    state[_ACT] = ACTIVITY_BASELINE * (1.0 + BETA_MALE_ACTIVITY * is_male)

    # ── Nature engagement ──
    # Scales with intervention prescription (0-6 hrs/week).
    state[_NAT] = nature_rx * 6.0

    # ── Resting heart rate ──
    # Cluster 1 mean from Paper 2; MH history raises baseline.
    state[_RHR] = 63.0 + 2.0 * mh_diagnosis

    # ── Heart rate variability ──
    # Paper 2: C1=63.32 ms, C2=67.18 ms; MH shifts downward.
    state[_HRV] = 65.0 - 4.0 * mh_diagnosis

    # ── Average respiratory rate ──
    # Paper 1 baseline.
    state[_ARR] = 15.6

    # ── Social jetlag ──
    # Late chronotypes have more SJL.
    state[_SJL] = max(0.0, (baseline_chronotype - 4.0) * 0.5)

    # ── Sleep shape (cluster membership) ──
    # 64% Cluster 1 baseline; shifts higher with MH history.
    state[_SHAPE] = 0.64 + 0.1 * mh_diagnosis

    # ── Well-being ──
    # Paper 7 control group baseline.
    state[_WB] = WEMWBS_BASELINE  # 46.66

    # ── Directed attention capacity ──
    state[_DAC] = DAC_BASELINE  # 0.8

    return state


# ── Derivatives ──────────────────────────────────────────────────────────────

def derivatives(
    state: np.ndarray,
    t: float,
    intervention: dict[str, float],
    patient: dict[str, float],
) -> np.ndarray:
    """How fast is every state variable changing right now?

    This function is the biological core of the LEMURS model.  Given the
    current state of all 14 variables, it computes how fast each one is
    rising or falling at this instant.  The six tiers of coupling run in
    order, each feeding into the next:

        Tier 1: Sleep -> Stress (Paper 3)
        Tier 2: Anxiety Markov state (Paper 4)
        Tier 3: Nature -> Stress -> HRV mediation (Paper 7)
        Tier 4: Sleep debt & activity coupling (Paper 6)
        Tier 5: Chronotype & sleep shape (Papers 2, 6)
        Tier 6: Attention restoration (Paper 8)
        Cross-tier: Depression, RHR, well-being

    The function is called hundreds of times during a simulation -- once
    for each day, multiple times per day for RK4 integration accuracy.
    All /7.0 divisions convert weekly rates to daily rates because
    dt = 1/7 week = 1 day.
    """
    d = np.zeros(N_STATES, dtype=np.float64)

    # ── Clip state to valid ranges ──
    TST = np.clip(state[_TST], 4.0, 12.0)
    SQ = np.clip(state[_SQ], 0.0, 100.0)
    PSS = np.clip(state[_PSS], 0.0, 40.0)
    GAD7 = np.clip(state[_GAD7], 0.0, 21.0)
    Depression = np.clip(state[_DEP], 0.0, 21.0)
    Activity = np.clip(state[_ACT], 0.0, 500.0)
    NatureEng = np.clip(state[_NAT], 0.0, 15.0)
    RHR = np.clip(state[_RHR], 45.0, 100.0)
    HRV = np.clip(state[_HRV], 15.0, 120.0)
    ARR = np.clip(state[_ARR], 10.0, 25.0)
    SJL = np.clip(state[_SJL], 0.0, 3.0)
    SleepShape = np.clip(state[_SHAPE], 0.0, 1.0)
    WEMWBS = np.clip(state[_WB], 14.0, 70.0)
    DAC = np.clip(state[_DAC], 0.0, 1.0)

    # ── Read intervention and patient params ──
    ctx = _semester_context(t)
    week = ctx["week"]
    is_wd = ctx["is_weekday"]
    is_school = ctx["is_school"]

    gender = patient.get("gender", 1.0)
    is_male = 1.0 if gender < 0.5 else 0.0
    is_nonmale = 1.0 - is_male
    is_nonbinary = 1.0 if gender > 1.5 else 0.0
    emotional_stability = patient.get("emotional_stability", 4.5)
    trauma_load = patient.get("trauma_load", 1.0)
    mh_diagnosis = patient.get("mh_diagnosis", 0.0)
    baseline_chronotype = patient.get("baseline_chronotype", 4.5)

    nature_rx = intervention.get("nature_rx", 0.0)
    exercise_rx = intervention.get("exercise_rx", 0.2)
    therapy_rx = intervention.get("therapy_rx", 0.0)
    sleep_hygiene = intervention.get("sleep_hygiene", 0.3)
    caffeine_reduction = intervention.get("caffeine_reduction", 0.0)
    academic_load = intervention.get("academic_load", 0.5)

    # ═══════════════════════════════════════════════════════════
    # TIER 1: SLEEP -> STRESS (Paper 3, strongest empirical support)
    # ═══════════════════════════════════════════════════════════
    # PSS driven by biometric deviations from reference values.
    # Within-person deviations are 2.2x stronger than absolute levels.

    pss_drive = (
        PSS_SECULAR_TREND / 7.0                           # +0.077/week -> daily
        + BETA_TST_PSS * (TST - TST_REFERENCE) / 7.0     # sleep protective
        + BETA_RHR_PSS * (RHR - RHR_REFERENCE) / 7.0     # high RHR harmful
        + BETA_HRV_PSS * (HRV - HRV_REFERENCE) / 7.0     # HRV protective
        + BETA_ARR_PSS * (ARR - ARR_REFERENCE) / 7.0      # tachypnea harmful
    )
    d[_PSS] = pss_drive - PSS_REVERSION * (PSS - 16.84) / 7.0
    # Note: gender effect is baked into initial_state, not derivatives

    # ═══════════════════════════════════════════════════════════
    # TIER 2: ANXIETY MARKOV STATE (Paper 4)
    # ═══════════════════════════════════════════════════════════
    # GAD-7 has threshold-crossing dynamics at >= 10.
    # Below threshold: stochastic risk of developing anxiety.
    # Above threshold: recovery with hysteresis (gets harder over semester).

    academic_stressor = 1.0 if academic_load > 0.6 else 0.0
    has_trauma = 1.0 if trauma_load >= 2.0 else 0.0

    if GAD7 < GAD7_THRESHOLD:
        # Development: push toward anxious state
        dev_rate = GAD7_DEVELOPMENT_RATE / 7.0  # daily rate
        modifier = (AOR_EMOTIONAL_STABILITY ** (-emotional_stability / 7.0)
                    * AOR_MH_DIAGNOSIS ** mh_diagnosis
                    * AOR_TRAUMA ** has_trauma
                    * AOR_ACADEMIC_STRESSOR ** academic_stressor)
        d[_GAD7] = dev_rate * modifier * (GAD7_ANXIOUS_ATTRACTOR - GAD7)
    else:
        # Recovery: pull toward healthy state (with hysteresis)
        semester_frac = min(week / 14.0, 1.0)
        rec_rate = GAD7_RECOVERY_RATE / 7.0
        hysteresis = RECOVERY_HYSTERESIS ** semester_frac
        modifier = (RECOVERY_STABILITY_AOR ** (-emotional_stability / 7.0)
                    * RECOVERY_STRESSOR_AOR ** academic_stressor)
        d[_GAD7] = rec_rate * hysteresis * modifier * (GAD7_RECOVERY_ATTRACTOR - GAD7)

    # ═══════════════════════════════════════════════════════════
    # TIER 3: NATURE -> STRESS -> HRV MEDIATION (Paper 7)
    # ═══════════════════════════════════════════════════════════
    # Nature intervention effect accrues over time (interaction with week).
    # Mediation: nature -> PSS reduction -> HRV improvement.

    engagement_quality = DAC * (
        1.0 if NatureEng > 0.5
        else NatureEng / 0.5 if NatureEng > 0
        else 0.0
    )
    time_factor = min(week / 14.0, 1.0)  # effect accrues over semester

    nature_pss_effect = (
        nature_rx * engagement_quality * BETA_NATURE_PSS / 7.0 * time_factor
    )
    d[_PSS] += nature_pss_effect

    # HRV: direct nature effect + mediation via PSS
    d[_HRV] = (
        NATURE_HRV_DIRECT * nature_rx / (14.0 * 7.0)       # +9.13 ms over 14 weeks
        + BETA_PSS_HRV * d[_PSS] / 40.0                    # mediation: PSS reduction -> HRV
        + HRV_REVERSION_TRAIT * (65.0 - HRV) / 7.0         # trait reversion
    )

    # Therapy effects
    d[_PSS] += therapy_rx * BETA_THERAPY_STRESS / 7.0 * time_factor
    d[_ARR] = (
        BETA_THERAPY_ARR * therapy_rx / 7.0                 # therapy reduces ARR
        + ARR_REVERSION * (15.6 - ARR) / 7.0               # trait reversion
    )

    # Well-being
    d[_WB] = (
        BETA_NATURE_WEMWBS * nature_rx / 7.0               # nature boosts well-being
        + WEMWBS_REVERSION * (WEMWBS_BASELINE - WEMWBS) / 7.0  # mean-reversion
        + 0.05 * exercise_rx / 7.0                          # exercise helps
        - 0.1 * (PSS / 40.0) / 7.0                         # stress hurts
    )

    # ═══════════════════════════════════════════════════════════
    # TIER 4: SLEEP DEBT & ACTIVITY COUPLING (Paper 6)
    # ═══════════════════════════════════════════════════════════
    # Weekday/weekend modulation.
    # School weeks: chronic 37-55 min/night sleep debt.
    # Weekends: partial recovery.

    sleep_debt = SLEEP_DEBT_WEEKDAY if (is_wd and is_school) else 0.0
    stress_sleep_disruption = 0.055 * max(PSS - 15.0, 0.0) / 40.0

    d[_TST] = (
        TST_REVERSION * (TST_BASELINE - TST) / 7.0         # recovery toward baseline
        - sleep_debt / 7.0                                  # school-day sleep loss
        - stress_sleep_disruption / 7.0                     # stress disrupts sleep
        + sleep_hygiene * 0.3 / 7.0                         # hygiene helps
        + caffeine_reduction * 0.1 / 7.0                    # caffeine reduction helps
    )

    # Activity
    activity_mods = (
        BETA_SJL_ACTIVITY * SJL
        + BETA_TST_ACTIVITY * TST
        + BETA_WEEKDAY_ACTIVITY * (1.0 if is_wd else 0.0)
        + BETA_SCHOOL_ACTIVITY * (1.0 if is_school else 0.0)
        + BETA_MALE_ACTIVITY * is_male
        + BETA_MH_ACTIVITY * mh_diagnosis
        + 0.1 * exercise_rx
    )
    d[_ACT] = (
        ACTIVITY_BASELINE * activity_mods / 7.0
        + ACTIVITY_REVERSION * (ACTIVITY_BASELINE - Activity) / 7.0
    )

    # Sleep quality tracks TST
    sq_target = SLEEP_QUALITY_BASELINE + SLEEP_QUALITY_TST_COUPLING * (TST - TST_BASELINE)
    d[_SQ] = SLEEP_QUALITY_REVERSION * (sq_target - SQ) / 7.0

    # ═══════════════════════════════════════════════════════════
    # TIER 5: CHRONOTYPE & SLEEP SHAPE (Papers 2, 6)
    # ═══════════════════════════════════════════════════════════

    sjl_target = SJL_WEEKDAY_FORCING * (1.0 if is_wd else 0.0)
    sjl_chrono = max(0.0, (baseline_chronotype - 4.0) * 0.2)
    sjl_mh = BETA_MH_CHRONOTYPE * mh_diagnosis / 7.0
    d[_SJL] = SJL_REVERSION * (sjl_target + sjl_chrono + sjl_mh - SJL) / 7.0

    # Sleep shape: gender-modulated coupling
    if is_male < 0.5 and is_nonbinary < 0.5:  # female
        shape_target = (
            0.64
            + SHAPE_FEMALE_MH_COEFF * mh_diagnosis
            + SHAPE_FEMALE_TRAUMA_COEFF * has_trauma
        )
    elif is_nonbinary > 0.5:  # non-binary
        shape_target = 0.64 + SHAPE_NB_TRAUMA_COEFF * has_trauma
    else:  # male
        shape_target = 0.64  # weak/no coupling
    d[_SHAPE] = SHAPE_REVERSION * (shape_target - SleepShape) / 7.0

    # ═══════════════════════════════════════════════════════════
    # TIER 6: ATTENTION RESTORATION (Paper 8)
    # ═══════════════════════════════════════════════════════════

    d[_DAC] = (
        -DAC_DEPLETION * academic_load / 7.0                # depletion
        + DAC_RESTORATION * NatureEng * engagement_quality / 7.0  # restoration
        + DAC_REVERSION * (DAC_BASELINE - DAC) / 7.0        # natural recovery
    )

    # Nature engagement dynamics
    nature_target = nature_rx * 6.0  # 0-6 hrs/week for nature intervention
    d[_NAT] = NATURE_ENGAGEMENT_REVERSION * (nature_target - NatureEng) / 7.0

    # ═══════════════════════════════════════════════════════════
    # CROSS-TIER: Depression & RHR
    # ═══════════════════════════════════════════════════════════

    # Depression driven by stress + anxiety + poor sleep
    dep_drive = (
        DEPRESSION_PSS_COUPLING * (PSS - 16.84) / 40.0
        + DEPRESSION_GAD7_COUPLING * (GAD7 - 7.5) / 21.0
        + DEPRESSION_SLEEP_COUPLING * (7.0 - TST)
    )
    d[_DEP] = dep_drive / 7.0 + DEPRESSION_REVERSION * (8.42 - Depression) / 7.0
    # Secular decrease (Paper 7: all groups depression decreasing p=0.026)
    d[_DEP] -= 0.05 / 7.0

    # RHR: very slow trait variable
    d[_RHR] = RHR_REVERSION * (63.0 - RHR) / 7.0

    return d


# ── RK4 integrator ───────────────────────────────────────────────────────────

def _rk4_step(
    state: np.ndarray,
    t: float,
    dt: float,
    intervention: dict[str, float],
    patient: dict[str, float],
) -> np.ndarray:
    """Advance the simulation by one day using Runge-Kutta 4th order.

    RK4 estimates the slope at several points within the step and takes a
    weighted average, producing much more accurate results than a simple
    forward step.  This matters when variables are changing quickly (like
    PSS spiking during exam week).

    After stepping, clamps all state variables to their biological bounds
    using _LOWER and _UPPER arrays from constants.py.
    """
    k1 = derivatives(state, t, intervention, patient)
    k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt, intervention, patient)
    k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt, intervention, patient)
    k4 = derivatives(state + dt * k3, t + dt, intervention, patient)
    new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Clamp to biological bounds: TST can't drop below 4 hours or exceed
    # 12 hours; GAD7 stays in [0, 21]; RHR stays in [45, 100 bpm]; etc.
    new_state = np.maximum(new_state, _LOWER)
    new_state = np.minimum(new_state, _UPPER)

    return new_state


# ── Main simulation loop ────────────────────────────────────────────────────

def simulate(
    intervention: dict[str, float] | None = None,
    patient: dict[str, float] | None = None,
    sim_weeks: int | None = None,
) -> dict:
    """Run a complete semester simulation.

    Takes a description of the student (patient parameters) and their
    semester-long intervention package (intervention parameters), then
    simulates what happens day by day for 15 weeks (one semester),
    tracking 14 coupled state variables.

    Returns a dictionary containing the full trajectory -- the value of
    all 14 state variables at every daily timestep.
    """
    intv = {**DEFAULT_INTERVENTION, **(intervention or {})}
    pat = {**DEFAULT_PATIENT, **(patient or {})}
    weeks = sim_weeks if sim_weeks is not None else SIM_WEEKS
    n_steps = int(weeks * 7)
    dt = DT

    # Merge nature_rx into patient dict for initial_state
    pat_with_rx = {**pat, "nature_rx": intv.get("nature_rx", 0.0)}

    # Set up the initial state: Day 0, start of semester
    state = initial_state(pat_with_rx)

    # Pre-allocate arrays for the full trajectory (14 variables x n_steps days)
    states = np.zeros((n_steps + 1, N_STATES), dtype=np.float64)
    times = np.zeros(n_steps + 1, dtype=np.float64)
    states[0] = state
    times[0] = 0.0

    # Step forward one day at a time for the entire simulation
    for i in range(n_steps):
        t = i * dt
        state = _rk4_step(state, t, dt, intv, pat)
        states[i + 1] = state
        times[i + 1] = (i + 1) * dt

    return {
        "states": states,         # shape (n_steps+1, 14) -- the full trajectory
        "times": times,           # shape (n_steps+1,) -- time in weeks
        "intervention": intv,     # the interventions applied
        "patient": pat,           # the student's characteristics
    }


# ── Standalone test ──────────────────────────────────────────────────────────
# Running "python simulator.py" exercises all 8 student archetypes and prints
# a summary of key metrics for each one.  This is a quick sanity check.

if __name__ == "__main__":
    from constants import STUDENT_ARCHETYPES, STATE_NAMES

    print("LEMURS Simulator -- Student Archetype Test")
    print("=" * 70)

    for arch in STUDENT_ARCHETYPES:
        intv = arch.get("intervention", {})
        pat = arch.get("patient", {})
        result = simulate(intervention=intv, patient=pat)
        states = result["states"]

        # Extract key metrics
        pss_mean = np.mean(states[:, _PSS])
        pss_final = states[-1, _PSS]
        gad7_mean = np.mean(states[:, _GAD7])
        gad7_max = np.max(states[:, _GAD7])
        hrv_mean = np.mean(states[:, _HRV])
        tst_mean = np.mean(states[:, _TST])

        print(f"\n{arch['name']}: {arch['description']}")
        print(f"  PSS:  mean={pss_mean:.2f}  final={pss_final:.2f}")
        print(f"  GAD7: mean={gad7_mean:.2f}  max={gad7_max:.2f}")
        print(f"  HRV:  mean={hrv_mean:.2f} ms")
        print(f"  TST:  mean={tst_mean:.2f} hrs")

    print("\n" + "=" * 70)
    print("Done.")
