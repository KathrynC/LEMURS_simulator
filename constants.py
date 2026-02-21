"""Central configuration for the LEMURS college-student well-being simulator.

This file defines everything the simulator needs to know before it runs:
who the student is, what interventions they receive, the biological and
psychological rate constants that govern how sleep, stress, anxiety, and
well-being co-evolve over a 15-week college semester.

The model draws on 9 published studies from the LEMURS research program
(Longitudinal Ecological Momentary Understanding of Real-time Sleep):

    Paper 2: Bloomfield 2024 — Sleep phenotyping via clustering (Oura Ring)
    Paper 3: Bloomfield 2024 — Wearable biomarkers predict perceived stress
             (PLOS Digital Health)
    Paper 4: Bloomfield 2024 — Anxiety prevalence & persistence in college
             students (JAACAP Open)
    Paper 6: Fudolig 2025  — Sleep debt, social jetlag, and physical activity
             (npj Complexity)
    Paper 7: Bloomfield 2025 — Nature engagement as stress mediator
    Paper 8: Bloomfield 2025 — Attention restoration and perceived nature

The 12 input parameters (6 describing the student + 6 describing their
interventions) define a 12-dimensional space. Every possible student and
every possible semester-long intervention package is a single point in that
space. The simulator takes that point and runs the student forward in time
for 15 weeks (one semester), tracking 14 coupled state variables daily.
"""
from __future__ import annotations

import numpy as np

# == Type aliases ============================================================
# These are just shorthand names for "a dictionary mapping parameter names
# to numbers." They help the code read more clearly but don't change behavior.

ParamDict = dict[str, float]
InterventionDict = dict[str, float]
PatientDict = dict[str, float]

# == Simulation config =======================================================

SIM_WEEKS: int = 15             # One college semester (15 weeks)
DT: float = 1.0 / 7.0          # One day, expressed as a fraction of a week
N_STEPS: int = 105              # Total number of daily steps (15 weeks x 7 days)

# == State variables (14D) ===================================================
# These are the 14 things the simulator tracks for each student, day by day.
# Together they tell the story of how sleep, stress, anxiety, physical
# activity, nature engagement, and well-being co-evolve across a semester.

N_STATES: int = 14
STATE_NAMES: list[str] = [
    "TST",           # 0  Total Sleep Time (hours/night) — Paper 3
                     #    The most direct measure of sleep quantity. College
                     #    students average 6.5-7.5 hours on school nights but
                     #    accumulate significant sleep debt during the week.
                     #    (Bloomfield 2024, PLOS Digital Health)

    "SleepQuality",  # 1  Oura sleep score [0-100] — Paper 2
                     #    A composite measure from the Oura Ring that integrates
                     #    sleep duration, efficiency, latency, REM, and deep
                     #    sleep proportions. Cluster analysis reveals distinct
                     #    sleep phenotypes that predict health outcomes.
                     #    (Bloomfield 2024, Sleep phenotyping)

    "PSS",           # 2  Perceived Stress Scale [0-40] — Papers 3, 7, 8
                     #    Cohen's PSS measures the degree to which situations
                     #    in a student's life are appraised as stressful. This
                     #    is the central hub of the model: sleep, wearable
                     #    biomarkers, and nature engagement all converge here.
                     #    (Bloomfield 2024, PLOS Digital Health)

    "GAD7",          # 3  Generalized Anxiety [0-21] — Paper 4
                     #    Spitzer's GAD-7 screener for generalized anxiety
                     #    disorder. In the LEMURS cohort, 30% of students
                     #    screened above the clinical threshold (>=10) at
                     #    some point during the semester. Anxiety shows
                     #    Markov-chain dynamics: once above threshold,
                     #    recovery is slow and state-dependent.
                     #    (Bloomfield 2024, JAACAP Open)

    "Depression",    # 4  DASS-21 depression subscale [0-21] — Papers 7, 8
                     #    The Depression-Anxiety-Stress Scales depression
                     #    subscale. Driven by stress, anxiety comorbidity,
                     #    and poor sleep. Nature engagement provides a
                     #    protective pathway.

    "Activity",      # 5  Active calories [0-500] — Paper 6
                     #    Daily energy expenditure from physical activity,
                     #    measured by Oura Ring. Paradoxically, sleep-deprived
                     #    students are MORE active (compensation behavior),
                     #    and social jetlag increases daytime activity.
                     #    (Fudolig 2025, npj Complexity)

    "NatureEngagement",  # 6  Perceived hours in nature/week [0-15] — Papers 7, 8
                         #    Self-reported time spent in natural environments.
                         #    Crucially, perceived nature engagement differs from
                         #    GPS-measured green space exposure: perceived nature
                         #    predicts well-being improvements, while GPS-tracked
                         #    exposure alone does not.
                         #    (Bloomfield 2025, Nature & attention)

    "RHR",           # 7  Resting Heart Rate [45-100] bpm — Paper 3
                     #    24-hour average from wrist-worn sensor. Elevated RHR
                     #    predicts higher perceived stress even after controlling
                     #    for sleep and activity. A proxy for sympathetic tone.
                     #    (Bloomfield 2024, PLOS Digital Health)

    "HRV",           # 8  Heart Rate Variability RMSSD [15-120] ms — Papers 3, 7
                     #    Root-mean-square of successive differences in heartbeat
                     #    intervals. Higher HRV = better parasympathetic regulation.
                     #    Nature engagement increases HRV via the PSS mediation
                     #    pathway (nature -> lower stress -> higher HRV).
                     #    (Bloomfield 2024, 2025)

    "ARR",           # 9  Average Respiratory Rate [10-25] breaths/min — Papers 3, 7
                     #    Nocturnal respiratory rate from Oura Ring. Higher ARR
                     #    predicts higher perceived stress and partially mediates
                     #    the therapy -> stress reduction pathway.

    "SocialJetlag",  # 10 MSF_free - MSF_school [0-3] hours — Paper 6
                     #    The difference between free-day and school-day mid-sleep
                     #    time. College students show 0.5-2.5 hours of social
                     #    jetlag. Higher SJL paradoxically increases daytime
                     #    physical activity but at the cost of sleep quality.
                     #    (Fudolig 2025, npj Complexity)

    "SleepShape",    # 11 Fraction of nights in Cluster 1 [0-1] — Paper 2
                     #    Sleep phenotype cluster membership. Cluster 1 represents
                     #    shorter, more disrupted sleep patterns. Female students
                     #    with MH diagnoses are overrepresented in Cluster 1.
                     #    (Bloomfield 2024, Sleep phenotyping)

    "WEMWBS",        # 12 Warwick-Edinburgh Well-Being [14-70] — Paper 7
                     #    A validated measure of positive mental well-being. Nature
                     #    engagement is the primary driver of WEMWBS improvement
                     #    across the semester. Control group baseline: 46.66.
                     #    (Bloomfield 2025, Nature engagement)

    "DAC",           # 13 Directed Attention Capacity [0-1] — Paper 8
                     #    Kaplan & Kaplan's Attention Restoration Theory: directed
                     #    attention is a finite resource depleted by academic work
                     #    and restored by nature engagement. When DAC is low,
                     #    students report more stress and worse academic
                     #    performance. Perceived (not GPS-measured) nature contact
                     #    drives restoration.
                     #    (Bloomfield 2025, Attention restoration)
]

# ── State index constants ───────────────────────────────────────────────────
# Shorthand for indexing into state vectors. Using underscore-prefix
# module-level constants following the grief-simulator convention.

_TST = 0
_SQ = 1
_PSS = 2
_GAD7 = 3
_DEP = 4
_ACT = 5
_NAT = 6
_RHR = 7
_HRV = 8
_ARR = 9
_SJL = 10
_SHAPE = 11
_WB = 12
_DAC = 13

# ── State bounds ────────────────────────────────────────────────────────────
# Physical/clinical bounds for each state variable. The simulator clamps
# values to these ranges after each integration step.

_LOWER = np.array([
    4.0,    # TST: minimum viable sleep (hours)
    0.0,    # SleepQuality: worst possible Oura score
    0.0,    # PSS: no perceived stress
    0.0,    # GAD7: no anxiety symptoms
    0.0,    # Depression: no depression symptoms
    0.0,    # Activity: sedentary
    0.0,    # NatureEngagement: no nature contact
    45.0,   # RHR: athletic resting heart rate (bpm)
    15.0,   # HRV: very low vagal tone (ms RMSSD)
    10.0,   # ARR: slow resting respiratory rate (breaths/min)
    0.0,    # SocialJetlag: no circadian misalignment (hours)
    0.0,    # SleepShape: all nights in healthy cluster
    14.0,   # WEMWBS: minimum possible score
    0.0,    # DAC: completely depleted attention
])

_UPPER = np.array([
    12.0,   # TST: excessive sleep (hours)
    100.0,  # SleepQuality: perfect Oura score
    40.0,   # PSS: maximum perceived stress
    21.0,   # GAD7: maximum anxiety score
    21.0,   # Depression: maximum depression score
    500.0,  # Activity: very high active calories
    15.0,   # NatureEngagement: maximum nature hours/week
    100.0,  # RHR: tachycardic resting heart rate (bpm)
    120.0,  # HRV: excellent vagal tone (ms RMSSD)
    25.0,   # ARR: rapid respiratory rate (breaths/min)
    3.0,    # SocialJetlag: extreme circadian misalignment (hours)
    1.0,    # SleepShape: all nights in disrupted cluster
    70.0,   # WEMWBS: maximum well-being score
    1.0,    # DAC: fully restored attention
])

# == Intervention parameters (6D) ============================================
# These are the behavioral levers that can be applied during the semester.
# Each is a dial from 0 (not doing it / absent) to 1 (maximum engagement).
# The default values represent a typical student with no special intervention.

INTERVENTION_NAMES: list[str] = [
    "nature_rx",      # Nature prescription — structured time in natural
                      #   environments (campus gardens, nearby parks, forest
                      #   trails). The primary driver of attention restoration
                      #   and well-being improvement in Paper 7 and Paper 8.

    "exercise_rx",    # Exercise prescription — structured physical activity
                      #   beyond baseline campus walking. Increases active
                      #   calories, improves HRV, reduces RHR over time.

    "therapy_rx",     # Therapy/counseling engagement — campus mental health
                      #   services. Reduces DASS stress and anxiety through
                      #   direct pathway; also lowers ARR through relaxation
                      #   training component. (Paper 7, therapy arm)

    "sleep_hygiene",  # Sleep hygiene intervention — consistent bedtime,
                      #   blue-light reduction, bedroom environment. Improves
                      #   TST and sleep quality directly. (Papers 2, 3)

    "caffeine_reduction",  # Caffeine reduction — lowering stimulant intake
                           #   that disrupts sleep quality and elevates RHR.
                           #   College students average 200-400mg caffeine/day.

    "academic_load",  # Academic load intensity — course load, exam pressure,
                      #   assignment density. 0=light load, 0.5=typical,
                      #   1.0=overloaded. This is the primary DEPLETOR of
                      #   directed attention capacity (Paper 8) and the source
                      #   of the semester stress trajectory.
]

INTERVENTION_BOUNDS: dict[str, tuple[float, float]] = {
    "nature_rx":          (0.0, 1.0),
    "exercise_rx":        (0.0, 1.0),
    "therapy_rx":         (0.0, 1.0),
    "sleep_hygiene":      (0.0, 1.0),
    "caffeine_reduction": (0.0, 1.0),
    "academic_load":      (0.0, 1.0),
}

# Grids define the discrete values we test at when sweeping parameters.
# These aren't the only possible values — they're sampling points.
INTERVENTION_GRIDS: dict[str, list[float]] = {
    "nature_rx":          [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "exercise_rx":        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "therapy_rx":         [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "sleep_hygiene":      [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "caffeine_reduction": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "academic_load":      [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
}

# The "default student" is taking a typical course load with no special
# interventions — no nature prescription, minimal exercise, no therapy,
# moderate sleep hygiene, no caffeine reduction. This is the baseline
# against which we measure the impact of interventions.
DEFAULT_INTERVENTION: InterventionDict = {
    "nature_rx":          0.0,    # no structured nature engagement
    "exercise_rx":        0.2,    # minimal exercise (some campus walking)
    "therapy_rx":         0.0,    # no counseling
    "sleep_hygiene":      0.3,    # some awareness but inconsistent
    "caffeine_reduction": 0.0,    # typical caffeine consumption
    "academic_load":      0.5,    # standard semester load
}

# == Patient / student parameters (6D) =======================================
# These describe WHO the student is. Unlike the intervention parameters
# (which can be changed), these are fixed characteristics that determine
# vulnerability, resilience, and baseline trajectories.

PATIENT_NAMES: list[str] = [
    "age",             # Age in years. College students are 18-25. Younger
                       #   students (freshmen) may have less coping experience.

    "gender",          # Gender identity coded as: 0=male, 1=female, 2=nonbinary.
                       #   The LEMURS cohort is ~65% female. Gender modifies
                       #   stress perception (Paper 3: females report +2.956 PSS
                       #   points), chronotype (Paper 6: males sleep later), and
                       #   sleep phenotype (Paper 2: female+MH -> Cluster 1).

    "emotional_stability",  # Big Five emotional stability / neuroticism (reversed),
                            #   on a 1-7 Likert scale. Higher = more stable. This
                            #   is the strongest protective factor against anxiety
                            #   persistence (Paper 4: AOR=0.58 per point).

    "trauma_load",     # Cumulative adverse experiences on a 0-5 scale. ACE-like
                       #   measure. Trauma_load >= 2 is a risk factor for anxiety
                       #   occurrence (Paper 4: AOR=1.80) and modifies sleep
                       #   phenotype in females (Paper 2).

    "mh_diagnosis",    # Prior mental health diagnosis (0=no, 1=yes). The
                       #   single strongest predictor of anxiety occurrence
                       #   (Paper 4: AOR=2.10) and modifies chronotype, activity
                       #   level, and sleep phenotype.

    "baseline_chronotype",  # Midsleep time on free days (MSF_free) in hours,
                            #   on a 2-7 scale (2am-7am). Later chronotype =
                            #   more social jetlag on school days. College students
                            #   average MSF ~4.5h. Modified by gender (males
                            #   +28.3 min) and MH diagnosis (+16.4 min).
                            #   (Paper 6, Fudolig 2025)
]

PATIENT_BOUNDS: dict[str, tuple[float, float]] = {
    "age":                  (18.0, 25.0),
    "gender":               (0.0, 2.0),
    "emotional_stability":  (1.0, 7.0),
    "trauma_load":          (0.0, 5.0),
    "mh_diagnosis":         (0.0, 1.0),
    "baseline_chronotype":  (2.0, 7.0),
}

PATIENT_GRIDS: dict[str, list[float]] = {
    "age":                  [18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
    "gender":               [0.0, 1.0, 2.0],
    "emotional_stability":  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    "trauma_load":          [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    "mh_diagnosis":         [0.0, 1.0],
    "baseline_chronotype":  [2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 7.0],
}

# The "default student" represents the modal student in the LEMURS cohort:
# a 19-year-old female (65% female sample), moderate emotional stability,
# low trauma load, no prior MH diagnosis, average chronotype.
DEFAULT_PATIENT: PatientDict = {
    "age":                  19.0,
    "gender":               1.0,    # female (matching 65% female sample)
    "emotional_stability":  4.5,    # moderate stability
    "trauma_load":          1.0,    # low trauma exposure
    "mh_diagnosis":         0.0,    # no prior diagnosis
    "baseline_chronotype":  4.5,    # average college midsleep time
}

# ============================================================================
# COUPLING CONSTANTS
# ============================================================================
# These numbers control how strongly each process drives the others. They
# were derived from the regression coefficients, adjusted odds ratios, and
# effect sizes reported in the 9 LEMURS papers. Where papers report
# standardized betas, we convert to natural units (PSS points, hours, bpm)
# for interpretability.
#
# Think of each constant as a dial that controls the strength of a connection
# between two variables. For example, BETA_TST_PSS = -0.877 means "each
# additional hour of sleep reduces perceived stress by 0.877 PSS points."

# -- Tier 1: Sleep -> Stress (Paper 3: Bloomfield 2024, PLOS Digital Health) -
# The core finding: wearable biomarkers (RHR, HRV, ARR) predict perceived
# stress beyond what sleep duration alone captures. Within-person deviations
# from a student's own baseline are 2.2x more predictive than between-person
# differences, suggesting that the model should track deviations from
# individual reference points.

PSS_SECULAR_TREND: float = 0.077
"""PSS increase per week across the semester (Paper 3, Table S2).
Captures the general semester stress ramp-up: assignments accumulate,
exams approach, and social demands increase. This is the background
pressure that interventions must counteract."""

BETA_TST_PSS: float = -0.877
"""PSS points per hour of total sleep time (Paper 3, Table 2).
Each additional hour of sleep predicts 0.877 fewer PSS points. This is
the single strongest wearable predictor of perceived stress. The effect
operates primarily through within-person deviations: a student who sleeps
1 hour less than THEIR OWN average reports nearly 2 more PSS points
(after within-person amplification)."""

BETA_RHR_PSS: float = 0.055
"""PSS points per bpm resting heart rate (Paper 3, Table 2).
Higher RHR = higher stress. RHR captures sympathetic arousal that the
student may not consciously perceive. Even after controlling for sleep
and activity, RHR independently predicts stress."""

BETA_HRV_PSS: float = -0.012
"""PSS points per ms RMSSD heart rate variability (Paper 3, Table 2).
Higher HRV = lower stress. The vagal tone pathway: better parasympathetic
regulation enables more effective stress buffering."""

BETA_ARR_PSS: float = 0.270
"""PSS points per breath/min average respiratory rate (Paper 3, Table 2).
Faster breathing predicts higher stress. ARR is the second-strongest
wearable predictor after TST. Nocturnal respiratory rate may reflect
subclinical anxiety or sleep-disordered breathing."""

BETA_GENDER_PSS: float = 2.956
"""PSS points for nonmale gender (Paper 3, Table 2).
Female and nonbinary students report nearly 3 more PSS points at
identical wearable biomarker levels. This is a level shift, not a
slope difference — the sensitivity to sleep loss is the same, but the
baseline stress is higher."""

TST_REFERENCE: float = 7.0
"""Reference point for TST deviation (hours).
Within-person effects are computed as deviations from this reference.
7.0 hours represents the approximate cohort mean TST on school nights."""

RHR_REFERENCE: float = 65.0
"""Reference point for RHR deviation (bpm).
The approximate cohort mean 24-hour resting heart rate. Deviations
from a student's own RHR baseline are more predictive than absolute
RHR levels."""

HRV_REFERENCE: float = 60.0
"""Reference point for HRV deviation (ms RMSSD).
Approximate cohort mean RMSSD. Within-person HRV drops predict
same-week stress increases."""

ARR_REFERENCE: float = 15.0
"""Reference point for ARR deviation (breaths/min).
Approximate cohort mean nocturnal respiratory rate."""

PSS_REVERSION: float = 0.3
"""Mean-reversion rate for PSS (calibrated).
PSS does not drift arbitrarily — students have a stress setpoint
determined by their personality and circumstances. This reversion
rate pulls PSS back toward the individual's equilibrium."""

WITHIN_PERSON_AMPLIFICATION: float = 2.2
"""Within-person deviation effects are 2.2x stronger than between-person
differences (Paper 3, Results section). When a student deviates from
their OWN baseline sleep/HRV/RHR, the stress effect is 2.2x what
you'd estimate from comparing two different students with those levels.
This is the key finding that justifies a dynamical model: a student
who normally sleeps 8 hours and drops to 6 is hit harder than a student
who always sleeps 6."""

# -- Tier 2: Anxiety Markov Chain (Paper 4: Bloomfield 2024, JAACAP Open) ----
# Anxiety in college students follows Markov dynamics: the probability of
# being anxious THIS week depends on whether you were anxious LAST week.
# Once above the GAD-7 clinical threshold (>=10), recovery is slow and
# influenced by different risk factors than initial occurrence.

GAD7_THRESHOLD: float = 10.0
"""Clinical threshold for GAD-7 (Paper 4).
A score of 10 or above indicates "moderate" generalized anxiety and
is the standard clinical screening cutoff. In the LEMURS cohort,
~30% of students cross this threshold at some point."""

GAD7_DEVELOPMENT_RATE: float = 0.12
"""Base probability per week of transitioning from below-threshold to
above-threshold anxiety (Paper 4, Table 3). This is the "occurrence"
transition rate before adjusting for individual risk factors."""

GAD7_RECOVERY_RATE: float = 0.65
"""Base probability per week of transitioning from above-threshold to
below-threshold anxiety (Paper 4, Table 3). Recovery is much more
likely than development — most anxiety episodes are transient. But
this rate is modified by persistence factors below."""

AOR_EMOTIONAL_STABILITY: float = 0.58
"""Adjusted odds ratio for anxiety occurrence per point of emotional
stability (Paper 4, Table 3). Each additional point of emotional
stability (1-7 scale) reduces the odds of crossing the anxiety
threshold by 42%. This is the strongest protective factor."""

AOR_MH_DIAGNOSIS: float = 2.10
"""Adjusted odds ratio for anxiety occurrence given prior MH diagnosis
(Paper 4, Table 3). Students with a prior mental health diagnosis are
2.1x more likely to develop above-threshold anxiety in any given week.
This captures both biological vulnerability and residual symptoms."""

AOR_TRAUMA: float = 1.80
"""Adjusted odds ratio for anxiety occurrence when trauma_load >= 2
(Paper 4, Table 3). Prior traumatic experiences lower the threshold
for anxiety activation."""

AOR_ACADEMIC_STRESSOR: float = 1.68
"""Adjusted odds ratio for anxiety occurrence when academic stressors
are present (Paper 4, Table 3). Exams, deadlines, and academic
pressure are a significant anxiety trigger in this population."""

RECOVERY_HYSTERESIS: float = 0.92
"""Recovery rate decay per semester fraction (calibrated from Paper 4).
The longer a student stays anxious, the harder recovery becomes. Each
passing week that anxiety persists reduces the recovery probability
by this factor. This creates the persistence observed clinically."""

RECOVERY_STABILITY_AOR: float = 0.82
"""Emotional stability AOR for anxiety RECOVERY (Paper 4).
Higher emotional stability also predicts faster recovery from anxiety
episodes, but the effect is weaker than for prevention (0.82 vs 0.58)."""

RECOVERY_STRESSOR_AOR: float = 0.69
"""Academic stressor AOR for anxiety RECOVERY (Paper 4).
When academic stressors persist, they actively prevent recovery from
anxiety episodes, reducing recovery probability by 31%."""

GAD7_ANXIOUS_ATTRACTOR: float = 15.0
"""Where the anxious state pulls the GAD-7 score toward.
Once in the anxious regime (above threshold), the GAD-7 score
gravitates toward this value, representing a stable anxious state."""

GAD7_RECOVERY_ATTRACTOR: float = 5.0
"""Where recovery pulls the GAD-7 score toward.
In the sub-threshold regime, GAD-7 gravitates toward this value,
representing a mildly stressed but not clinically anxious state."""

# -- Tier 3: Nature Mediation (Paper 7: Bloomfield 2025) ---------------------
# Nature engagement operates through multiple parallel pathways:
# direct stress reduction, HRV improvement (via PSS mediation), and
# well-being enhancement. The key insight is that nature does NOT work
# by direct physiological mechanisms alone — the stress-reduction pathway
# (nature -> lower PSS -> better HRV) is the primary mediator.

BETA_NATURE_PSS: float = -1.507
"""PSS reduction per unit nature engagement (Paper 7, mediation model).
The direct effect of nature engagement on perceived stress. Each unit
increase in nature_rx reduces PSS by 1.507 points. This is the entry
point for nature's cascading benefits."""

BETA_PSS_HRV: float = -0.618
"""HRV change per PSS point (Paper 7, mediation model).
The stress-to-HRV pathway: each PSS point reduces HRV by 0.618 ms
RMSSD. This mediates the nature -> HRV effect: nature reduces stress,
lower stress improves HRV."""

NATURE_HRV_DIRECT: float = 9.13
"""Direct HRV gain from nature engagement over 14 weeks (Paper 7).
The total HRV improvement attributable to nature engagement in the
intervention group vs control, measured as ms RMSSD gain over the
semester. This includes both direct and mediated pathways."""

BETA_NATURE_WEMWBS: float = 0.104
"""Well-being (WEMWBS) gain per week from nature engagement (Paper 7).
A slow, cumulative effect: nature engagement gradually builds positive
well-being over the semester. The effect is small per week but
compounds over 15 weeks."""

BETA_THERAPY_STRESS: float = -0.102
"""DASS stress reduction per week from therapy engagement (Paper 7).
The therapy arm showed significant reductions in DASS-21 stress
subscale, independent of the nature pathway."""

BETA_THERAPY_ARR: float = -0.012
"""ARR reduction per week from therapy engagement (Paper 7).
Therapy (particularly relaxation training components) reduces
nocturnal respiratory rate, which is both a stress marker and
an independent predictor of PSS."""

# -- Tier 4: Sleep Debt & Activity (Paper 6: Fudolig 2025, npj Complexity) --
# Sleep debt accumulates on weekdays (school nights are 37-55 minutes
# shorter than free nights) and drives compensatory changes in physical
# activity. Paradoxically, sleep-deprived students are MORE active, not
# less — social jetlag and school schedules force movement.

SLEEP_DEBT_WEEKDAY: float = 45.0 / 60.0
"""Hours of sleep lost per school night relative to free nights (Paper 6).
The midpoint of the 37-55 minute range reported in the paper. This
creates a weekly sleep debt of ~3.75 hours that partially recovers
on weekends."""

BETA_SJL_ACTIVITY: float = 0.023
"""+2.3% activity per hour of social jetlag (Paper 6, Table 2).
More social jetlag = more forced daytime activity. This is a
compensatory mechanism, not a health benefit."""

BETA_TST_ACTIVITY: float = -0.021
"""-2.1% activity per hour of sleep (Paper 6, Table 2).
Less sleep = more activity. Counterintuitive but robust:
sleep-deprived students are compelled to be more active by
their schedules and compensatory arousal."""

BETA_WEEKDAY_ACTIVITY: float = 0.051
"""+5.1% activity on weekdays vs weekends (Paper 6, Table 2).
Weekday schedules (classes, labs, campus transit) force more
physical movement than weekend rest days."""

BETA_SCHOOL_ACTIVITY: float = 0.137
"""+13.7% activity during school vs break periods (Paper 6, Table 2).
The largest activity modifier: school is simply more physically
demanding than vacation, driven by campus geography and structured
schedules."""

BETA_MALE_ACTIVITY: float = 0.105
"""+10.5% activity for male students (Paper 6, Table 2).
Males show consistently higher physical activity, likely reflecting
both greater lean body mass and higher participation in recreational
sports."""

BETA_MH_ACTIVITY: float = -0.028
"""-2.8% activity if MH impairment present (Paper 6, Table 2).
Mental health conditions reduce physical activity through
amotivation, social withdrawal, and medication side effects."""

ACTIVITY_BASELINE: float = 200.0
"""Baseline active calories per day (kcal).
Approximate average daily active calories for a college student
during the school term, from Oura Ring data."""

TST_BASELINE: float = 7.5
"""Baseline total sleep time (hours).
The Thanksgiving/break sleep level from Paper 6 — what students
would sleep if unconstrained by school schedules. During the
semester, TST drops below this due to sleep debt."""

TST_REVERSION: float = 0.5
"""Sleep recovery rate toward baseline.
On free days (weekends, breaks), TST recovers toward TST_BASELINE
at this rate. Partial recovery on weekends creates the oscillating
sleep pattern observed in Paper 6."""

ACTIVITY_REVERSION: float = 0.3
"""Activity recovery rate toward baseline.
Activity returns toward ACTIVITY_BASELINE when acute modifiers
(weekday, school) are removed."""

# -- Tier 5: Chronotype & Sleep Shape (Papers 2, 6) -------------------------
# Chronotype determines social jetlag, which drives the weekday sleep debt
# cycle. Sleep shape (cluster membership) captures qualitative differences
# in sleep architecture that predict health outcomes beyond duration alone.

BETA_MALE_CHRONOTYPE: float = 0.472
"""+28.3 minutes later midsleep for males (Paper 6, Table 1).
Males have later chronotypes, sleeping ~28 minutes later than females
on free days. Converted to hours: 28.3/60 = 0.472."""

BETA_MH_CHRONOTYPE: float = 0.273
"""+16.4 minutes later midsleep for MH impairment (Paper 6, Table 1).
Students with mental health conditions sleep later, possibly due to
evening rumination, medication effects, or circadian disruption.
Converted to hours: 16.4/60 = 0.273."""

SJL_WEEKDAY_FORCING: float = 0.924
"""Hours earlier on weekdays due to school schedules (Paper 6).
The social jetlag forcing function: school schedules push wake time
0.924 hours earlier than the student's free-day preference, creating
the SJL gap. Derived from the mean weekday-weekend MSF difference."""

SJL_REVERSION: float = 0.5
"""SJL recovery rate on free days.
Social jetlag partially resolves on weekends and fully resolves
during breaks, when school schedule forcing disappears."""

SHAPE_REVERSION: float = 0.1
"""Sleep shape adaptation rate.
Cluster membership changes slowly — sleep architecture is a
relatively stable trait that responds to sustained changes in
sleep habits, stress, and treatment."""

SHAPE_FEMALE_MH_COEFF: float = 0.3
"""Female + MH diagnosis coupling to Cluster 1 (Paper 2).
Female students with mental health diagnoses are more likely to
exhibit the disrupted sleep phenotype (Cluster 1). This interaction
effect is not seen in males."""

SHAPE_FEMALE_TRAUMA_COEFF: float = 0.2
"""Female + trauma load coupling to Cluster 1 (Paper 2).
Trauma history in female students shifts sleep architecture toward
the disrupted cluster, independent of MH diagnosis."""

SHAPE_NB_TRAUMA_COEFF: float = 0.2
"""Nonbinary + trauma load coupling to Cluster 1.
Nonbinary students with trauma history show a similar (estimated)
shift toward disrupted sleep patterns."""

# -- Tier 6: Attention Restoration (Paper 8: Bloomfield 2025) ----------------
# Kaplan & Kaplan's Attention Restoration Theory operationalized: directed
# attention is a finite cognitive resource depleted by effortful academic
# work and restored by engagement with natural environments. Critically,
# PERCEIVED nature engagement (not GPS-measured green space exposure)
# drives restoration — the subjective experience of "being in nature"
# matters more than physical proximity to green space.

DAC_DEPLETION: float = 0.3
"""Attention depletion rate from academic load (calibrated from Paper 8).
Academic work (lectures, studying, exams) depletes directed attention
capacity. At full academic load (1.0), DAC decreases by 0.3 per week
before recovery mechanisms act."""

DAC_RESTORATION: float = 0.2
"""Nature engagement restoration rate (Paper 8).
Perceived nature engagement restores directed attention capacity.
At full nature engagement (1.0), DAC recovers by 0.2 per week."""

DAC_REVERSION: float = 0.1
"""Natural DAC recovery rate.
Even without nature engagement, DAC slowly recovers through rest,
sleep, and passive restorative experiences (soft fascination)."""

DAC_BASELINE: float = 0.8
"""Baseline directed attention capacity.
A well-rested student at the start of semester has ~80% of maximum
attention capacity. Perfect 1.0 is rarely achieved even under
ideal conditions."""

BETA_PERCEIVED_NATURE_DEP: float = -0.066
"""Depression reduction per perceived hour of nature engagement (Paper 8).
Self-reported nature time predicts lower depression scores. This
effect is specific to perceived engagement — GPS-tracked green space
exposure without subjective engagement does NOT predict improvement."""

BETA_GPS_NATURE_DEP: float = 0.032
"""Paradoxical GPS effect: GPS-measured green space WITHOUT engagement
predicts slightly HIGHER depression (Paper 8). Students who pass
through green spaces without perceiving nature benefit may be
commuting or distracted, and the GPS signal may be confounded with
campus layout (paths through green areas to lecture halls)."""

# -- Cross-tier coupling constants -------------------------------------------
# These constants couple the tiers together, creating the feedback loops
# that make the model a dynamical system rather than a collection of
# independent regressions.

DEPRESSION_PSS_COUPLING: float = 0.3
"""How strongly perceived stress drives depression.
Sustained high PSS gradually increases depression scores through
the cognitive pathway: appraised stress -> helplessness ->
depressive cognition."""

DEPRESSION_GAD7_COUPLING: float = 0.2
"""How strongly anxiety comorbidity drives depression.
Anxiety and depression are highly comorbid in college students.
Sustained above-threshold anxiety increases depression risk through
shared worry/rumination pathways."""

DEPRESSION_SLEEP_COUPLING: float = 0.2
"""How strongly poor sleep drives depression.
Short sleep and disrupted sleep architecture directly increase
depression risk through inflammatory and serotonergic pathways,
independent of the stress mediator."""

DEPRESSION_REVERSION: float = 0.2
"""Depression recovery rate.
Depression scores revert toward baseline when drivers (stress,
anxiety, poor sleep) are removed. Recovery is gradual, reflecting
the persistence of depressive cognitive patterns."""

WEMWBS_BASELINE: float = 46.66
"""Control group baseline WEMWBS (Paper 7).
The average well-being score for students in the control condition
(no nature or therapy intervention) across the semester. This is
the attractor point for well-being in the absence of intervention."""

WEMWBS_REVERSION: float = 0.2
"""Well-being recovery rate.
WEMWBS reverts toward WEMWBS_BASELINE when active drivers (nature
engagement, therapy) are removed. Like depression, well-being has
inertia and changes gradually."""

RHR_REVERSION: float = 0.05
"""Very slow trait reversion for resting heart rate.
RHR is a relatively stable physiological trait that changes slowly
in response to sustained exercise, stress reduction, or caffeine
changes. Week-to-week fluctuations are small."""

HRV_REVERSION_TRAIT: float = 0.05
"""Very slow trait reversion for heart rate variability.
Like RHR, HRV is a slow-moving trait. Acute changes in HRV reflect
state (stress, recovery), but the baseline drifts slowly."""

ARR_REVERSION: float = 0.03
"""Slowest trait reversion for average respiratory rate.
Nocturnal respiratory rate is the most stable wearable biomarker —
it changes meaningfully only with sustained therapy/relaxation
practice or chronic stress accumulation."""

SLEEP_QUALITY_TST_COUPLING: float = 10.0
"""Sleep quality tracks TST at 10 Oura score points per hour.
A rough coupling: each additional hour of sleep improves the Oura
sleep score by about 10 points. This captures the strong but
imperfect correlation between sleep duration and sleep quality."""

SLEEP_QUALITY_BASELINE: float = 75.0
"""Baseline Oura sleep score.
A typical college student's sleep quality score when sleeping
their normal amount. Not perfect (100) but good (75)."""

SLEEP_QUALITY_REVERSION: float = 0.5
"""Sleep quality recovery rate.
Sleep quality reverts toward the TST-determined equilibrium
at this rate, smoothing out day-to-day fluctuations."""

NATURE_ENGAGEMENT_REVERSION: float = 0.3
"""Nature engagement decay rate.
Without active nature prescription, a student's nature engagement
decays toward their baseline (usually near zero for the typical
student). Maintaining nature engagement requires ongoing effort."""

# == Semester Calendar Functions ==============================================
# The semester has a weekly and daily structure that drives the sleep debt
# and activity cycles. School nights (Sun-Thu) impose earlier wake times;
# weekends allow recovery; spring break provides a full week off.

SPRING_BREAK_WEEK: int = 8
"""Spring break occurs in week 8 of the 15-week semester (1-indexed).
During spring break, school schedule forcing disappears: no classes,
no early alarms, no academic stressors. Sleep debt recovers and
activity drops to break-period levels."""


def day_of_week(day: int) -> int:
    """Return day of week (0=Monday, 6=Sunday). Semester starts on a Monday."""
    return day % 7


def is_weekday(day: int) -> bool:
    """Return True if day is Monday-Friday (weekday)."""
    return day_of_week(day) < 5


def is_school_day(day: int) -> bool:
    """Return True if day is a weekday AND not during spring break.

    Spring break is week 8 (0-indexed: week index 7), so days 49-55
    are break days even though Mon-Fri would normally be school days.
    """
    week = day // 7
    if week == SPRING_BREAK_WEEK - 1:  # 0-indexed week, so week 8 -> index 7
        return False
    return is_weekday(day)


def week_of_semester(day: int) -> int:
    """Return the week number (0-indexed) for a given day."""
    return day // 7


# == Student Archetypes ======================================================
# These are 8 representative student profiles drawn from the LEMURS cohort
# characteristics. Each one tells a different story and produces a different
# trajectory through the 14-dimensional state space.

STUDENT_ARCHETYPES: list[dict] = [
    # The resilient male — high emotional stability, no prior MH, moderate
    # chronotype. This student absorbs the semester stress without crossing
    # clinical thresholds. Represents the ~50% of male students who maintain
    # sub-clinical anxiety throughout.
    {"name": "resilient_male",
     "description": "High stability, no MH, moderate chronotype",
     "patient": {"age": 19.0, "gender": 0.0, "emotional_stability": 6.0,
                 "trauma_load": 0.0, "mh_diagnosis": 0.0,
                 "baseline_chronotype": 4.0}},

    # The resilient female — high stability, early chronotype (less social
    # jetlag), active lifestyle. Despite the +2.956 PSS gender penalty,
    # her high emotional stability and good sleep habits keep her in the
    # healthy range.
    {"name": "resilient_female",
     "description": "High stability, early chronotype, active",
     "patient": {"age": 19.0, "gender": 1.0, "emotional_stability": 6.0,
                 "trauma_load": 0.0, "mh_diagnosis": 0.0,
                 "baseline_chronotype": 3.5}},

    # The vulnerable female — the high-risk profile from Paper 4. Low
    # emotional stability, prior MH diagnosis, trauma history, and late
    # chronotype compound to create the highest anxiety occurrence and
    # lowest recovery rates. Represents the ~15% of female students who
    # develop persistent anxiety.
    {"name": "vulnerable_female",
     "description": "Low stability, MH history, trauma, late chronotype",
     "patient": {"age": 20.0, "gender": 1.0, "emotional_stability": 3.0,
                 "trauma_load": 3.0, "mh_diagnosis": 1.0,
                 "baseline_chronotype": 5.5}},

    # The anxious male — prior MH diagnosis but moderate stability. Tests
    # the male anxiety trajectory: lower baseline stress (no gender PSS
    # penalty) but MH history creates vulnerability.
    {"name": "anxious_male",
     "description": "Prior anxiety, moderate stability",
     "patient": {"age": 19.0, "gender": 0.0, "emotional_stability": 4.0,
                 "trauma_load": 1.0, "mh_diagnosis": 1.0,
                 "baseline_chronotype": 4.5}},

    # The sleep-deprived student — extreme late chronotype (MSF=6.5h) means
    # massive social jetlag on school days. Tests the sleep debt cascade:
    # high SJL -> sleep debt -> stress -> anxiety risk.
    {"name": "sleep_deprived",
     "description": "Extreme late chronotype, high SJL",
     "patient": {"age": 19.0, "gender": 1.0, "emotional_stability": 4.5,
                 "trauma_load": 0.0, "mh_diagnosis": 0.0,
                 "baseline_chronotype": 6.5}},

    # The nature seeker — average vulnerability but high nature engagement
    # and reduced academic load. Tests the attention restoration pathway:
    # nature -> restored DAC -> lower stress -> better well-being.
    {"name": "nature_seeker",
     "description": "High nature engagement, outdoor orientation",
     "patient": {"age": 19.0, "gender": 1.0, "emotional_stability": 5.0,
                 "trauma_load": 0.0, "mh_diagnosis": 0.0,
                 "baseline_chronotype": 4.0},
     "intervention": {"nature_rx": 0.8, "academic_load": 0.3}},

    # The digital-immersed student — minimal nature contact, high academic
    # load, moderate vulnerability. Tests the attention depletion pathway:
    # high load -> depleted DAC -> higher stress -> worse outcomes.
    {"name": "digital_immersed",
     "description": "Low nature, high academic load",
     "patient": {"age": 19.0, "gender": 0.0, "emotional_stability": 4.0,
                 "trauma_load": 1.0, "mh_diagnosis": 0.0,
                 "baseline_chronotype": 5.0},
     "intervention": {"nature_rx": 0.0, "academic_load": 0.9}},

    # The recovery trajectory — starts with high anxiety risk factors but
    # receives the full intervention package (nature + exercise + therapy).
    # Tests whether the combined intervention can overcome high vulnerability.
    {"name": "recovery_trajectory",
     "description": "Starts anxious, interventions applied",
     "patient": {"age": 20.0, "gender": 1.0, "emotional_stability": 4.0,
                 "trauma_load": 2.0, "mh_diagnosis": 1.0,
                 "baseline_chronotype": 5.0},
     "intervention": {"nature_rx": 0.6, "exercise_rx": 0.5,
                      "therapy_rx": 0.4}},
]

# == Grid snapping ============================================================
# When exploring parameter space, we sometimes want to "snap" a continuous
# value to the nearest grid point. This is useful for creating clean
# comparison groups and for matching the discrete levels used in clinical
# studies (e.g., "low / medium / high" nature engagement).

_ALL_GRIDS: dict[str, list[float]] = {**INTERVENTION_GRIDS, **PATIENT_GRIDS}


def snap_param(name: str, value: float) -> float:
    """Snap a parameter value to the nearest grid point.

    For example, snap_param("nature_rx", 0.37) returns 0.4 — the nearest
    value on the nature prescription grid [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].
    """
    grid = _ALL_GRIDS[name]
    return min(grid, key=lambda g: abs(g - value))


def snap_all(params: dict[str, float]) -> dict[str, float]:
    """Snap all recognized parameters to their nearest grid points.

    Parameters not found in any grid are left unchanged.
    """
    result = dict(params)
    for name in result:
        if name in _ALL_GRIDS:
            result[name] = snap_param(name, result[name])
    return result
