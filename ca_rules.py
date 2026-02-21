"""Tiered rule table for the LEMURS semantic cellular automaton.

Rules are organized by ODE coupling tier, mirroring the 6-tier cascade in
derivatives(). Each rule specifies input bin conditions, output bin updates,
a confidence weight, and the source paper. Cross-tier compound rules capture
emergent multi-variable dynamics (burnout cascade, spring break reset).

The rule format follows the ER project's rulebook pattern: JSON-serializable
dicts that can be inspected, edited, and versioned independently of code.
"""
from __future__ import annotations

import json
from copy import deepcopy

from ca_schema import BIN_SCHEMA, _classify, bin_index


# ── Rule table ────────────────────────────────────────────────────────────
# Each rule is a dict with:
#   tier: int           -- which ODE coupling tier (1-6, 0 for cross-tier)
#   name: str           -- human-readable rule name
#   inputs: dict        -- {var_name: bin_label} conditions (all must match)
#   context: dict       -- {context_key: value} conditions on calendar/patient
#   outputs: dict       -- {var_name: bin_label_or_direction} updates
#   confidence: float   -- rule strength [0, 1]
#   citation: str       -- source paper reference

RULE_TABLE: list[dict] = [
    # ═══════════════════════════════════════════════════════════════════
    # TIER 1: Sleep -> Stress (Paper 3)
    # ═══════════════════════════════════════════════════════════════════
    {
        "tier": 1,
        "name": "sleep_debt_stress_up",
        "inputs": {"TST": "deprived", "SleepQuality": "poor"},
        "context": {},
        "outputs": {"PSS": "+1"},
        "confidence": 0.9,
        "citation": "Paper 3: BETA_TST_PSS=-0.877, within-person 2.2x",
    },
    {
        "tier": 1,
        "name": "sleep_adequate_stress_stable",
        "inputs": {"TST": "adequate", "SleepQuality": "fair"},
        "context": {},
        "outputs": {"PSS": "0"},
        "confidence": 0.8,
        "citation": "Paper 3: adequate sleep stabilizes stress",
    },
    {
        "tier": 1,
        "name": "sleep_good_stress_down",
        "inputs": {"TST": "adequate", "SleepQuality": "good"},
        "context": {},
        "outputs": {"PSS": "-1"},
        "confidence": 0.85,
        "citation": "Paper 3: good sleep quality protects against stress",
    },
    {
        "tier": 1,
        "name": "excess_sleep_stress_stable",
        "inputs": {"TST": "excess"},
        "context": {},
        "outputs": {"PSS": "0"},
        "confidence": 0.6,
        "citation": "Paper 3: excess sleep neutral for stress",
    },
    {
        "tier": 1,
        "name": "elevated_rhr_stress_up",
        "inputs": {"RHR": "elevated"},
        "context": {},
        "outputs": {"PSS": "+1"},
        "confidence": 0.7,
        "citation": "Paper 3: BETA_RHR_PSS=+0.055",
    },
    {
        "tier": 1,
        "name": "high_hrv_stress_down",
        "inputs": {"HRV": "high"},
        "context": {},
        "outputs": {"PSS": "-1"},
        "confidence": 0.65,
        "citation": "Paper 3: BETA_HRV_PSS=-0.012, parasympathetic buffer",
    },

    # ═══════════════════════════════════════════════════════════════════
    # TIER 2: Anxiety Markov (Paper 4)
    # ═══════════════════════════════════════════════════════════════════
    {
        "tier": 2,
        "name": "anxiety_development_risk",
        "inputs": {"GAD7": "sub_threshold", "PSS": "high"},
        "context": {"emotional_stability_low": True},
        "outputs": {"GAD7": "+1"},
        "confidence": 0.85,
        "citation": "Paper 4: AOR_ES=0.58, AOR_MH=2.10, dev_rate=0.12/wk",
    },
    {
        "tier": 2,
        "name": "anxiety_development_moderate",
        "inputs": {"GAD7": "sub_threshold", "PSS": "moderate"},
        "context": {"mh_diagnosis": True},
        "outputs": {"GAD7": "+1"},
        "confidence": 0.7,
        "citation": "Paper 4: AOR_MH=2.10 amplifies moderate stress",
    },
    {
        "tier": 2,
        "name": "anxiety_persistence_hysteresis",
        "inputs": {"GAD7": "clinical"},
        "context": {"week_gt_4": True},
        "outputs": {"GAD7": "0"},
        "confidence": 0.8,
        "citation": "Paper 4: RECOVERY_HYSTERESIS=0.92, recovery decays",
    },
    {
        "tier": 2,
        "name": "anxiety_recovery_stable",
        "inputs": {"GAD7": "clinical", "PSS": "low"},
        "context": {"emotional_stability_high": True},
        "outputs": {"GAD7": "-1"},
        "confidence": 0.75,
        "citation": "Paper 4: RECOVERY_STABILITY_AOR=0.82, low stress aids recovery",
    },
    {
        "tier": 2,
        "name": "anxiety_stable_subthreshold",
        "inputs": {"GAD7": "sub_threshold", "PSS": "low"},
        "context": {},
        "outputs": {"GAD7": "0"},
        "confidence": 0.9,
        "citation": "Paper 4: low stress, sub-threshold = stable",
    },

    # ═══════════════════════════════════════════════════════════════════
    # TIER 3: Nature -> Stress -> HRV Mediation (Paper 7)
    # ═══════════════════════════════════════════════════════════════════
    {
        "tier": 3,
        "name": "nature_restoration",
        "inputs": {"NatureEngagement": "engaged", "DAC": "available"},
        "context": {},
        "outputs": {"PSS": "-1", "HRV": "+1"},
        "confidence": 0.85,
        "citation": "Paper 7: BETA_NATURE_PSS=-1.507, mediation pathway",
    },
    {
        "tier": 3,
        "name": "nature_attention_trap",
        "inputs": {"NatureEngagement": "low", "DAC": "depleted"},
        "context": {},
        "outputs": {},
        "confidence": 0.9,
        "citation": "Paper 8: depleted attention blocks restoration",
    },
    {
        "tier": 3,
        "name": "nature_partial_restoration",
        "inputs": {"NatureEngagement": "engaged", "DAC": "depleted"},
        "context": {},
        "outputs": {"PSS": "0"},
        "confidence": 0.7,
        "citation": "Paper 8: engaged but depleted = reduced benefit",
    },
    {
        "tier": 3,
        "name": "therapy_depression_reduction",
        "inputs": {},
        "context": {"therapy_active": True},
        "outputs": {"Depression": "-1", "ARR": "-1"},
        "confidence": 0.8,
        "citation": "Paper 7: BETA_THERAPY_STRESS=-0.102, BETA_THERAPY_ARR=-0.012",
    },
    {
        "tier": 3,
        "name": "nature_wellbeing_boost",
        "inputs": {"NatureEngagement": "engaged"},
        "context": {},
        "outputs": {"WEMWBS": "+1"},
        "confidence": 0.75,
        "citation": "Paper 7: BETA_NATURE_WEMWBS=0.104/week",
    },

    # ═══════════════════════════════════════════════════════════════════
    # TIER 4: Sleep Debt & Activity (Paper 6)
    # ═══════════════════════════════════════════════════════════════════
    {
        "tier": 4,
        "name": "school_day_sleep_debt",
        "inputs": {},
        "context": {"is_school_day": True},
        "outputs": {"TST": "-1"},
        "confidence": 0.95,
        "citation": "Paper 6: SLEEP_DEBT_WEEKDAY=45min/school night",
    },
    {
        "tier": 4,
        "name": "weekend_sleep_recovery",
        "inputs": {},
        "context": {"is_weekend": True},
        "outputs": {"TST": "+1"},
        "confidence": 0.85,
        "citation": "Paper 6: partial recovery on weekends",
    },
    {
        "tier": 4,
        "name": "sleep_deprived_activity_compensation",
        "inputs": {"TST": "deprived"},
        "context": {},
        "outputs": {"Activity": "+1"},
        "confidence": 0.7,
        "citation": "Paper 6: BETA_TST_ACTIVITY=-0.021, paradoxical compensation",
    },
    {
        "tier": 4,
        "name": "school_day_activity_boost",
        "inputs": {},
        "context": {"is_school_day": True},
        "outputs": {"Activity": "+1"},
        "confidence": 0.8,
        "citation": "Paper 6: BETA_SCHOOL_ACTIVITY=+0.137",
    },

    # ═══════════════════════════════════════════════════════════════════
    # TIER 5: Chronotype & Sleep Shape (Papers 2, 6)
    # ═══════════════════════════════════════════════════════════════════
    {
        "tier": 5,
        "name": "late_chronotype_jetlag",
        "inputs": {},
        "context": {"late_chronotype": True, "is_school_day": True},
        "outputs": {"SocialJetlag": "misaligned"},
        "confidence": 0.9,
        "citation": "Paper 6: SJL_WEEKDAY_FORCING=0.924h",
    },
    {
        "tier": 5,
        "name": "female_mh_disrupted_sleep",
        "inputs": {},
        "context": {"female": True, "mh_diagnosis": True},
        "outputs": {"SleepShape": "disrupted"},
        "confidence": 0.8,
        "citation": "Paper 2: SHAPE_FEMALE_MH_COEFF=0.3",
    },
    {
        "tier": 5,
        "name": "female_trauma_disrupted_sleep",
        "inputs": {},
        "context": {"female": True, "trauma_high": True},
        "outputs": {"SleepShape": "disrupted"},
        "confidence": 0.75,
        "citation": "Paper 2: SHAPE_FEMALE_TRAUMA_COEFF=0.2",
    },
    {
        "tier": 5,
        "name": "weekend_jetlag_recovery",
        "inputs": {},
        "context": {"is_weekend": True},
        "outputs": {"SocialJetlag": "aligned"},
        "confidence": 0.7,
        "citation": "Paper 6: SJL recovers on free days",
    },

    # ═══════════════════════════════════════════════════════════════════
    # TIER 6: Attention Restoration (Paper 8)
    # ═══════════════════════════════════════════════════════════════════
    {
        "tier": 6,
        "name": "academic_attention_depletion",
        "inputs": {},
        "context": {"academic_load_high": True},
        "outputs": {"DAC": "-1"},
        "confidence": 0.9,
        "citation": "Paper 8: DAC_DEPLETION=0.3/week at full load",
    },
    {
        "tier": 6,
        "name": "nature_attention_restoration",
        "inputs": {"NatureEngagement": "engaged", "DAC": "available"},
        "context": {},
        "outputs": {"DAC": "+1"},
        "confidence": 0.8,
        "citation": "Paper 8: DAC_RESTORATION=0.2/week",
    },
    {
        "tier": 6,
        "name": "depleted_engagement_gate",
        "inputs": {"DAC": "depleted"},
        "context": {},
        "outputs": {},
        "confidence": 0.85,
        "citation": "Paper 8: depleted DAC closes engagement quality gate",
    },
    {
        "tier": 6,
        "name": "nature_dac_partial_restore",
        "inputs": {"NatureEngagement": "engaged", "DAC": "depleted"},
        "context": {},
        "outputs": {"DAC": "+1"},
        "confidence": 0.6,
        "citation": "Paper 8: nature restores even from depleted, but slowly",
    },

    # ═══════════════════════════════════════════════════════════════════
    # CROSS-TIER COMPOUND RULES
    # ═══════════════════════════════════════════════════════════════════
    {
        "tier": 0,
        "name": "burnout_cascade",
        "inputs": {
            "TST": "deprived",
            "PSS": "high",
            "DAC": "depleted",
            "GAD7": "clinical",
        },
        "context": {},
        "outputs": {},
        "confidence": 0.95,
        "citation": "Cross-tier: all restoration pathways blocked (absorbing state)",
    },
    {
        "tier": 0,
        "name": "spring_break_reset",
        "inputs": {},
        "context": {"is_spring_break": True},
        "outputs": {"TST": "+1", "PSS": "-1", "Activity": "-1", "SocialJetlag": "aligned"},
        "confidence": 0.9,
        "citation": "Cross-tier: spring break removes institutional forcing",
    },
    {
        "tier": 0,
        "name": "within_person_amplification",
        "inputs": {"TST": "deprived"},
        "context": {"tst_bin_dropped": True},
        "outputs": {"PSS": "+1"},
        "confidence": 0.85,
        "citation": "Paper 3: within-person deviation 2.2x amplification",
    },
    {
        "tier": 0,
        "name": "stress_depression_cascade",
        "inputs": {"PSS": "high", "GAD7": "clinical"},
        "context": {},
        "outputs": {"Depression": "+1"},
        "confidence": 0.8,
        "citation": "Cross-tier: DEPRESSION_PSS=0.3, DEPRESSION_GAD7=0.2",
    },
    {
        "tier": 0,
        "name": "sleep_quality_tracks_duration",
        "inputs": {"TST": "deprived"},
        "context": {},
        "outputs": {"SleepQuality": "-1"},
        "confidence": 0.8,
        "citation": "Cross-tier: SLEEP_QUALITY_TST_COUPLING=10.0 points/hour",
    },
    {
        "tier": 0,
        "name": "good_sleep_quality_improvement",
        "inputs": {"TST": "adequate"},
        "context": {},
        "outputs": {"SleepQuality": "+1"},
        "confidence": 0.7,
        "citation": "Cross-tier: adequate sleep improves quality score",
    },
]


def _evaluate_context(context_spec: dict, context: dict) -> bool:
    """Check whether all context conditions in a rule are satisfied."""
    for key, expected in context_spec.items():
        if key == "is_school_day":
            if context.get("is_school_day", False) != expected:
                return False
        elif key == "is_weekend":
            if context.get("is_weekday", True) == expected:
                return False
        elif key == "is_spring_break":
            if context.get("is_spring_break", False) != expected:
                return False
        elif key == "week_gt_4":
            if expected and context.get("week", 0) <= 4:
                return False
        elif key == "emotional_stability_low":
            if expected and context.get("emotional_stability", 4.5) >= 4.0:
                return False
        elif key == "emotional_stability_high":
            if expected and context.get("emotional_stability", 4.5) < 5.0:
                return False
        elif key == "mh_diagnosis":
            if expected and context.get("mh_diagnosis", 0.0) < 0.5:
                return False
        elif key == "therapy_active":
            if expected and context.get("therapy_rx", 0.0) < 0.2:
                return False
        elif key == "late_chronotype":
            if expected and context.get("baseline_chronotype", 4.5) < 5.0:
                return False
        elif key == "female":
            if expected and context.get("gender", 1.0) > 0.5:
                return False
        elif key == "trauma_high":
            if expected and context.get("trauma_load", 0.0) < 2.0:
                return False
        elif key == "academic_load_high":
            if expected and context.get("academic_load", 0.5) <= 0.6:
                return False
        elif key == "tst_bin_dropped":
            if expected and not context.get("tst_bin_dropped", False):
                return False
    return True


def _evaluate_inputs(input_spec: dict, discrete_state: dict) -> bool:
    """Check whether all input bin conditions in a rule match."""
    for var_name, required_label in input_spec.items():
        if discrete_state.get(var_name) != required_label:
            return False
    return True


def _apply_direction(
    current_label: str,
    direction: str,
    var_name: str,
) -> str:
    """Apply a directional update (+1, -1, 0) or absolute bin assignment."""
    schema = BIN_SCHEMA[var_name]
    labels = schema["labels"]

    # Absolute assignment: the direction IS the target bin label
    if direction in labels:
        return direction

    # Directional: "+1" means move up one bin, "-1" means move down
    if direction == "0":
        return current_label

    current_idx = labels.index(current_label)
    if direction == "+1":
        new_idx = min(current_idx + 1, len(labels) - 1)
    elif direction == "-1":
        new_idx = max(current_idx - 1, 0)
    else:
        return current_label

    return labels[new_idx]


def get_applicable_rules(
    discrete_state: dict[str, str],
    context: dict,
    rules: list[dict] | None = None,
) -> list[dict]:
    """Return the subset of rules whose conditions are satisfied.

    Parameters
    ----------
    discrete_state : dict[str, str]
        Current discretized state {var_name: bin_label}.
    context : dict
        Calendar and patient context (day, week, is_weekday, is_school_day,
        patient params, intervention params, etc.).
    rules : list[dict] or None
        Rule table to evaluate. Defaults to RULE_TABLE.

    Returns
    -------
    list[dict]
        Rules whose input and context conditions all match.
    """
    if rules is None:
        rules = RULE_TABLE
    applicable = []
    for rule in rules:
        if not _evaluate_inputs(rule["inputs"], discrete_state):
            continue
        if not _evaluate_context(rule["context"], context):
            continue
        applicable.append(rule)
    return applicable


def apply_rules(
    discrete_state: dict[str, str],
    context: dict,
    rules: list[dict] | None = None,
) -> tuple[dict[str, str], list[dict]]:
    """Apply all matching rules to produce the next discrete state.

    Rules are applied in tier order (1-6, then cross-tier 0). Within a tier,
    all matching rules contribute. If multiple rules try to update the same
    variable, the one with higher confidence wins. For the burnout cascade
    (absorbing state), no outputs are applied -- the state is frozen.

    Parameters
    ----------
    discrete_state : dict[str, str]
        Current discretized state.
    context : dict
        Calendar and patient context.
    rules : list[dict] or None
        Rule table. Defaults to RULE_TABLE.

    Returns
    -------
    tuple[dict[str, str], list[dict]]
        (new_state, fired_rules) — the updated discrete state and the list
        of rules that fired.
    """
    applicable = get_applicable_rules(discrete_state, context, rules)

    # Check for burnout cascade (absorbing state) — freeze the state
    for rule in applicable:
        if rule["name"] == "burnout_cascade":
            return dict(discrete_state), applicable

    # Collect proposed updates, resolve conflicts by confidence
    # {var_name: (direction, confidence, rule_name)}
    proposals: dict[str, tuple[str, float, str]] = {}
    for rule in applicable:
        for var_name, direction in rule["outputs"].items():
            if var_name not in proposals or rule["confidence"] > proposals[var_name][1]:
                proposals[var_name] = (direction, rule["confidence"], rule["name"])

    # Apply winning proposals
    new_state = dict(discrete_state)
    for var_name, (direction, confidence, _) in proposals.items():
        new_state[var_name] = _apply_direction(
            new_state[var_name], direction, var_name
        )

    return new_state, applicable


def save_rules(path: str, rules: list[dict] | None = None) -> None:
    """Save rule table to JSON."""
    if rules is None:
        rules = RULE_TABLE
    with open(path, "w") as f:
        json.dump(rules, f, indent=2)


def load_rules(path: str) -> list[dict]:
    """Load rule table from JSON."""
    with open(path) as f:
        return json.load(f)
