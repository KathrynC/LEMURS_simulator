"""Zimmerman toolkit full analysis for the LEMURS semester simulator.

Runs all 14 Zimmerman interrogation tools against LEMURSBridge and saves
results to artifacts/zimmerman/. Generates a unified dashboard report.

Usage:
    python zimmerman_analysis.py                           # all tools
    python zimmerman_analysis.py --tools sobol             # single tool
    python zimmerman_analysis.py --tools sobol,falsifier   # multiple tools
    python zimmerman_analysis.py --tools sobol --n-base 256  # full Sobol
    python zimmerman_analysis.py --student vulnerable_female # archetype
    python zimmerman_analysis.py --intervention-only       # 6D mode

Requires:
    zimmerman-toolkit (at ~/zimmerman-toolkit)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# -- Path setup ---------------------------------------------------------------

PROJECT = Path(__file__).resolve().parent
ZIMMERMAN_PATH = PROJECT.parent / "zimmerman-toolkit"
if str(ZIMMERMAN_PATH) not in sys.path:
    sys.path.insert(0, str(ZIMMERMAN_PATH))

# Project imports
from constants import (
    INTERVENTION_NAMES, PATIENT_NAMES,
    INTERVENTION_BOUNDS, PATIENT_BOUNDS,
    DEFAULT_INTERVENTION, DEFAULT_PATIENT,
    GAD7_THRESHOLD,
    STUDENT_ARCHETYPES,
)
from analytics import NumpyEncoder
from zimmerman_bridge import LEMURSBridge

# Zimmerman imports
from zimmerman.sobol import sobol_sensitivity
from zimmerman.falsifier import Falsifier
from zimmerman.contrastive import ContrastiveGenerator
from zimmerman.contrast_set_generator import ContrastSetGenerator
from zimmerman.pds import PDSMapper
from zimmerman.posiwid import POSIWIDAuditor
from zimmerman.prompts import PromptBuilder
from zimmerman.locality_profiler import LocalityProfiler
from zimmerman.relation_graph_extractor import RelationGraphExtractor
from zimmerman.diegeticizer import Diegeticizer
from zimmerman.token_extispicy import TokenExtispicyWorkbench
from zimmerman.prompt_receptive_field import PromptReceptiveField
from zimmerman.supradiegetic_benchmark import SuperdiegeticBenchmark
from zimmerman.meaning_construction_dashboard import MeaningConstructionDashboard

# -- Configuration ------------------------------------------------------------

ARTIFACTS_DIR = PROJECT / "artifacts" / "zimmerman"

# Student profiles for intervention-only mode (from STUDENT_ARCHETYPES)
STUDENT_PROFILES = {
    a["name"]: a.get("patient", {}) for a in STUDENT_ARCHETYPES
}
STUDENT_PROFILES["default"] = dict(DEFAULT_PATIENT)

# Tool names in execution order
ALL_TOOLS = [
    "sobol", "falsifier", "contrastive", "contrast_sets",
    "pds", "posiwid", "prompts",
    "locality", "relation_graph", "diegeticizer",
    "token_extispicy", "receptive_field", "supradiegetic_benchmark",
    "dashboard",
]


# -- Helper: outcome function for contrastive/contrast set tools ---------------

def _anxiety_outcome(result: dict) -> str:
    """Classify simulation outcome by GAD-7 threshold crossing.

    Paper 4: GAD-7 >= 10 defines clinical anxiety. The model has bistable
    dynamics at this threshold -- once crossed, recovery probability decays
    each week (RECOVERY_HYSTERESIS=0.92). This outcome function captures
    that clinical boundary.
    """
    gad7 = result.get("stress_anxiety_gad7_mean", 7.0)
    pss = result.get("stress_anxiety_pss_mean", 16.0)
    if gad7 >= GAD7_THRESHOLD:
        return "anxious"
    elif pss >= 20.0:
        return "stressed"
    else:
        return "healthy"


# -- Helper: default midpoint params ------------------------------------------

def _midpoint_params(sim: LEMURSBridge) -> dict[str, float]:
    """Return parameter values at the midpoint of each range."""
    spec = sim.param_spec()
    return {k: (lo + hi) / 2 for k, (lo, hi) in spec.items()}


def _default_full_params(sim: LEMURSBridge | None = None) -> dict[str, float]:
    """Return default parameter dict aligned to a simulator's param spec.

    If sim is None, returns the canonical 12D LEMURS defaults.
    If sim is provided, returns a full parameter dict matching sim.param_spec():
      - known LEMURS keys use project defaults
      - unknown keys are initialized to range midpoints
    """
    base = {**DEFAULT_INTERVENTION, **DEFAULT_PATIENT}
    if sim is None:
        return base

    spec = sim.param_spec()
    out = {}
    for k, (lo, hi) in spec.items():
        if k in base:
            out[k] = float(base[k])
        else:
            out[k] = float((lo + hi) / 2.0)
    return out


# -- Tool Runners --------------------------------------------------------------


def run_sobol(sim: LEMURSBridge, n_base: int = 256, seed: int = 42) -> dict:
    """Run Sobol global sensitivity analysis.

    Total sims: n_base * (2*D + 2) where D = number of params.
    For 12D with n_base=256: 256 * 26 = 6,656 sims.
    For 6D with n_base=256: 256 * 14 = 3,584 sims.
    """
    print(f"  Sobol: n_base={n_base}, D={len(sim.param_spec())}")
    result = sobol_sensitivity(sim, n_base=n_base, seed=seed)
    print(f"  Sobol: {result.get('n_total_sims', '?')} sims completed")
    return result


def run_falsifier(sim: LEMURSBridge, seed: int = 42) -> dict:
    """Run systematic falsification with LEMURS-specific assertions.

    Checks biological plausibility: PSS stays in [0, 40], GAD-7 in [0, 21],
    TST in [4, 12], HRV positive, no NaN/Inf in outputs.
    """

    def pss_in_range(result):
        pss = result.get("stress_anxiety_pss_mean", 16.0)
        return 0.0 <= pss <= 40.0

    def gad7_in_range(result):
        gad7 = result.get("stress_anxiety_gad7_peak", 7.0)
        return 0.0 <= gad7 <= 21.0

    def tst_in_range(result):
        tst = result.get("sleep_quality_tst_mean", 7.0)
        return 4.0 <= tst <= 12.0

    def hrv_positive(result):
        hrv = result.get("physiological_hrv_mean", 65.0)
        return hrv > 0.0

    def no_nan(result):
        return all(
            np.isfinite(v) for v in result.values()
            if isinstance(v, (int, float))
        )

    falsifier = Falsifier(
        sim,
        assertions=[pss_in_range, gad7_in_range, tst_in_range,
                     hrv_positive, no_nan],
    )
    result = falsifier.falsify(n_random=100, n_boundary=50, seed=seed)
    n_violations = result.get("summary", {}).get("violations_found", 0)
    n_tests = result.get("summary", {}).get("total_tests", 0)
    print(f"  Falsifier: {n_violations}/{n_tests} violations")
    return result


def run_contrastive(sim: LEMURSBridge, seed: int = 42) -> dict:
    """Find minimal parameter changes that flip anxiety outcome.

    Searches for the smallest perturbation that pushes a student across
    the GAD-7 >= 10 clinical threshold -- the boundary between "managing"
    and "clinically anxious."
    """
    spec = sim.param_spec()
    dim = len(spec)

    starts = [
        _default_full_params(sim),
        _midpoint_params(sim),
    ]
    # Add a near-threshold starting point (elevated stress, borderline GAD-7)
    near_threshold = _default_full_params(sim)
    near_threshold["academic_load"] = 0.8
    near_threshold["emotional_stability"] = 3.5
    starts.append(near_threshold)

    gen = ContrastiveGenerator(sim, outcome_fn=_anxiety_outcome)
    n_per_point = 30 if dim <= 12 else 8
    pairs = gen.contrastive_pairs(starts, n_per_point=n_per_point, seed=seed)
    sensitivity = gen.sensitivity_from_contrastives(pairs) if pairs else {}

    result = {
        "n_pairs": len(pairs),
        "pairs": pairs,
        "sensitivity": sensitivity,
    }
    print(f"  Contrastive: {len(pairs)} flip pairs found "
          f"(n_per_point={n_per_point}, D={dim})")
    return result


def run_contrast_sets(sim: LEMURSBridge, seed: int = 42) -> dict:
    """Find minimal ordered edit sequences that flip anxiety outcome."""
    gen = ContrastSetGenerator(sim, outcome_fn=_anxiety_outcome)
    base = _default_full_params(sim)
    result = gen.batch_contrast_sets(base, n_paths=10, n_edits=20, seed=seed)
    n_tips = len(result.get("pairs", []))
    print(f"  ContrastSets: {n_tips} tipping points found, "
          f"mean flip size: {result.get('mean_flip_size', 'N/A')}")
    return result


def run_pds(sim: LEMURSBridge, seed: int = 42) -> dict:
    """Map Power/Danger/Structure dimensions to LEMURS parameters.

    PDS mapping for college student well-being (full 12D):
      Power     -> nature_rx, exercise_rx, therapy_rx (protective interventions)
      Danger    -> academic_load, trauma_load, mh_diagnosis (risk factors)
      Structure -> sleep_hygiene, baseline_chronotype, age, gender, emotional_stability

    In intervention-only mode (6D), adapts to available params:
      Power     -> nature_rx, exercise_rx, therapy_rx
      Danger    -> academic_load
      Structure -> sleep_hygiene, caffeine_reduction
    """
    available = set(sim.param_spec().keys())

    # Full 12D mapping
    full_mapping = {
        "power": {
            "nature_rx": 0.35,
            "exercise_rx": 0.30,
            "therapy_rx": 0.35,
        },
        "danger": {
            "academic_load": 0.35,
            "trauma_load": 0.30,
            "mh_diagnosis": 0.35,
        },
        "structure": {
            "sleep_hygiene": 0.25,
            "baseline_chronotype": 0.20,
            "age": 0.15,
            "gender": 0.15,
            "emotional_stability": 0.25,
        },
    }

    # Filter to available params and renormalize weights
    mapping = {}
    for dim_name, dim_params in full_mapping.items():
        filtered = {k: v for k, v in dim_params.items() if k in available}
        if filtered:
            total = sum(filtered.values())
            mapping[dim_name] = {k: v / total for k, v in filtered.items()}

    if not mapping:
        print("  PDS: no mappable parameters found")
        return {"error": "no mappable parameters"}
    pds = PDSMapper(sim, dimension_names=["power", "danger", "structure"],
                     dimension_to_param_mapping=mapping)
    audit = pds.audit_mapping(n_samples=100, seed=seed)
    sensitivity = pds.sensitivity_per_dimension(n_samples=100, seed=seed)
    result = {
        "mapping": mapping,
        "audit": audit,
        "sensitivity": sensitivity,
    }
    print(f"  PDS: variance explained: "
          f"{', '.join(f'{k}={v:.3f}' for k, v in audit.get('variance_explained', {}).items())}"
          [:80])
    return result


def run_posiwid(sim: LEMURSBridge, seed: int = 42) -> dict:
    """Audit alignment between intended and actual outcomes.

    Tests clinical intention scenarios: "I intend to reduce stress by X"
    against what the simulator actually produces when interventions are applied.
    """
    auditor = POSIWIDAuditor(sim)

    scenarios = [
        {
            "label": "Nature prescription for stressed student",
            "intended": {
                "stress_anxiety_pss_mean": 13.0,
                "physiological_hrv_mean": 70.0,
            },
            "params": {
                "nature_rx": 0.8, "exercise_rx": 0.0,
                "therapy_rx": 0.0, "sleep_hygiene": 0.3,
                "caffeine_reduction": 0.0, "academic_load": 0.5,
                **DEFAULT_PATIENT,
            },
        },
        {
            "label": "Full protocol for vulnerable student",
            "intended": {
                "stress_anxiety_pss_mean": 12.0,
                "stress_anxiety_gad7_mean": 6.0,
            },
            "params": {
                "nature_rx": 0.8, "exercise_rx": 0.6,
                "therapy_rx": 0.4, "sleep_hygiene": 0.8,
                "caffeine_reduction": 0.5, "academic_load": 0.5,
                "emotional_stability": 3.0, "mh_diagnosis": 1.0,
                "trauma_load": 2.0, "age": 20.0,
                "gender": 1.0, "baseline_chronotype": 4.5,
            },
        },
        {
            "label": "Sleep hygiene only for late chronotype",
            "intended": {
                "sleep_quality_tst_mean": 7.5,
                "sleep_quality_social_jetlag_mean": 0.5,
            },
            "params": {
                "nature_rx": 0.0, "exercise_rx": 0.0,
                "therapy_rx": 0.0, "sleep_hygiene": 1.0,
                "caffeine_reduction": 0.5, "academic_load": 0.5,
                "baseline_chronotype": 6.0,
                **{k: v for k, v in DEFAULT_PATIENT.items()
                   if k != "baseline_chronotype"},
            },
        },
        {
            "label": "Exercise-only prescription",
            "intended": {
                "stress_anxiety_pss_mean": 14.0,
                "intervention_response_pss_benefit": 2.0,
            },
            "params": {
                "nature_rx": 0.0, "exercise_rx": 0.8,
                "therapy_rx": 0.0, "sleep_hygiene": 0.3,
                "caffeine_reduction": 0.0, "academic_load": 0.5,
                **DEFAULT_PATIENT,
            },
        },
        {
            "label": "Therapy for anxious male student",
            "intended": {
                "stress_anxiety_gad7_mean": 7.0,
                "stress_anxiety_depression_mean": 7.0,
            },
            "params": {
                "nature_rx": 0.0, "exercise_rx": 0.0,
                "therapy_rx": 0.8, "sleep_hygiene": 0.3,
                "caffeine_reduction": 0.0, "academic_load": 0.5,
                "gender": 0.0, "mh_diagnosis": 1.0,
                "emotional_stability": 3.0, "trauma_load": 1.0,
                "age": 20.0, "baseline_chronotype": 4.5,
            },
        },
    ]

    result = auditor.batch_audit(scenarios)
    overall = result.get("aggregate", {}).get("mean_overall", 0.0)
    print(f"  POSIWID: {len(scenarios)} scenarios, mean alignment={overall:.3f}")
    return result


def run_prompts(sim: LEMURSBridge) -> dict:
    """Build prompt templates for LLM-mediated intervention design."""
    builder = PromptBuilder(sim, context={
        "domain": "College student biopsychosocial dynamics (LEMURS, UVM 2023-2025)",
        "goal": "Design semester intervention packages to reduce stress and prevent anxiety threshold crossing",
    })
    scenario = ("Sophomore female student, late chronotype (MSF_free=6.0), "
                "prior anxiety diagnosis, moderate trauma history, taking 18 credits. "
                "Design a semester-long intervention package.")

    result = {
        "numeric": builder.build_numeric(scenario),
        "diegetic": builder.build_diegetic(
            scenario,
            state_description="Current state: PSS=18.5, GAD7=9.2, TST=6.8hr, HRV=58ms"),
        "contrastive": builder.build_contrastive(
            scenario,
            agent_a="cautious campus counselor",
            agent_b="proactive wellness coach"),
    }
    print(f"  Prompts: 3 styles generated "
          f"(numeric={len(result['numeric'])}ch, "
          f"diegetic={len(result['diegetic'])}ch, "
          f"contrastive={len(result['contrastive'])}ch)")
    return result


def run_locality(sim: LEMURSBridge, seed: int = 42) -> dict:
    """Profile perturbation decay: how local are the system's responses?"""
    profiler = LocalityProfiler(sim)
    base = _default_full_params(sim)
    result = profiler.profile(task={"base_params": base}, n_seeds=10, seed=seed)
    n_sims = result.get("n_sims", "?")
    print(f"  Locality: {n_sims} sims, profiled decay curves")
    return result


def run_relation_graph(sim: LEMURSBridge, seed: int = 42) -> dict:
    """Build causal relation graph: param -> output influence."""
    extractor = RelationGraphExtractor(sim)
    base = _default_full_params(sim)
    result = extractor.extract(base, n_probes=50, seed=seed)
    n_causal = len(result.get("edges", {}).get("causal", []))
    print(f"  RelationGraph: {n_causal} causal edges, "
          f"{result.get('n_sims', '?')} sims")
    return result


def run_diegeticizer(sim: LEMURSBridge) -> dict:
    """Roundtrip diegeticization with campus well-being lexicon."""
    lexicon = {
        "nature_rx": "nature_prescription",
        "exercise_rx": "exercise_prescription",
        "therapy_rx": "counseling_engagement",
        "sleep_hygiene": "sleep_routine_quality",
        "caffeine_reduction": "stimulant_reduction",
        "academic_load": "course_pressure",
        "age": "student_age",
        "gender": "gender_identity",
        "emotional_stability": "emotional_resilience",
        "trauma_load": "adverse_experience_history",
        "mh_diagnosis": "prior_mental_health_diagnosis",
        "baseline_chronotype": "natural_sleep_timing",
    }
    for key in sim.param_spec().keys():
        lexicon.setdefault(key, key)
    dieg = Diegeticizer(sim, lexicon=lexicon, n_bins=5)
    params = _default_full_params(sim)
    narrative = dieg.diegeticize(params)
    recovered = dieg.re_diegeticize(narrative["narrative"])
    roundtrip = dieg.run(params)
    result = {
        "narrative": narrative,
        "recovered": recovered,
        "roundtrip": roundtrip,
        "lexicon": lexicon,
    }
    error = narrative.get("roundtrip_error", "?")
    print(f"  Diegeticizer: roundtrip error={error}")
    return result


def run_token_extispicy(sim: LEMURSBridge, seed: int = 42) -> dict:
    """Quantify tokenization-induced flattening as hazard surface."""
    workbench = TokenExtispicyWorkbench(sim)
    result = workbench.analyze(n_samples=100, seed=seed)
    frag_corr = result.get("fragmentation_output_correlation", "?")
    print(f"  TokenExtispicy: frag-output correlation={frag_corr}, "
          f"{result.get('n_sims', '?')} sims")
    return result


def run_receptive_field(sim: LEMURSBridge, seed: int = 42) -> dict:
    """Sobol analysis over parameter segment groupings.

    Segments for LEMURS:
      behavioral   -> nature_rx, exercise_rx, therapy_rx
      lifestyle    -> sleep_hygiene, caffeine_reduction, academic_load
      demographics -> age, gender, baseline_chronotype
      vulnerability -> emotional_stability, trauma_load, mh_diagnosis
    """
    lemurs_segments = [
        {"name": "behavioral",
         "params": ["nature_rx", "exercise_rx", "therapy_rx"]},
        {"name": "lifestyle",
         "params": ["sleep_hygiene", "caffeine_reduction", "academic_load"]},
        {"name": "demographics",
         "params": ["age", "gender", "baseline_chronotype"]},
        {"name": "vulnerability",
         "params": ["emotional_stability", "trauma_load", "mh_diagnosis"]},
    ]

    def lemurs_segmenter(spec):
        """Group LEMURS params into clinical segments."""
        available = set(spec.keys())
        result = []
        for seg in lemurs_segments:
            params = [p for p in seg["params"] if p in available]
            if params:
                result.append({"name": seg["name"], "params": params})
        return result

    field = PromptReceptiveField(sim, segmenter=lemurs_segmenter)
    base = _default_full_params()
    result = field.analyze(base_params=base, n_base=64, seed=seed)
    rankings = result.get("rankings", [])
    if isinstance(rankings, dict):
        print(f"  ReceptiveField: most influential={rankings.get('most_influential', '?')}")
    else:
        print(f"  ReceptiveField: rankings={rankings[:5] if rankings else '?'}")
    return result


def run_supradiegetic_benchmark(sim: LEMURSBridge, seed: int = 42) -> dict:
    """Run form-vs-meaning benchmark."""
    bench = SuperdiegeticBenchmark(sim)
    result = bench.run_benchmark(seed=seed)
    gain = result.get("summary", {}).get("mean_gain", "?")
    print(f"  SuperdiegeticBenchmark: mean diegeticization gain={gain}")
    return result


def run_dashboard(reports: dict, sim: LEMURSBridge) -> dict:
    """Compile all reports into unified dashboard."""
    dashboard = MeaningConstructionDashboard(sim)
    result = dashboard.compile(reports)
    coverage = result.get("coverage", {})
    n_recs = len(result.get("recommendations", []))
    print(f"  Dashboard: {coverage.get('tools_present', 0)}/"
          f"{coverage.get('tools_total', 0)} tools, "
          f"{n_recs} recommendations")
    return result


# -- Tool dispatcher -----------------------------------------------------------

TOOL_RUNNERS = {
    "sobol": lambda sim, args: run_sobol(sim, n_base=args.n_base),
    "falsifier": lambda sim, args: run_falsifier(sim),
    "contrastive": lambda sim, args: run_contrastive(sim),
    "contrast_sets": lambda sim, args: run_contrast_sets(sim),
    "pds": lambda sim, args: run_pds(sim),
    "posiwid": lambda sim, args: run_posiwid(sim),
    "prompts": lambda sim, args: run_prompts(sim),
    "locality": lambda sim, args: run_locality(sim),
    "relation_graph": lambda sim, args: run_relation_graph(sim),
    "diegeticizer": lambda sim, args: run_diegeticizer(sim),
    "token_extispicy": lambda sim, args: run_token_extispicy(sim),
    "receptive_field": lambda sim, args: run_receptive_field(sim),
    "supradiegetic_benchmark": lambda sim, args: run_supradiegetic_benchmark(sim),
    # dashboard is special -- needs all other reports
}


# -- Generate markdown report --------------------------------------------------

def _generate_markdown(reports: dict, dashboard: dict | None) -> str:
    """Generate a markdown summary of all Zimmerman analysis results."""
    lines = [
        "# Zimmerman Toolkit Analysis -- LEMURS Semester Simulator",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # Sobol summary
    if "sobol" in reports:
        sobol = reports["sobol"]
        lines.append("## Sobol Global Sensitivity")
        lines.append(f"- Base samples: {sobol.get('n_base', '?')}")
        lines.append(f"- Total sims: {sobol.get('n_total_sims', '?')}")
        lines.append(f"- Parameters: {sobol.get('parameter_names', [])}")
        for key in sobol.get("output_keys", [])[:5]:
            if key in sobol:
                s1 = sobol[key].get("S1", {})
                top3 = sorted(s1.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                lines.append(f"- **{key}** top S1: " +
                             ", ".join(f"{k}={v:.3f}" for k, v in top3))
        lines.append("")

    # Falsifier summary
    if "falsifier" in reports:
        summary = reports["falsifier"].get("summary", {})
        lines.append("## Falsification")
        lines.append(f"- Tests run: {summary.get('total_tests', '?')}")
        lines.append(f"- Violations: {summary.get('violations_found', 0)}")
        lines.append(f"- Violation rate: {summary.get('violation_rate', 0):.1%}")
        lines.append("")

    # Contrastive summary
    if "contrastive" in reports:
        contrastive = reports["contrastive"]
        lines.append("## Contrastive Analysis")
        lines.append(f"- Flip pairs found: {contrastive.get('n_pairs', 0)}")
        sens = contrastive.get("sensitivity", {})
        if "rankings" in sens:
            lines.append(f"- Most flip-prone params: {sens['rankings'][:5]}")
        lines.append("")

    # POSIWID summary
    if "posiwid" in reports:
        agg = reports["posiwid"].get("aggregate", {})
        lines.append("## POSIWID Alignment")
        lines.append(f"- Mean overall alignment: {agg.get('mean_overall', 0):.3f}")
        lines.append(f"- Direction accuracy: {agg.get('mean_direction_accuracy', 0):.3f}")
        lines.append(f"- Magnitude accuracy: {agg.get('mean_magnitude_accuracy', 0):.3f}")
        lines.append("")

    # PDS summary
    if "pds" in reports:
        pds_audit = reports["pds"].get("audit", {})
        lines.append("## PDS Mapping")
        ve = pds_audit.get("variance_explained", {})
        for k, v in ve.items():
            lines.append(f"- {k} variance explained: {v:.3f}")
        lines.append("")

    # Locality summary
    if "locality" in reports:
        locality = reports["locality"]
        lines.append("## Locality Profile")
        lines.append(f"- Simulations: {locality.get('n_sims', '?')}")
        horizon = locality.get("effective_horizon", "?")
        lines.append(f"- Effective horizon: {horizon}")
        lines.append("")

    # Dashboard summary
    if dashboard:
        coverage = dashboard.get("coverage", {})
        lines.append("## Dashboard Summary")
        lines.append(f"- Coverage: {coverage.get('tools_present', 0)}/"
                      f"{coverage.get('tools_total', 0)} tools "
                      f"({coverage.get('coverage_pct', 0):.0f}%)")
        recs = dashboard.get("recommendations", [])
        if recs:
            lines.append("### Recommendations")
            for rec in recs[:10]:
                if isinstance(rec, dict):
                    lines.append(f"- **{rec.get('finding', '')}** -> {rec.get('action', '')}")
                else:
                    lines.append(f"- {rec}")
        lines.append("")

    return "\n".join(lines)


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Zimmerman toolkit analysis for LEMURS semester simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available tools: {', '.join(ALL_TOOLS)}\n"
               f"Available students: {', '.join(STUDENT_PROFILES.keys())}",
    )
    parser.add_argument("--tools", type=str, default=None,
                        help="Comma-separated tool names (default: all)")
    parser.add_argument("--student", type=str, default=None,
                        help="Student archetype for intervention-only mode")
    parser.add_argument("--intervention-only", action="store_true",
                        help="Use 6D intervention-only mode (default student)")
    parser.add_argument("--n-base", type=int, default=256,
                        help="Sobol base sample count (default: 256)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    # Determine which tools to run
    if args.tools:
        tools = [t.strip() for t in args.tools.split(",")]
        for t in tools:
            if t not in ALL_TOOLS:
                print(f"Unknown tool: {t}. Available: {', '.join(ALL_TOOLS)}")
                sys.exit(1)
    else:
        tools = ALL_TOOLS

    # Create simulator
    if args.student or args.intervention_only:
        student_name = args.student or "default"
        patient = STUDENT_PROFILES.get(student_name, DEFAULT_PATIENT)
        sim = LEMURSBridge(intervention_only=True, patient_override=patient)
        print(f"Mode: intervention-only (student={student_name})")
    else:
        sim = LEMURSBridge()
        print(f"Mode: full 12D (intervention + student)")

    print(f"Parameters: {len(sim.param_spec())}D")
    print(f"Tools: {', '.join(tools)}")
    print(f"Sobol n_base: {args.n_base}")
    print()

    # Create output directory
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run tools
    reports = {}
    total_t0 = time.time()

    for tool_name in tools:
        if tool_name == "dashboard":
            continue  # run last
        if tool_name not in TOOL_RUNNERS:
            print(f"  Skipping unknown tool: {tool_name}")
            continue

        print(f"[{tool_name}]")
        t0 = time.time()
        try:
            result = TOOL_RUNNERS[tool_name](sim, args)
            reports[tool_name] = result
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s")

            # Save individual report
            out_path = ARTIFACTS_DIR / f"{tool_name}_report.json"
            out_path.write_text(json.dumps(result, indent=2, cls=NumpyEncoder,
                                            default=str))
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  ERROR: {e}")
            reports[tool_name] = {"error": str(e)}
        print()

    # Dashboard compilation
    dashboard_result = None
    if "dashboard" in tools:
        print("[dashboard]")
        t0 = time.time()
        try:
            dashboard_input = {}
            key_mapping = {
                "sobol": "sobol",
                "falsifier": "falsifier",
                "posiwid": "posiwid",
                "contrast_sets": "contrast_sets",
                "contrastive": "contrastive",
                "locality": "locality",
                "relation_graph": "relation_graph",
                "receptive_field": "receptive_field",
                "diegeticizer": "diegeticizer",
                "supradiegetic_benchmark": "benchmark",
                "token_extispicy": "token_extispicy",
                "pds": "pds",
            }
            for our_key, dash_key in key_mapping.items():
                if our_key in reports and "error" not in reports[our_key]:
                    dashboard_input[dash_key] = reports[our_key]

            dashboard_result = run_dashboard(dashboard_input, sim)
            reports["dashboard"] = dashboard_result
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s")

            out_path = ARTIFACTS_DIR / "dashboard.json"
            out_path.write_text(json.dumps(dashboard_result, indent=2,
                                            cls=NumpyEncoder, default=str))
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    # Generate markdown report
    md = _generate_markdown(reports, dashboard_result)
    md_path = ARTIFACTS_DIR / "dashboard.md"
    md_path.write_text(md)
    print(f"Markdown report: {md_path}")

    total_elapsed = time.time() - total_t0
    print(f"\nTotal time: {total_elapsed:.1f}s")
    print(f"Reports saved to: {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
