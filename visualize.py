"""Visualization for the LEMURS semester simulator.

┌─────────────────────────────────────────────────────────────────────────────┐
│                     WATCHING A SEMESTER UNFOLD                             │
│                                                                            │
│  A semester is 105 days of coupled dynamics: sleep debt accumulating on   │
│  weeknights, stress ratcheting up toward finals, anxiety crossing and    │
│  recrossing clinical thresholds, directed attention draining under       │
│  academic load and restoring in green spaces. These visualizations       │
│  organize the 14 state variables into four panels that together tell     │
│  the story of how a student's body and mind co-evolve from move-in      │
│  day to finals week.                                                     │
│                                                                            │
│    Panel 1 — Sleep: Is the student sleeping enough? How is their Oura    │
│      score trending? How much social jetlag are they carrying?           │
│                                                                            │
│    Panel 2 — Stress & Mood: Are they stressed? Anxious? Depressed?      │
│      Has GAD-7 crossed the clinical threshold of 10?                     │
│                                                                            │
│    Panel 3 — Physiology: What do the wearable biomarkers show? Is RHR   │
│      creeping up? Is HRV falling? Is respiratory rate elevated?          │
│                                                                            │
│    Panel 4 — Behavioral: Are they active? Engaging with nature? Is      │
│      well-being holding? Is directed attention capacity depleted?        │
│                                                                            │
│  The comparison plots overlay all 8 student archetypes — the resilient   │
│  male, the vulnerable female, the sleep-deprived student, the nature     │
│  seeker — so you can see where their trajectories diverge. Same campus,  │
│  same 15 weeks, dramatically different outcomes.                         │
└─────────────────────────────────────────────────────────────────────────────┘

Uses the Agg (non-interactive) matplotlib backend so plots can be generated
on servers and in automated pipelines without needing a display.
All output goes to PNG files in the output/ directory.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless backend — no GUI window needed
import matplotlib.pyplot as plt
import numpy as np

from constants import STUDENT_ARCHETYPES, SPRING_BREAK_WEEK
from simulator import simulate


# ── Spring break bounds in weeks ──────────────────────────────────────────
# SPRING_BREAK_WEEK is 1-indexed (week 8), so the 0-indexed week is 7.
# That means days 49-55, or weeks 7.0-8.0.
_BREAK_START = SPRING_BREAK_WEEK - 1  # week 7 (0-indexed start)
_BREAK_END = SPRING_BREAK_WEEK        # week 8 (0-indexed end)


def _add_spring_break(ax, label: bool = True) -> None:
    """Shade the spring break region on an axis.

    The gold band marks the week when school-day forcing disappears:
    no classes, no early alarms, no academic stressors. Sleep debt
    recovers, activity drops, and stress temporarily abates.
    """
    ax.axvspan(
        _BREAK_START, _BREAK_END,
        alpha=0.15, color="gold",
        label="Spring Break" if label else None,
    )


def plot_trajectory(result: dict, title: str, output_path: str) -> None:
    """Generate a 4-panel trajectory plot for a single simulation.

    This is the workhorse visualization. Given one student's simulated
    semester trajectory, it creates four panels that together tell the
    whole story — from sleep patterns through the stress cascade to
    behavioral outcomes and attention dynamics.

    Reading the panels:
      Top-left (Sleep): The blue TST line should hover around 7 hours.
        Weekday dips and weekend recoveries create an oscillating pattern.
        During spring break (gold band), sleep recovers fully. The orange
        sleep score tracks TST but also captures quality (REM, efficiency).
        The green dashed social jetlag line shows circadian misalignment.

      Top-right (Stress & Mood): PSS (red) is the central hub — driven by
        sleep loss, biomarker deviations, and academic load. GAD-7 (purple)
        has threshold dynamics: once it crosses the gray dashed line at 10,
        recovery becomes harder. Depression (brown) tracks stress and
        anxiety comorbidity.

      Bottom-left (Physiology): The wearable biomarkers. RHR (left axis)
        rises with sustained stress. HRV (right axis, green) is vagal
        tone — higher is healthier. ARR (right axis, orange dashed) is
        nocturnal respiratory rate — elevated ARR signals stress even when
        the student doesn't report it.

      Bottom-right (Behavioral & Well-Being): Activity (blue) shows the
        paradoxical sleep-deprivation compensation (less sleep = more
        movement). Nature engagement (green) drives attention restoration.
        WEMWBS well-being (purple dashed) is the positive-psychology
        outcome. DAC (red dotted) is directed attention — watch for
        depletion under heavy academic load.
    """
    states = result["states"]
    times = result["times"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── Panel 1: Sleep ────────────────────────────────────────────────
    # TST is the raw hours; SleepQuality (Oura score) captures quality
    # beyond duration; SocialJetlag shows circadian misalignment that
    # drives the weekday sleep debt cycle (Paper 6).
    ax = axes[0, 0]
    ax.plot(times, states[:, 0], label="TST (hours)", color="C0")
    ax.plot(times, states[:, 1] / 10.0, label="Sleep Score /10", color="C1")
    ax.plot(times, states[:, 10], label="Social Jetlag (hrs)",
            color="C2", linestyle="--")
    _add_spring_break(ax)
    ax.set_xlabel("Time (weeks)")
    ax.set_ylabel("Hours / Score")
    ax.set_title("Sleep")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Stress & Mood ────────────────────────────────────────
    # PSS is the central hub of the model. GAD-7 has Markov threshold
    # dynamics: crossing 10 activates a different dynamical regime
    # where recovery is harder (Paper 4). Depression co-evolves with
    # stress and anxiety comorbidity.
    ax = axes[0, 1]
    ax.plot(times, states[:, 2], label="PSS (0-40)", color="C3")
    ax.plot(times, states[:, 3], label="GAD-7 (0-21)", color="C4")
    ax.plot(times, states[:, 4], label="Depression (0-21)", color="C5")
    ax.axhline(y=10.0, color="gray", linestyle="--", linewidth=1,
               label="GAD-7 threshold")
    ax.set_xlabel("Time (weeks)")
    ax.set_ylabel("Score")
    ax.set_title("Stress & Mood")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Physiology ───────────────────────────────────────────
    # Wearable biomarkers from Paper 3. RHR on the left axis (bpm);
    # HRV (ms RMSSD) and ARR (breaths/min) on the right axis because
    # they live on different scales. RHR up + HRV down = stress.
    ax = axes[1, 0]
    ax.plot(times, states[:, 7], label="RHR (bpm)", color="C0")
    ax.set_xlabel("Time (weeks)")
    ax.set_ylabel("RHR (bpm)")
    ax.set_title("Physiology")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(times, states[:, 8], label="HRV (ms RMSSD)", color="C2")
    ax2.plot(times, states[:, 9], label="ARR (breaths/min)",
             color="C1", linestyle="--")
    ax2.set_ylabel("HRV / ARR")

    # Combine legends from both y-axes
    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = ax2.get_legend_handles_labels()
    ax.legend(lines_l + lines_r, labels_l + labels_r, fontsize=8,
              loc="upper right")

    # ── Panel 4: Behavioral & Well-Being ──────────────────────────────
    # Activity (scaled by /10 for readability), nature engagement,
    # WEMWBS well-being (/10 for readability), and directed attention
    # capacity (DAC). The spring break band shows where academic load
    # vanishes and DAC can recover.
    ax = axes[1, 1]
    ax.plot(times, states[:, 5] / 10.0, label="Activity (kcal/10)",
            color="C0")
    ax.plot(times, states[:, 6], label="Nature (hrs/wk)", color="C2")
    ax.plot(times, states[:, 12] / 10.0, label="Well-Being /10",
            color="C4", linestyle="--")
    ax.plot(times, states[:, 13], label="Attention (DAC)",
            color="C3", linestyle=":")
    _add_spring_break(ax)
    ax.set_xlabel("Time (weeks)")
    ax.set_title("Behavioral & Well-Being")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_comparison(
    results: list[dict],
    labels: list[str],
    output_path: str,
    title: str = "Semester Comparison",
) -> None:
    """Overlay key trajectories from multiple student archetypes side by side.

    This is the "aha" plot — the one that shows how different students with
    different vulnerabilities and interventions diverge across the same
    15-week semester.

    Three panels, each showing all archetypes overlaid:

      Left — PSS (Perceived Stress): Who stays calm and who spirals?
        The resilient students maintain low, stable PSS. The vulnerable
        students ratchet upward across the semester.

      Center — GAD-7 (Anxiety): Who crosses the clinical threshold (10)?
        The gray dashed line marks the cutoff. Students who cross it
        enter a different dynamical regime where recovery is harder —
        the Markov persistence from Paper 4.

      Right — HRV (Heart Rate Variability): Who maintains good vagal
        tone? HRV is the body's resilience indicator. Nature engagement
        and low stress keep HRV high; chronic stress and poor sleep
        erode it.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for i, (result, label) in enumerate(zip(results, labels)):
        times = result["times"]
        states = result["states"]
        axes[0].plot(times, states[:, 2], label=label, color=colors[i])
        axes[1].plot(times, states[:, 3], label=label, color=colors[i])
        axes[2].plot(times, states[:, 8], label=label, color=colors[i])

    # Panel 1: PSS
    axes[0].set_title("PSS (Perceived Stress)")
    axes[0].set_xlabel("Time (weeks)")
    axes[0].set_ylabel("PSS (0-40)")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: GAD-7
    axes[1].set_title("GAD-7 (Anxiety)")
    axes[1].set_xlabel("Time (weeks)")
    axes[1].set_ylabel("GAD-7 (0-21)")
    axes[1].axhline(y=10.0, color="gray", linestyle="--", linewidth=1,
                     label="Clinical threshold")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: HRV
    axes[2].set_title("HRV (Heart Rate Variability)")
    axes[2].set_xlabel("Time (weeks)")
    axes[2].set_ylabel("HRV (ms RMSSD)")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_all_scenarios(output_dir: str = "output") -> None:
    """Run all 8 student archetypes and generate every plot.

    This function is the "generate everything" button. It:

      1. Runs each of the 8 student archetypes through the full ODE simulator
         (resilient male, vulnerable female, sleep-deprived, nature seeker, etc.)

      2. Saves an individual 4-panel trajectory plot for each archetype,
         so you can study each student's journey through the semester in detail.

      3. Creates a comparison overlay showing all 8 archetypes side by side,
         so you can see how different vulnerabilities and interventions lead
         to different outcomes across the same 15-week semester.

    The 8 archetypes were designed to span the space of student experiences
    described in the LEMURS papers — from the most resilient to the most
    vulnerable, from minimal intervention to full nature+exercise+therapy.
    Together, they paint a picture of how variable the college experience
    really is, and where interventions can change the trajectory.
    """
    os.makedirs(output_dir, exist_ok=True)

    results = []
    labels = []

    for seed in STUDENT_ARCHETYPES:
        patient = seed["patient"]
        intervention = seed.get("intervention", None)
        result = simulate(patient=patient, intervention=intervention)
        results.append(result)
        labels.append(seed["name"])

        path = os.path.join(output_dir, f"trajectory_{seed['name']}.png")
        plot_trajectory(result, f"{seed['name']}: {seed['description']}", path)

    # Comparison overlay — all archetypes on one plot
    plot_comparison(
        results, labels,
        os.path.join(output_dir, "comparison_all_scenarios.png"),
        title="All Student Archetypes — Semester Comparison",
    )


if __name__ == "__main__":
    print("Generating LEMURS simulator visualizations...")
    plot_all_scenarios()
    print("Done.")
