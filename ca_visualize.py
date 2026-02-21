"""Visualization for the LEMURS semantic cellular automaton.

A semester is 105 days of discrete state transitions: each day, tiered
rules evaluate the current clinical bins (deprived/adequate/excess for
sleep, sub_threshold/clinical for anxiety, etc.) and fire updates that
propagate through the coupling tiers. These visualizations render the
CA dynamics in ways that complement the continuous ODE trajectory plots.

  Trajectory heatmap -- A bird's-eye view of one student's entire semester
    as a 14-row x 105-column grid. Each cell is colored by bin index,
    making it easy to spot phase transitions (a row changing color) and
    spring break resets (a vertical column of simultaneous changes).

  Rule firing timeline -- Shows WHICH rules fired on WHICH days, colored
    by tier. Reveals the temporal structure of the rule cascade: Tier 1
    (sleep->stress) fires daily on school days, Tier 2 (anxiety Markov)
    activates only when conditions align, cross-tier compounds (burnout,
    spring break) are rare but dramatic.

  Fidelity comparison -- How well does the CA's discrete abstraction
    track the continuous ODE? Per-variable bar chart of bin agreement
    rates reveals which variables the CA captures faithfully (sleep,
    stress) and which it oversimplifies (HRV, respiratory rate).

  Population grid -- An NxN grid of students at selected days, colored
    by attractor category (healthy/struggling/stressed/burnout). Watch
    the grid evolve from mostly green on day 0 to a mosaic of outcomes
    by finals week, with spring break causing a visible relaxation.

Uses the Agg (non-interactive) matplotlib backend so plots can be generated
on servers and in automated pipelines without needing a display.
All output goes to PNG files in the output/ca/ directory.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless backend -- no GUI window needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from constants import STUDENT_ARCHETYPES, SPRING_BREAK_WEEK
from simulator import simulate
from ca_schema import BIN_SCHEMA, _VAR_ORDER, bin_index, bin_count
from ca_simulator import run_single_cell, run_population_grid
from ca_rules import RULE_TABLE
from ca_analytics import _classify_attractor, _fidelity_stats


# ── Spring break bounds in days ──────────────────────────────────────────
# SPRING_BREAK_WEEK is 1-indexed (week 8), so 0-indexed week index is 7.
# That means days 49-55.
_BREAK_DAY_START = (SPRING_BREAK_WEEK - 1) * 7   # day 49
_BREAK_DAY_END = SPRING_BREAK_WEEK * 7            # day 56


def plot_ca_trajectory(
    ca_result: dict,
    title: str,
    output_path: str,
) -> None:
    """Generate a state timeline showing each variable's clinical bin over time.

    Like the rule timeline, each row is a variable with colored horizontal
    bands showing which clinical bin it occupies. The bin LABEL is written
    inside each band so you can read the clinical meaning directly. Colors
    are semantically consistent: green = healthy/good, yellow = intermediate,
    red = clinically concerning — regardless of whether the bin index is
    high or low for that variable.

    Spring break (days 49-55) is highlighted with a gold band, making it
    easy to see the institutional-forcing removal and recovery effects.
    """
    trajectory = ca_result["trajectory"]
    n_days = len(trajectory)
    n_vars = len(_VAR_ORDER)

    # Semantic severity: for each variable, map bin labels to 0=good ... N=bad
    # This ensures green always means "healthy" and red always means "concerning"
    _SEVERITY = {
        "TST":               {"deprived": 2, "adequate": 0, "excess": 1},
        "SleepQuality":      {"poor": 2, "fair": 1, "good": 0},
        "PSS":               {"low": 0, "moderate": 1, "high": 2},
        "GAD7":              {"sub_threshold": 0, "clinical": 1},
        "Depression":        {"normal": 0, "mild": 1, "moderate_plus": 2},
        "Activity":          {"sedentary": 1, "moderate": 0, "active": 0},
        "NatureEngagement":  {"low": 1, "engaged": 0},
        "RHR":               {"low": 0, "normal": 0, "elevated": 1},
        "HRV":               {"low": 2, "moderate": 1, "high": 0},
        "ARR":               {"normal": 0, "elevated": 1},
        "SocialJetlag":      {"aligned": 0, "misaligned": 1},
        "SleepShape":        {"stable": 0, "disrupted": 1},
        "WEMWBS":            {"low": 2, "moderate": 1, "high": 0},
        "DAC":               {"depleted": 1, "available": 0},
    }

    # Severity colors: 0=green, 1=yellow, 2=red
    sev_colors = {0: "#2E8B57", 1: "#FFD700", 2: "#DC143C"}

    fig, ax = plt.subplots(figsize=(18, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for v, var_name in enumerate(_VAR_ORDER):
        # Find contiguous runs of the same bin label
        labels_seq = [
            trajectory[t].get(var_name, BIN_SCHEMA[var_name]["labels"][0])
            for t in range(n_days)
        ]
        sev_map = _SEVERITY.get(var_name, {})

        run_start = 0
        for t in range(1, n_days + 1):
            if t == n_days or labels_seq[t] != labels_seq[run_start]:
                label = labels_seq[run_start]
                severity = sev_map.get(label, 0)
                color = sev_colors[severity]

                # Draw colored band
                ax.barh(
                    v, t - run_start, left=run_start, height=0.8,
                    color=color, edgecolor="white", linewidth=0.3,
                )

                # Write bin label in the band if wide enough
                run_width = t - run_start
                if run_width >= 4:
                    ax.text(
                        run_start + run_width / 2, v, label,
                        ha="center", va="center", fontsize=6,
                        fontweight="bold", color="black" if severity < 2 else "white",
                    )

                run_start = t

    # Spring break band
    ax.axvspan(_BREAK_DAY_START, _BREAK_DAY_END, alpha=0.2, color="gold",
               label="Spring Break", zorder=0)

    # Y-axis: variable names with bin options
    ylabels = []
    for var_name in _VAR_ORDER:
        bins = "/".join(BIN_SCHEMA[var_name]["labels"])
        ylabels.append(f"{var_name}  ({bins})")
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(ylabels, fontsize=7.5)

    # X-axis: week markers
    week_days = list(range(0, n_days, 7))
    week_labels = [f"W{d // 7 + 1}" for d in week_days]
    ax.set_xticks(week_days)
    ax.set_xticklabels(week_labels, fontsize=8)
    ax.set_xlabel("Week of Semester")

    ax.set_xlim(-0.5, n_days + 0.5)
    ax.set_ylim(-0.5, n_vars - 0.5)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.15, axis="x")
    ax.legend(loc="upper right", fontsize=8)

    # Legend for severity colors
    from matplotlib.patches import Patch
    sev_legend = [
        Patch(facecolor="#2E8B57", label="Healthy/Good"),
        Patch(facecolor="#FFD700", label="Intermediate"),
        Patch(facecolor="#DC143C", label="Concerning"),
    ]
    ax.legend(handles=sev_legend + [Patch(facecolor="gold", alpha=0.3, label="Spring Break")],
              loc="upper right", fontsize=7, ncol=2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_rule_timeline(
    ca_result: dict,
    title: str,
    output_path: str,
) -> None:
    """Generate a rule firing timeline for the CA simulation.

    The x-axis is days (0-104), the y-axis lists every rule that fired
    at least once during the simulation. Each cell is colored by the
    rule's tier: Tier 1 (sleep->stress) in blue, Tier 2 (anxiety Markov)
    in purple, Tier 3 (nature mediation) in green, Tier 4 (sleep debt)
    in orange, Tier 5 (chronotype) in cyan, Tier 6 (attention) in red,
    and cross-tier compounds (tier 0) in gold.

    Reading the timeline:
      - Dense horizontal bands = rules that fire nearly every day
        (e.g., school_day_sleep_debt fires 75+ times on school days)
      - Sparse dots = conditional rules that activate only under specific
        state/context combinations (e.g., burnout_cascade)
      - Gaps at days 49-55 = spring break (school-day rules stop firing,
        spring_break_reset activates)
      - Vertical clusters = cascade events where one rule's output
        triggers multiple downstream rules on the same day
    """
    rule_log = ca_result["rule_log"]
    n_days = len(rule_log)

    # Collect all unique rule names that actually fired
    fired_rules_set: set[str] = set()
    for day_rules in rule_log:
        fired_rules_set.update(day_rules)

    if not fired_rules_set:
        # No rules fired -- create a minimal placeholder plot
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.text(0.5, 0.5, "No rules fired", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_path}")
        return

    # Build a name->tier lookup from RULE_TABLE
    rule_tier: dict[str, int] = {}
    for rule in RULE_TABLE:
        rule_tier[rule["name"]] = rule["tier"]

    # Sort fired rules by tier, then name
    fired_rules = sorted(fired_rules_set, key=lambda r: (rule_tier.get(r, 99), r))
    rule_to_idx = {name: i for i, name in enumerate(fired_rules)}
    n_rules = len(fired_rules)

    # Tier color map
    tier_colors = {
        0: "#DAA520",  # gold -- cross-tier compounds
        1: "#4169E1",  # royal blue -- sleep->stress
        2: "#8A2BE2",  # blue-violet -- anxiety Markov
        3: "#228B22",  # forest green -- nature mediation
        4: "#FF8C00",  # dark orange -- sleep debt & activity
        5: "#00CED1",  # dark turquoise -- chronotype
        6: "#DC143C",  # crimson -- attention restoration
    }

    # Build the firing grid and color grid
    fig_height = max(4, 0.3 * n_rules + 1.5)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for day_idx, day_rules in enumerate(rule_log):
        for rule_name in day_rules:
            if rule_name in rule_to_idx:
                y = rule_to_idx[rule_name]
                tier = rule_tier.get(rule_name, 99)
                color = tier_colors.get(tier, "#808080")
                ax.plot(day_idx, y, "s", color=color, markersize=3)

    # Y-axis: rule names
    ax.set_yticks(range(n_rules))
    ylabels = []
    for name in fired_rules:
        tier = rule_tier.get(name, "?")
        ylabels.append(f"[T{tier}] {name}")
    ax.set_yticklabels(ylabels, fontsize=7)

    # X-axis: week markers
    week_days = list(range(0, n_days, 7))
    week_labels = [f"W{d // 7 + 1}" for d in week_days]
    ax.set_xticks(week_days)
    ax.set_xticklabels(week_labels, fontsize=8)
    ax.set_xlabel("Week")
    ax.set_ylabel("Rule")

    # Spring break highlight
    ax.axvspan(_BREAK_DAY_START, _BREAK_DAY_END, alpha=0.15, color="gold",
               label="Spring Break")

    ax.set_xlim(-0.5, n_days - 0.5)
    ax.set_ylim(-0.5, n_rules - 0.5)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis="x")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_ca_fidelity(
    ca_result: dict,
    ode_result: dict,
    title: str,
    output_path: str,
) -> None:
    """Generate a two-panel fidelity report: per-variable bars + summary donut.

    Left panel: Horizontal bars showing CA-ODE bin agreement per variable,
    grouped by ODE coupling tier so you can see which tiers the CA handles
    well. Bars are color-coded green/yellow/red by agreement quality.

    Right panel: Donut chart with overall agreement percentage in the center,
    giving an instant read on the CA's fidelity as a whole.

    Reading the chart:
      - Green bars (>=80%): CA captures this variable faithfully
      - Yellow bars (>=60%): CA tracks the trend but misses some transitions
      - Red bars (<60%): CA's bins are too coarse for this variable
      - Tier groupings reveal whether entire coupling tiers are well-captured
    """
    fidelity = _fidelity_stats(ca_result["trajectory"], ode_result)
    per_var = fidelity["per_variable_agreement"]
    overall = fidelity["overall_agreement"]

    # Group variables by ODE coupling tier for readability
    tier_groups = [
        ("Tier 1: Sleep→Stress", ["TST", "SleepQuality", "PSS", "RHR", "HRV", "ARR"]),
        ("Tier 2: Anxiety",      ["GAD7", "Depression"]),
        ("Tier 3: Nature",       ["NatureEngagement", "WEMWBS"]),
        ("Tier 4-5: Rhythm",     ["Activity", "SocialJetlag", "SleepShape"]),
        ("Tier 6: Attention",    ["DAC"]),
    ]

    # Flatten into ordered list with group separators
    var_names = []
    group_boundaries = []
    for group_label, vars_in_group in reversed(tier_groups):
        group_boundaries.append((len(var_names), group_label))
        for v in reversed(vars_in_group):
            var_names.append(v)

    agreements = [per_var.get(v, 0.0) for v in var_names]

    # Color by agreement quality
    def _fidelity_color(a):
        if a >= 0.8:
            return "#2E8B57"   # green
        elif a >= 0.6:
            return "#FFD700"   # yellow
        return "#DC143C"       # red

    colors = [_fidelity_color(a) for a in agreements]

    fig, (ax_bars, ax_donut) = plt.subplots(
        1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [3, 1]}
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ── Left panel: horizontal bars ──
    y_pos = range(len(var_names))
    ax_bars.barh(y_pos, agreements, color=colors, edgecolor="white", height=0.7)

    # Overall agreement line
    ax_bars.axvline(x=overall, color="navy", linestyle="--", linewidth=1.5,
                    label=f"Overall: {overall:.0%}")

    # Tier group labels as horizontal spans
    for start_idx, group_label in group_boundaries:
        ax_bars.text(-0.02, start_idx - 0.1, group_label,
                     fontsize=7, color="#555", ha="right", va="top",
                     transform=ax_bars.get_yaxis_transform())

    ax_bars.set_yticks(y_pos)
    ax_bars.set_yticklabels(var_names, fontsize=9)
    ax_bars.set_xlabel("CA-ODE Bin Agreement (%)", fontsize=10)
    ax_bars.set_xlim(0, 1.08)
    xticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax_bars.set_xticks(xticks)
    ax_bars.set_xticklabels([f"{int(x*100)}%" for x in xticks])
    ax_bars.grid(True, alpha=0.2, axis="x")

    # Percentage labels on bars
    for i, a in enumerate(agreements):
        ax_bars.text(a + 0.01, i, f"{a:.0%}", va="center", fontsize=8,
                     fontweight="bold" if a >= 0.8 else "normal")

    ax_bars.legend(loc="lower right", fontsize=9)

    # ── Right panel: donut chart ──
    good = sum(1 for a in agreements if a >= 0.8)
    moderate = sum(1 for a in agreements if 0.6 <= a < 0.8)
    poor = sum(1 for a in agreements if a < 0.6)

    sizes = [good, moderate, poor]
    donut_colors = ["#2E8B57", "#FFD700", "#DC143C"]
    labels_d = [f"Good ({good})", f"Moderate ({moderate})", f"Poor ({poor})"]

    # Only plot non-zero slices
    nonzero = [(s, c, l) for s, c, l in zip(sizes, donut_colors, labels_d) if s > 0]
    if nonzero:
        sz, cl, lb = zip(*nonzero)
        wedges, texts = ax_donut.pie(
            sz, colors=cl, labels=lb, startangle=90,
            wedgeprops={"width": 0.4, "edgecolor": "white", "linewidth": 2},
            textprops={"fontsize": 9},
        )

    # Center text: overall agreement
    ax_donut.text(0, 0, f"{overall:.0%}", ha="center", va="center",
                  fontsize=28, fontweight="bold", color="navy")
    ax_donut.text(0, -0.15, "agreement", ha="center", va="center",
                  fontsize=10, color="#666")
    ax_donut.set_title("Overall", fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_population_grid(
    pop_result: dict,
    days: list[int] | None = None,
    output_path: str = "output/ca/population_grid.png",
) -> None:
    """Generate NxN grid snapshots with labeled cells and count summaries.

    Each panel shows the population at a semester moment. Each cell has a
    one-letter label (H/S/X/B) so the attractor is readable even in small
    grids. Below each grid panel is a stacked bar showing the attractor
    distribution as counts and percentages.

    Attractor key (matches trajectory severity colors):
      H = Healthy (green)  -- low stress, sub-threshold anxiety
      S = Struggling (yellow) -- moderate stress or sleep-deprived
      X = Stressed (orange) -- high PSS or clinical GAD-7
      B = Burnout (red) -- absorbing state (all four conditions met)

    Default days capture key semester phases:
      Day 0: Start (mostly H)
      Day 49: Pre-spring-break (stress accumulated)
      Day 56: Post-spring-break (reset visible)
      Day 104: Finals (final attractor distribution)
    """
    if days is None:
        days = [0, 49, 56, 104]

    grid_states = pop_result["grid_states"]
    grid_size = pop_result["grid_size"]
    n_panels = len(days)
    total_students = grid_size * grid_size

    # Attractor styling
    attractor_info = {
        "healthy":    {"code": 0, "color": "#2E8B57", "letter": "H", "text_color": "white"},
        "struggling": {"code": 1, "color": "#FFD700", "letter": "S", "text_color": "black"},
        "stressed":   {"code": 2, "color": "#FF8C00", "letter": "X", "text_color": "white"},
        "burnout":    {"code": 3, "color": "#DC143C", "letter": "B", "text_color": "white"},
    }
    attractor_order = ["healthy", "struggling", "stressed", "burnout"]
    attractor_cmap = mcolors.ListedColormap(
        [attractor_info[a]["color"] for a in attractor_order]
    )
    attractor_norm = mcolors.BoundaryNorm([0, 1, 2, 3, 4], attractor_cmap.N)

    # Two rows: top = grid, bottom = count summary bar
    fig, axes = plt.subplots(
        2, n_panels, figsize=(4.5 * n_panels, 7),
        gridspec_kw={"height_ratios": [4, 1]},
    )
    fig.suptitle("Population Grid — Attractor Snapshots", fontsize=14,
                 fontweight="bold", y=0.98)

    if n_panels == 1:
        axes = axes.reshape(2, 1)

    for idx, day in enumerate(days):
        ax_grid = axes[0, idx]
        ax_bar = axes[1, idx]

        t_idx = min(day, len(grid_states) - 1)
        grid_snapshot = grid_states[t_idx]

        # Build numeric grid and classify each cell
        attractor_grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        attractor_labels = [[None] * grid_size for _ in range(grid_size)]
        counts = {a: 0 for a in attractor_order}

        for r in range(grid_size):
            for c in range(grid_size):
                state = grid_snapshot[r][c]
                attr = _classify_attractor(state)
                attractor_grid[r, c] = attractor_info.get(attr, attractor_info["healthy"])["code"]
                attractor_labels[r][c] = attr
                counts[attr] = counts.get(attr, 0) + 1

        # Draw grid with imshow
        ax_grid.imshow(
            attractor_grid, cmap=attractor_cmap, norm=attractor_norm,
            interpolation="nearest",
        )

        # Grid lines between cells
        for i in range(grid_size + 1):
            ax_grid.axhline(y=i - 0.5, color="white", linewidth=1.5)
            ax_grid.axvline(x=i - 0.5, color="white", linewidth=1.5)

        # Letter labels inside each cell
        font_size = max(7, min(14, 60 // grid_size))
        for r in range(grid_size):
            for c in range(grid_size):
                attr = attractor_labels[r][c]
                info = attractor_info.get(attr, attractor_info["healthy"])
                ax_grid.text(
                    c, r, info["letter"],
                    ha="center", va="center",
                    fontsize=font_size, fontweight="bold",
                    color=info["text_color"],
                )

        week = day // 7 + 1
        is_break = _BREAK_DAY_START <= day < _BREAK_DAY_END
        day_label = f"Day {day} (Week {week})"
        if is_break:
            day_label += " *break*"
        ax_grid.set_title(day_label, fontsize=10, fontweight="bold")
        ax_grid.set_xticks([])
        ax_grid.set_yticks([])

        # ── Summary bar below each grid ──
        left = 0
        for attr in attractor_order:
            count = counts[attr]
            if count == 0:
                continue
            width = count / total_students
            info = attractor_info[attr]
            ax_bar.barh(
                0, width, left=left, height=0.6,
                color=info["color"], edgecolor="white", linewidth=0.5,
            )
            if width >= 0.12:  # only label if wide enough
                pct = count / total_students * 100
                ax_bar.text(
                    left + width / 2, 0,
                    f"{info['letter']}:{count}\n({pct:.0f}%)",
                    ha="center", va="center",
                    fontsize=7, fontweight="bold",
                    color=info["text_color"],
                )
            left += width

        ax_bar.set_xlim(0, 1)
        ax_bar.set_ylim(-0.5, 0.5)
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        ax_bar.set_frame_on(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=attractor_info[a]["color"],
              label=f"{attractor_info[a]['letter']} = {a.title()}")
        for a in attractor_order
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# HERO IMAGE CANDIDATES — competition-grade visualizations
# ══════════════════════════════════════════════════════════════════════════════


def plot_diverging_fates(output_path: str = "output/ca/hero_diverging_fates.png") -> None:
    """Hero image #1: 'Diverging Fates' — 8 patient chart panels.

    A 2×4 grid where each panel is one student archetype's "semester chart."
    Each panel shows 5 key variables as severity-colored sparkline rows
    (green=healthy, yellow=intermediate, red=concerning). A large final-state
    badge shows the student's outcome (H/S/X/B). Spring break is marked.

    This design gives each student their own space, avoids overlapping lines,
    and makes the contrast between resilient and vulnerable students immediate.
    """
    from ca_analytics import _classify_attractor

    # ── Run all 8 archetypes through both ODE and CA ──
    archetype_data = []
    for seed in STUDENT_ARCHETYPES:
        ode_result = simulate(patient=seed["patient"],
                              intervention=seed.get("intervention"))
        ca_result = run_single_cell(patient=seed["patient"],
                                    intervention=seed.get("intervention"))
        attractor = _classify_attractor(ca_result["final_state"])
        archetype_data.append({
            "name": seed["name"],
            "description": seed["description"],
            "states": ode_result["states"],
            "ca_attractor": attractor,
            "ca_final": ca_result["final_state"],
        })

    n_days = archetype_data[0]["states"].shape[0]

    # Key variables to sparkline (the most clinically telling)
    spark_vars = [
        {"label": "Sleep",    "idx": 0,  "lo": 4,  "hi": 12,  "good": "high",
         "thresholds": [6.0]},
        {"label": "Stress",   "idx": 2,  "lo": 0,  "hi": 40,  "good": "low",
         "thresholds": [14.0, 27.0]},
        {"label": "Anxiety",  "idx": 3,  "lo": 0,  "hi": 21,  "good": "low",
         "thresholds": [10.0]},
        {"label": "Attention","idx": 13, "lo": 0,  "hi": 1,   "good": "high",
         "thresholds": [0.3]},
        {"label": "Wellbeing","idx": 12, "lo": 14, "hi": 70,  "good": "high",
         "thresholds": [40.0]},
    ]

    # Severity coloring function
    def _severity_color(val, vinfo):
        """Return RGB hex for a value based on clinical severity."""
        good_dir = vinfo["good"]
        lo, hi = vinfo["lo"], vinfo["hi"]
        norm = (val - lo) / (hi - lo + 1e-9)
        if good_dir == "low":
            norm = 1.0 - norm  # invert so 0 = bad, 1 = good
        # Green (good) → Yellow → Red (bad)
        if norm > 0.6:
            return "#2E8B57"
        elif norm > 0.35:
            return "#FFD700"
        else:
            return "#DC143C"

    # Attractor info
    attractor_style = {
        "healthy":    {"letter": "H", "color": "#2E8B57", "label": "Healthy"},
        "struggling": {"letter": "S", "color": "#FFD700", "label": "Struggling"},
        "stressed":   {"letter": "X", "color": "#FF8C00", "label": "Stressed"},
        "burnout":    {"letter": "B", "color": "#DC143C", "label": "Burnout"},
    }

    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    fig.patch.set_facecolor("#0D1117")

    for a_idx, ad in enumerate(archetype_data):
        row, col = divmod(a_idx, 4)
        ax = axes[row, col]
        ax.set_facecolor("#13171F")

        states = ad["states"]
        name = ad["name"].replace("_", " ").title()
        desc = ad["description"]

        # Use CA attractor classification (more differentiated than ODE endpoints)
        attractor = ad["ca_attractor"]
        att_info = attractor_style.get(attractor, attractor_style["healthy"])

        # Annotation of key CA final bins that differ between archetypes
        ca_final = ad["ca_final"]
        highlights = []
        if ca_final.get("GAD7") == "clinical":
            highlights.append("GAD-7: clinical")
        if ca_final.get("DAC") == "depleted":
            highlights.append("DAC: depleted")
        if ca_final.get("WEMWBS") == "high":
            highlights.append("Well-being: high")

        n_spark = len(spark_vars)
        spark_h = 1.0 / (n_spark + 1.5)  # fraction of vertical space per sparkline

        for s_idx, sv in enumerate(spark_vars):
            trace = states[:, sv["idx"]]
            lo, hi = sv["lo"], sv["hi"]
            trace_norm = np.clip((trace - lo) / (hi - lo + 1e-9), 0, 1)

            # Y position for this sparkline row
            y_base = 1.0 - (s_idx + 1.2) * spark_h
            y_height = spark_h * 0.7

            # Smooth
            kernel = np.ones(3) / 3
            trace_s = np.convolve(trace_norm, kernel, mode="same")

            # Draw colored segments — each day-segment colored by severity
            for d in range(n_days - 1):
                val = trace[d]
                color = _severity_color(val, sv)
                x0 = d / n_days
                x1 = (d + 1) / n_days
                y0 = y_base + trace_s[d] * y_height
                y1 = y_base + trace_s[d+1] * y_height
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.8,
                        solid_capstyle="round", transform=ax.transAxes)

            # Spring break marker
            sb_x = _BREAK_DAY_START / n_days
            sb_w = (_BREAK_DAY_END - _BREAK_DAY_START) / n_days
            from matplotlib.patches import Rectangle
            rect = Rectangle((sb_x, y_base - 0.005), sb_w, y_height + 0.01,
                              facecolor="gold", alpha=0.1,
                              transform=ax.transAxes, zorder=0)
            ax.add_patch(rect)

            # Variable label (left side)
            ax.text(-0.02, y_base + y_height * 0.5, sv["label"],
                    fontsize=8, color="#AAA", va="center", ha="right",
                    transform=ax.transAxes, fontfamily="monospace")

            # Thin baseline
            ax.plot([0, 1], [y_base, y_base], color="#333", linewidth=0.3,
                    transform=ax.transAxes)

        # ── Archetype name (large, top of panel) ──
        ax.text(0.5, 0.97, name, ha="center", va="top",
                fontsize=13, color="white", fontweight="bold",
                transform=ax.transAxes, fontfamily="monospace")
        ax.text(0.5, 0.90, desc, ha="center", va="top",
                fontsize=7.5, color="#888", style="italic",
                transform=ax.transAxes)

        # ── Final attractor badge (bottom-right, large) ──
        badge_x, badge_y = 0.88, 0.08
        badge = plt.Circle((badge_x, badge_y), 0.06, facecolor=att_info["color"],
                            edgecolor="white", linewidth=2,
                            transform=ax.transAxes, zorder=20)
        ax.add_patch(badge)
        ax.text(badge_x, badge_y, att_info["letter"],
                ha="center", va="center", fontsize=16, color="white",
                fontweight="bold", transform=ax.transAxes, zorder=21)
        ax.text(badge_x, badge_y - 0.09, att_info["label"],
                ha="center", fontsize=7, color=att_info["color"],
                fontweight="bold", transform=ax.transAxes)

        # Key differentiating CA bins below badge
        if highlights:
            hl_text = "\n".join(highlights)
            ax.text(badge_x, badge_y - 0.18, hl_text,
                    ha="center", fontsize=5.5, color="#AAA",
                    transform=ax.transAxes, linespacing=1.3)

        # Week markers along bottom
        for w in range(0, 16, 2):
            wx = (w * 7) / n_days
            if 0 <= wx <= 1:
                ax.text(wx, -0.02, f"W{w+1}", fontsize=5.5, color="#555",
                        ha="center", transform=ax.transAxes)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Panel border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#333")
            spine.set_linewidth(0.5)

    # ── Title ──
    fig.text(0.5, 0.99, "DIVERGING FATES", ha="center", fontsize=26,
             fontweight="bold", color="white", fontfamily="monospace",
             va="top")
    fig.text(0.5, 0.955,
             "Eight students enter the same semester. Same campus, same 15 weeks — radically different outcomes.",
             ha="center", fontsize=11, color="#888", style="italic")

    # ── Legend bar ──
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#2E8B57", label="Healthy range"),
        Patch(facecolor="#FFD700", label="Intermediate"),
        Patch(facecolor="#DC143C", label="Concerning"),
        Patch(facecolor="gold", alpha=0.3, label="Spring Break"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=4, fontsize=8,
               facecolor="#1A1A2E", edgecolor="#333", labelcolor="white",
               bbox_to_anchor=(0.5, 0.002))

    fig.text(0.98, 0.003,
             "LEMURS Simulator  ·  14D ODE  ·  9 papers (UVM 2023-2025)",
             ha="right", fontsize=6, color="#444", fontfamily="monospace")

    plt.tight_layout(rect=[0.04, 0.03, 1.0, 0.94], h_pad=1.0, w_pad=0.8)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_architecture_of_vulnerability(
    output_path: str = "output/ca/hero_architecture.png",
) -> None:
    """Hero image #2: 'Architecture of Vulnerability' radial coupling diagram.

    14 state variables arranged radially, grouped by coupling tier. Curved
    ribbons show ODE coupling paths with thickness proportional to |coefficient|.
    PSS emerges as the central hub. Sparkline trajectories on the outer rim
    show the vulnerable_female archetype's semester arc.
    """
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.colors import LinearSegmentedColormap

    # ── Variable positions: arranged by tier, radially ──
    # Group by tier for visual clustering
    tier_groups = {
        1: ["TST", "SleepQuality", "RHR", "HRV", "ARR"],      # sleep→stress biomarkers
        "hub": ["PSS"],                                          # central hub
        2: ["GAD7", "Depression"],                               # anxiety/depression
        3: ["NatureEngagement", "WEMWBS"],                       # nature mediation
        4: ["Activity", "SocialJetlag", "SleepShape"],           # rhythm
        6: ["DAC"],                                              # attention
    }

    tier_colors = {
        1: "#4169E1",   # royal blue
        "hub": "#FF4500", # orange-red for PSS hub
        2: "#8A2BE2",   # blue-violet
        3: "#228B22",   # forest green
        4: "#FF8C00",   # dark orange
        6: "#DC143C",   # crimson
    }

    # Assign angular positions: evenly spaced around circle with tier gaps
    all_vars = []
    var_tier = {}
    var_angles = {}

    tier_order = [1, "hub", 2, 3, 4, 6]
    # Equal spacing per variable, with extra gap between tier groups
    n_total_vars = sum(len(tier_groups[t]) for t in tier_order)
    n_gaps = len(tier_order)
    gap_angle = 0.25  # radians between tier groups
    var_angle = (2 * np.pi - n_gaps * gap_angle) / n_total_vars

    current_angle = np.pi / 2  # start at top
    for tier_key in tier_order:
        vars_in_tier = tier_groups[tier_key]
        for var in vars_in_tier:
            var_angles[var] = current_angle
            var_tier[var] = tier_key
            all_vars.append(var)
            current_angle -= var_angle
        current_angle -= gap_angle

    # ── Coupling connections (from ODE derivatives) ──
    # (source, target, |coefficient|, tier_color_key, label)
    couplings = [
        # Tier 1: biomarkers → PSS
        ("TST", "PSS", 0.877, 1, "β=-0.877"),
        ("RHR", "PSS", 0.055, 1, ""),
        ("HRV", "PSS", 0.012, 1, ""),
        ("ARR", "PSS", 0.270, 1, "β=+0.27"),
        # Tier 2: PSS → anxiety/depression
        ("PSS", "GAD7", 0.5, 2, ""),
        ("PSS", "Depression", 0.3, 2, ""),
        ("GAD7", "Depression", 0.2, 2, ""),
        # Tier 3: nature mediation
        ("NatureEngagement", "PSS", 1.507, 3, "β=-1.51"),
        ("PSS", "HRV", 0.618, 3, "β=-0.62"),
        ("NatureEngagement", "WEMWBS", 0.104, 3, ""),
        ("NatureEngagement", "DAC", 0.2, 3, ""),
        # Tier 4: sleep/activity
        ("TST", "SleepQuality", 0.5, 4, ""),
        ("TST", "Activity", 0.021, 4, ""),
        ("SocialJetlag", "Activity", 0.023, 4, ""),
        ("SocialJetlag", "TST", 0.4, 4, ""),
        ("TST", "SleepShape", 0.3, 4, ""),
        # Tier 6: attention
        ("DAC", "NatureEngagement", 0.3, 6, "gate"),
        # Cross: within-person amplification
        ("PSS", "PSS", 0.5, 1, "2.2x"),
    ]

    # ── Run vulnerable_female for sparklines ──
    vuln = next(s for s in STUDENT_ARCHETYPES if s["name"] == "vulnerable_female")
    result = simulate(patient=vuln["patient"], intervention=vuln.get("intervention"))
    vuln_states = result["states"]

    # Variable index map
    var_idx_map = {name: i for i, name in enumerate([
        "TST", "SleepQuality", "PSS", "GAD7", "Depression", "Activity",
        "NatureEngagement", "RHR", "HRV", "ARR", "SocialJetlag",
        "SleepShape", "WEMWBS", "DAC"
    ])}
    var_ranges = {
        "TST": (4, 12), "SleepQuality": (0, 100), "PSS": (0, 40),
        "GAD7": (0, 21), "Depression": (0, 21), "Activity": (0, 500),
        "NatureEngagement": (0, 15), "RHR": (45, 100), "HRV": (15, 120),
        "ARR": (10, 25), "SocialJetlag": (0, 3), "SleepShape": (0, 1),
        "WEMWBS": (14, 70), "DAC": (0, 1),
    }

    fig, ax = plt.subplots(figsize=(16, 16))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    radius = 4.2
    node_radius = 0.28

    # ── Draw coupling ribbons ──
    for src, tgt, coeff, tier_key, label in couplings:
        if src == tgt:
            continue  # handle self-loops separately
        angle_s = var_angles[src]
        angle_t = var_angles[tgt]
        x_s = radius * np.cos(angle_s)
        y_s = radius * np.sin(angle_s)
        x_t = radius * np.cos(angle_t)
        y_t = radius * np.sin(angle_t)

        # Curved path through center-ish (Bezier control point)
        # Pull toward center for longer connections, less for short
        dist = np.sqrt((x_t - x_s)**2 + (y_t - y_s)**2)
        pull = 0.3 + 0.1 * dist / (2 * radius)
        cx = (x_s + x_t) / 2 * (1 - pull)
        cy = (y_s + y_t) / 2 * (1 - pull)

        # Ribbon width from coefficient
        lw = 0.5 + 4.0 * min(coeff / 1.5, 1.0)

        color = tier_colors.get(tier_key, "#808080")
        alpha = 0.4 + 0.3 * min(coeff / 1.0, 1.0)

        # Draw as quadratic Bezier approximation
        t_param = np.linspace(0, 1, 50)
        bx = (1-t_param)**2 * x_s + 2*(1-t_param)*t_param * cx + t_param**2 * x_t
        by = (1-t_param)**2 * y_s + 2*(1-t_param)*t_param * cy + t_param**2 * y_t
        ax.plot(bx, by, color=color, linewidth=lw, alpha=alpha, solid_capstyle="round")

        # Arrowhead at target
        arrow_idx = 42  # near end
        dx = bx[-1] - bx[arrow_idx]
        dy = by[-1] - by[arrow_idx]
        ax.annotate("", xy=(bx[-1], by[-1]),
                     xytext=(bx[arrow_idx], by[arrow_idx]),
                     arrowprops=dict(arrowstyle="-|>", color=color,
                                     lw=lw * 0.6, mutation_scale=8 + lw * 2),
                     zorder=3)

        # Label on ribbon
        if label:
            mid = len(t_param) // 2
            ax.text(bx[mid], by[mid], label, fontsize=5.5, color=color,
                    ha="center", va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#0D1117",
                              edgecolor="none", alpha=0.8))

    # Self-loop for PSS within-person amplification
    pss_angle = var_angles["PSS"]
    pss_x = radius * np.cos(pss_angle)
    pss_y = radius * np.sin(pss_angle)
    loop_r = 0.5
    theta = np.linspace(0, 2 * np.pi * 0.85, 40)
    loop_cx = pss_x + 0.6 * np.cos(pss_angle)
    loop_cy = pss_y + 0.6 * np.sin(pss_angle)
    lx = loop_cx + loop_r * np.cos(theta)
    ly = loop_cy + loop_r * np.sin(theta)
    ax.plot(lx, ly, color="#FF4500", linewidth=2, alpha=0.6, linestyle="--")
    ax.text(loop_cx, loop_cy + loop_r + 0.15, "2.2x", fontsize=6,
            color="#FF4500", ha="center", fontweight="bold")

    # ── Draw variable nodes + external labels ──
    _DISPLAY_NAMES = {
        "TST": "Total Sleep\nTime",
        "SleepQuality": "Sleep\nQuality",
        "PSS": "Perceived\nStress (PSS)",
        "GAD7": "Anxiety\n(GAD-7)",
        "Depression": "Depression",
        "Activity": "Physical\nActivity",
        "NatureEngagement": "Nature\nEngagement",
        "RHR": "Resting\nHeart Rate",
        "HRV": "Heart Rate\nVariability",
        "ARR": "Respiratory\nRate",
        "SocialJetlag": "Social\nJetlag",
        "SleepShape": "Sleep\nShape",
        "WEMWBS": "Well-Being\n(WEMWBS)",
        "DAC": "Directed\nAttention",
    }

    for var in all_vars:
        angle = var_angles[var]
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        tier_key = var_tier[var]
        color = tier_colors[tier_key]

        # Node: colored dot (PSS is bigger as the hub)
        r = node_radius * (2.0 if var == "PSS" else 1.0)
        circle = plt.Circle((x, y), r, facecolor=color, edgecolor="white",
                             linewidth=2.0 if var == "PSS" else 1.2,
                             alpha=0.9, zorder=10)
        ax.add_patch(circle)

        # ── External label: the ONLY text (no abbreviation inside node) ──
        # Position label radially outward from node
        label_offset = 0.65 if var != "PSS" else 0.8
        lx = (radius + label_offset) * np.cos(angle)
        ly = (radius + label_offset) * np.sin(angle)

        display_name = _DISPLAY_NAMES.get(var, var)

        # Alignment based on angular position
        cos_a = np.cos(angle)
        if abs(cos_a) < 0.25:
            ha = "center"
        elif cos_a > 0:
            ha = "left"
        else:
            ha = "right"

        fontsize = 11 if var == "PSS" else 9
        ax.text(lx, ly, display_name, ha=ha, va="center",
                fontsize=fontsize, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=color,
                          edgecolor="white", alpha=0.85, linewidth=0.8),
                zorder=12)

        # ── Sparkline on outer rim ──
        spark_radius = radius + 1.5
        spark_len = 0.2   # angular extent
        spark_height = 0.3

        idx = var_idx_map[var]
        trace = vuln_states[:, idx]
        lo, hi = var_ranges[var]
        trace_norm = np.clip((trace - lo) / (hi - lo + 1e-9), 0, 1)

        spark_angles = np.linspace(angle + spark_len/2, angle - spark_len/2, len(trace_norm))
        spark_x = (spark_radius + trace_norm * spark_height) * np.cos(spark_angles)
        spark_y = (spark_radius + trace_norm * spark_height) * np.sin(spark_angles)

        # Baseline + trace
        base_x = spark_radius * np.cos(spark_angles)
        base_y = spark_radius * np.sin(spark_angles)
        ax.plot(base_x, base_y, color="#333", linewidth=0.5)
        ax.plot(spark_x, spark_y, color=color, linewidth=1.2, alpha=0.8)

    # ── Center text ──
    ax.text(0, 0.3, "ARCHITECTURE", ha="center", fontsize=18, color="white",
            fontweight="bold", fontfamily="monospace")
    ax.text(0, -0.05, "OF VULNERABILITY", ha="center", fontsize=18, color="white",
            fontweight="bold", fontfamily="monospace")
    ax.text(0, -0.5, "6 tiers  ·  9 papers  ·  14 variables  ·  32 rules",
            ha="center", fontsize=8, color="#888", fontfamily="monospace")

    # ── Tier legend ──
    from matplotlib.patches import Patch
    tier_labels = {
        1: "T1: Sleep→Stress", 2: "T2: Anxiety Markov",
        3: "T3: Nature Mediation", 4: "T4: Sleep Debt & Rhythm",
        6: "T6: Attention Restoration", "hub": "PSS Hub",
    }
    legend_patches = [
        Patch(facecolor=tier_colors[k], label=tier_labels[k])
        for k in tier_order
    ]
    ax.legend(handles=legend_patches, loc="lower center", fontsize=7,
              facecolor="#1A1A2E", edgecolor="#333", labelcolor="white",
              ncol=3, bbox_to_anchor=(0.5, -0.02))

    ax.set_xlim(-8.0, 8.0)
    ax.set_ylim(-8.0, 8.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Source citation
    fig.text(0.5, 0.02,
             "Sparklines: vulnerable_female archetype  |  LEMURS Simulator  |  UVM 2023-2025",
             ha="center", fontsize=7, color="#555", fontfamily="monospace")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_probability_terrain(
    output_path: str = "output/ca/hero_probability_terrain.png",
    n_trials: int = 500,
) -> None:
    """Hero image #3: 'Probability Terrain' stochastic ridgeline landscape.

    Six key variables as full-width panels, each showing the probability
    distribution of final-state bins from 500 Monte Carlo stochastic CA runs.
    Three student archetypes overlaid as colored ridgelines, with severity-
    colored background zones spanning each bin region. The design emphasizes
    separation between archetypes (where do their distributions diverge?)
    and clinical thresholds (what fraction crosses into concerning territory?).

    Selected variables are those showing the most archetype differentiation:
    GAD-7 (bistable anxiety), Depression, PSS (stress hub), DAC (attention
    burnout), SleepShape (phenotype), Activity (paradoxical compensation).
    """
    from ca_stochastic import run_single_cell_stochastic

    # ── Three archetype ensembles ──
    archetypes_to_run = [
        ("resilient_male",      "#2E8B57", "Resilient Male"),
        ("vulnerable_female",   "#DC143C", "Vulnerable Female"),
        ("recovery_trajectory", "#4169E1", "Recovery (intervention)"),
    ]

    ensemble_data = {}
    for arch_name, color, label in archetypes_to_run:
        seed = next(s for s in STUDENT_ARCHETYPES if s["name"] == arch_name)
        patient = seed["patient"]
        intervention = seed.get("intervention")
        result = run_single_cell_stochastic(
            patient=patient, intervention=intervention,
            n_trials=n_trials, seed=42,
        )
        ensemble_data[arch_name] = result

    # ── Severity map ──
    _SEVERITY = {
        "GAD7":              {"sub_threshold": 0, "clinical": 2},
        "PSS":               {"low": 0, "moderate": 1, "high": 2},
        "Depression":        {"normal": 0, "mild": 1, "moderate_plus": 2},
        "TST":               {"deprived": 2, "adequate": 0, "excess": 1},
        "NatureEngagement":  {"low": 2, "engaged": 0},
        "WEMWBS":            {"low": 2, "moderate": 1, "high": 0},
    }
    sev_colors = {0: "#2E8B57", 1: "#FFD700", 2: "#DC143C"}
    sev_bg_alpha = {0: 0.08, 1: 0.06, 2: 0.08}

    # ── Display names for readability ──
    _DISPLAY = {
        "GAD7":              "Anxiety (GAD-7)",
        "PSS":               "Perceived Stress (PSS)",
        "Depression":        "Depression (DASS-21)",
        "TST":               "Total Sleep Time",
        "NatureEngagement":  "Nature Engagement",
        "WEMWBS":            "Well-Being (WEMWBS)",
    }

    # Variables showing distributional spread and/or cross-archetype differentiation
    # GAD7: dramatic binary split (threshold story)
    # PSS: 3-way spread with intervention effect (the star variable)
    # Depression: cross-archetype + within-archetype spread
    # TST: 3-bin sleep debt distribution
    # NatureEngagement: intervention effect
    # WEMWBS: intervention effect
    display_vars = ["GAD7", "PSS", "Depression", "TST", "NatureEngagement", "WEMWBS"]

    n_vars = len(display_vars)
    fig, axes = plt.subplots(n_vars, 1, figsize=(16, 18), sharex=True)
    fig.patch.set_facecolor("#0D1117")

    for v_idx, var_name in enumerate(display_vars):
        ax = axes[v_idx]
        ax.set_facecolor("#0D1117")

        schema = BIN_SCHEMA[var_name]
        labels = schema["labels"]
        n_bins = len(labels)
        sev_map = _SEVERITY.get(var_name, {})

        # X-axis: bin positions evenly spaced 0 to 1, then scaled to 0-10
        x_positions = np.linspace(0, 1, n_bins) if n_bins > 1 else np.array([0.5])
        x_scale = 10.0  # visual width

        # ── Severity zone backgrounds (colored vertical bands) ──
        for i, lbl in enumerate(labels):
            sev = sev_map.get(lbl, 0)
            fill_color = sev_colors[sev]
            alpha_bg = sev_bg_alpha[sev]

            # Compute zone boundaries
            if n_bins == 1:
                x_left, x_right = 0.0, 1.0
            elif i == 0:
                x_left = -0.15
                x_right = (x_positions[0] + x_positions[1]) / 2
            elif i == n_bins - 1:
                x_left = (x_positions[i-1] + x_positions[i]) / 2
                x_right = 1.15
            else:
                x_left = (x_positions[i-1] + x_positions[i]) / 2
                x_right = (x_positions[i] + x_positions[i+1]) / 2

            ax.axvspan(x_left * x_scale, x_right * x_scale,
                       facecolor=fill_color, alpha=alpha_bg, zorder=0)

            # Bin label at the top of the zone, large and readable
            zone_center = (x_left + x_right) / 2 * x_scale
            display_lbl = lbl.replace("_", " ").replace("plus", "+").upper()
            ax.text(zone_center, 0.93, display_lbl,
                    ha="center", va="top", fontsize=12, color=fill_color,
                    fontweight="bold", alpha=0.7, transform=ax.get_xaxis_transform(),
                    zorder=1,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#0D1117",
                              edgecolor="none", alpha=0.5))

        # ── Draw ridgelines for each archetype ──
        max_peak_across = 0.0  # track for normalization
        ridge_data = []

        for arch_name, color, arch_label in archetypes_to_run:
            final_states = ensemble_data[arch_name]["final_states"]
            bin_counts = {lbl: 0 for lbl in labels}
            for fs in final_states:
                lbl = fs.get(var_name, labels[0])
                if lbl in bin_counts:
                    bin_counts[lbl] += 1
            total = len(final_states)
            fractions = [bin_counts[lbl] / total for lbl in labels]

            # Smooth ridge curve
            x_fine = np.linspace(-0.15, 1.15, 300)
            y_fine = np.zeros_like(x_fine)
            bandwidth = 0.055 if n_bins > 2 else 0.09

            for xp, frac in zip(x_positions, fractions):
                y_fine += frac * np.exp(-0.5 * ((x_fine - xp) / bandwidth)**2)

            max_peak_across = max(max_peak_across, y_fine.max())
            ridge_data.append((arch_name, color, arch_label, x_fine, y_fine, fractions))

        # Normalize and draw
        for arch_name, color, arch_label, x_fine, y_fine, fractions in ridge_data:
            if max_peak_across > 0:
                y_norm = y_fine / max_peak_across
            else:
                y_norm = y_fine

            # Filled ridge with archetype color
            ax.fill_between(x_fine * x_scale, 0, y_norm,
                            color=color, alpha=0.2, linewidth=0, zorder=3)
            # Ridge outline
            ax.plot(x_fine * x_scale, y_norm, color=color, linewidth=2.5,
                    alpha=0.85, zorder=4, solid_capstyle="round")

            # Percentage annotations near each peak
            # Small dark background box for readability against colored fills
            arch_nudge = {"resilient_male": -0.25, "vulnerable_female": 0.0,
                          "recovery_trajectory": 0.25}
            x_nudge = arch_nudge.get(arch_name, 0.0)
            for i, (xp, frac) in enumerate(zip(x_positions, fractions)):
                if frac >= 0.05:  # only annotate bins with >= 5%
                    closest_idx = np.argmin(np.abs(x_fine - xp))
                    peak_y = y_fine[closest_idx] / max_peak_across if max_peak_across > 0 else 0
                    ax.text(xp * x_scale + x_nudge, peak_y + 0.06,
                            f"{frac:.0%}", ha="center", va="bottom",
                            fontsize=7.5, color=color, fontweight="bold",
                            alpha=0.95, zorder=5,
                            bbox=dict(boxstyle="round,pad=0.1",
                                      facecolor="#0D1117", edgecolor="none",
                                      alpha=0.6))

        # Variable name as left label
        ax.set_ylabel(_DISPLAY.get(var_name, var_name),
                      fontsize=13, color="white", fontweight="bold",
                      labelpad=12)

        # Style
        ax.set_ylim(0, 1.45)
        ax.set_xlim(-1.5, 11.5)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#333")
        ax.tick_params(colors="#666")

        # Thin baseline
        ax.axhline(0, color="#444", linewidth=0.5, zorder=1)

    # Remove x-axis tick labels (bins are labeled in each panel)
    axes[-1].set_xticks([])
    axes[-1].set_xlabel("")

    # ── GAD-7 threshold annotation on the GAD7 panel ──
    gad_ax = axes[0]  # GAD7 is first
    # The boundary between sub_threshold and clinical
    gad_labels = BIN_SCHEMA["GAD7"]["labels"]
    if len(gad_labels) == 2:
        boundary_x = 0.5 * 10.0  # midpoint between two bins
        gad_ax.axvline(boundary_x, color="#FF6B6B", linewidth=1.5,
                       linestyle="--", alpha=0.6, zorder=6)
        gad_ax.text(boundary_x + 0.3, 0.75, "GAD-7 ≥ 10\nbistable threshold",
                    fontsize=9, color="#FF6B6B", fontweight="bold",
                    va="center", zorder=7)

    # ── Title ──
    fig.text(0.5, 0.975, "WHERE DO 500 SEMESTERS END?",
             ha="center", fontsize=24, fontweight="bold",
             color="white", fontfamily="monospace")
    fig.text(0.5, 0.955,
             "Probability distributions from 500 stochastic CA trajectories  "
             "·  three student archetypes  ·  six clinical dimensions",
             ha="center", fontsize=10, color="#888", style="italic")

    # ── Legend ──
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#DC143C", lw=3, label="Vulnerable Female"),
        Line2D([0], [0], color="#2E8B57", lw=3, label="Resilient Male"),
        Line2D([0], [0], color="#4169E1", lw=3, label="Recovery (with intervention)"),
        Patch(facecolor="#2E8B57", alpha=0.25, label="Healthy zone"),
        Patch(facecolor="#FFD700", alpha=0.25, label="Intermediate zone"),
        Patch(facecolor="#DC143C", alpha=0.25, label="Concerning zone"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", fontsize=10,
               facecolor="#1A1A2E", edgecolor="#555", labelcolor="white",
               ncol=3, bbox_to_anchor=(0.5, 0.005),
               handlelength=2.5, columnspacing=2.0)

    fig.text(0.98, 0.003,
             "LEMURS Stochastic CA  ·  Confidence-weighted rule sampling  ·  Semantic severity mapping",
             ha="right", fontsize=7, color="#444", fontfamily="monospace")

    plt.tight_layout(rect=[0.08, 0.04, 1.0, 0.94], h_pad=0.5)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_ca_all_scenarios(output_dir: str = "output/ca") -> None:
    """Run all 8 student archetypes through both CA and ODE, generate everything.

    This is the CA visualization equivalent of visualize.py's plot_all_scenarios().
    For each of the 8 student archetypes defined in constants.STUDENT_ARCHETYPES,
    it runs both the cellular automaton (run_single_cell) and the continuous ODE
    (simulate), then generates three plots per archetype:

      1. Trajectory heatmap -- the CA's 14-variable discrete state over 105 days
      2. Rule firing timeline -- which rules fired when and at what tier
      3. Fidelity comparison -- how well the CA tracks the ODE, variable by variable

    Finally, it runs a 5x5 population grid with social coupling (0.3) and plots
    attractor snapshots at four key semester moments: start, pre-break, post-break,
    and finals.

    The full set of plots provides a comprehensive view of how well the semantic
    CA captures the ODE dynamics, where it agrees, and where it diverges. The
    population grid adds the social-coupling dimension that the single-cell
    simulations cannot show.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Generating CA trajectory heatmaps, rule timelines, and fidelity plots...")

    for seed in STUDENT_ARCHETYPES:
        name = seed["name"]
        patient = seed["patient"]
        intervention = seed.get("intervention", None)

        print(f"\n  Archetype: {name} -- {seed['description']}")

        # Run CA
        ca_result = run_single_cell(patient=patient, intervention=intervention)

        # Run ODE
        ode_result = simulate(patient=patient, intervention=intervention)

        # 1. Trajectory heatmap
        traj_path = os.path.join(output_dir, f"ca_trajectory_{name}.png")
        plot_ca_trajectory(
            ca_result,
            f"CA Trajectory: {name}",
            traj_path,
        )

        # 2. Rule firing timeline
        rule_path = os.path.join(output_dir, f"ca_rules_{name}.png")
        plot_rule_timeline(
            ca_result,
            f"Rule Firings: {name}",
            rule_path,
        )

        # 3. Fidelity comparison
        fidelity_path = os.path.join(output_dir, f"ca_fidelity_{name}.png")
        plot_ca_fidelity(
            ca_result,
            ode_result,
            f"CA-ODE Fidelity: {name}",
            fidelity_path,
        )

    # 4. Population grid
    print("\n  Running 5x5 population grid (coupling=0.3)...")
    pop_result = run_population_grid(grid_size=5, social_coupling=0.3)
    pop_path = os.path.join(output_dir, "ca_population_grid.png")
    plot_population_grid(pop_result, output_path=pop_path)


if __name__ == "__main__":
    print("Generating LEMURS CA visualizations...")
    plot_ca_all_scenarios()
    print("\nDone.")
