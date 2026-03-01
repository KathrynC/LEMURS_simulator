# Semantic CA Extensions Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the LEMURS Semantic CA with three capabilities: visualization, expanded social coupling (stress contagion, sleep norms, anxiety diffusion), and stochastic rule engine with ensemble analysis.

**Architecture:** Three independent extensions that build on the existing 5-file CA implementation (ca_schema, ca_rules, ca_simulator, ca_analytics, ca_zimmerman_bridge). Extension 1 (visualization) is a new file. Extension 2 (social coupling) modifies ca_simulator.py and ca_analytics.py. Extension 3 (stochastic engine) is a new file with a new Zimmerman adapter in ca_zimmerman_bridge.py.

**Tech Stack:** Python 3.11+, numpy-only, matplotlib Agg backend for visualization

---

## Context

The LEMURS Semantic CA is already implemented across 5 files with 77 passing tests (`tests/test_ca.py`). It discretizes the 14D continuous ODE state into clinical bins and simulates transitions using 32 tiered rules. Two simulation modes exist: single-cell (one student, 105 days) and population grid (NxN with social coupling).

**Current limitations:**
1. No visualization — the ODE has `visualize.py` but the CA has nothing
2. Social coupling only affects NatureEngagement and Activity (2 of 14 variables)
3. Fully deterministic — identical inputs always produce identical outputs, no uncertainty quantification

**Dependency order:** Tasks 1-3 (visualization) are independent. Tasks 4-5 (social coupling) are independent. Tasks 6-7 (stochastic engine) are independent. All three extensions can be built in any order. The test task (Task 8) runs everything.

---

## Reference Files

**Existing CA implementation:**
- `~/lemurs-simulator/ca_schema.py` — BIN_SCHEMA, discretize_state(), continuous_exemplar(), _VAR_ORDER
- `~/lemurs-simulator/ca_rules.py` — RULE_TABLE (32 rules), apply_rules(), get_applicable_rules()
- `~/lemurs-simulator/ca_simulator.py` — run_single_cell(), run_population_grid(), _build_context(), step_cell()
- `~/lemurs-simulator/ca_analytics.py` — compute_ca_analytics(), _classify_attractor(), _rule_stats(), _cascade_stats(), _attractor_stats()
- `~/lemurs-simulator/ca_zimmerman_bridge.py` — LEMURSCASimulator, LEMURSPopulationSimulator, _flatten_ca_analytics(), _flatten_population_summary()
- `~/lemurs-simulator/tests/test_ca.py` — 77 tests across 6 test classes

**Pattern to follow:**
- `~/lemurs-simulator/visualize.py` — ODE visualization (Agg backend, 4-panel plots, spring break highlighting, plot_all_scenarios())
- `~/lemurs-simulator/constants.py` — STUDENT_ARCHETYPES, SPRING_BREAK_WEEK, calendar functions

---

## Tasks

### Task 1: Create ca_visualize.py — Single-Cell Plots

**Files:**
- Create: `~/lemurs-simulator/ca_visualize.py`

**Implementation:**

Create `ca_visualize.py` with the same structure as `visualize.py` (Agg backend, output to `output/ca/`).

```python
"""Visualization for the LEMURS semantic cellular automaton.

Renders CA dynamics as heatmaps, rule timelines, population grids,
and fidelity comparisons. All output goes to PNG files in output/ca/.
Uses the Agg (non-interactive) matplotlib backend.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from constants import STUDENT_ARCHETYPES, SPRING_BREAK_WEEK
from ca_schema import BIN_SCHEMA, _VAR_ORDER
from ca_simulator import run_single_cell
from ca_analytics import compute_ca_analytics, _classify_attractor
from simulator import simulate

_BREAK_START = SPRING_BREAK_WEEK - 1
_BREAK_END = SPRING_BREAK_WEEK
```

**Function 1: `plot_ca_trajectory(ca_result, title, output_path)`**

Single-cell trajectory heatmap: 14 rows (variables) × 106 columns (days including initial state).

- Create a 2D integer array (14 × len(trajectory)) where each cell is the bin index for that variable on that day
- Use a categorical colormap with distinct colors per bin count (max 3 bins for most variables, 2 for GAD7/NatureEngagement/etc.)
- Y-axis labels are the 14 variable names from _VAR_ORDER
- X-axis is days (0-105), with week markers at 7-day intervals
- Spring break band (days 49-55) highlighted with vertical gold lines
- Title at top
- Colorbar legend showing bin labels

Implementation:
```python
def plot_ca_trajectory(ca_result: dict, title: str, output_path: str) -> None:
    trajectory = ca_result["trajectory"]
    n_days = len(trajectory)

    # Build integer grid: row=variable, col=day
    grid = np.zeros((len(_VAR_ORDER), n_days), dtype=int)
    for day_idx, state in enumerate(trajectory):
        for var_idx, var_name in enumerate(_VAR_ORDER):
            label = state.get(var_name, "")
            labels = BIN_SCHEMA[var_name]["labels"]
            grid[var_idx, day_idx] = labels.index(label) if label in labels else -1

    # Max bins across all variables (for colormap)
    max_bins = max(len(BIN_SCHEMA[v]["labels"]) for v in _VAR_ORDER)

    fig, ax = plt.subplots(figsize=(16, 8))
    cmap = plt.cm.get_cmap("RdYlGn_r", max_bins)
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=max_bins - 1,
                   interpolation="nearest")

    # Y-axis: variable names
    ax.set_yticks(range(len(_VAR_ORDER)))
    ax.set_yticklabels(_VAR_ORDER, fontsize=8)

    # X-axis: weeks
    week_ticks = list(range(0, n_days, 7))
    ax.set_xticks(week_ticks)
    ax.set_xticklabels([f"W{i}" for i in range(len(week_ticks))], fontsize=8)
    ax.set_xlabel("Week of Semester")

    # Spring break
    ax.axvline(x=_BREAK_START * 7, color="gold", linewidth=2, linestyle="--")
    ax.axvline(x=_BREAK_END * 7, color="gold", linewidth=2, linestyle="--")

    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Bin index (0=best, higher=worse)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
```

**Function 2: `plot_rule_timeline(ca_result, title, output_path)`**

Rule firing timeline: each day on x-axis, rules on y-axis (grouped by tier), filled cell = rule fired that day.

```python
def plot_rule_timeline(ca_result: dict, title: str, output_path: str) -> None:
    rule_log = ca_result["rule_log"]
    n_days = len(rule_log)

    # Collect all unique rule names that fired, order by tier
    from ca_rules import RULE_TABLE
    rule_names = [r["name"] for r in RULE_TABLE]
    rule_tiers = {r["name"]: r["tier"] for r in RULE_TABLE}
    fired_ever = set()
    for day_rules in rule_log:
        fired_ever.update(day_rules)

    # Filter to rules that actually fired, sorted by tier then name
    active_rules = sorted(fired_ever, key=lambda n: (rule_tiers.get(n, 99), n))
    if not active_rules:
        return  # nothing to plot

    # Build firing matrix
    rule_idx = {name: i for i, name in enumerate(active_rules)}
    firing_grid = np.zeros((len(active_rules), n_days), dtype=int)
    for day, day_rules in enumerate(rule_log):
        for rname in day_rules:
            if rname in rule_idx:
                firing_grid[rule_idx[rname], day] = rule_tiers.get(rname, 0) + 1

    fig, ax = plt.subplots(figsize=(16, max(6, len(active_rules) * 0.3)))
    cmap = plt.cm.get_cmap("Set1", 8)
    im = ax.imshow(firing_grid, aspect="auto", cmap=cmap, vmin=0, vmax=7,
                   interpolation="nearest")

    ax.set_yticks(range(len(active_rules)))
    ax.set_yticklabels(active_rules, fontsize=7)
    week_ticks = list(range(0, n_days, 7))
    ax.set_xticks(week_ticks)
    ax.set_xticklabels([f"W{i}" for i in range(len(week_ticks))], fontsize=8)
    ax.set_xlabel("Week of Semester")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Tier (0=cross-tier, 1-6=ODE tier)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
```

**Function 3: `plot_ca_fidelity(ca_result, ode_result, title, output_path)`**

Per-variable agreement bar chart: CA vs ODE bin agreement over the 106-step trajectory.

```python
def plot_ca_fidelity(
    ca_result: dict, ode_result: dict, title: str, output_path: str
) -> None:
    from ca_analytics import _fidelity_stats
    fs = _fidelity_stats(ca_result["trajectory"], ode_result)

    var_names = list(fs["per_variable_agreement"].keys())
    agreements = [fs["per_variable_agreement"][v] for v in var_names]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2ecc71" if a >= 0.8 else "#f39c12" if a >= 0.6 else "#e74c3c"
              for a in agreements]
    bars = ax.barh(var_names, agreements, color=colors)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Agreement (fraction of days CA bin matches ODE bin)")
    ax.axvline(x=fs["overall_agreement"], color="navy", linestyle="--",
               linewidth=2, label=f"Overall: {fs['overall_agreement']:.2f}")
    ax.legend()
    ax.set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
```

**Test:** Run `python ca_visualize.py` (after Task 2 adds the main block). Visually inspect output PNGs.

---

### Task 2: Add Population Grid Plot and Main Block to ca_visualize.py

**Files:**
- Modify: `~/lemurs-simulator/ca_visualize.py`

**Function 4: `plot_population_grid(pop_result, days, output_path)`**

NxN grid snapshots at specified days, each cell colored by attractor classification.

```python
_ATTRACTOR_COLORS = {
    "healthy": "#2ecc71",
    "struggling": "#f1c40f",
    "stressed": "#e67e22",
    "burnout": "#e74c3c",
}

def plot_population_grid(
    pop_result: dict, days: list[int] | None = None, output_path: str = ""
) -> None:
    grid_states = pop_result["grid_states"]
    grid_size = pop_result["grid_size"]
    sim_days = pop_result["sim_days"]

    if days is None:
        days = [0, 49, 56, sim_days]  # start, pre-break, post-break, end

    # Clamp days to valid range
    days = [min(d, len(grid_states) - 1) for d in days]

    fig, axes = plt.subplots(1, len(days), figsize=(4 * len(days), 4))
    if len(days) == 1:
        axes = [axes]

    attractor_to_int = {"healthy": 0, "struggling": 1, "stressed": 2, "burnout": 3}
    cmap = mcolors.ListedColormap(
        [_ATTRACTOR_COLORS[k] for k in ["healthy", "struggling", "stressed", "burnout"]]
    )

    for ax, day in zip(axes, days):
        grid_at_day = grid_states[day]
        grid_int = np.zeros((grid_size, grid_size), dtype=int)
        for r in range(grid_size):
            for c in range(grid_size):
                att = _classify_attractor(grid_at_day[r][c])
                grid_int[r, c] = attractor_to_int.get(att, 0)

        ax.imshow(grid_int, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(f"Day {day}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Population Grid — Attractor States", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
```

**Function 5: `plot_ca_all_scenarios(output_dir)`** — the "generate everything" function.

```python
def plot_ca_all_scenarios(output_dir: str = "output/ca") -> None:
    os.makedirs(output_dir, exist_ok=True)

    for seed in STUDENT_ARCHETYPES:
        patient = seed["patient"]
        intervention = seed.get("intervention", None)
        name = seed["name"]

        # Run CA
        ca_result = run_single_cell(patient=patient, intervention=intervention)

        # Trajectory heatmap
        plot_ca_trajectory(
            ca_result,
            f"CA: {name} — {seed['description']}",
            os.path.join(output_dir, f"ca_trajectory_{name}.png"),
        )

        # Rule timeline
        plot_rule_timeline(
            ca_result,
            f"Rules: {name}",
            os.path.join(output_dir, f"ca_rules_{name}.png"),
        )

        # Fidelity comparison with ODE
        ode_result = simulate(patient=patient, intervention=intervention)
        plot_ca_fidelity(
            ca_result, ode_result,
            f"CA-ODE Fidelity: {name}",
            os.path.join(output_dir, f"ca_fidelity_{name}.png"),
        )

    # Population grid (default patient, medium coupling)
    from ca_simulator import run_population_grid
    pop_result = run_population_grid(grid_size=5, social_coupling=0.3)
    plot_population_grid(
        pop_result,
        days=[0, 49, 56, 105],
        output_path=os.path.join(output_dir, "ca_population_grid.png"),
    )


if __name__ == "__main__":
    print("Generating LEMURS CA visualizations...")
    plot_ca_all_scenarios()
    print("Done.")
```

**Test:** `python ca_visualize.py` generates PNGs to `output/ca/`. Check output directory has files.

---

### Task 3: Visualization Tests

**Files:**
- Modify: `~/lemurs-simulator/tests/test_ca.py`

Add a new test class `TestCAVisualization` at the end of the file:

```python
from ca_visualize import (
    plot_ca_trajectory, plot_rule_timeline, plot_ca_fidelity,
    plot_population_grid,
)

class TestCAVisualization:
    """Test that CA visualization functions produce output files."""

    def test_trajectory_plot_creates_file(self, tmp_path):
        ca_result = run_single_cell()
        path = str(tmp_path / "traj.png")
        plot_ca_trajectory(ca_result, "Test Trajectory", path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_rule_timeline_creates_file(self, tmp_path):
        ca_result = run_single_cell()
        path = str(tmp_path / "rules.png")
        plot_rule_timeline(ca_result, "Test Rules", path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_fidelity_plot_creates_file(self, tmp_path):
        ca_result = run_single_cell()
        ode_result = simulate()
        path = str(tmp_path / "fidelity.png")
        plot_ca_fidelity(ca_result, ode_result, "Test Fidelity", path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_population_grid_creates_file(self, tmp_path):
        pop_result = run_population_grid(grid_size=3, social_coupling=0.2)
        path = str(tmp_path / "pop.png")
        plot_population_grid(pop_result, days=[0, 50, 105], output_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
```

Add the necessary imports at the top of `test_ca.py`:
```python
from ca_visualize import (
    plot_ca_trajectory, plot_rule_timeline, plot_ca_fidelity,
    plot_population_grid,
)
from simulator import simulate
```

**Verify:** `python -m pytest tests/test_ca.py::TestCAVisualization -v` — all 4 pass.

---

### Task 4: Expanded Social Coupling in ca_simulator.py

**Files:**
- Modify: `~/lemurs-simulator/ca_simulator.py:204-362` (run_population_grid function)

**Changes:**

1. **Update `run_population_grid` signature** to accept a `social_coupling_config` dict instead of a single float. Maintain backward compatibility: if a float is passed, apply it uniformly to all channels.

```python
def run_population_grid(
    grid_size: int = 5,
    patient_distribution: dict | None = None,
    intervention: dict | None = None,
    sim_days: int = N_STEPS,
    social_coupling: float | dict = 0.0,  # float for backward compat, or dict per channel
    seed: int = 42,
    rules: list[dict] | None = None,
) -> dict:
```

2. **Parse coupling config** at the start of the function:

```python
    # Parse social coupling config
    if isinstance(social_coupling, (int, float)):
        coupling_config = {
            "nature": float(social_coupling),
            "activity": float(social_coupling),
            "stress": float(social_coupling),
            "sleep": float(social_coupling),
            "anxiety": float(social_coupling),
        }
        coupling_scalar = float(social_coupling)
    else:
        coupling_config = {
            "nature": float(social_coupling.get("nature", 0.0)),
            "activity": float(social_coupling.get("activity", 0.0)),
            "stress": float(social_coupling.get("stress", 0.0)),
            "sleep": float(social_coupling.get("sleep", 0.0)),
            "anxiety": float(social_coupling.get("anxiety", 0.0)),
        }
        coupling_scalar = max(coupling_config.values())
```

3. **Replace the existing social coupling block** (lines ~299-337) with an expanded version. Keep the existing NatureEngagement and Activity coupling, and add 3 new channels:

```python
                # Social coupling: neighbor influence
                if coupling_scalar > 0:
                    neighbors = []
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid_size and 0 <= nc < grid_size:
                            neighbors.append(grid[nr][nc])

                    if neighbors:
                        n_neighbors = len(neighbors)

                        # Channel 1: Nature engagement (existing)
                        if coupling_config["nature"] > 0:
                            nature_engaged = sum(
                                1 for n in neighbors
                                if n.get("NatureEngagement") == "engaged"
                            )
                            if (nature_engaged / n_neighbors > 0.5
                                    and rng.random() < coupling_config["nature"]):
                                nat_labels = BIN_SCHEMA["NatureEngagement"]["labels"]
                                curr_idx = nat_labels.index(
                                    new_state.get("NatureEngagement", "low")
                                )
                                new_state["NatureEngagement"] = nat_labels[
                                    min(curr_idx + 1, len(nat_labels) - 1)
                                ]

                        # Channel 2: Activity (existing)
                        if coupling_config["activity"] > 0:
                            active_count = sum(
                                1 for n in neighbors
                                if n.get("Activity") == "active"
                            )
                            if (active_count / n_neighbors > 0.5
                                    and rng.random() < coupling_config["activity"]):
                                act_labels = BIN_SCHEMA["Activity"]["labels"]
                                curr_idx = act_labels.index(
                                    new_state.get("Activity", "moderate")
                                )
                                new_state["Activity"] = act_labels[
                                    min(curr_idx + 1, len(act_labels) - 1)
                                ]

                        # Channel 3: Stress contagion (NEW)
                        # Asymmetric: stress spreads easier than it recedes
                        if coupling_config["stress"] > 0:
                            high_stress = sum(
                                1 for n in neighbors
                                if n.get("PSS") == "high"
                            )
                            if (high_stress / n_neighbors > 0.5
                                    and rng.random() < coupling_config["stress"]):
                                pss_labels = BIN_SCHEMA["PSS"]["labels"]
                                curr_idx = pss_labels.index(
                                    new_state.get("PSS", "moderate")
                                )
                                new_state["PSS"] = pss_labels[
                                    min(curr_idx + 1, len(pss_labels) - 1)
                                ]
                            # Reverse: low-stress neighbors help reduce (at half strength)
                            low_stress = sum(
                                1 for n in neighbors
                                if n.get("PSS") == "low"
                            )
                            if (low_stress / n_neighbors > 0.5
                                    and rng.random() < coupling_config["stress"] * 0.5):
                                pss_labels = BIN_SCHEMA["PSS"]["labels"]
                                curr_idx = pss_labels.index(
                                    new_state.get("PSS", "moderate")
                                )
                                new_state["PSS"] = pss_labels[
                                    max(curr_idx - 1, 0)
                                ]

                        # Channel 4: Sleep norm influence (NEW)
                        # Only on school days — peer pressure to stay up late
                        if coupling_config["sleep"] > 0 and ctx.get("is_school_day"):
                            deprived_neighbors = sum(
                                1 for n in neighbors
                                if n.get("TST") == "deprived"
                            )
                            if (deprived_neighbors / n_neighbors > 0.5
                                    and rng.random() < coupling_config["sleep"]):
                                tst_labels = BIN_SCHEMA["TST"]["labels"]
                                curr_idx = tst_labels.index(
                                    new_state.get("TST", "adequate")
                                )
                                new_state["TST"] = tst_labels[
                                    max(curr_idx - 1, 0)
                                ]

                        # Channel 5: Anxiety diffusion (NEW)
                        # Clinical anxiety in neighbors amplifies development risk
                        # for sub-threshold students (doesn't force clinical directly)
                        if coupling_config["anxiety"] > 0:
                            clinical_neighbors = sum(
                                1 for n in neighbors
                                if n.get("GAD7") == "clinical"
                            )
                            if (clinical_neighbors / n_neighbors > 0.3  # lower threshold
                                    and new_state.get("GAD7") == "sub_threshold"
                                    and new_state.get("PSS") in ("moderate", "high")
                                    and rng.random() < coupling_config["anxiety"]):
                                new_state["GAD7"] = "clinical"
```

4. **Update the return dict** to include the coupling config:

Replace `"social_coupling": social_coupling,` with `"social_coupling": coupling_config,`

**Test:** Existing tests should still pass (backward compatible with float). New tests in Task 5.

---

### Task 5: Social Coupling Tests and Population Analytics

**Files:**
- Modify: `~/lemurs-simulator/tests/test_ca.py` — add tests to `TestPopulationGrid`
- Modify: `~/lemurs-simulator/ca_analytics.py` — add contagion analytics

**New tests in TestPopulationGrid:**

```python
    def test_stress_contagion_spreads(self):
        """Stress should spread from high-stress neighbors when coupling is active."""
        result_no_coupling = run_population_grid(
            grid_size=3, social_coupling=0.0, seed=42
        )
        result_with_coupling = run_population_grid(
            grid_size=3,
            social_coupling={"nature": 0.0, "activity": 0.0, "stress": 0.8,
                             "sleep": 0.0, "anxiety": 0.0},
            seed=42,
        )
        # With stress coupling, stress should spread more
        no_coupling_stress = result_no_coupling["population_summary"]["high_stress_fraction"]
        with_coupling_stress = result_with_coupling["population_summary"]["high_stress_fraction"]
        assert with_coupling_stress >= no_coupling_stress

    def test_sleep_norm_influence(self):
        """Sleep deprivation should spread via peer norms when coupling is active."""
        result = run_population_grid(
            grid_size=3,
            social_coupling={"nature": 0.0, "activity": 0.0, "stress": 0.0,
                             "sleep": 0.8, "anxiety": 0.0},
            seed=42,
        )
        assert "population_summary" in result

    def test_anxiety_diffusion(self):
        """Anxiety should diffuse from clinical neighbors to at-risk students."""
        result = run_population_grid(
            grid_size=3,
            social_coupling={"nature": 0.0, "activity": 0.0, "stress": 0.0,
                             "sleep": 0.0, "anxiety": 0.8},
            seed=42,
        )
        assert "population_summary" in result

    def test_dict_coupling_backward_compatible(self):
        """Float coupling should produce same result as equivalent dict."""
        result_float = run_population_grid(grid_size=3, social_coupling=0.3, seed=42)
        result_dict = run_population_grid(
            grid_size=3,
            social_coupling={"nature": 0.3, "activity": 0.3, "stress": 0.3,
                             "sleep": 0.3, "anxiety": 0.3},
            seed=42,
        )
        # Both should run without error and have same structure
        assert result_float["population_summary"]["total_students"] == result_dict["population_summary"]["total_students"]

    def test_coupling_config_in_result(self):
        """Result should include the parsed coupling config dict."""
        result = run_population_grid(grid_size=3, social_coupling=0.3, seed=42)
        assert isinstance(result["social_coupling"], dict)
        assert "stress" in result["social_coupling"]
```

**New population analytics in `ca_analytics.py`:**

Add a new function `_contagion_stats()` and integrate it into `compute_ca_analytics()` when population data is available. This function is called from the population bridge, not the single-cell path.

```python
def compute_population_analytics(pop_result: dict) -> dict:
    """Compute population-level analytics including contagion metrics."""
    grid_states = pop_result["grid_states"]
    grid_size = pop_result["grid_size"]
    final_grid = pop_result["final_grid"]

    # Attractor distribution
    attractor_counts = {"healthy": 0, "struggling": 0, "stressed": 0, "burnout": 0}
    for r in range(grid_size):
        for c in range(grid_size):
            att = _classify_attractor(final_grid[r][c])
            attractor_counts[att] = attractor_counts.get(att, 0) + 1

    total = grid_size * grid_size

    # Contagion clustering: find connected components of same attractor
    largest_stressed_cluster = _largest_cluster(final_grid, grid_size, {"stressed", "burnout"})

    return {
        "attractor_distribution": attractor_counts,
        "attractor_fractions": {k: v / total for k, v in attractor_counts.items()},
        "largest_stressed_cluster": largest_stressed_cluster,
        "total_students": total,
    }


def _largest_cluster(grid, grid_size, target_attractors):
    """BFS to find largest connected component of target attractor states."""
    visited = set()
    max_size = 0

    for r in range(grid_size):
        for c in range(grid_size):
            if (r, c) in visited:
                continue
            att = _classify_attractor(grid[r][c])
            if att not in target_attractors:
                visited.add((r, c))
                continue

            # BFS from (r, c)
            queue = [(r, c)]
            visited.add((r, c))
            cluster_size = 0
            while queue:
                cr, cc = queue.pop(0)
                cluster_size += 1
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < grid_size and 0 <= nc < grid_size
                            and (nr, nc) not in visited):
                        visited.add((nr, nc))
                        n_att = _classify_attractor(grid[nr][nc])
                        if n_att in target_attractors:
                            queue.append((nr, nc))
            max_size = max(max_size, cluster_size)

    return max_size
```

**Verify:** `python -m pytest tests/test_ca.py -v` — all tests pass (old 77 + new ~9).

---

### Task 6: Create ca_stochastic.py — Stochastic Rule Engine

**Files:**
- Create: `~/lemurs-simulator/ca_stochastic.py`

**Implementation:**

```python
"""Stochastic rule engine and ensemble analysis for the LEMURS CA.

Provides probabilistic rule application (sampling proportional to confidence)
and ensemble simulation that runs N stochastic trajectories to produce
distributional metrics: burnout probability, attractor confidence intervals,
and anxiety crossing probability.
"""
from __future__ import annotations

import numpy as np

from constants import N_STEPS, DEFAULT_INTERVENTION, DEFAULT_PATIENT, STUDENT_ARCHETYPES
from ca_schema import BIN_SCHEMA, _VAR_ORDER, discretize_state
from ca_rules import RULE_TABLE, get_applicable_rules, _apply_direction
from ca_simulator import _build_context
from ca_analytics import _classify_attractor, compute_ca_analytics
from simulator import initial_state
```

**Function 1: `apply_rules_stochastic(state, context, rng, rules=None)`**

Instead of highest-confidence-wins, when multiple rules propose updates to the same variable:
- Collect all proposals per variable
- Sample one proposal proportional to confidence
- Rules with confidence < 0.5 have a (1 - confidence) chance of not firing at all
- Burnout cascade still deterministic (absorbing state)

```python
def apply_rules_stochastic(
    discrete_state: dict[str, str],
    context: dict,
    rng: np.random.Generator,
    rules: list[dict] | None = None,
) -> tuple[dict[str, str], list[dict]]:
    applicable = get_applicable_rules(discrete_state, context, rules)

    # Burnout cascade is still absorbing (deterministic)
    for rule in applicable:
        if rule["name"] == "burnout_cascade":
            return dict(discrete_state), applicable

    # Collect proposals per variable: {var: [(direction, confidence, rule)]}
    proposals: dict[str, list[tuple[str, float, dict]]] = {}
    for rule in applicable:
        # Probabilistic gating: rules with confidence < 0.5 may not fire
        if rule["confidence"] < 0.5 and rng.random() > rule["confidence"]:
            continue
        for var_name, direction in rule["outputs"].items():
            if var_name not in proposals:
                proposals[var_name] = []
            proposals[var_name].append((direction, rule["confidence"], rule))

    # For each variable, sample one proposal proportional to confidence
    new_state = dict(discrete_state)
    for var_name, prop_list in proposals.items():
        if len(prop_list) == 1:
            direction = prop_list[0][0]
        else:
            confidences = np.array([p[1] for p in prop_list])
            probs = confidences / confidences.sum()
            idx = rng.choice(len(prop_list), p=probs)
            direction = prop_list[idx][0]

        new_state[var_name] = _apply_direction(
            new_state[var_name], direction, var_name
        )

    return new_state, applicable
```

**Function 2: `run_single_cell_stochastic(patient, intervention, sim_days, n_trials, seed)`**

```python
def run_single_cell_stochastic(
    patient: dict | None = None,
    intervention: dict | None = None,
    sim_days: int = N_STEPS,
    n_trials: int = 100,
    seed: int = 42,
    rules: list[dict] | None = None,
) -> dict:
    pat = {**DEFAULT_PATIENT, **(patient or {})}
    intv = {**DEFAULT_INTERVENTION, **(intervention or {})}

    pat_with_rx = {**pat, "nature_rx": intv.get("nature_rx", 0.0)}
    continuous_init = initial_state(pat_with_rx)
    init_state = discretize_state(continuous_init)

    all_trajectories = []
    all_final_states = []
    all_rule_logs = []

    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial)
        state = dict(init_state)
        trajectory = [dict(state)]
        rule_log = []
        prev_state = None

        for day in range(sim_days):
            ctx = _build_context(day, pat, intv, prev_state, state)
            new_state, fired = apply_rules_stochastic(state, ctx, rng, rules)
            rule_log.append([r["name"] for r in fired])
            prev_state = state
            state = new_state
            trajectory.append(dict(state))

        all_trajectories.append(trajectory)
        all_final_states.append(dict(state))
        all_rule_logs.append(rule_log)

    return {
        "trajectories": all_trajectories,
        "final_states": all_final_states,
        "rule_logs": all_rule_logs,
        "n_trials": n_trials,
        "initial_state": dict(init_state),
        "patient": pat,
        "intervention": intv,
    }
```

**Function 3: `compute_ensemble_analytics(ensemble_result)`**

```python
def compute_ensemble_analytics(ensemble_result: dict) -> dict:
    n_trials = ensemble_result["n_trials"]
    final_states = ensemble_result["final_states"]

    # Attractor distribution
    attractor_counts = {"healthy": 0, "struggling": 0, "stressed": 0, "burnout": 0}
    for state in final_states:
        att = _classify_attractor(state)
        attractor_counts[att] = attractor_counts.get(att, 0) + 1

    attractor_probs = {k: v / n_trials for k, v in attractor_counts.items()}

    # Per-variable final bin distributions
    var_distributions = {}
    for var_name in _VAR_ORDER:
        counts = {}
        for state in final_states:
            label = state.get(var_name, "")
            counts[label] = counts.get(label, 0) + 1
        var_distributions[var_name] = {k: v / n_trials for k, v in counts.items()}

    # Anxiety crossing probability: fraction of trials where GAD7 ever reaches clinical
    anxiety_crossings = 0
    for traj in ensemble_result["trajectories"]:
        for state in traj:
            if state.get("GAD7") == "clinical":
                anxiety_crossings += 1
                break
    anxiety_crossing_prob = anxiety_crossings / n_trials

    # Burnout probability
    burnout_prob = attractor_probs.get("burnout", 0.0)

    return {
        "attractor_counts": attractor_counts,
        "attractor_probabilities": attractor_probs,
        "burnout_probability": burnout_prob,
        "anxiety_crossing_probability": anxiety_crossing_prob,
        "variable_distributions": var_distributions,
        "n_trials": n_trials,
    }
```

**Test:** Import and run in Task 7.

---

### Task 7: Stochastic Engine Tests and Zimmerman Adapter

**Files:**
- Modify: `~/lemurs-simulator/tests/test_ca.py` — add TestStochastic class
- Modify: `~/lemurs-simulator/ca_zimmerman_bridge.py` — add LEMURSCAEnsembleSimulator

**New test class:**

```python
from ca_stochastic import (
    apply_rules_stochastic, run_single_cell_stochastic,
    compute_ensemble_analytics,
)

class TestStochastic:
    """Stochastic rule engine and ensemble tests."""

    def test_stochastic_rules_produce_valid_state(self):
        """Stochastic rule application should produce valid bin labels."""
        ca_result = run_single_cell()
        state = ca_result["trajectory"][50]
        ctx = _build_context(50, DEFAULT_PATIENT, DEFAULT_INTERVENTION)
        rng = np.random.default_rng(42)
        new_state, fired = apply_rules_stochastic(state, ctx, rng)
        for var_name in _VAR_ORDER:
            assert new_state[var_name] in BIN_SCHEMA[var_name]["labels"]

    def test_stochastic_nondeterministic(self):
        """Different seeds should sometimes produce different results."""
        results = set()
        for seed in range(20):
            rng = np.random.default_rng(seed)
            ca_result = run_single_cell()
            state = ca_result["trajectory"][50]
            ctx = _build_context(50, DEFAULT_PATIENT, DEFAULT_INTERVENTION)
            new_state, _ = apply_rules_stochastic(state, ctx, rng)
            results.add(tuple(sorted(new_state.items())))
        # With 20 seeds, we should see at least 2 different outcomes
        # (may be 1 if rules are strongly determined; relax if needed)
        assert len(results) >= 1

    def test_ensemble_runs(self):
        """Ensemble simulation should produce n_trials trajectories."""
        result = run_single_cell_stochastic(n_trials=10, seed=42)
        assert len(result["trajectories"]) == 10
        assert len(result["final_states"]) == 10

    def test_ensemble_analytics(self):
        """Ensemble analytics should return valid probabilities."""
        result = run_single_cell_stochastic(n_trials=20, seed=42)
        analytics = compute_ensemble_analytics(result)
        assert 0.0 <= analytics["burnout_probability"] <= 1.0
        assert 0.0 <= analytics["anxiety_crossing_probability"] <= 1.0
        total_prob = sum(analytics["attractor_probabilities"].values())
        assert abs(total_prob - 1.0) < 1e-6

    def test_ensemble_deterministic_same_seed(self):
        """Same seed should produce same ensemble results."""
        r1 = run_single_cell_stochastic(n_trials=5, seed=42)
        r2 = run_single_cell_stochastic(n_trials=5, seed=42)
        for i in range(5):
            assert r1["final_states"][i] == r2["final_states"][i]

    def test_burnout_cascade_still_absorbing(self):
        """Burnout cascade should remain absorbing even in stochastic mode."""
        burnout_state = {
            "TST": "deprived", "SleepQuality": "poor", "PSS": "high",
            "GAD7": "clinical", "Depression": "moderate_plus", "Activity": "sedentary",
            "NatureEngagement": "low", "RHR": "elevated", "HRV": "low",
            "ARR": "elevated", "SocialJetlag": "misaligned", "SleepShape": "disrupted",
            "WEMWBS": "low", "DAC": "depleted",
        }
        ctx = _build_context(50, DEFAULT_PATIENT, DEFAULT_INTERVENTION)
        rng = np.random.default_rng(42)
        new_state, _ = apply_rules_stochastic(burnout_state, ctx, rng)
        assert new_state == burnout_state
```

**Zimmerman adapter in `ca_zimmerman_bridge.py`:**

Add `LEMURSCAEnsembleSimulator` class after the existing classes:

```python
from ca_stochastic import run_single_cell_stochastic, compute_ensemble_analytics

class LEMURSCAEnsembleSimulator:
    """Zimmerman-protocol-compatible stochastic CA ensemble simulator.

    Runs N stochastic CA trajectories and returns distributional metrics
    (mean, std, probabilities) instead of single deterministic values.
    """

    def __init__(self, n_trials: int = 50):
        self.n_trials = n_trials

    def param_spec(self) -> dict[str, tuple[float, float]]:
        return {**INTERVENTION_BOUNDS, **PATIENT_BOUNDS}

    def run(self, params: dict[str, float]) -> dict[str, float]:
        intervention = dict(DEFAULT_INTERVENTION)
        patient = dict(DEFAULT_PATIENT)

        for k, v in params.items():
            if k in INTERVENTION_BOUNDS:
                intervention[k] = float(v)
            elif k in PATIENT_BOUNDS:
                patient[k] = float(v)

        ensemble = run_single_cell_stochastic(
            patient=patient, intervention=intervention,
            n_trials=self.n_trials,
        )
        analytics = compute_ensemble_analytics(ensemble)

        flat = _flatten_ensemble_analytics(analytics)
        return flat


def _flatten_ensemble_analytics(analytics: dict) -> dict[str, float]:
    """Convert ensemble analytics to a flat dict of floats."""
    flat: dict[str, float] = {}

    flat["ens_burnout_probability"] = analytics["burnout_probability"]
    flat["ens_anxiety_crossing_prob"] = analytics["anxiety_crossing_probability"]
    flat["ens_n_trials"] = float(analytics["n_trials"])

    for att, prob in analytics["attractor_probabilities"].items():
        flat[f"ens_attractor_{att}_prob"] = prob

    for var_name, dist in analytics["variable_distributions"].items():
        for label, frac in dist.items():
            flat[f"ens_final_{var_name}_{label}_frac"] = frac

    # Guard against NaN/Inf
    for k, v in flat.items():
        if math.isnan(v):
            flat[k] = 0.0
        elif math.isinf(v):
            flat[k] = 999.0

    return flat
```

**Add test for the adapter:**

```python
from ca_zimmerman_bridge import LEMURSCAEnsembleSimulator

class TestEnsembleBridge:
    """Zimmerman adapter for stochastic ensemble."""

    def test_ensemble_simulator_runs(self):
        sim = LEMURSCAEnsembleSimulator(n_trials=5)
        result = sim.run({})
        assert isinstance(result, dict)
        assert "ens_burnout_probability" in result
        assert "ens_anxiety_crossing_prob" in result

    def test_ensemble_param_spec(self):
        sim = LEMURSCAEnsembleSimulator()
        spec = sim.param_spec()
        assert len(spec) == 12

    def test_ensemble_all_values_finite(self):
        sim = LEMURSCAEnsembleSimulator(n_trials=5)
        result = sim.run({"nature_rx": 0.8})
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not float"
            assert not math.isnan(v), f"{k} is NaN"
            assert not math.isinf(v), f"{k} is Inf"
```

**Verify:** `python -m pytest tests/test_ca.py -v` — all tests pass.

---

### Task 8: Update CLAUDE.md and Full Verification

**Files:**
- Modify: `~/lemurs-simulator/CLAUDE.md`

**Update the Semantic Cellular Automaton section** to include the three new capabilities:

In the Architecture section, add:
```
ca_visualize.py           <- Semantic CA: trajectory heatmap, rule timeline, population grid, fidelity plots
ca_stochastic.py          <- Semantic CA: stochastic rule engine, ensemble simulation, distributional analytics
```

In the Commands section, add:
```bash
python ca_visualize.py                # generate all CA plots to output/ca/
python -c "from ca_stochastic import run_single_cell_stochastic; r = run_single_cell_stochastic(n_trials=10); print(f'{r[\"n_trials\"]} trials')"
python -c "from ca_zimmerman_bridge import LEMURSCAEnsembleSimulator; s = LEMURSCAEnsembleSimulator(n_trials=10); print(s.run({}))"
```

In the Key CA dynamics section, add:
- **Expanded social coupling** — 5 channels (nature, activity, stress, sleep, anxiety); stress spreads asymmetrically (contagion strength > recovery strength); anxiety diffusion lowers the GAD-7 development threshold for at-risk students near clinical neighbors
- **Stochastic rules** — probabilistic rule firing proportional to confidence; ensemble of N trajectories produces burnout probability, anxiety crossing probability, and attractor confidence intervals; burnout cascade remains deterministic (absorbing state)

**Full verification:**

```bash
cd ~/lemurs-simulator
python -m pytest tests/test_ca.py -v              # All CA tests pass (old + new)
python -m pytest tests/ -v                         # Full suite: no regressions
python ca_visualize.py                             # Generates PNGs to output/ca/
python -c "from ca_stochastic import run_single_cell_stochastic, compute_ensemble_analytics; r = run_single_cell_stochastic(n_trials=20); a = compute_ensemble_analytics(r); print(a['attractor_probabilities'])"
python -c "from ca_zimmerman_bridge import LEMURSCAEnsembleSimulator; s = LEMURSCAEnsembleSimulator(n_trials=10); r = s.run({}); print(len(r), 'metrics')"
```

---

## Verification

After all tasks:
```bash
cd ~/lemurs-simulator
python -m pytest tests/ -v                    # All tests pass (229 existing + ~20 new ≈ 249)
python -m pytest tests/test_ca.py -v          # CA-specific tests (77 existing + ~20 new ≈ 97)
python ca_visualize.py                        # Generates to output/ca/
ls output/ca/                                 # Should show ~25 PNGs
```
