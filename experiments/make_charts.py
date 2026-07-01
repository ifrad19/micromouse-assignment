"""Generate bar charts for the bang-bang RRT* research report.

Reads:
  - experiments/results/summary_<latest>.csv
  - experiments/results/controller_comparison.csv

Writes:
  - experiments/results/charts/arrival_time.png
  - experiments/results/charts/path_length.png
  - experiments/results/charts/controller_time.png
  - experiments/results/charts/controller_cte.png
"""

import csv
import os
from collections import defaultdict
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CHART_DIR = os.path.join(RESULTS_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

PLANNER_ORDER = [
    "astar",
    "theta_star",
    "rrt",
    "rrt_smooth",
    "rrt_grid_smooth",
    "rrt_grid_smooth_strategic",
    "rrt_smooth_bangbang_circles",
    "bb_rrt_star",
]
PLANNER_LABEL = {
    "astar": "A*",
    "theta_star": "Theta*",
    "rrt": "RRT",
    "rrt_smooth": "RRT+smooth",
    "rrt_grid_smooth": "RRT+grid-smooth",
    "rrt_grid_smooth_strategic": "RRT+strategic",
    "rrt_smooth_bangbang_circles": "RRT+smooth+BB",
    "bb_rrt_star": "BB-RRT*",
}
PLANNER_COLOR = {
    "astar": "#888888",
    "theta_star": "#1f77b4",
    "rrt": "#ff7f0e",
    "rrt_smooth": "#2ca02c",
    "rrt_grid_smooth": "#9467bd",
    "rrt_grid_smooth_strategic": "#e377c2",
    "rrt_smooth_bangbang_circles": "#8c564b",
    "bb_rrt_star": "#d62728",
}


def load_latest_summary():
    paths = sorted(glob(os.path.join(RESULTS_DIR, "summary_*.csv")))
    if not paths:
        raise SystemExit("no summary_*.csv found")
    with open(paths[-1]) as f:
        return list(csv.DictReader(f)), os.path.basename(paths[-1])


def load_controller_comparison():
    path = os.path.join(RESULTS_DIR, "controller_comparison.csv")
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_arrival_time(rows, source):
    by_maze = defaultdict(dict)
    for r in rows:
        by_maze[r["maze"]][r["planner"]] = float(r["path_length"]) if r["path_length"] else float("inf")

    mazes = sorted(by_maze.keys())
    planners = [p for p in PLANNER_ORDER if any(p in by_maze[m] for m in mazes)]
    n_maze = len(mazes)
    n_planner = len(planners)
    width = 0.8 / n_planner
    x = np.arange(n_maze)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, p in enumerate(planners):
        vals = [by_maze[m].get(p, 0) for m in mazes]
        vals = [v if np.isfinite(v) else 0 for v in vals]
        ax.bar(
            x + (i - n_planner / 2) * width + width / 2,
            vals,
            width,
            label=PLANNER_LABEL[p],
            color=PLANNER_COLOR[p],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(mazes)
    ax.set_ylabel("arrival_time_s  (path_length for non-bang-bang planners)")
    ax.set_title(f"Arrival time per planner per maze\n(source: {source})")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    out = os.path.join(CHART_DIR, "arrival_time.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def plot_path_length(rows, source):
    by_maze = defaultdict(dict)
    for r in rows:
        by_maze[r["maze"]][r["planner"]] = float(r["path_length_cells"])

    mazes = sorted(by_maze.keys())
    planners = [p for p in PLANNER_ORDER if any(p in by_maze[m] for m in mazes)]
    n_maze = len(mazes)
    n_planner = len(planners)
    width = 0.8 / n_planner
    x = np.arange(n_maze)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, p in enumerate(planners):
        vals = [by_maze[m].get(p, 0) for m in mazes]
        ax.bar(
            x + (i - n_planner / 2) * width + width / 2,
            vals,
            width,
            label=PLANNER_LABEL[p],
            color=PLANNER_COLOR[p],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(mazes)
    ax.set_ylabel("path_length_cells")
    ax.set_title(f"Geometric path length per planner per maze\n(source: {source})")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    out = os.path.join(CHART_DIR, "path_length.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def plot_controller_time(rows):
    by_maze_tracker = defaultdict(dict)
    for r in rows:
        by_maze_tracker[r["maze"]][r["tracker"]] = {
            "time_s": float(r["time_s"]) if r["status"] == "GOAL" else float("nan"),
            "max_cte": float(r["max_cte"]) if r["max_cte"] else float("nan"),
            "mean_cte": float(r["mean_cte"]) if r["mean_cte"] else float("nan"),
            "status": r["status"],
        }

    mazes = sorted(by_maze_tracker.keys())
    trackers = ["pure_pursuit", "stanley"]
    tracker_label = {"pure_pursuit": "Pure Pursuit", "stanley": "Stanley"}
    tracker_color = {"pure_pursuit": "#1f77b4", "stanley": "#d62728"}

    x = np.arange(len(mazes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, t in enumerate(trackers):
        times = [by_maze_tracker[m][t]["time_s"] for m in mazes]
        ax.bar(
            x + (i - 0.5) * width,
            times,
            width,
            label=tracker_label[t],
            color=tracker_color[t],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(mazes)
    ax.set_ylabel("time to goal (s)  [nan = COLLISION]")
    ax.set_title("Controller comparison: time to goal\n(0.5 m/s, centered start)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out = os.path.join(CHART_DIR, "controller_time.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def plot_controller_cte(rows):
    by_maze_tracker = defaultdict(dict)
    for r in rows:
        by_maze_tracker[r["maze"]][r["tracker"]] = {
            "max_cte": float(r["max_cte"]) if r["max_cte"] else float("nan"),
            "mean_cte": float(r["mean_cte"]) if r["mean_cte"] else float("nan"),
        }

    mazes = sorted(by_maze_tracker.keys())
    trackers = ["pure_pursuit", "stanley"]
    tracker_label = {"pure_pursuit": "Pure Pursuit", "stanley": "Stanley"}
    tracker_color = {"pure_pursuit": "#1f77b4", "stanley": "#d62728"}
    x = np.arange(len(mazes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, t in enumerate(trackers):
        max_cte = [by_maze_tracker[m][t]["max_cte"] for m in mazes]
        ax.bar(
            x + (i - 0.5) * width,
            max_cte,
            width,
            label=tracker_label[t],
            color=tracker_color[t],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(mazes)
    ax.set_ylabel("max cross-track error (cells)  [0 = not measured]")
    ax.set_title("Controller comparison: max cross-track error")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.5, label="1-cell corridor")

    out = os.path.join(CHART_DIR, "controller_cte.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    summary, source = load_latest_summary()
    print(f"using summary: {source}")
    plot_arrival_time(summary, source)
    plot_path_length(summary, source)

    try:
        controller = load_controller_comparison()
        plot_controller_time(controller)
        plot_controller_cte(controller)
    except FileNotFoundError:
        print("controller_comparison.csv not found, skipping controller charts")


if __name__ == "__main__":
    main()
