#!/usr/bin/env python3
"""Visualize the explicit C-space obstacle map produced by
build_cspace_obstacle_map.

Usage:
    python3 experiments/visualize_cspace.py
    python3 experiments/visualize_cspace.py --maze bench_mazes/corridor.csv
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import micromouse
from bangbang_rrt_star import build_cspace_obstacle_map


def main():
    parser = argparse.ArgumentParser(description="Visualize C-space obstacle inflation")
    parser.add_argument("--maze", default="bench_mazes/narrow.csv",
                        help="Path to maze CSV (default: narrow.csv)")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--output-dir", default="experiments/results/charts",
                        help="Directory for output PNG")
    parser.add_argument("--robot-radius", type=float, default=None,
                        help="Override robot_radius (default: read from config)")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    bb_cfg = config.get("bb_rrt_star", {})
    robot_radius = args.robot_radius if args.robot_radius is not None else bb_cfg.get("robot_radius", 0.28)

    maze_obj = micromouse.Map(micromouse.load_maze(args.maze))
    grid = maze_obj.get_grid_representation()
    grid_walls = (grid == 1).astype(int)

    cspace_blocked = build_cspace_obstacle_map(grid_walls, robot_radius)

    maze_name = os.path.splitext(os.path.basename(args.maze))[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # --- left panel: original maze ---
    ax1.imshow(grid_walls, cmap="gray_r", interpolation="nearest", origin="upper")
    ax1.set_title(f"Original maze — {maze_name} ({grid.shape[0]}x{grid.shape[1]})")
    ax1.set_xlabel("column")
    ax1.set_ylabel("row")

    # --- right panel: C-space obstacles ---
    vis = np.zeros((*grid_walls.shape, 3))
    # original walls in dark grey
    vis[grid_walls == 1] = [0.2, 0.2, 0.2]
    # inflation ring (blocked but not wall) in orange
    inflation = cspace_blocked & (grid_walls == 0)
    vis[inflation] = [1.0, 0.6, 0.2]
    # free space in white
    vis[~cspace_blocked] = [1.0, 1.0, 1.0]

    ax2.imshow(vis, interpolation="nearest", origin="upper")
    ax2.set_title(f"C-space obstacles  (R = {robot_radius:.2f} cells)\n"
                   f"orange = inflation ring, dark = original walls")
    ax2.set_xlabel("column")
    ax2.set_ylabel("row")

    n_wall = int(grid_walls.sum())
    n_inflation = int(inflation.sum())
    n_free = int((~cspace_blocked).sum())
    fig.suptitle(f"Minkowski inflation  —  walls: {n_wall},  "
                 f"inflation: {n_inflation},  free: {n_free}", fontsize=11)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"cspace_{maze_name}_R{robot_radius:.1f}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
