"""
Create a small 16x16 maze suite for benchmarking.

Output: /tmp/bench/<name>.csv

Each maze has:
    0 = free
    1 = wall
    2 = start
    4 = end
"""

import csv
import os
import sys

BENCH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bench_mazes')
os.makedirs(BENCH_DIR, exist_ok=True)


def write_maze(name, maze):
    path = os.path.join(BENCH_DIR, f'{name}.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        for row in maze:
            w.writerow(row)
    return path


def in_bounds(maze, r, c):
    return 0 <= r < len(maze) and 0 <= c < len(maze[0])


def make_corridor_16x16():
    """16x16 with a few horizontal/vertical walls; start top-left, goal bottom-right."""
    g = [[0] * 16 for _ in range(16)]
    # Outer wall
    for i in range(16):
        g[0][i] = g[15][i] = 1
        g[i][0] = g[i][15] = 1
    # Interior wall block
    for c in range(4, 12):
        g[7][c] = 1
    for r in range(3, 8):
        g[r][4] = 1
    # Start (1, 1) and goal (14, 14)
    g[1][1] = 2
    g[14][14] = 4
    return g


def make_open_field_16x16():
    """Wide open 16x16 with just one pillar."""
    g = [[0] * 16 for _ in range(16)]
    for i in range(16):
        g[0][i] = g[15][i] = 1
        g[i][0] = g[i][15] = 1
    # central pillar
    g[8][7] = g[8][8] = g[7][7] = g[7][8] = 1
    g[1][1] = 2
    g[14][14] = 4
    return g


def make_spiral_16x16():
    """Spiral walls forcing a long, curved path."""
    g = [[0] * 16 for _ in range(16)]
    for i in range(16):
        g[0][i] = g[15][i] = 1
        g[i][0] = g[i][15] = 1
    # horizontal wall
    for c in range(1, 14):
        g[4][c] = 1
    # vertical wall going down from row 4 col 13
    for r in range(4, 14):
        g[r][13] = 1
    # horizontal wall going left
    for c in range(2, 14):
        g[11][c] = 1
    # vertical wall going up
    for r in range(4, 11):
        g[r][2] = 1
    g[1][1] = 2
    g[10][1] = 4
    return g


def make_narrow_16x16():
    """Tight zig-zag corridors (1-cell wide) — solvable path through."""
    g = [[0] * 16 for _ in range(16)]
    for i in range(16):
        g[0][i] = g[15][i] = 1
        g[i][0] = g[i][15] = 1
    # Long horizontal walls that force a long detour.
    # Leave a 1-cell gap in the middle for passage.
    for c in range(1, 15):
        g[5][c] = 1
        g[10][c] = 1
    g[5][8] = 0  # gap at col 8 in row 5
    g[10][7] = 0  # gap at col 7 in row 10 (offset for zig-zag)
    # Vertical walls creating 1-cell-wide channels
    for r in range(5, 10):
        g[r][10] = 1
    for r in range(2, 5):  # leave row 1 open so start can escape
        g[r][3] = 1
    g[1][1] = 2
    g[13][14] = 4
    return g


if __name__ == '__main__':
    for name, m in [
        ('corridor', make_corridor_16x16()),
        ('open_field', make_open_field_16x16()),
        ('spiral', make_spiral_16x16()),
        ('narrow', make_narrow_16x16()),
    ]:
        p = write_maze(name, m)
        print(f'  wrote {p}  ({len(m)}x{len(m[0])})')
