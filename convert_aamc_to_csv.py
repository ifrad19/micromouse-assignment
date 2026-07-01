"""Convert AAMC .txt maze to a small .csv grid suitable for BB-RRT*.

The AAMC format (o, |, ---, S, G) represents a 16x16-cell maze.
This converter produces a compact grid where each logical cell becomes
a SCALE x SCALE block. Walls between cells are rendered as 1-cell-thick
lines of wall pixels.

Usage:
    python convert_aamc_to_csv.py mazefiles/classic/AAMC15Maze.txt output.csv [SCALE]
"""

import sys
import csv


def convert_aamc_to_grid(txt_path, scale=4):
    """Convert AAMC .txt maze to a numeric grid.

    Args:
        txt_path: path to .txt file in AAMC format
        scale: cells per logical AAMC cell (default 4 → 65x65 grid)

    Returns:
        (grid, start_rc, goal_rc) where grid is list-of-lists of ints,
        start_rc and goal_rc are (row, col) tuples.
    """
    with open(txt_path) as f:
        lines = [line.rstrip('\n') for line in f if line.strip()]

    size = 16  # AAMC mazes are 16x16 cells
    g = size * scale + 1  # grid dimension

    # Start all-free
    grid = [[0] * g for _ in range(g)]

    # --- Parse horizontal walls (even lines: 0, 2, 4, ...) ---
    for i in range(0, min(len(lines), 2 * size + 1), 2):
        line = lines[i].strip()
        wall_row = (i // 2) * scale
        for c in range(size):
            seg = line[4 * c + 1 : 4 * c + 4] if 4 * c + 4 <= len(line) else '---'
            if '-' in seg:
                for j in range(c * scale, min(g, c * scale + scale + 1)):
                    if 0 <= wall_row < g:
                        grid[wall_row][j] = 1

    # --- Parse vertical walls and cell content (odd lines: 1, 3, 5, ...) ---
    start_rc = None
    goal_rc = None
    for i in range(1, min(len(lines), 2 * size + 1), 2):
        line = lines[i].strip()
        row_idx = (i - 1) // 2

        # Vertical walls
        for c in range(size + 1):
            if 4 * c < len(line) and line[4 * c] == '|':
                col = c * scale
                for r in range(row_idx * scale, min(g, row_idx * scale + scale + 1)):
                    if 0 <= col < g:
                        grid[r][col] = 1

        # Cell content
        for c in range(size):
            content = line[4 * c + 1 : 4 * c + 4] if 4 * c + 4 <= len(line) else '   '
            cr = row_idx * scale + scale // 2
            cc = c * scale + scale // 2
            if 'S' in content:
                start_rc = (cr, cc)
                grid[cr][cc] = 2
            elif 'G' in content:
                goal_rc = (cr, cc)
                grid[cr][cc] = 4

    # Borders (should already be walls from parsing, but ensure)
    for i in range(g):
        grid[0][i] = grid[g - 1][i] = grid[i][0] = grid[i][g - 1] = 1

    # Defaults if no S/G found
    if start_rc is None:
        sc = scale // 2
        start_rc = (sc, sc)
        grid[sc][sc] = 2
    if goal_rc is None:
        ec = (size - 1) * scale + scale // 2
        goal_rc = (ec, ec)
        grid[ec][ec] = 4

    return grid, start_rc, goal_rc


def save_csv(grid, csv_path):
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        for row in grid:
            w.writerow(row)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python convert_aamc_to_csv.py input.txt output.csv [scale]")
        sys.exit(1)
    txt_path = sys.argv[1]
    csv_path = sys.argv[2]
    scale = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    grid, start, goal = convert_aamc_to_grid(txt_path, scale)
    save_csv(grid, csv_path)

    rows = len(grid)
    cols = len(grid[0])
    free = sum(1 for r in grid for c in r if c == 0)
    walls = sum(1 for r in grid for c in r if c == 1)
    print(f"Converted {txt_path} -> {csv_path}")
    print(f"  Grid: {rows}x{cols}, start={start}, goal={goal}")
    print(f"  Free cells: {free}, Wall cells: {walls}")
