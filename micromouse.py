"""
Micromouse Simulation Program
----------------------------
This program simulates a micromouse robot navigating through a maze using:
- A* algorithm for path planning
- Pure Pursuit controller for path following
- Collision detection and reporting

The program reads maze configuration from a CSV file and parameters from a YAML config file.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import csv
from sys import maxsize
import time
import yaml

# Path planning algorithms - import additional planners as you implement them
from pathPlanning import AStar
try:
    from pathPlanning import ThetaStar
except ImportError:
    ThetaStar = None
    print("Note: ThetaStar not available. Set path_planner to 'astar' in config.yaml")

try:
    from pathPlanning import RRT
except ImportError:
    RRT = None
    print("Note: RRT not available. Implement in pathPlanning.py")

from pathTracking import PurePursuit

def optimize_path(path, map, min_distance=0.3):
    """
    Optimize path using clothoid curves for smooth corners.
    
    Args:
        path: List of (x, y) waypoints from planner
        map: Maze map object
        min_distance: Minimum clearance from walls
        
    Returns:
        Optimized path as list of (x, y) points
        
    See clothoids.md for implementation details.
    """
    # Your clothoid-based smoothing implementation
    return path  # Placeholder: returns unmodified path

class State:
    """
    Represents a single cell in the maze with position, state, and pathfinding information
    """
    def __init__(self, x, y):
        self.x = x  # x-coordinate in the maze
        self.y = y  # y-coordinate in the maze
        self.parent = None  # Parent state in pathfinding
        self.state = "."  # Cell state: ".", "#", "s", "e", or "*"
        self.t = "new"  # State tag for algorithms
        # For A* search:
        # g = cost from start to this state
        # h = heuristic estimate from this state to goal
        # k = f = g + h (total estimated cost)
        self.g = 0.0
        self.h = 0.0
        self.k = 0.0

    def cost(self, state):
        """Calculate movement cost to another state (infinity if wall)"""
        if self.state == "#" or state.state == "#":
            return maxsize  # Infinite cost for walls
        return math.sqrt((self.x - state.x)**2 + (self.y - state.y)**2)

    def set_state(self, state):
        """Set the cell state if valid"""
        if state in ["s", ".", "#", "e", "*"]:
            self.state = state

class Map:
    """
    Represents the maze environment and provides maze-related operations
    """
    def __init__(self, maze_array):
        self.row = len(maze_array)
        self.col = len(maze_array[0]) if self.row > 0 else 0
        self.map = self.init_map()
        self.process_maze_array(maze_array)

    def init_map(self):
        """Initialize 2D grid of State objects"""
        return [[State(i, j) for j in range(self.col)] for i in range(self.row)]

    def process_maze_array(self, maze_array):
        """Convert numerical maze array to State objects with proper states. If multiple goal cells (4) exist, set only the centroid as the goal."""
        self.start = None
        goal_cells = []
        for i in range(self.row):
            for j in range(self.col):
                if maze_array[i][j] == 1:  # Wall
                    self.map[i][j].set_state("#")
                elif maze_array[i][j] == 2:  # Start
                    self.map[i][j].set_state("s")
                    self.start = (i, j)
                elif maze_array[i][j] == 4:  # End
                    goal_cells.append((i, j))
                else:
                    self.map[i][j].set_state(".")
        if not goal_cells:
            raise ValueError("Maze must contain at least one goal (4) cell")
        # Compute centroid of all goal cells
        centroid_i = int(round(sum(i for i, _ in goal_cells) / len(goal_cells)))
        centroid_j = int(round(sum(j for _, j in goal_cells) / len(goal_cells)))
        self.goal = (centroid_i, centroid_j)
        # Set all cells to free except the centroid, which is set as goal
        for i, j in goal_cells:
            if (i, j) == self.goal:
                self.map[i][j].set_state("e")
            else:
                self.map[i][j].set_state(".")
        if self.start is None or self.goal is None:
            raise ValueError("Maze must contain start (2) and goal (4) positions")

    def get_neighbors(self, state):
        """Get valid neighboring states (up, down, left, right)"""
        neighbors = []
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:  # 4-directional movement
            x, y = state.x + dx, state.y + dy
            if 0 <= x < self.row and 0 <= y < self.col:
                neighbors.append(self.map[x][y])
        return neighbors

    def get_grid_representation(self):
        """Convert State map to numerical grid for visualization"""
        grid = np.zeros((self.row, self.col))
        for i in range(self.row):
            for j in range(self.col):
                if self.map[i][j].state == "#":  # Wall
                    grid[i][j] = 1
                elif self.map[i][j].state == "s":  # Start
                    grid[i][j] = 3
                elif self.map[i][j].state == "e":  # End
                    grid[i][j] = 4
                elif self.map[i][j].state == "*":  # Path
                    grid[i][j] = 2
        return grid

    def compute_wall_distance_map(self):
        """Compute approximate Euclidean distance from every cell to the
        nearest wall using a two-pass chamfer distance transform.

        Result is stored in ``self.wall_distance`` as a numpy array of shape
        (row, col).  Wall cells have distance 0; free cells have positive
        distance values.
        """
        grid = np.asarray(self.get_grid_representation())
        is_wall = (grid == 1)

        dist = np.full((self.row, self.col), np.inf)
        dist[is_wall] = 0.0

        ORTH = 1.0     # orthogonal step cost
        DIAG = 1.414   # diagonal step cost

        # Forward pass (top-left -> bottom-right)
        for i in range(self.row):
            for j in range(self.col):
                if is_wall[i, j]:
                    continue
                d = dist[i, j]
                if i > 0:
                    d = min(d, dist[i - 1, j] + ORTH)
                if j > 0:
                    d = min(d, dist[i, j - 1] + ORTH)
                if i > 0 and j > 0:
                    d = min(d, dist[i - 1, j - 1] + DIAG)
                if i > 0 and j < self.col - 1:
                    d = min(d, dist[i - 1, j + 1] + DIAG)
                dist[i, j] = d

        # Backward pass (bottom-right -> top-left)
        for i in range(self.row - 1, -1, -1):
            for j in range(self.col - 1, -1, -1):
                if is_wall[i, j]:
                    continue
                d = dist[i, j]
                if i < self.row - 1:
                    d = min(d, dist[i + 1, j] + ORTH)
                if j < self.col - 1:
                    d = min(d, dist[i, j + 1] + ORTH)
                if i < self.row - 1 and j < self.col - 1:
                    d = min(d, dist[i + 1, j + 1] + DIAG)
                if i < self.row - 1 and j > 0:
                    d = min(d, dist[i + 1, j - 1] + DIAG)
                dist[i, j] = d

        self.wall_distance = dist
        return dist

def read_maze_from_csv(filename):
    """Read maze configuration from CSV file.
    Format: 1=wall, 0=free, 2=start, 4=end
    """
    maze = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Convert valid cells to integers, default to 1 (wall) for invalid cells
            filtered_row = [int(cell) if cell.strip().isdigit() else 1 for cell in row if cell.strip() != '']
            if filtered_row:
                maze.append(filtered_row)
    return maze

def upsample_maze(maze_array, factor=2):
    """
    Upsample maze by repeating each cell factor×factor times.
    Args:
        maze_array: Original maze array
        factor: Upsampling factor (2 = double density)
    """
    upsampled = []
    for row in maze_array:
        # Repeat each row 'factor' times
        for _ in range(factor):
            new_row = []
            for cell in row:
                # Repeat each cell 'factor' times
                new_row.extend([cell] * factor)
            upsampled.append(new_row)
    return upsampled


def read_maze_from_txt(filename):
    """Read maze configuration from TXT file.
    Supports two formats:
    1. Numeric: 1=wall, 0=free, 2=start, 4=end (space or comma separated)
    2. AAMC ASCII: IEEE micromouse format with o, |, -, S, G
    """
    with open(filename, 'r') as file:
        lines = [line.rstrip('\n') for line in file if line.strip()]
    if not lines:
        return []
    # Detect AAMC format: starts with 'o' and contains '---' or '|'
    first = lines[0].strip()
    if first.startswith('o') and ('---' in first or '|' in lines[1] if len(lines) > 1 else False):
        return _parse_aamc_maze(lines)
    return _parse_numeric_txt(lines)


def _parse_numeric_txt(lines):
    """Parse numeric TXT format (1=wall, 0=free, 2=start, 4=end)."""
    maze = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if ',' in line:
            cells = [c.strip() for c in line.split(',') if c.strip()]
        else:
            cells = line.split()
        row = [int(c) if str(c).strip().isdigit() and int(c) in (0, 1, 2, 4) else 1 for c in cells]
        if row:
            maze.append(row)
    return maze


def _parse_aamc_maze(lines):
    """Parse AAMC/IEEE micromouse ASCII format. Returns grid with 1=wall, 0=free, 2=start, 4=end.
    Produces a full occupancy map like maze.csv with explicit free space (0) in corridors
    and walls (1) at boundaries. Uses scale 8 per cell for ~129x129 resolution like maze.csv.
    """
    # AAMC: 16x16 cells, 33 lines. Even=horizontal walls, odd=vertical walls + cell content
    SCALE = 8  # ~8 cells per AAMC cell for resolution similar to maze.csv (129x129)
    size = 16
    g = size * SCALE + 1  # 129
    # Start with all free space (0), then add walls (1)
    maze = [[0] * g for _ in range(g)]
    # Parse horizontal walls (even lines)
    for i in range(0, min(len(lines), 33), 2):
        line = lines[i].strip()
        row = i // 2 * SCALE
        for c in range(size):
            seg = line[4*c+1:4*c+4] if 4*c+4 <= len(line) else '---'
            if '-' in seg:
                for j in range(c*SCALE, min(g, c*SCALE + SCALE + 1)):
                    if 0 <= row < g:
                        maze[row][j] = 1
    # Parse vertical walls and cell content (odd lines)
    start_pos, goal_pos = None, None
    for i in range(1, min(len(lines), 33), 2):
        line = lines[i].strip()
        row_idx = (i - 1) // 2
        for c in range(size + 1):
            if 4*c < len(line) and line[4*c] == '|':
                col = c * SCALE
                for r in range(row_idx*SCALE, min(g, row_idx*SCALE + SCALE + 1)):
                    if 0 <= col < g:
                        maze[r][col] = 1
        for c in range(size):
            content = line[4*c+1:4*c+4] if 4*c+4 <= len(line) else '   '
            # Center of cell - use middle of the SCALE x SCALE free-space block
            cr = row_idx * SCALE + SCALE // 2
            cc = c * SCALE + SCALE // 2
            if 'S' in content:
                start_pos = (cr, cc)
                maze[cr][cc] = 2
            elif 'G' in content:
                goal_pos = (cr, cc)
                maze[cr][cc] = 4
    # Borders
    for i in range(g):
        maze[0][i] = maze[g-1][i] = maze[i][0] = maze[i][g-1] = 1
    if start_pos is None:
        maze[SCALE//2][SCALE//2] = 2
    if goal_pos is None:
        maze[g//2][g//2] = 4
    return maze


def convert_txt_to_csv(txt_path, csv_path=None):
    """Convert a TXT maze file to CSV format.
    TXT format: 1=wall, 0=free, 2=start, 4=end.
    If csv_path is None, returns the maze array (same format as CSV would have).
    """
    maze = read_maze_from_txt(txt_path)
    if csv_path is None:
        return maze
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in maze:
            writer.writerow(row)
    return maze


def load_maze(filename):
    """Load maze from file. Auto-detects format from extension.
    Supports .csv and .txt (numeric format: 1=wall, 0=free, 2=start, 4=end).
    """
    if filename.lower().endswith('.txt'):
        return read_maze_from_txt(filename)
    return read_maze_from_csv(filename)

class Micromouse:
    """
    Micromouse robot simulation
    Combines path planning and path following with collision detection
    """
    def __init__(self, maps, start_pos, config=None):
        self.map = maps
        self.position = list(start_pos)
        
        # Load configuration with defaults
        self.config = config.get('micromouse', {}) if config else {}
        self.heading = self.config.get('initial_heading', math.pi/4)
        self.debug = self.config.get('debug', False)
        self.visualize_every = self.config.get('visualize_every', 5)
        self.dt = self.config.get('dt', 0.1)
        self.goal_threshold = self.config.get('goal_threshold', 0.5)
        self.min_speed_threshold = self.config.get('min_speed_threshold', 0.01)
        
        self.trajectory = [list(start_pos)]  # Record of all positions
        self.speed = 0
        self.steering = 0
        self.counter = 0
        self.pp = PurePursuit(maps, config)  # Pure Pursuit controller
        
        if self.debug:
            plt.ion()
            figsize = config.get('visualization', {}).get('figure_size', [12, 12])
            self.fig, self.ax = plt.subplots(figsize=figsize)
            cmap_colors = config.get('visualization', {}).get('cmap_colors', ['white', 'black', 'green', 'red', 'blue'])
            self.cmap = ListedColormap(cmap_colors)
            self.initialize_debug_plot()

    def initialize_debug_plot(self):
        """Set up real-time visualization for debugging"""
        grid = self.map.get_grid_representation()
        self.background = self.ax.imshow(grid, cmap=self.cmap)
        # Visualization elements for micromouse
        self.path_plot = self.ax.plot([], [], 'y-', linewidth=2, label='Planned Path')[0]
        self.traj_plot = self.ax.plot([], [], 'r-', linewidth=1, alpha=0.7, label='Trajectory')[0]
        self.pos_plot = self.ax.plot([], [], 'ro', markersize=8, label='Position')[0]
        self.heading_plot = self.ax.plot([], [], 'g-', linewidth=2, label='Heading')[0]
        self.ax.set_title("Micromouse Simulation")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_debug_plot(self, path=None):
        """Update the real-time visualization"""
        if not self.debug or self.counter % self.visualize_every != 0:
            self.counter += 1
            return
            
        grid = self.map.get_grid_representation()
        self.background.set_array(grid)
        
        if path is not None and len(path) > 0:
            path_arr = np.array(path)
            self.path_plot.set_data(path_arr[:,1], path_arr[:,0])
        
        traj = np.array(self.trajectory)
        if len(traj) > 0:
            self.traj_plot.set_data(traj[:,1], traj[:,0])
        
        self.pos_plot.set_data([self.position[1]], [self.position[0]])
        
        heading_length = 2
        self.heading_plot.set_data(
            [self.position[1], self.position[1] + heading_length * math.sin(self.heading)],
            [self.position[0], self.position[0] + heading_length * math.cos(self.heading)])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.counter += 1

    def update(self, dt, path=None):
        """Update robot position based on current speed and steering"""
        self.position[0] += self.speed * math.cos(self.heading) * dt
        self.position[1] += self.speed * math.sin(self.heading) * dt
        self.heading += self.speed * math.tan(self.steering) * dt
        self.heading = (self.heading + math.pi) % (2*math.pi) - math.pi  # Normalize angle
        self.trajectory.append(list(self.position))  # Record new position
        
        if self.debug:
            self.update_debug_plot(path)

    def check_collision(self, position):
        """
        Check if current position collides with a wall
        Returns True if collision detected, False otherwise
        """
        x, y = position
        # Check if position is out of bounds
        if not (0 <= x < self.map.row and 0 <= y < self.map.col):
            return True
        # Check if position is inside a wall
        return self.map.map[int(x)][int(y)].state == "#"

    def follow_path(self, path, dt=0.1):
        """
        Follow the given path using Pure Pursuit controller
        Returns the total distance traveled before stopping
        """
        self.pp.set_path(path)
        traversed_distance = 0
        
        while True:
            prev_position = list(self.position)
            speed, steering = self.pp.get_control(self.position, self.heading)
            self.speed = speed
            self.steering = steering
            self.update(dt, path)
            
            # Calculate distance moved in this step
            dx = self.position[0] - prev_position[0]
            dy = self.position[1] - prev_position[1]
            traversed_distance += math.sqrt(dx**2 + dy**2)
            
            # Check for collision with walls
            if self.check_collision(self.position):
                print(f"COLLISION DETECTED! Distance traveled: {traversed_distance:.2f} units")
                return traversed_distance
            
            if len(path) == 0:  # No path to follow
                break
                
            goal = path[-1]
            dist_to_goal = math.sqrt((self.position[0]-goal[0])**2 + 
                          (self.position[1]-goal[1])**2)
            
            if dist_to_goal < self.goal_threshold:  # Reached goal
                break
            
            if abs(self.speed) < self.min_speed_threshold:  # Stopped moving
                break
        
        return traversed_distance

def visualize(maze, path=None, mouse_traj=None, title="Micromouse Simulation"):
    """
    Create a static visualization of the maze, planned path, and actual trajectory
    """
    plt.figure(figsize=(12, 12))
    
    grid = maze.get_grid_representation()
    cmap = ListedColormap(['white', 'black', 'green', 'red', 'blue'])
    plt.imshow(grid, cmap=cmap)
    
    if path and len(path) > 0:  # Draw planned path
        path_arr = np.array(path)
        plt.plot(path_arr[:,1], path_arr[:,0], 'y-', linewidth=2, label='Planned Path')
    
    if mouse_traj:  # Draw actual trajectory
        traj = np.array(mouse_traj)
        plt.plot(traj[:,1], traj[:,0], 'r-', linewidth=1, label='Actual Path')
        plt.plot(traj[0,1], traj[0,0], 'go', markersize=10, label='Start')
        plt.plot(traj[-1,1], traj[-1,0], 'bx', markersize=10, label='End')
    
    plt.legend()
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def main(config_file='config.yaml'):
    """
    Main function to run the micromouse simulation
    """
    # Load configuration from YAML file
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    # Load and initialize maze first (supports .csv and .txt with 1=wall, 0=free, 2=start, 4=end)
    maze_file = config.get('maze_file', 'maze.csv')
    maze_array = load_maze(maze_file)
    
    # Optional maze upsampling (do this before metric conversion to get final grid size)
    upsampling_factor = 1
    if config.get('maze_upsampling', {}).get('enabled', False):
        upsampling_factor = config.get('maze_upsampling', {}).get('factor', 1)
        maze_array = upsample_maze(maze_array, upsampling_factor)
        print(f"Maze upsampled by factor {upsampling_factor}. New size: {len(maze_array)}x{len(maze_array[0])}")
    
    # =========================================================================
    # CONVERT M/S SPEEDS TO GRID UNITS (after upsampling, before auto-scaling)
    # =========================================================================
    phys_dims = config.get('physical_dimensions', {})
    use_metric_speeds = phys_dims.get('use_metric_speeds', False)
    
    if use_metric_speeds:
        maze_width_meters = phys_dims.get('maze_width_meters', 2.88)
        maze_height_meters = phys_dims.get('maze_height_meters', 2.88)
        
        # Calculate meters per grid cell
        num_cols = len(maze_array[0]) if len(maze_array) > 0 else 1
        num_rows = len(maze_array)
        meters_per_cell_x = maze_width_meters / num_cols
        meters_per_cell_y = maze_height_meters / num_rows
        meters_per_cell = (meters_per_cell_x + meters_per_cell_y) / 2.0
        
        print(f"\n{'='*70}")
        print(f"METRIC SPEED CONVERSION")
        print(f"{'='*70}")
        print(f"Maze dimensions: {num_rows}×{num_cols} grid cells")
        print(f"Physical size: {maze_height_meters}m × {maze_width_meters}m")
        print(f"Meters per cell: {meters_per_cell:.6f} m/cell")
        
        # Convert Pure Pursuit speeds from m/s to grid units/s
        if 'pure_pursuit' in config:
            pp = config['pure_pursuit']
            if 'max_speed' in pp:
                max_speed_ms = pp['max_speed']
                pp['max_speed'] = max_speed_ms / meters_per_cell
                print(f"\nPure Pursuit:")
                print(f"  max_speed: {max_speed_ms:.2f} m/s → {pp['max_speed']:.2f} grid units/s")
            
            if 'min_speed' in pp:
                min_speed_ms = pp['min_speed']
                pp['min_speed'] = min_speed_ms / meters_per_cell
                print(f"  min_speed: {min_speed_ms:.3f} m/s → {pp['min_speed']:.3f} grid units/s")
        
        # Convert Micromouse speed threshold from m/s to grid units/s
        if 'micromouse' in config:
            mm = config['micromouse']
            if 'min_speed_threshold' in mm:
                threshold_ms = mm['min_speed_threshold']
                mm['min_speed_threshold'] = threshold_ms / meters_per_cell
                print(f"\nMicromouse:")
                print(f"  min_speed_threshold: {threshold_ms:.3f} m/s → {mm['min_speed_threshold']:.3f} grid units/s")
        
        print(f"{'='*70}\n")
    
    # =========================================================================
    # AUTO-SCALE SPEEDS BASED ON UPSAMPLING FACTOR (deprecated - kept for compatibility)
    # =========================================================================
    if config.get('maze_upsampling', {}).get('enabled', False):
        auto_scale = config.get('maze_upsampling', {}).get('auto_scale_speeds', True)
        
        if upsampling_factor > 1 and auto_scale:
            print(f"\n{'='*70}")
            print(f"UPSAMPLING AUTO-SCALE: Factor = {upsampling_factor}×")
            print(f"{'='*70}")
            
            # Scale Pure Pursuit parameters
            if 'pure_pursuit' in config:
                pp = config['pure_pursuit']
                original_max_speed = pp.get('max_speed', 0.5)
                pp['max_speed'] *= upsampling_factor
                pp['min_speed'] *= upsampling_factor
                pp['lookahead_distance'] *= upsampling_factor
                
                print(f"Pure Pursuit:")
                print(f"  max_speed: {original_max_speed:.2f} → {pp['max_speed']:.2f} grid units/s")
                print(f"  min_speed: {pp['min_speed']:.2f} grid units/s")
                print(f"  lookahead: {pp['lookahead_distance']:.2f} grid units")
            
            # Scale Micromouse parameters
            if 'micromouse' in config:
                mm = config['micromouse']
                original_dt = mm.get('dt', 0.05)
                # Smaller timestep for stability with finer grid
                mm['dt'] /= upsampling_factor
                mm['goal_threshold'] *= upsampling_factor
                mm['min_speed_threshold'] *= upsampling_factor
                
                print(f"Micromouse:")
                print(f"  dt: {original_dt:.4f} → {mm['dt']:.4f}s (timestep)")
                print(f"  goal_threshold: {mm['goal_threshold']:.2f} grid units")
                print(f"  min_speed_threshold: {mm['min_speed_threshold']:.4f} grid units/s")
            
            # Scale Diff Drive wheel speeds
            if 'diff_drive' in config:
                dd = config['diff_drive']
                original_wheel_speed = dd.get('max_wheel_speed', 20.0)
                dd['max_wheel_speed'] *= upsampling_factor
                # Note: max_acceleration is NOT scaled (already in rad/s²)
                
                print(f"Differential Drive:")
                print(f"  max_wheel_speed: {original_wheel_speed:.1f} → {dd['max_wheel_speed']:.1f} rad/s")
                print(f"  (acceleration NOT scaled - already in rad/s²)")
            
            print(f"{'='*70}\n")
    
    # Create maze map from already loaded and upsampled maze_array
    maze = Map(maze_array)
    start_pos = maze.start
    goal_pos = maze.goal
    start_state = maze.map[start_pos[0]][start_pos[1]]
    goal_state = maze.map[goal_pos[0]][goal_pos[1]]
    
    # Plan path using A*
    print("Planning path with A*...")
    astar = AStar(maze, config)
    path_x, path_y = astar.plan_path(start_state, goal_state)
    path = list(zip(path_x, path_y)) if path_x and path_y else []
    
    # Calculate and display planned path distance
    planned_distance = astar.calculate_path_distance(path_x, path_y)
    print(f"Planned path distance: {planned_distance:.2f} grid units")
    if upsampling_factor > 1:
        # Convert to physical distance (AAMC cell = 18cm, base scale = 8 cells/AAMC cell)
        physical_distance = planned_distance * (0.18 / (8 * upsampling_factor))
        print(f"Physical distance: {physical_distance:.2f} meters")
    
    # Run micromouse simulation
    print("Running simulation with differential drive dynamics...")
    mouse = Micromouse(maze, start_pos, config)
    traversed_distance = mouse.follow_path(path)
    
    # Display results
    print(f"\nActual traversed distance: {traversed_distance:.2f} grid units")
    if upsampling_factor > 1:
        physical_traversed = traversed_distance * (0.18 / (8 * upsampling_factor))
        print(f"Physical distance: {physical_traversed:.2f} meters")
    print(f"Distance difference: {abs(traversed_distance - planned_distance):.2f} grid units")
    
    # Show final visualization if not in debug mode
    if not config.get('astar', {}).get('debug', False):
        visualize(maze, path, mouse.trajectory, 
                 title=config.get('visualization', {}).get('title', "Micromouse Simulation"))

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == 'convert':
        # Usage: python micromouse.py convert <input.txt> [output.csv]
        if len(sys.argv) < 3:
            print("Usage: python micromouse.py convert <input.txt> [output.csv]")
            sys.exit(1)
        txt_path = sys.argv[2]
        csv_path = sys.argv[3] if len(sys.argv) > 3 else txt_path.replace('.txt', '.csv')
        convert_txt_to_csv(txt_path, csv_path)
        print(f"Converted {txt_path} -> {csv_path}")
    elif len(sys.argv) >= 2 and sys.argv[1] == 'sim':
        # Launch interactive pygame simulation
        from simulation import main as sim_main
        sim_main()
    else:
        main()