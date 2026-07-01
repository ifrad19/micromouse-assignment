import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sys import maxsize

try:
    from theta_star import ThetaStarPlanner
except ImportError:
    ThetaStarPlanner = None
    print("Warning: theta_star.py not found. ThetaStar planner will not be available.")


class RRT:
    def __init__(self, map, max_iter=1000, step_size=0.5, goal_bias=0.1):
        # Your implementation here
        None # this is just added to avoid error, delete this

class ThetaStar:
    """
    Adapter wrapper for ThetaStarPlanner to work with Map class.
    
    Theta* is an any-angle path planning algorithm that produces smoother,
    shorter paths than A* by allowing direct line-of-sight connections between
    non-adjacent grid cells.
    
    See theta_star_integration.md for implementation details.
    """
    
    def __init__(self, maps, config=None):
        if ThetaStarPlanner is None:
            raise ImportError("ThetaStarPlanner not available. Ensure theta_star.py is present.")
        
        self.map = maps
        self.config = config.get('theta_star', {}) if config else {}
        
        # Configuration
        self.resolution = self.config.get('resolution', 1.0)
        self.robot_radius = self.config.get('robot_radius', 0.3)
        self.debug = self.config.get('debug', False)
        
        # Convert Map to obstacle lists
        self.ox, self.oy = self._map_to_obstacles()
        
        # Create underlying Theta* planner
        self.planner = ThetaStarPlanner(self.ox, self.oy, self.resolution, self.robot_radius)
        
        if self.debug:
            print(f"Theta* initialized: {len(self.ox)} obstacle points")
    
    def _map_to_obstacles(self):
        """
        Convert Map grid to obstacle point lists.
        
        The theta_star.py implementation expects lists of obstacle coordinates,
        while our Map class uses a 2D grid of State objects.
        """
        ox, oy = [], []
        
        for i in range(self.map.row):
            for j in range(self.map.col):
                if self.map.map[i][j].state == "#":
                    # Note: Coordinate system conversion
                    # Map uses (row, col) indexing where row=y, col=x
                    # Theta* expects (x, y) coordinates
                    ox.append(float(j))  # column -> x
                    oy.append(float(i))  # row -> y
        
        return ox, oy
    
    def plan(self):
        """
        Run Theta* planning and return path.
        
        Returns:
            List of (row, col) tuples representing the path from start to goal.
            Returns empty list if no path found.
        """
        # Get start and goal from map (row, col format)
        start_row, start_col = self.map.start
        goal_row, goal_col = self.map.goal
        
        # Convert to (x, y) for theta_star
        sx = float(start_col)
        sy = float(start_row)
        gx = float(goal_col)
        gy = float(goal_row)
        
        if self.debug:
            print(f"Theta* planning from ({sx}, {sy}) to ({gx}, {gy})")
        
        # Run Theta* planning
        # Returns (rx, ry) where rx[0] is goal x, rx[-1] is start x
        rx, ry = self.planner.planning(sx, sy, gx, gy)
        
        if not rx or not ry:
            print("Theta*: No path found!")
            return []
        
        # Convert back to (row, col) format and reverse order
        # theta_star returns path from goal to start, we want start to goal
        path = []
        for i in range(len(rx) - 1, -1, -1):  # Reverse iteration
            row = int(round(ry[i]))  # y -> row
            col = int(round(rx[i]))  # x -> col
            path.append((row, col))
        
        if self.debug:
            print(f"Theta* found path with {len(path)} waypoints")
        
        return path
    
    def get_path_length(self, path):
        """Calculate total Euclidean path length"""
        if len(path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][1] - path[i][1]  # col difference
            dy = path[i+1][0] - path[i][0]  # row difference
            length += math.sqrt(dx*dx + dy*dy)
        
        return length

class AStar:
    """
    A* Pathfinding Algorithm - Traditional Micromouse Approach
    
    Cost function: g(n) = distance_cost + wall_cost + turn_cost
    
    - distance_cost: Euclidean distance between cells
    - wall_cost: Penalty for proximity to walls (keeps robot safer)
    - turn_cost: Discrete penalty per direction change
        * 0 cost: continuing straight
        * 1× weight: 90-degree turn (e.g., North to East)
        * 2× weight: 180-degree turn (requires two 90° turns)
    
    This matches real micromouse competition algorithms where turns
    are counted as discrete events at cell boundaries.
    """
    def __init__(self, maps, config=None):
        self.map = maps
        self.open_list = set()  # States to be evaluated
        self.closed_list = set()  # Evaluated states
        
        # Load configuration with defaults
        self.config = config.get('astar', {}) if config else {}
        self.debug = self.config.get('debug', False)
        self.visualize_every = self.config.get('visualize_every', 10)
        self.heuristic_weight = self.config.get('heuristic_weight', 1.0)
        self.counter = 0

        # Wall proximity cost configuration
        wc = config.get('wall_cost', {}) if config else {}
        self.wall_cost_enabled   = wc.get('enabled', False)
        self.wc_weight           = wc.get('weight', 2.0)
        self.wc_decay            = wc.get('decay', 'exponential')
        self.wc_decay_rate       = wc.get('decay_rate', 0.5)
        self.wc_threshold        = wc.get('threshold', 5.0)
        if self.wall_cost_enabled:
            self.map.compute_wall_distance_map()
        
        # Metric scaling for turn and distance calculations (optional)
        # If physical maze dimensions are provided, use them to derive
        # meters-per-cell in each axis so turn angles are computed in
        # physical space rather than raw grid indices.
        phys_dims = config.get('physical_dimensions', {}) if config else {}
        maze_width_m  = phys_dims.get('maze_width_meters', None)
        maze_height_m = phys_dims.get('maze_height_meters', None)
        if maze_width_m is not None and maze_height_m is not None:
            num_cols = max(self.map.col, 1)
            num_rows = max(self.map.row, 1)
            # Columns span the physical maze width (x-axis), rows the height (y-axis)
            self.m_per_cell_x = maze_width_m / num_cols
            self.m_per_cell_y = maze_height_m / num_rows
        else:
            # Fallback to isotropic grid units if no physical dimensions given
            self.m_per_cell_x = 1.0
            self.m_per_cell_y = 1.0
        
        # Turn cost configuration
        self.turn_cost_enabled = self.config.get('turn_cost_enabled', False)
        self.turn_cost_weight = self.config.get('turn_cost_weight', 2.0)
        self.turn_cost_threshold = self.config.get('turn_cost_threshold', 45)  # degrees
        
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
        
        # Plot elements for visualization
        self.open_set_plot = self.ax.plot([], [], 'go', markersize=3, alpha=0.5, label='Open Set')[0]
        self.closed_set_plot = self.ax.plot([], [], 'ro', markersize=2, alpha=0.3, label='Closed Set')[0]
        self.path_plot = self.ax.plot([], [], 'b-', linewidth=2, label='Current Best')[0]
        
        self.ax.set_title("A* Pathfinding")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def heuristic(self, state, goal):
        """Manhattan distance heuristic"""
        return self.heuristic_weight * (abs(state.x - goal.x) + abs(state.y - goal.y))

    def wall_proximity_cost(self, state):
        """
        Calculate cost penalty for being near walls.
        Returns higher cost when closer to walls.
        """
        if not self.wall_cost_enabled:
            return 0.0
        
        d = self.map.wall_distance[state.x, state.y]
        if d >= self.wc_threshold:
            return 0.0
        
        if self.wc_decay == 'exponential':
            return self.wc_weight * math.exp(-self.wc_decay_rate * d)
        elif self.wc_decay == 'inverse':
            return self.wc_weight / (d + 0.1)
        elif self.wc_decay == 'linear':
            return self.wc_weight * (1.0 - d / self.wc_threshold)
        return 0.0
    
    def calculate_turn_cost(self, from_state, current_state, next_state):
        """
        Calculate turn cost based on direction change at cell level.
        Traditional micromouse approach: count discrete direction changes.
        
        A "turn" occurs when moving direction changes between cells:
        - North to East = 1 turn
        - North to South = 2 turns (180° requires two 90° turns)
        - North to North = 0 turns (straight)
        
        Args:
            from_state: Previous state (parent of current) - can be None
            current_state: Current state
            next_state: Proposed next state
            
        Returns:
            Turn cost based on number of 90-degree turns required
        """
        if not self.turn_cost_enabled or from_state is None:
            return 0.0
        
        # Calculate incoming direction (from -> current)
        dx1 = current_state.x - from_state.x
        dy1 = current_state.y - from_state.y
        
        # Calculate outgoing direction (current -> next)
        dx2 = next_state.x - current_state.x
        dy2 = next_state.y - current_state.y
        
        # Check if continuing in same direction (no turn)
        if dx1 == dx2 and dy1 == dy2:
            return 0.0
        
        # Check if reversing direction (180° turn = 2 turns)
        if dx1 == -dx2 and dy1 == -dy2:
            return self.turn_cost_weight * 2.0
        
        # Otherwise it's a 90° turn (perpendicular)
        # Verify it's actually perpendicular (dot product should be 0)
        dot = dx1 * dx2 + dy1 * dy2
        if abs(dot) < 0.01:  # Perpendicular (90-degree turn)
            return self.turn_cost_weight
        
        # Should never reach here with 4-directional movement
        return 0.0

    def update_debug_plot(self, current_state=None):
        """Update the real-time visualization"""
        if not self.debug or self.counter % self.visualize_every != 0:
            self.counter += 1
            return
        
        # Update visualization elements
        open_x = [s.x for s in self.open_list]
        open_y = [s.y for s in self.open_list]
        self.open_set_plot.set_data(open_y, open_x)

        closed_x = [s.x for s in self.closed_list]
        closed_y = [s.y for s in self.closed_list]
        self.closed_set_plot.set_data(closed_y, closed_x)

        if current_state:  # Draw current best path
            path_x, path_y = [], []
            temp_state = current_state
            while temp_state:
                path_x.append(temp_state.x)
                path_y.append(temp_state.y)
                temp_state = temp_state.parent
            self.path_plot.set_data(path_y, path_x)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.counter += 1

    def plan_path(self, start, goal):
        """Find optimal path from start to goal using A* algorithm with turn cost"""
        # Initialize start node costs
        self.open_list = set()
        self.closed_list = set()

        start.g = 0.0
        start.h = self.heuristic(start, goal)
        start.k = start.g + start.h
        start.parent = None
        self.open_list.add(start)

        while self.open_list:
            current = min(self.open_list, key=lambda x: x.k)  # Get state with lowest cost
            
            if current == goal:  # Path found
                return self.reconstruct_path(current)

            self.open_list.remove(current)
            self.closed_list.add(current)
            self.update_debug_plot(current)

            for neighbor in self.map.get_neighbors(current):
                if neighbor in self.closed_list or neighbor.state == "#":
                    continue  # Skip walls and already evaluated states

                # Calculate cost components for traditional micromouse pathfinding:
                base_cost = current.cost(neighbor)  # Distance (typically 1.0 for adjacent cells)
                wall_cost = self.wall_proximity_cost(neighbor)  # Safety margin from walls
                turn_cost = self.calculate_turn_cost(current.parent, current, neighbor)  # Discrete turn count
                
                # Total path cost: distance + safety + turn_penalty
                # Example: straight move = 1.0, 90° turn = 1.0 + turn_weight
                tentative_g = current.g + base_cost + wall_cost + turn_cost
                
                if neighbor not in self.open_list:
                    self.open_list.add(neighbor)
                elif tentative_g >= neighbor.g:
                    # Existing path to neighbor is better or equal
                    continue

                # Update neighbor with better path
                neighbor.parent = current
                neighbor.g = tentative_g
                neighbor.h = self.heuristic(neighbor, goal)
                neighbor.k = neighbor.g + neighbor.h

        return [], []  # Return empty path if none found

    def reconstruct_path(self, current):
        """Reconstruct path from goal to start by following parent pointers"""
        path_x, path_y = [], []
        total_turns = 0
        
        # Collect path
        while current:
            path_x.insert(0, current.x)
            path_y.insert(0, current.y)
            current = current.parent
        
        # Count turns in the final path
        if self.turn_cost_enabled and len(path_x) >= 3:
            for i in range(1, len(path_x) - 1):
                dx1 = path_x[i] - path_x[i-1]
                dy1 = path_y[i] - path_y[i-1]
                dx2 = path_x[i+1] - path_x[i]
                dy2 = path_y[i+1] - path_y[i]
                
                # Count direction changes
                if not (dx1 == dx2 and dy1 == dy2):
                    if dx1 == -dx2 and dy1 == -dy2:
                        total_turns += 2  # 180-degree turn
                    else:
                        total_turns += 1  # 90-degree turn
            
            print(f"\n=== Path Planning Complete ===")
            print(f"Path length: {len(path_x)} cells")
            print(f"Total turns: {total_turns}")
            print(f"Turn cost weight: {self.turn_cost_weight}")
            print(f"Turn cost contribution: {total_turns * self.turn_cost_weight}")
            print(f"==============================\n")
        
        return path_x, path_y

    def calculate_path_distance(self, path_x, path_y):
        """Calculate total Euclidean distance of the path"""
        if not path_x or len(path_x) < 2:
            return 0
        total_distance = 0
        for i in range(len(path_x) - 1):
            dx = path_x[i+1] - path_x[i]
            dy = path_y[i+1] - path_y[i]
            total_distance += math.sqrt(dx**2 + dy**2)
        return total_distance
    
    def calculate_path_turns(self, path_x, path_y):
        """
        Calculate total turn angle and number of turns in the path.
        Useful for evaluating path smoothness.
        
        Returns:
            (total_turn_angle, num_significant_turns)
        """
        if not path_x or len(path_x) < 3:
            return 0.0, 0
        
        total_angle = 0.0
        num_turns = 0
        
        for i in range(1, len(path_x) - 1):
            # Vectors before and after this waypoint
            # path_x stores row indices (y-axis in meters), path_y stores
            # column indices (x-axis in meters). Convert to metric space.
            v1_x = (path_y[i]   - path_y[i-1]) * self.m_per_cell_x
            v1_y = (path_x[i]   - path_x[i-1]) * self.m_per_cell_y
            v2_x = (path_y[i+1] - path_y[i])   * self.m_per_cell_x
            v2_y = (path_x[i+1] - path_x[i])   * self.m_per_cell_y
            
            # Calculate angle
            if (v1_x == 0 and v1_y == 0) or (v2_x == 0 and v2_y == 0):
                continue
            
            dot_product = v1_x * v2_x + v1_y * v2_y
            mag1 = math.sqrt(v1_x**2 + v1_y**2)
            mag2 = math.sqrt(v2_x**2 + v2_y**2)
            
            cos_angle = dot_product / (mag1 * mag2)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            
            angle_rad = math.acos(cos_angle)
            angle_deg = math.degrees(angle_rad)
            
            total_angle += angle_deg
            
            # Count turns above threshold
            if angle_deg > self.turn_cost_threshold:
                num_turns += 1
        
        return total_angle, num_turns