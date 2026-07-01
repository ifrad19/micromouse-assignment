# Micromouse Simulation Assignment: Path Planning and Tracking Enhancements

## Objective
Enhance the existing **micromouse simulation** by implementing advanced path planning and tracking algorithms while optimizing the path for smooth, safe navigation.

## Baseline Code
You will start with the provided codebase that includes:
- A* path planning (`pathPlanning.py`)
- Pure Pursuit path tracking (`pathTracking.py`)
- Main simulation framework (`Simulation.py`)

## Map
For this assignment, you will use the alljapan-045-2024-exp-fin.txt map file (already loaded in the config.yaml file)

## Assignment Requirements

### 1. Implement Advanced Path Planning (20 points)

**Choose and implement TWO of the following three algorithms:**

#### Option A: Theta* (Any-Angle Planning)
- **Add Theta* implementation** with adapter for existing Map structure
- **Key feature**: Produces smoother, shorter paths than A* by allowing line-of-sight connections
- **Advantage**: 5-20% shorter paths with fewer turns, less aggressive smoothing needed
- **Implementation**: Adapt provided `theta_star.py` to work with Map class

#### Option B: RRT (Sampling-Based Planning)  
- **Add RRT implementation** in `pathPlanning.py` (3 sample implementations provided)
- **Key feature**: Randomly samples the space, good for complex/high-dimensional problems
- **Advantage**: Probabilistically complete, doesn't require explicit grid representation
- **Implementation**: Adapt provided sample code (basic RRT, Sobol sampler variant, or smoothed variant) with Map class adapter
- **Reference**: See `rrt_integration.md` for adapter pattern and integration guide

#### Option C: A* with Enhancements
- **Baseline A* is provided**, enhance with advanced features:
  - Wall cost penalties (already in baseline, tune for better performance)
  - Turn cost optimization (already in baseline, demonstrate effectiveness)
  - Jump Point Search optimization
  - Bidirectional search

**Requirements:**
- Implement TWO algorithms (can include enhanced A* as one choice)
- Compare path length, computation time, and path smoothness
- Visualize both planning methods showing search progress
- Discuss trade-offs in your report

### 2. Implement Stanley Controller Path Tracking (20 points)
- **Add Stanley controller** in `pathTracking.py` file
- **Modify micromouse.py** to support both Pure Pursuit and Stanley
- **Compare performance** between tracking methods
- **Demonstrate stability** with different speed profiles and path curvatures

### 3. Path Optimization with Clothoids (20 points)
- **Clothoid curve smoothing**: Replace sharp corners with Euler spirals for constant curvature rate
- **Safety margins**: Ensure smoothed path stays at least 0.3 units from walls
- **Curvature-based velocity**: Adjust speed based on local path curvature
- **Visual comparison**: Show before/after path with curvature profiles

### 4. Analysis and Report (20 points)
- **Planner comparison**: Compare your two chosen path planning algorithms
  - Path length, computation time, number of waypoints
  - Path smoothness (total curvature or number of turns)
  - Success rate across different maze configurations
- **Controller comparison**: Analyze Pure Pursuit vs Stanley on both raw and optimized paths
- **Path optimization impact**: Show curvature profiles before/after clothoid smoothing
- **Integrated performance**: Show how planner choice affects controller tracking and overall speed
- **Parameter sensitivity**: Explore effects of algorithm parameters on performance
- **Failure analysis**: Document any collision cases and how they were resolved

### 5. Speed Competition (20 points)

**Compete against your classmates for the fastest micromouse!**

All submissions will be ranked based on **completion time**:

```
Score = Completion Time (seconds)
```

**Lower time is better** - reach the goal as quickly as possible!

#### Competition Rules:

1. **Standard Maze**: All submissions tested on `alljapan-045-2024-exp-fin.txt` (same maze as assignment)
2. **Qualification Requirements**:
   - Must reach goal successfully (no timeouts)
   - Zero collisions with walls
   - Uses your implemented algorithms (planner + Stanley + optimization)
3. **Timing**: Measured from simulation start until goal is reached
4. **Reproducibility**: Must run successfully with your submitted code and config.yaml

#### Scoring Distribution (Tier-Based):

| Tier | Percentile | Points | Description |
|------|------------|--------|-------------|
| **Elite** | Top 20% | 20 | Exceptional optimization |
| **Advanced** | Next 30% | 16 | Strong performance |
| **Proficient** | Middle 30% | 12 | Solid implementation |
| **Qualified** | Bottom 20% | 8 | Meets requirements |
| **Disqualified** | - | 0 | Failed requirements (collision/timeout/error) |

**Example**: If 25 students submit, rankings would be:
- Ranks 1-5: 20 points each
- Ranks 6-12: 16 points each
- Ranks 13-20: 12 points each
- Ranks 21-25: 8 points each

#### Tips for Competitive Performance:

1. **Use Benchmark Mode for Fast Iteration**:
   - Run `python simulation.py --benchmark` for instant feedback
   - No visualization overhead = faster testing cycles
   - Try multiple parameter combinations quickly

2. **Path Planning**:
   - Theta* typically produces shorter, faster paths than A*
   - RRT benefits from path smoothing variant
   - Tune goal bias and exploration parameters

3. **Path Optimization**:
   - Clothoid smoothing enables higher speeds through corners
   - Balance safety margins with aggressive trajectories
   - Smoother paths = higher sustained speeds

4. **Tracking Controller**:
   - Stanley typically allows higher speeds than Pure Pursuit
   - Tune gains for maximum speed while maintaining stability
   - Prioritize speed over smoothness

5. **Parameter Tuning**:
   - Increase max_speed aggressively while avoiding collisions
   - Reduce lookahead distance or increase Stanley k for faster response
   - Push the limits - balance speed vs safety

6. **Strategy**:
   - Focus on minimizing completion time above all else
   - Shorter paths complete faster (fewer meters to travel)
   - Higher speeds complete faster (but risk collisions)
   - Test many parameter combinations using benchmark mode

**Note**: Competition points are awarded based on relative performance within the class. Even if your implementation doesn't win, meeting the qualification requirements guarantees at least 8/20 points.

## Implementation Guide
## Configuration yaml file
In the baseline implementation provided, the configuration parameters are provided in the `config.yaml` file. 
- Explore the effects of changing these parameters on the micromouse navigation performance
- It is recommended to use the config.yaml file to store the parameters associated with your chosen path planning algorithms (Theta*, RRT, or A* enhancements), Stanley controller, and path optimization parameters

### Path Planning Implementation Guidelines

**For Theta* Implementation:**
1. Create adapter class to convert between Map format and obstacle lists:
   ```python
   class ThetaStar:
       def __init__(self, maps, config=None):
           # Convert Map grid to obstacle points for theta_star.py
           # See theta_star_integration.md for details
   ```
2. Handle coordinate system differences (row/col vs x/y)
3. Convert output path back to waypoint list
4. Refer to `theta_star_integration.md` for complete implementation guide

**For RRT Implementation:**
1. Choose one of the 3 provided sample implementations:
   - `rrt.py`: Basic RRT with uniform random sampling
   - `rrt_with_sobol_sampler.py`: RRT with Sobol quasi-random sampling (faster convergence)
   - `rrt_with_pathsmoothing.py`: RRT with post-processing path smoothing
2. Create adapter class similar to ThetaStar to convert Map format to RRT's obstacle list format
3. Handle coordinate conversions and parameter configuration
4. Integrate with main simulation loop
5. Refer to `rrt_integration.md` for complete implementation guide with adapter example

### Stanley Controller Implementation Guidelines
Assume the micromouse is a 2-wheeled differential drive mouse robot, as per example shown below:
![Micromouse Maze Visualization](images//Micromouse_Green_Giant_V1.3.jpg) 
*2-wheeled mouse*

**Note**: Stanley controller was originally designed for Ackermann steering (car-like) vehicles but can be adapted for differential drive robots by converting the steering angle output to left/right wheel velocities. See `stanley.md` Section 3 for the conversion formulas.

1. Design for these parameters:
   ```python
   class Stanley:
       def __init__(self, k=0.5, k_soft=1.0, max_steer=np.pi/3, wheel_base=0.08):
           # k: Cross-track error gain
           # k_soft: Softening constant for low speeds
           # max_steer: Maximum steering angle
           # wheel_base: Distance between wheels (for diff drive conversion)
   ```
2. Implement both heading error and cross-track error correction
3. Add differential drive conversion methods (steering angle → wheel velocities)
4. Consider velocity-dependent tuning for robust performance
5. Refer to `stanley.md` for detailed implementation guidance

### Path Optimization with Clothoids

**Clothoid curves (Euler spirals)** provide optimal smoothness for path tracking because their curvature increases linearly with arc length, matching the natural motion of differential drive robots.

1. Implement the path optimization function:
   ```python
   def optimize_path(path, map, min_distance=0.3):
       """
       Smooth path using clothoid curves at corners.
       
       Args:
           path: List of waypoints from A* or RRT
           map: Maze map for collision checking
           min_distance: Minimum clearance from walls
           
- https://github.com/AtsushiSakai/PythonRobotics/tree/master contains reference implementations of path planning and path tracking
- Theta* original paper: "Theta*: Any-Angle Path Planning on Grids" (Nash et al., AAAI 2007)
- RRT paper: "Rapidly-Exploring Random Trees: A New Tool for Path Planning" (LaValle, 1998)
           Smoothed path as list of closely-spaced points
       """
       # Your clothoid-based smoothing implementation
   ```

2. Implementation approaches:
   - **Recommended**: Use `pyclothoids` library for G² continuous curves
   - **Alternative**: Implement clothoid lookup tables with Fresnel integrals
   - **Hybrid**: Use clothoids for gentle turns, circular arcs for tight 90° corners

3. Key considerations:
   - Detect corner points requiring smoothing (large heading changes)
   - Fit clothoid segments that connect incoming/outgoing path directions
   - Verify all points maintain `min_distance` clearance
   - Resample at uniform intervals for controller compatibility

4. Refer to `clothoids.md` for detailed implementation guidance and examples

### Useful resources
- Refer to class notes for overview of path planning and path tracking. 
- https://github.com/AtsushiSakai/PythonRobotics/tree/master contains reference implementations of path planning and path tracking

## Testing Your Competition Performance

To test your implementation and view competition metrics:

### Quick Benchmark Mode (Recommended for Iteration)

For rapid testing without visualization:

```bash
python simulation.py --benchmark
```

This headless mode will:
- Run the simulation at maximum speed (no frame rate limiting)
- Skip all pygame visualization
- Print competition metrics to console
- Allow you to iterate quickly on parameter tuning

Output example:
```
============================================================
BENCHMARK MODE - Planning...
============================================================
[OK] Path found: 234 waypoints

Running simulation...
  Step 1000: time=2.5s, pos=(8.3, 12.1), traversed=15.2
  Step 2000: time=5.0s, pos=(14.7, 9.8), traversed=28.9

============================================================
COMPETITION RESULTS
============================================================
  Completion Time:    12.345 seconds [COMPETITION SCORE]
  Path Length:        89.234 units (reference)
  Planned Distance:   87.123 units
  Path Efficiency:    97.6%
  Collisions:         0
  Status:             [QUALIFIED]
============================================================
Lower time is better - be the fastest!
============================================================
```

### Interactive Mode (For Visualization)

To watch the simulation and debug:

```bash
python simulation.py
```

1. **Ensure correct maze is configured** in `config.yaml`:
   ```yaml
   maze_file: "mazefiles/classic/alljapan-045-2024-exp-fin.txt"
   ```

2. **Run the simulation** and press SPACE to start

3. **View competition metrics** - When the robot reaches the goal, metrics are printed to console

4. **Take a screenshot** of the console output for your submission

### Optimization Workflow

1. **Baseline test**: Run `--benchmark` with default parameters
2. **Adjust one parameter** (e.g., increase max_speed)
3. **Re-run benchmark** and compare completion time
4. **Iterate**: Keep changes that improve time, revert those that don't
5. **Validate**: Run interactive mode to ensure no collisions
6. **Document**: Note which parameters gave best results for your report

**Tips for best results:**
- Enable Stanley controller (typically faster than Pure Pursuit)
- Use Theta* or smoothed RRT for shorter paths
- Aggressive clothoid smoothing (lower min_clearance = shorter paths)
- High max_speed with well-tuned controller gains
- Test, measure, adjust, repeat!

## Deliverables

1. **Code Submission**:
   - All Python files with your implementations
   - Updated `requirements.txt` with any additional dependencies
   - Your tuned `config.yaml` file (competition configuration)
   
2. **Competition Results**:
   - **Console output** showing:
     - Completion time (seconds) - this is your competition score
     - Path length (units) - for reference
     - Collision status (must be zero)
   - **Screenshot or video** of successful maze completion
   - Must use `alljapan-045-2024-exp-fin.txt` maze
   
3. **Report** (PDF, 5-10 pages):
   - **Results**: Comparative analysis with metrics (path length, curvature, tracking error, computation time)
   - **Visualizations**: 
     - Path comparisons between your two planning algorithms (raw vs optimized)
     - Curvature profiles showing smoothing effectiveness
     - Controller tracking performance plots (Pure Pursuit vs Stanley)
     - Algorithm-specific visualizations (e.g., RRT tree growth, Theta* line-of-sight, A* search)
   - **Discussion**: 
     - Why you chose your two planning algorithms
     - Algorithm trade-offs observed in practice
     - Parameter tuning insights and sensitivity analysis
     - Failure cases and solutions
     - Competition strategy and optimization approach
   - **Conclusion**: Recommended algorithm/controller combinations for different scenarios

## Grading Rubric

| **Component** | **Points** | **Criteria** |
|---------------|------------|--------------|
| **Path Planning (2 algorithms)** | **20** | |
| - First algorithm            | 9      | Correct implementation with proper search/sampling logic |
| - Second algorithm           | 9      | Correct implementation integrated with framework |
| - Comparative analysis       | 2      | Quantitative comparison of both algorithms |
| **Stanley Controller**      | **20** | |
| - Correct implementation     | 9      | Heading error + cross-track error with proper formulation |
| - Path tracking performance  | 6      | Successfully follows paths with reasonable error |
| - Visualization & debugging  | 3      | Shows vehicle state, tracking points, debug output |
| - Parameter tuning           | 2      | Demonstrates effect of k, k_soft parameters |
| **Path Optimization**       | **20** | |
| - Clothoid implementation    | 10     | Smooths corners using clothoid curves or fallback strategy |
| - Safety verification        | 5      | Maintains min_clearance from walls |
| - Performance improvement    | 3      | Measurable reduction in curvature, improved tracking |
| - Visualization              | 2      | Clear before/after comparison with curvature plots |
| **Analysis and Report**     | **20** | |
| - Methodology clarity        | 5      | Clear explanation of all implementations |
| - Quantitative results       | 6      | Tables/plots comparing algorithms with metrics |
| - Critical analysis          | 5      | Insights on parameter effects, failure cases, tradeoffs |
| - Report quality             | 4      | Well-organized, professional, proper citations |
| **Speed Competition**       | **20** | |
| - Elite (Top 20%)           | 20     | Fastest completion times in class |
| - Advanced (Next 30%)       | 16     | Strong competitive performance |
| - Proficient (Middle 30%)   | 12     | Solid implementation and tuning |
| - Qualified (Bottom 20%)    | 8      | Meets all requirements, completes successfully |
| - Disqualified              | 0      | Collision, timeout, or execution error |
| **Total**                   | **100**| |

**Note on Competition**: Choose TWO from: Theta*, RRT, or Enhanced A* (see Section 1 for details)

**Percentage of module grade: 20%**

## Deadline
Submit your work by **Midnight, 6th April, 2025** via Canvas. 



