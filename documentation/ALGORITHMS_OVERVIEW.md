# Micromouse Algorithms Overview

This document provides a comprehensive overview of all path planning and path tracking algorithms implemented in the micromouse navigation system.

## Table of Contents

- [Path Planning Algorithms](#path-planning-algorithms)
  - [A* Algorithm](#a-pathfinding-algorithm)
  - [Theta* Algorithm](#theta-pathfinding-algorithm)
- [Path Tracking Controllers](#path-tracking-controllers)
  - [Pure Pursuit Controller](#pure-pursuit-controller)
  - [Stanley Controller](#stanley-controller)
- [Path Optimization](#path-optimization)
  - [Clothoid Curves](#clothoid-curves-for-path-optimization)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)

---

# Path Planning Algorithms

## A* Pathfinding Algorithm

### What is A*?

A* (pronounced "A-star") is an informed search algorithm that finds the shortest path between two points in a graph or grid. It's widely used in robotics, video games, and navigation systems because it's both optimal and efficient.

### How A* Works

A* combines two key pieces of information:

- **g(n)**: The actual cost from the start node to the current node
- **h(n)**: The estimated cost (heuristic) from the current node to the goal
- **f(n) = g(n) + h(n)**: The total estimated cost of the path through this node

The algorithm maintains two lists:

1. **Open List**: Nodes to be evaluated
2. **Closed List**: Nodes already evaluated

### Algorithm Steps

1. Add the start node to the open list
2. While the open list is not empty:
   - Select the node with the lowest f(n) from the open list
   - If this node is the goal, reconstruct and return the path
   - Move the current node to the closed list
   - For each neighbor of the current node:
     - Skip if it's a wall or in the closed list
     - Calculate tentative g score
     - If the neighbor is not in the open list or this path is better:
       - Update the neighbor's parent, g, and f values
       - Add to open list if not already present

### Key Features

#### Heuristic Function

The algorithm uses **Manhattan distance** (taxicab geometry) as the heuristic, which is ideal for grid-based movement where diagonal moves aren't allowed or are more expensive. The heuristic is admissible (never overestimates) and consistent, ensuring optimal paths.

**Formula**: h(a,b) = |a.x - b.x| + |a.y - b.y|

#### Wall Proximity Cost

An advanced feature that adds extra cost to paths near walls, encouraging the robot to stay away from walls. This is useful for:
- Avoiding sensor noise near walls
- Reducing collision risk
- Creating smoother paths for the controller

**Decay modes**:
- **Exponential**: Cost decreases exponentially with distance
- **Inverse**: Cost is inversely proportional to distance
- **Linear**: Cost decreases linearly to zero at threshold

### Complexity

- **Time Complexity**: O(b^d) in worst case, but typically much better with a good heuristic
- **Space Complexity**: O(b^d) where b is the branching factor and d is the depth

### Advantages and Limitations

**Advantages**:
- Guarantees optimal path (shortest)
- Efficient with good heuristic
- Complete (finds a solution if one exists)

**Limitations**:
- Memory intensive for large maps
- Recomputation needed if obstacles change
- Path may have sharp angles requiring smoothing

---

## Theta* Pathfinding Algorithm

### What is Theta*?

Theta* is an **any-angle path planning algorithm** that extends A* to produce shorter, more realistic paths by allowing paths to take any angle, not just grid-constrained moves. Unlike A* which is restricted to moving along grid edges, Theta* can create paths that cut across grid cells when there is line of sight.

### Why Theta* Over A*?

**A* Limitation**: Standard A* produces paths that follow grid edges, resulting in:
- Unnecessarily long paths with many waypoints
- Unnatural "staircase" patterns on diagonal routes
- Requires post-processing smoothing

**Theta* Solution**: 
- Produces paths with fewer waypoints
- Allows straight-line segments across multiple grid cells
- Creates more natural, direct routes
- Generally 5-15% shorter paths than A*

### How Theta* Works

Theta* extends A* with a key modification called **path compression** using line-of-sight checks:

1. **Standard A* Expansion**: Expand neighbors normally from current node
2. **Line of Sight Check**: For each neighbor, check if the grandparent (parent's parent) has direct line of sight
3. **Path Compression**: If line of sight exists, connect neighbor directly to grandparent, bypassing the intermediate parent

### Key Innovation: Line of Sight

The algorithm uses **Bresenham's line algorithm** to efficiently check if two grid cells have unobstructed line of sight. This algorithm traverses the grid cells between two points and verifies that none contain obstacles.

### Theta* Modification to A*

The critical difference occurs during neighbor expansion:

**Standard A***: Connect neighbor to current node using the cost from start to current plus the edge cost.

**Theta* Addition**: If the grandparent (parent's parent) has line of sight to the neighbor, skip the current node and connect the neighbor directly to the grandparent. This "path compression" creates straighter paths.

### Algorithm Steps

1. Initialize open set with start node
2. While open set is not empty:
   - Select node with minimum f(n) = g(n) + h(n)
   - If goal reached, return path
   - Move current to closed set
   - For each neighbor:
     - If not valid or in closed set, skip
     - **Theta* Modification**: Check if grandparent has line of sight to neighbor
       - If yes: Connect neighbor directly to grandparent
       - If no: Connect neighbor to current node (standard A*)
     - Update open set with better path

### Visual Comparison

```
A* Path (grid-constrained):          Theta* Path (any-angle):
S---+                                S
    |                                 \
    +---+                              \
        |                               \
        +---G                            G

Total distance: 8 units              Total distance: 5.66 units
Waypoints: 6                         Waypoints: 2
```

### Mathematical Foundation

**Euclidean Distance**: Unlike A* which uses Manhattan distance for grid-based movement, Theta* uses Euclidean distance since it allows any-angle paths:

**Formula**: h(n1, n2) = √[(n1.x - n2.x)² + (n1.y - n2.y)²]

### Complexity

- **Time Complexity**: O(b^d) similar to A*, but with additional line-of-sight checks
- **Space Complexity**: O(b^d) same as A*
- **Line of Sight Check**: O(max(dx, dy)) per check using Bresenham's algorithm

### Advantages and Limitations

**Advantages**:
- Produces shorter, more natural paths than A*
- Fewer waypoints = simpler path following
- No post-processing smoothing needed
- Still guarantees shortest path in continuous space
- Maintains A*'s completeness and optimality properties

**Limitations**:
- Slightly slower than A* due to line-of-sight checks
- Still grid-based (resolution dependent)
- Assumes straight-line motion between waypoints
- May require path tracking controllers that handle sparse waypoints

### When to Use Theta* vs A*

**Use Theta* when**:
- Path naturalness and length are critical
- You need fewer waypoints for controller efficiency
- The environment has large open spaces
- Post-processing overhead needs to be minimized

**Use A* when**:
- Computational resources are extremely limited
- Grid-aligned movement is acceptable
- You already have smoothing in your pipeline
- The environment is highly cluttered (less benefit from line of sight)

---

# Path Tracking Controllers

## Pure Pursuit Controller

### What is Pure Pursuit?

Pure Pursuit is a path tracking algorithm that calculates the curvature needed to reach a "lookahead point" on the path. It's geometrically intuitive and widely used in mobile robotics.

### Core Concept

Imagine you're driving a car and looking at a point on the road ahead (not directly in front, but some distance forward). You steer to aim toward that point. As you move, you continuously update which point you're looking at. This is the essence of Pure Pursuit.

### How Pure Pursuit Works

The controller uses a **lookahead distance** to select a target point ahead on the path, then calculates the steering angle needed to reach that point.

### Algorithm Steps

1. **Find Lookahead Point**: Search along the path for the first point that is at least `lookahead_distance` away from the robot

2. **Transform to Local Coordinates**: Convert the target point from global coordinates to the robot's local reference frame

3. **Calculate Curvature**: Using the lookahead point's position, calculate the path curvature:
   ```
   curvature = 2 * y_local / lookahead_distance²
   ```

4. **Compute Steering Angle**: Convert curvature to steering angle:
   ```
   steering_angle = atan(curvature * steering_gain)
   ```

5. **Adjust Speed**: Reduce speed when making sharp turns

### Mathematical Foundation

The Pure Pursuit algorithm is based on finding the arc of a circle that:
- Passes through the robot's current position
- Is tangent to the robot's current heading
- Passes through the lookahead point

The curvature formula `2y/L²` comes from circle geometry, where:
- `y` is the lateral offset of the lookahead point in robot coordinates
- `L` is the lookahead distance

### Key Characteristics

- **Lookahead Point Selection**: Searches forward along the path for the first point at least the lookahead distance away
- **Coordinate Transformation**: Uses rotation matrices to transform the target point from global to robot's local reference frame  
- **Adaptive Speed Control**: Reduces speed for sharp turns (typically >  45°) to maintain stability and tracking accuracy

### Key Parameters

- **lookahead_distance**: Larger values create smoother paths but may cut corners; smaller values track more precisely but may oscillate
- **steering_gain**: Amplifies the steering response; typically set to 1.0
- **max_speed / min_speed**: Define the speed range based on path curvature
- **slow_steering_threshold**: Angle threshold for speed reduction (default: π/4 or 45°)

### Advantages and Limitations

**Advantages**:
- Simple and intuitive
- Smooth control outputs
- Computationally lightweight
- Works well for most paths

**Limitations**:
- May cut corners on sharp turns
- Lookahead distance requires tuning
- Can oscillate with poor parameter choices
- No obstacle avoidance (follows path blindly)

---

## Stanley Controller

### What is Stanley Controller?

The Stanley Controller is a sophisticated path tracking algorithm developed by Stanford University for their winning entry in the 2005 DARPA Grand Challenge. It's more advanced than Pure Pursuit and particularly effective for higher-speed navigation.

### Why Stanley Controller?

- **Proven Performance**: Used in real autonomous vehicle competitions
- **Works with Waypoints**: Directly handles discrete path points (no reference trajectory needed)
- **Geometric Intuition**: Combines heading alignment + cross-track error correction
- **Velocity Adaptive**: Performance automatically adjusts with speed
- **Simple Implementation**: More sophisticated than Pure Pursuit, simpler than LQR/MPC

### How Stanley Works

The Stanley controller computes a steering angle `δ` that drives the vehicle toward the path using two components:

```math
\delta(t) = \psi(t) + \arctan\left(\frac{k \cdot e(t)}{v(t) + k_{soft}}\right)
```

Where:
- **ψ(t)**: Heading error (angle between vehicle heading and path tangent)
- **e(t)**: Cross-track error (lateral distance from path)
- **v(t)**: Current velocity
- **k**: Gain parameter (typical: 0.5 - 2.5)
- **k_soft**: Softening constant to prevent division by zero at low speeds (typical: 0.5 - 1.0)

### Physical Interpretation

1. **First term (ψ)**: Points the vehicle along the path direction
   - Aligns the robot's heading with the path tangent
   - Ensures the robot is oriented correctly

2. **Second term (arctan)**: Points the vehicle toward the path
   - Large cross-track error → steer more aggressively
   - High velocity → reduce steering sensitivity (stability)
   - Low velocity → increase steering gain (responsiveness)

### Adaptation for Differential Drive Robots

**Important**: Stanley was originally designed for **Ackermann steering** (car-like) vehicles and outputs a **steering angle δ**. For **differential drive** robots (like micromouse), the steering angle must be converted to **left/right wheel velocities**.

**Conversion Process**:
1. Convert steering angle to angular velocity: ω = v × tan(δ) / L
2. Calculate wheel velocities:
   - v_left = v - (ω × L / 2)
   - v_right = v + (ω × L / 2)

Where v is forward velocity and L is the wheelbase.

### Key Parameters

- **k (gain)**: Controls sensitivity to cross-track error
  - Higher k: More aggressive correction, faster convergence
  - Lower k: Gentler correction, smoother motion
  - Typical range: 0.5 - 2.5

- **k_soft (softening)**: Prevents instability at low speeds
  - Acts as a lower bound on denominator
  - Typical range: 0.5 - 1.0

### Advantages and Limitations

**Advantages**:
- Velocity-dependent gain provides stability at all speeds
- Handles sharp turns better than Pure Pursuit
- Proven in real-world high-speed applications
- Simple mathematical formulation

**Limitations**:
- Requires accurate velocity information
- Needs conversion for differential drive
- May oscillate if k is too high
- Performance depends on parameter tuning

---

# Path Optimization

## Clothoid Curves for Path Optimization

### What are Clothoids?

**Clothoids** (also called **Euler spirals**) are special curves where curvature changes linearly with arc length. They represent the most natural and comfortable path for vehicles and robots to follow.

### Why Clothoids (Euler Spirals)?

- **Linear Curvature Growth**: Curvature κ increases linearly with arc length → smooth steering commands
- **Kinematically Optimal**: Matches natural motion of differential drive robots (constant angular acceleration)
- **G² Continuity**: Smooth position, tangent, AND curvature at all points
- **Minimizes Jerk**: Reduces mechanical stress and improves tracking accuracy
- **Industry Standard**: Used in highway design, railway curves, and autonomous vehicle planning

### Mathematical Foundation

A **clothoid** (Euler spiral) is defined by the property:

```math
\kappa(s) = \kappa_0 + \frac{s}{L} \cdot (\kappa_f - \kappa_0)
```

Where:
- **κ(s)**: Curvature at arc length `s`
- **κ₀**: Initial curvature (often 0 for straight entry)
- **κ_f**: Final curvature (often 0 for straight exit)
- **L**: Total arc length of the spiral
- **s**: Distance along curve from start

Unlike circular arcs (constant κ) or Bézier curves (arbitrary κ), clothoids have **monotonic curvature** which is naturally trackable by wheeled robots.

### Parametric Form

Using Fresnel integrals:

```math
x(s) = \int_0^s \cos\left(\frac{\pi t^2}{2L^2}\right) dt
```

```math
y(s) = \int_0^s \sin\left(\frac{\pi t^2}{2L^2}\right) dt
```

These integrals don't have closed-form solutions, requiring numerical methods or lookup tables.

### Why Clothoids for Path Smoothing?

After A* or Theta* produces a waypoint path, clothoids can:
1. **Connect waypoints smoothly** with continuous curvature
2. **Eliminate sharp corners** that are difficult to track
3. **Produce feasible trajectories** for differential drive robots
4. **Minimize control effort** through smooth steering profiles

### Implementation Approaches

Clothoid fitting typically requires:
1. **Boundary value solving** to determine clothoid parameters that connect two poses
2. **Numerical integration** of Fresnel integrals to compute curve points
3. **Optimization algorithms** to minimize path length or curvature variation

Specialized libraries (such as `pyclothoids`) provide robust implementations with pre-computed lookup tables for efficient computation.

### Comparison: Circular Arcs vs Clothoids

| Property | Circular Arc | Clothoid |
|----------|-------------|----------|
| Curvature | Constant | Linear transition |
| G² Continuity | No (discontinuous κ) | Yes (smooth κ) |
| Steering Command | Step change | Ramp change |
| Tracking Accuracy | Poor at transitions | Excellent |
| Jerk | High | Minimal |

### Applications in Micromouse

1. **Post-processing A* paths**: Smooth sharp 90° turns
2. **Connecting Theta* waypoints**: Create continuous curvature paths
3. **Speed profile optimization**: Match robot dynamics to path curvature
4. **Corner cutting**: Safe, smooth corner navigation

---

# Configuration Parameters

## Path Planning Parameters

**A* Algorithm:**
- **Heuristic weight**: Scales the heuristic (> 1.0 for faster but suboptimal paths)
- **Wall proximity**: Optional cost to discourage paths near walls
- **Decay mode**: Exponential, inverse, or linear decay for wall costs

**Theta* Algorithm:**
- **Grid resolution**: Trade-off between path quality and computation time
- **Robot radius**: Safety margin for collision checking

## Path Tracking Parameters

**Pure Pursuit:**
- **Lookahead distance**: Larger for smoother paths, smaller for precision
- **Speed limits**: Maximum and minimum speeds
- **Steering gain**: Amplification factor for steering response
- **Turn threshold**: Angle at which to reduce speed

**Stanley Controller:**
- **Gain (k)**: Cross-track error sensitivity (typical: 0.5-2.5)
- **Softening (k_soft)**: Prevents instability at low speeds (typical: 0.5-1.0)
- **Maximum steering**: Safety limit on steering angle

---

# Parameter Tuning Guidelines

## For A* Planning

- Increase `heuristic_weight` (> 1.0) for faster but potentially suboptimal paths
- Enable `wall_cost` for safer navigation in narrow spaces
- Adjust `wall_cost.weight` to balance safety vs path length
- Use `exponential` decay for smoother cost gradients

## For Theta* Planning

- Decrease `resolution` for finer paths but slower computation
- Increase `robot_radius` for more conservative collision avoidance
- Best used in open environments with long sight lines

## For Pure Pursuit

- Increase `lookahead_distance` for smoother but less precise tracking
- Decrease `lookahead_distance` for tighter corners but potential oscillation
- Adjust speed limits based on robot dynamics and sensor capabilities
- Tune `steering_gain` if the robot under/over-steers
- Works best with dense waypoint paths (A* output)

## For Stanley Controller

- Increase `k` for faster convergence but potential oscillation
- Decrease `k` for smoother motion but slower convergence
- Adjust `k_soft` if robot is unstable at low speeds
- Works best with sparse waypoint paths (Theta* output)

---

# Algorithm Selection Guide

## Path Planning: A* vs Theta*

| Scenario | Recommended Algorithm | Reason |
|----------|----------------------|--------|
| Dense mazes, many walls | **A*** | Less benefit from line-of-sight, faster computation |
| Open environments | **Theta*** | Significantly shorter paths, fewer waypoints |
| Need guaranteed grid-optimal | **A*** | Optimal on grid graph |
| Need shortest Euclidean path | **Theta*** | Any-angle, closer to true optimal |
| Tight computational budget | **A*** | No line-of-sight checks |
| Post-processing available | **A*** + smoothing | Can match Theta* quality |

## Path Tracking: Pure Pursuit vs Stanley

| Scenario | Recommended Controller | Reason |
|----------|------------------------|--------|
| Dense waypoint paths | **Pure Pursuit** | Simpler, effective with many points |
| Sparse waypoint paths | **Stanley** | Better handles large gaps between waypoints |
| High-speed navigation | **Stanley** | Velocity-adaptive, more stable |
| Simple implementation needed | **Pure Pursuit** | Easier to understand and tune |
| Differential drive robot | **Pure Pursuit** | Direct output, no conversion needed |
| Precise path following | **Stanley** | Better cross-track error handling |

---

# References

## A* Algorithm
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). "A Formal Basis for the Heuristic Determination of Minimum Cost Paths." *IEEE Transactions on Systems Science and Cybernetics*.

## Theta* Algorithm
- Nash, A., et al. (2007). "Theta*: Any-Angle Path Planning on Grids." *AAAI Conference on Artificial Intelligence*. [PDF](https://cdn.aaai.org/AAAI/2007/AAAI07-187.pdf)
- Nash, A., & Koenig, S. (2013). "Any-Angle Path Planning." *AI Magazine*, 34(4), 85-107.

## Pure Pursuit
- Coulter, R. C. (1992). "Implementation of the Pure Pursuit Path Tracking Algorithm." *Carnegie Mellon University Robotics Institute Technical Report*.

## Stanley Controller
- Thrun, S., et al. (2006). "Stanley: The robot that won the DARPA Grand Challenge." *Journal of Field Robotics*, 23(9), 661-692.
- Snider, J. M. (2009). "Automatic Steering Methods for Autonomous Automobile Path Tracking." CMU-RI-TR-09-08.
- Hoffmann, G. M., et al. (2007). "Autonomous automobile trajectory tracking for off-road driving: Controller design, experimental validation and racing." *American Control Conference*.

## Clothoid Curves
- Bertolazzi, E., & Frego, M. (2015). "G¹ fitting with clothoids." *Mathematical Methods in the Applied Sciences*.
- Walton, D. J., & Meek, D. S. (2005). "A controlled clothoid spline." *Computers & Graphics*, 29(3), 353-363.
- Scheuer, A., & Fraichard, T. (1997). "Continuous-curvature path planning for car-like vehicles." *IEEE/RSJ International Conference on Intelligent Robots and Systems*.
- PyClothoids Documentation: https://github.com/philippeller/pyclothoids

---

## License

This implementation is provided for educational purposes.
