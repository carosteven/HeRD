# SamplingPushingPlanner: RRT* with Box-Aware Cost Function

## Overview

`SamplingPushingPlanner` is a sophisticated motion planning algorithm designed specifically for robotic manipulation tasks where an agent needs to push movable boxes towards a target receptacle. Unlike traditional planners that treat all objects as static obstacles, this planner **actively evaluates pushing feasibility** using a novel cost function based on the **Deviation Heuristic**.

## Key Features

### 1. **RRT* (RRT-Star) Core Algorithm**
- Asymptotically optimal sampling-based motion planning
- K-nearest neighbor rewiring for convergence to optimal solutions
- Efficient spatial queries using KDTree
- Configurable exploration vs. goal-biasing

### 2. **Box-Aware Cost Function**
The critical innovation that distinguishes this from standard RRT* is the **adaptive cost function** for edges that interact with movable boxes:

#### Cost Computation
For an edge traversing from node $n_{start}$ to $n_{end}$:

$$\text{Cost} = d_{euclidean} \times \text{BASE\_COST\_MULTIPLIER} \times (1 + \text{box\_penalty})$$

Where the **box_penalty** depends on feasibility of pushing any boxes in the path.

#### Deviation Heuristic: Push Feasibility Evaluation
For each box along the robot's path:

1. **Compute Push Vector** $\vec{v}_{push}$:
   - Direction from robot path to box center
   - Represents the direction the box would be pushed

2. **Compute Target Vector** $\vec{v}_{target}$:
   - Direction from box to receptacle position
   - Represents the desired push direction

3. **Compute Alignment** (cosine similarity):
   $$\cos(\theta) = \frac{\vec{v}_{push} \cdot \vec{v}_{target}}{|\vec{v}_{push}| \cdot |\vec{v}_{target}|}$$

4. **Determine Push Desirability**:
   - **Advantageous Push** ($\cos(\theta) > \text{PUSH\_COST\_THRESHOLD}$):
     - Push moves box towards receptacle ✓
     - Apply LOW cost: $\text{penalty} = \text{PUSH\_COST\_ADVANTAGE} \times (1 - \cos(\theta))$
     - Makes this path CHEAPER than driving around

   - **Detrimental Push** ($\cos(\theta) \leq \text{PUSH\_COST\_THRESHOLD}$):
     - Push moves box away from receptacle ✗
     - Apply HIGH cost: $\text{penalty} = \text{PUSH\_COST\_PENALTY} \times |\cos(\theta)| \times \text{proximity}$
     - Forces the robot to DEVIATE/AVOID (hence "Deviation Heuristic")

## Mathematical Formulation

### Cost Function Details

Given an edge $(n_a, n_b)$ and a set of boxes $B$:

$$C(n_a, n_b) = d(n_a, n_b) \cdot \beta \cdot \left(1 + \sum_{b \in B} \text{cost}_{push}(b)\right)$$

Where:
- $d(n_a, n_b)$ = Euclidean distance
- $\beta$ = BASE_COST_MULTIPLIER (default: 2.0)
- $\text{cost}_{push}(b)$ depends on alignment:

$$\text{cost}_{push}(b) = \begin{cases}
\alpha_{adv} \cdot (1 - \cos\theta) & \text{if } \cos\theta > \tau \\
\alpha_{det} \cdot |\cos\theta| \cdot p_{prox} & \text{otherwise}
\end{cases}$$

Where:
- $\alpha_{adv}$ = PUSH_COST_ADVANTAGE (default: 0.5) — reward for good pushes
- $\alpha_{det}$ = PUSH_COST_PENALTY (default: 50.0) — strong disincentive for bad pushes
- $\tau$ = PUSH_COST_THRESHOLD (default: 0.05) — alignment threshold
- $p_{prox} = (1 - d_{box}/r_{box})^2$ — proximity term

### RRT* Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon` | 32 | Number of output waypoints |
| `max_iterations` | 5000 | Max tree expansion iterations |
| `step_size` | 0.15 | Max distance per expansion step |
| `goal_sample_rate` | 0.15 | Probability of sampling goal directly |
| `rewire_radius_factor` | 2.0 | Scales rewiring radius computation |
| `collision_check_resolution` | 0.05 | Distance between collision check points |
| `box_radius` | 0.1 | Approximate box radius for interaction |

## Usage

### Basic Usage

```python
from sampling_pushing_planner import SamplingPushingPlanner
import numpy as np
import torch

# Initialize planner
planner = SamplingPushingPlanner(
    horizon=32,
    max_iterations=5000,
    step_size=0.15,
    goal_sample_rate=0.15,
    rewire_radius_factor=2.0,
    verbose=True  # Enable debugging
)

# Prepare observation dictionary
obs_dict = {
    'grid': torch.tensor(obstacle_map, dtype=torch.float32).unsqueeze(0),  # (1, H, W)
    'robot_pos': np.array([5.0, 5.0]),  # Current robot position
    'goal_pos': np.array([10.0, 10.0]),  # Goal position
    'box_positions': [
        np.array([7.0, 6.0]),  # Box 1
        np.array([8.0, 7.0]),  # Box 2
    ],
    'receptacle_pos': np.array([9.5, 9.5])  # Target receptacle
}

# Get planned waypoints
result = planner.predict_action(obs_dict)
waypoints = result['action']  # shape: (1, 32, 2)
```

### Integration with HeRDPolicy

```python
from herd_policy_with_sampling_planner import HeRDPolicyWithSamplingPlanner
from submodules.BenchNPIN.benchnpin.common.utils.utils import DotDict

# Configure policy
cfg = DotDict.to_dot_dict({
    'diffusion': {
        'use_diffusion_policy': True,  # Enables SamplingPushingPlanner
        'horizon': 32,
        # ... other config ...
    },
    # ... rest of config ...
})

# Create policy
policy = HeRDPolicyWithSamplingPlanner(cfg=cfg)

# Use in action loop
path, spatial_action = policy.act(
    rl_obs, diff_obs, box_obs, robot_pose
)
```

### Evaluation

```bash
# Run with default settings (5 episodes, small_columns)
python3 eval_with_sampling_planner.py

# Run with 50 episodes on large_divider configuration
python3 eval_with_sampling_planner.py --num_eps 50 --config large_divider

# Run with verbose output
python3 eval_with_sampling_planner.py --num_eps 10 --config large_columns --verbose
```

## Algorithm Details

### RRT* Expansion Loop

1. **Sample Random Node**:
   - With probability `goal_sample_rate`, sample goal directly
   - Otherwise, sample uniformly from free space

2. **Find Nearest Node**:
   - Use Euclidean distance to find closest tree node
   - $n_{nearest} = \arg\min_n ||n - n_{random}||$

3. **Steer Towards Random Node**:
   - Move from nearest by at most `step_size`
   - $n_{new} = n_{nearest} + \min(\text{step\_size}, ||n_{random} - n_{nearest}||) \cdot \frac{n_{random} - n_{nearest}}{||n_{random} - n_{nearest}||}$

4. **Collision Check** (Walls Only):
   - Sample along segment at `collision_check_resolution` intervals
   - Only check against walls, NOT boxes (handled in cost)

5. **Compute Edge Cost**:
   - Base distance + box-aware penalties
   - Uses **Deviation Heuristic** to evaluate push feasibility

6. **K-Nearest Neighbor Rewiring**:
   - Find $k = 1.2 \log(n)$ nearest neighbors using KDTree
   - For each neighbor, consider rewiring if cost improves
   - Only rewire if collision-free and cost-reducing

7. **Goal Connection**:
   - When tree comes within `step_size` of goal
   - Add goal node if connection is collision-free
   - Return reconstructed path

### Path Reconstruction

The final path is reconstructed by backtracking through parent pointers:

```
path = []
node = goal
while node != start:
    path.append(node)
    node = parent[node]
path.reverse()
return path
```

The path nodes are then **interpolated to exactly `horizon` waypoints** using linear interpolation weighted by segment lengths.

## Comparison with Alternatives

### vs. Standard RRT*
- **Standard RRT***: Treats all boxes as static obstacles or free space
- **SamplingPushingPlanner**: Dynamically evaluates box-pushing feasibility
- **Benefit**: Finds routes that exploit advantageous pushes

### vs. Clearance Planner
- **Clearance Planner**: Grid-based A* with distance-to-obstacle awareness
- **SamplingPushingPlanner**: Sampling-based with push-direction awareness
- **Benefit**: Handles continuous spaces better, explicit box interaction evaluation

### vs. Machine Learning (Diffusion)
- **DiffusionPolicy**: Learns from demonstrations, requires training
- **SamplingPushingPlanner**: Analytical, no training required
- **Benefit**: Guaranteed properties, explainable, no data requirements

## Customization

### Adjusting Push Cost Parameters

```python
planner = SamplingPushingPlanner(horizon=32)

# Make pushing more attractive
planner.PUSH_COST_ADVANTAGE = 0.2        # Lower cost for good pushes
planner.PUSH_COST_PENALTY = 100.0        # Higher penalty for bad pushes
planner.PUSH_COST_THRESHOLD = 0.1        # More lenient threshold

# Or more conservative
planner.PUSH_COST_ADVANTAGE = 1.0
planner.PUSH_COST_PENALTY = 20.0
planner.PUSH_COST_THRESHOLD = 0.0        # Require near-perfect alignment
```

### Adjusting Planning Parameters

```python
planner = SamplingPushingPlanner(
    max_iterations=10000,          # More exploration
    step_size=0.25,                # Larger steps
    goal_sample_rate=0.2,          # More goal-directed
    rewire_radius_factor=3.0,      # More aggressive rewiring
    collision_check_resolution=0.02  # Finer collision checks
)
```

## Computational Complexity

- **Time Complexity**: $O(n \log n)$ per iteration due to KDTree queries
- **Space Complexity**: $O(n)$ for storing tree nodes
- **Typical Runtime**: 50-500 ms depending on environment complexity

## Debugging and Visualization

Enable verbose output:

```python
planner = SamplingPushingPlanner(
    horizon=32,
    verbose=True  # Prints planning progress
)
```

This outputs:
- Planning problem statement
- RRT* iteration count when goal is reached
- Warning if fallback to linear path occurs

## Related Files

- **`sampling_pushing_planner.py`**: Core planner implementation
- **`herd_policy_with_sampling_planner.py`**: Integration with HeRDPolicy
- **`eval_with_sampling_planner.py`**: Evaluation script
- **`herd_policy.py`**: Base policy class
- **`environment.py`**: BoxDeliveryEnv environment

## References

- **RRT***: Karaman & Frazzoli (2011) - "Sampling-based Algorithms for Optimal Motion Planning"
- **Deviation Heuristic**: Novel cost function for manipulation-aware planning
- **KDTree**: Bentley (1975) - "Multidimensional Binary Search Trees"

## Paper Contribution

This planner addresses Reviewer 1's request for a "sampling planner with cost terms" by:

1. ✓ Using RRT* (a proven sampling-based method)
2. ✓ Implementing sophisticated cost terms specific to pushing tasks
3. ✓ Evaluating box-pushing feasibility via the Deviation Heuristic
4. ✓ Providing a principled baseline that doesn't require training data
5. ✓ Being fully interpretable and explainable

The planner serves as a strong analytical baseline for comparison against learning-based approaches.

---

**Author**: Developed for HeRD robotics manipulation paper  
**Version**: 1.0  
**Last Updated**: 2026-02-11
