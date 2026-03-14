# Hierarchical RRT Policy

This document explains how the hierarchical policy works when using the RRT-based low-level planner.

## Policy Structure

The policy is hierarchical with two levels:

1. **High-level controller (RL)** selects a spatial action / target region.
2. **Low-level controller (RRT planner)** generates a waypoint path that executes that high-level intent while respecting geometry and task constraints.

In code, this is implemented by combining the standard `HeRDPolicy` logic with `SamplingPushingPlanner` in `benchmark/rrt/herd_policy_with_sampling_planner.py`.

## End-to-End Action Flow

At each decision step:

1. The environment provides observations (`rl_obs`, `diff_obs`, `box_obs`, `robot_pose`).
2. The high-level RL policy predicts a spatial action.
3. The environment position controller maps that action to a nominal target.
4. If low-level refinement is active, the RRT planner is invoked to generate a trajectory to the target.
5. The resulting trajectory is feasibility-conditioned and converted to controller-ready format.
6. The final path is executed by the environment.

## RRT Planner Role

`SamplingPushingPlanner` performs tree-based planning with:

- Goal-biased sampling
- Collision checking against static obstacles (with obstacle inflation / robot radius buffer)
- Box-aware edge costs (Deviation Heuristic) to prefer beneficial interactions and penalize harmful pushes
- Path reconstruction and interpolation to fixed horizon

The output trajectory format expected by the policy interface is:

- `action`: tensor with shape `[1, T, 2]` where `T = horizon`

## Feasibility Conditioning

To match diffusion-style post-processing behavior, the RRT trajectory can run through the same conditioning functions used in the base policy:

- `ensure_waypoint_feasibility`
- `ensure_path_feasibility`

This ensures the sampled trajectory is projected/adjusted to valid traversable geometry before execution.

## Coordinate Conventions

The integration uses two coordinate spaces:

- **World coordinates (meters):** policy/environment/controller space
- **Grid coordinates (pixels/indices):** occupancy map and planning space

The low-level integration must preserve this pipeline:

1. Convert world positions to grid indices before RRT planning.
2. Plan in grid space.
3. Convert planned waypoints back to world-space meters.
4. Apply feasibility conditioning in the expected policy format.
5. Append headings and send to controller.

## Why this hierarchy helps

- High-level RL handles strategic decisions (what to do next).
- Low-level RRT handles geometric execution (how to move safely and effectively).
- Feasibility conditioning provides robustness against invalid segments.
- The combined system stays modular: high-level and low-level components can be swapped independently.

## Relevant Files

- `herd_policy.py` — base hierarchical policy and feasibility functions
- `benchmark/rrt/herd_policy_with_sampling_planner.py` — hierarchical policy variant using RRT low-level planning
- `benchmark/rrt/sampling_pushing_planner.py` — RRT planner implementation
- `scripts/run_experiments.py` — experiment/evaluation entry point
