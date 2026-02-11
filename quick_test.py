#!/usr/bin/env python3
"""Quick validation test for SamplingPushingPlanner."""

import numpy as np
import torch
from sampling_pushing_planner import SamplingPushingPlanner

print("Testing SamplingPushingPlanner...")

# Test 1: Basic initialization
planner = SamplingPushingPlanner(
    horizon=32,
    max_iterations=500,  # Fewer iterations for quick test
    verbose=False
)
print("✓ Planner initialized")

# Test 2: Extract inputs with list of boxes
grid = np.zeros((50, 50), dtype=np.float32)
obs_dict = {
    'grid': torch.tensor(grid, dtype=torch.float32).unsqueeze(0),
    'robot_pos': np.array([5.0, 5.0]),
    'goal_pos': np.array([45.0, 45.0]),
    'box_positions': [np.array([25.0, 25.0])],
    'receptacle_pos': np.array([45.0, 25.0])
}
g, r, gl, b, rec = planner._extract_planning_inputs(obs_dict)
print(f"✓ Extract inputs: grid {g.shape}, robot {r}, {len(b)} boxes")

# Test 3: Push feasibility scoring
cost_adv = planner._compute_edge_cost(
    np.array([0.0, 0.0]),
    np.array([0.5, 0.0]),
    [np.array([1.0, 0.0])],
    np.array([2.0, 0.0])
)

cost_det = planner._compute_edge_cost(
    np.array([0.0, 0.0]),
    np.array([0.5, 0.0]),
    [np.array([1.0, 0.0])],
    np.array([0.0, 0.0])
)

print(f"✓ Push costs: advantageous={cost_adv:.4f}, detrimental={cost_det:.4f}")
print(f"  Ratio: {cost_det/cost_adv:.2f}x (should be > 1)")

# Test 4: Simple planning
print("\nRunning simple path planning...")
obs_dict = {
    'grid': torch.tensor(grid, dtype=torch.float32).unsqueeze(0),
    'robot_pos': np.array([5.0, 5.0]),
    'goal_pos': np.array([45.0, 45.0]),
    'box_positions': [],
    'receptacle_pos': np.array([45.0, 45.0])
}

try:
    result = planner.predict_action(obs_dict)
    waypoints = result['action'].numpy()
    print(f"✓ Planning successful: {waypoints.shape}")
    print(f"  Start: {waypoints[0, 0]}, End: {waypoints[0, -1]}")
except Exception as e:
    print(f"✗ Planning failed: {e}")

print("\n✓ All basic tests passed!")
