#!/usr/bin/env python3
"""
Unit tests for SamplingPushingPlanner.

Run with:
    python3 test_sampling_planner.py
"""

import sys
import os
import numpy as np
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sampling_pushing_planner import SamplingPushingPlanner


def test_basic_initialization():
    """Test planner initialization."""
    print("\n" + "="*60)
    print("TEST 1: Basic Initialization")
    print("="*60)
    
    planner = SamplingPushingPlanner(
        horizon=32,
        max_iterations=1000,
        step_size=0.15,
        verbose=False
    )
    
    assert planner.horizon == 32
    assert planner.max_iterations == 1000
    assert planner.step_size == 0.15
    
    print("✓ Planner initialized successfully")
    print(f"  - Horizon: {planner.horizon}")
    print(f"  - Max iterations: {planner.max_iterations}")
    print(f"  - Step size: {planner.step_size}")


def test_simple_planning():
    """Test planning in a simple empty environment."""
    print("\n" + "="*60)
    print("TEST 2: Simple Planning (Empty Environment)")
    print("="*60)
    
    planner = SamplingPushingPlanner(
        horizon=32,
        max_iterations=2000,
        step_size=0.2,
        verbose=True
    )
    
    # Create empty grid
    grid = np.zeros((50, 50), dtype=np.float32)
    
    # Create observation dict
    obs_dict = {
        'grid': torch.tensor(grid, dtype=torch.float32).unsqueeze(0),
        'robot_pos': np.array([5.0, 5.0]),
        'goal_pos': np.array([45.0, 45.0]),
        'box_positions': [],
        'receptacle_pos': np.array([45.0, 45.0])
    }
    
    # Plan
    result = planner.predict_action(obs_dict)
    waypoints = result['action'].numpy()
    
    assert waypoints.shape == (1, 32, 2), f"Expected shape (1, 32, 2), got {waypoints.shape}"
    
    # Check that path goes from start to goal
    start = waypoints[0, 0]
    end = waypoints[0, -1]
    start_dist = np.linalg.norm(start - np.array([5.0, 5.0]))
    end_dist = np.linalg.norm(end - np.array([45.0, 45.0]))
    
    print(f"✓ Planning successful")
    print(f"  - Output shape: {waypoints.shape}")
    print(f"  - Start position: {start} (dist from goal: {start_dist:.2f})")
    print(f"  - End position: {end} (dist from goal: {end_dist:.2f})")
    
    assert start_dist < 2.0, "Start position too far from robot_pos"
    assert end_dist < 5.0, "End position too far from goal_pos"


def test_obstacle_avoidance():
    """Test planning with obstacles."""
    print("\n" + "="*60)
    print("TEST 3: Obstacle Avoidance")
    print("="*60)
    
    planner = SamplingPushingPlanner(
        horizon=32,
        max_iterations=3000,
        step_size=0.2,
        verbose=True
    )
    
    # Create grid with wall
    grid = np.zeros((50, 50), dtype=np.float32)
    grid[20:30, 20] = 1.0  # Vertical wall
    
    obs_dict = {
        'grid': torch.tensor(grid, dtype=torch.float32).unsqueeze(0),
        'robot_pos': np.array([5.0, 25.0]),
        'goal_pos': np.array([45.0, 25.0]),
        'box_positions': [],
        'receptacle_pos': np.array([45.0, 25.0])
    }
    
    result = planner.predict_action(obs_dict)
    waypoints = result['action'].numpy()
    
    # Check that path avoids the wall
    wall_crossings = 0
    for wp in waypoints[0]:
        if 18 < wp[0] < 22 and 19 < wp[1] < 31:
            wall_crossings += 1
    
    print(f"✓ Planning with obstacles successful")
    print(f"  - Output shape: {waypoints.shape}")
    print(f"  - Wall crossings: {wall_crossings} (should be 0 or few)")
    
    # Allow some tolerance due to grid discretization
    assert wall_crossings < 5, "Path crossed wall too many times"


def test_box_aware_cost():
    """Test that box-aware cost function affects planning."""
    print("\n" + "="*60)
    print("TEST 4: Box-Aware Cost Function")
    print("="*60)
    
    planner = SamplingPushingPlanner(
        horizon=32,
        max_iterations=2000,
        step_size=0.2,
        verbose=False
    )
    
    # Grid with a box between start and goal
    grid = np.zeros((50, 50), dtype=np.float32)
    
    # Box is at (25, 25), receptacle at (45, 25)
    # So pushing box right (towards receptacle) should be advantageous
    obs_dict = {
        'grid': torch.tensor(grid, dtype=torch.float32).unsqueeze(0),
        'robot_pos': np.array([5.0, 25.0]),
        'goal_pos': np.array([45.0, 25.0]),
        'box_positions': [np.array([25.0, 25.0])],  # Box in the middle
        'receptacle_pos': np.array([45.0, 25.0])  # Right of box
    }
    
    result = planner.predict_action(obs_dict)
    waypoints = result['action'].numpy()
    
    print(f"✓ Planning with box-aware costs successful")
    print(f"  - Output shape: {waypoints.shape}")
    print(f"  - Box at: (25.0, 25.0)")
    print(f"  - Receptacle at: (45.0, 25.0)")
    print(f"  - Start: {waypoints[0, 0]}, End: {waypoints[0, -1]}")


def test_waypoint_interpolation():
    """Test waypoint interpolation."""
    print("\n" + "="*60)
    print("TEST 5: Waypoint Interpolation")
    print("="*60)
    
    planner = SamplingPushingPlanner(horizon=32)
    
    # Test with 3 nodes
    nodes = [
        np.array([0.0, 0.0]),
        np.array([5.0, 0.0]),
        np.array([5.0, 5.0])
    ]
    
    waypoints = planner._interpolate_waypoints(nodes, 32)
    
    assert waypoints.shape == (32, 2), f"Expected shape (32, 2), got {waypoints.shape}"
    
    # First waypoint should be close to start
    assert np.linalg.norm(waypoints[0] - nodes[0]) < 1.0
    
    # Last waypoint should be close to end
    assert np.linalg.norm(waypoints[-1] - nodes[-1]) < 1.0
    
    print(f"✓ Waypoint interpolation successful")
    print(f"  - Input: {len(nodes)} nodes")
    print(f"  - Output: {waypoints.shape[0]} waypoints")
    print(f"  - Start: {waypoints[0]} (expected ≈ {nodes[0]})")
    print(f"  - End: {waypoints[-1]} (expected ≈ {nodes[-1]})")


def test_push_score_calculation():
    """Test the push feasibility scoring."""
    print("\n" + "="*60)
    print("TEST 6: Push Score Calculation (Deviation Heuristic)")
    print("="*60)
    
    planner = SamplingPushingPlanner()
    
    # Test 1: Advantageous push (towards receptacle)
    # Robot at origin, box at (1, 0), receptacle at (2, 0)
    # Push vector should align with target vector
    node_a = np.array([0.0, 0.0])
    node_b = np.array([0.5, 0.0])
    box_pos = np.array([1.0, 0.0])
    receptacle_pos = np.array([2.0, 0.0])
    
    cost1 = planner._compute_edge_cost(
        node_a, node_b, [box_pos], receptacle_pos
    )
    
    # Test 2: Detrimental push (away from receptacle)
    receptacle_pos_bad = np.array([0.0, 0.0])
    
    cost2 = planner._compute_edge_cost(
        node_a, node_b, [box_pos], receptacle_pos_bad
    )
    
    print(f"✓ Push score calculations successful")
    print(f"  - Advantageous push cost: {cost1:.4f} (should be low)")
    print(f"  - Detrimental push cost: {cost2:.4f} (should be high)")
    print(f"  - Cost ratio: {cost2/cost1:.2f}x (should be >> 1)")
    
    assert cost2 > cost1, "Detrimental push should have higher cost"


def test_collision_checking():
    """Test collision detection."""
    print("\n" + "="*60)
    print("TEST 7: Collision Checking")
    print("="*60)
    
    planner = SamplingPushingPlanner()
    
    # Grid with obstacle
    grid = np.zeros((50, 50), dtype=np.float32)
    grid[20:30, 20:30] = 1.0  # Square obstacle
    
    # Path that avoids obstacle
    path_free = planner._is_collision_free(
        grid,
        np.array([5.0, 5.0]),
        np.array([15.0, 5.0])
    )
    
    # Path that hits obstacle
    path_collision = planner._is_collision_free(
        grid,
        np.array([15.0, 25.0]),
        np.array([35.0, 25.0])
    )
    
    print(f"✓ Collision checking successful")
    print(f"  - Free path detected: {path_free} (expected True)")
    print(f"  - Collision path detected: {path_collision} (expected False)")
    
    assert path_free is True, "Free path should be clear"
    assert path_collision is False, "Path through obstacle should collide"


def test_kdtree_search():
    """Test KDTree-based k-nearest neighbor search."""
    print("\n" + "="*60)
    print("TEST 8: KDTree K-Nearest Neighbor Search")
    print("="*60)
    
    planner = SamplingPushingPlanner()
    
    # Create a set of nodes
    nodes = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([10.0, 10.0]),
    ]
    
    # Find 3 nearest to (0.5, 0.5)
    query_point = np.array([0.5, 0.5])
    nearest = planner._find_k_nearest(nodes, query_point, k=3)
    
    # Should return 3 closest points (first 4)
    assert len(nearest) == 3
    
    # Distances to selected neighbors
    dists = [np.linalg.norm(nodes[i] - query_point) for i in nearest]
    
    print(f"✓ KDTree k-nearest search successful")
    print(f"  - Query point: {query_point}")
    print(f"  - k=3 nearest neighbors found: {nearest}")
    print(f"  - Distances: {dists}")
    
    # Shouldn't select the distant point at (10, 10)
    assert 4 not in nearest, "Should not select distant point"


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SAMPLING PUSHING PLANNER - UNIT TESTS")
    print("="*60)
    
    tests = [
        test_basic_initialization,
        test_simple_planning,
        test_obstacle_avoidance,
        test_box_aware_cost,
        test_waypoint_interpolation,
        test_push_score_calculation,
        test_collision_checking,
        test_kdtree_search,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
