"""
SamplingPushingPlanner: RRT* with Box-Aware Cost Function

This planner uses RRT-Star to generate paths from robot_pos to goal_pos,
with a sophisticated cost function that:
1. Treats walls as infinitely costly (forbidden)
2. Evaluates box-pushing feasibility via cosine similarity between push 
   and target vectors
3. Uses KDTree-based k-nearest neighbor search for asymptotically optimal rewiring

Author: Implemented for HeRD robotic manipulation baseline
"""

import numpy as np
from scipy.spatial import KDTree
from typing import List, Tuple, Optional, Callable
import time
from scipy.ndimage import binary_dilation

class RRTNode:
    """Node in the RRT* tree."""
    def __init__(self, pos: np.ndarray, parent: Optional['RRTNode'] = None, cost: float = 0.0):
        self.pos = pos  # (x, y)
        self.parent = parent
        self.cost = cost
        self.children: List['RRTNode'] = []

class SamplingPushingPlanner:
    """
    RRT* planner with box-aware cost function (Deviation Heuristic).
    
    Unlike standard RRT that treats all objects as obstacles, this planner
    evaluates whether pushing a box is advantageous (moves it toward receptacle)
    or detrimental (moves it away). This creates a cost landscape that encourages
    exploitation of beneficial pushes while penalizing harmful ones.
    """
    
    def __init__(
        self,
        horizon: int = 32,
        max_iterations: int = 5000,
        step_size: float = 0.15,
        goal_sample_rate: float = 0.15,
        rewire_radius_factor: float = 2.0,
        collision_check_resolution: float = 0.05,
        box_radius: float = 0.1,
        robot_radius: float = 0.25,
        verbose: bool = False,
        # Conditioning controls to match diffusion policy behavior
        condition_trajectory: bool = False,
        conditioning_functions: Optional[List[Callable]] = None,
    ):
        """
        Initialize the RRT* planner with box-aware cost parameters.
        
        Args:
            horizon: Number of output waypoints (must match DiffusionPolicy)
            max_iterations: Maximum RRT* expansion iterations
            step_size: Maximum edge extension distance
            goal_sample_rate: Probability of sampling goal vs. random
            rewire_radius_factor: Multiplier for k-nearest rewiring radius
            collision_check_resolution: Distance between collision check points
            box_radius: Approximate radius of movable boxes
            robot_radius: Robot footprint radius (for collision buffer - IMPORTANT!)
            verbose: Print debug information
        """
        self.horizon = horizon
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.collision_check_resolution = collision_check_resolution
        self.box_radius = box_radius
        self.robot_radius = robot_radius
        # Conditioning (optional) to apply the same feasibility conditioning
        # that the diffusion policy uses (e.g., ensure_waypoint_feasibility)
        self.condition_trajectory = condition_trajectory
        self.conditioning_functions = conditioning_functions
        self.verbose = verbose
        
        # Derived parameters
        self.search_radius = rewire_radius_factor * step_size
        self.k_neighbors = 10
        self.push_alignment_threshold = 0.05
        self.advantageous_push_cost = 0.5
        self.detrimental_push_cost = 50.0
        self.box_proximity_threshold = box_radius * 2.0
        
        self.rng = np.random.RandomState(42)
        
        # Environment state (set by plan())
        self.obstacle_map = None
        self.inflated_obstacle_map = None  # NEW: Obstacles with robot_radius buffer
        self.grid_width = None
        self.grid_height = None
        self.box_positions = []
        self.receptacle_pos = None
    
    def _inflate_obstacles(self, obstacle_map: np.ndarray) -> np.ndarray:
        """
        Inflate fixed obstacles by robot_radius to ensure collision-free paths.
        
        Args:
            obstacle_map: Binary grid (1 = obstacle, 0 = free)
        
        Returns:
            Inflated obstacle map with robot_radius buffer
        """
        # Convert robot_radius (in meters/grid units) to pixels
        # Assuming 1 grid cell = 0.1m (adjust if your resolution differs)
        inflation_pixels = int(np.ceil(self.robot_radius / 0.1))
        
        # Create circular structuring element
        y, x = np.ogrid[-inflation_pixels:inflation_pixels+1, -inflation_pixels:inflation_pixels+1]
        structure = x**2 + y**2 <= inflation_pixels**2
        
        # Dilate obstacles
        inflated = binary_dilation(obstacle_map, structure=structure)
        return inflated.astype(np.uint8)
    
    def _is_valid(self, pos: np.ndarray) -> bool:
        """Check if a position is collision-free with FIXED obstacles (with robot_radius buffer)."""
        x, y = int(pos[0]), int(pos[1])
        
        # Bounds check
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return False
        
        # Check inflated obstacle map (already accounts for robot_radius)
        return self.inflated_obstacle_map[y, x] == 0
    
    def _line_collision_check(self, pos1: np.ndarray, pos2: np.ndarray, num_checks: int = 20) -> bool:
        """
        Check if the line segment between pos1 and pos2 is collision-free.
        
        Args:
            pos1: Start position (x, y)
            pos2: End position (x, y)
            num_checks: Number of interpolation points to check
        
        Returns:
            True if collision-free, False otherwise
        """
        for alpha in np.linspace(0, 1, num_checks):
            intermediate = pos1 + alpha * (pos2 - pos1)
            if not self._is_valid(intermediate):
                return False
        return True
    
    def _compute_edge_cost(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Compute the cost of traversing from pos1 to pos2.
        
        This is where the "Deviation Heuristic" is applied:
        - Base cost = Euclidean distance
        - If path intersects a box:
            * Compute push vector (robot → box)
            * Compute target vector (box → receptacle)
            * If aligned (cos θ > threshold): LOW COST (exploit push)
            * If misaligned: HIGH COST (force deviation)
        
        Args:
            pos1: Start position
            pos2: End position
        
        Returns:
            Total cost for this edge
        """
        # Base cost: Euclidean distance
        base_cost = np.linalg.norm(pos2 - pos1)
        
        # Check for box interactions
        push_cost_multiplier = 1.0
        
        for box_pos in self.box_positions:
            # Compute minimum distance from line segment to box center
            dist_to_box = self._point_to_segment_distance(box_pos, pos1, pos2)
            
            # If robot path comes close to box, evaluate push alignment
            if dist_to_box < self.box_proximity_threshold:
                # Push vector: direction from robot to box
                push_vector = box_pos - pos1
                push_vector_norm = np.linalg.norm(push_vector)
                
                if push_vector_norm > 1e-6:
                    push_vector = push_vector / push_vector_norm
                    
                    # Target vector: direction from box to receptacle
                    target_vector = self.receptacle_pos - box_pos
                    target_vector_norm = np.linalg.norm(target_vector)
                    
                    if target_vector_norm > 1e-6:
                        target_vector = target_vector / target_vector_norm
                        
                        # Compute alignment (cosine similarity)
                        alignment = np.dot(push_vector, target_vector)
                        
                        # Apply Deviation Heuristic
                        if alignment > self.push_alignment_threshold:
                            # Advantageous push: REDUCE cost (encourage this path)
                            push_cost_multiplier = min(push_cost_multiplier, self.advantageous_push_cost)
                        else:
                            # Detrimental push: INCREASE cost (penalize this path)
                            push_cost_multiplier = max(push_cost_multiplier, self.detrimental_push_cost)
        
        return base_cost * push_cost_multiplier
    
    def _point_to_segment_distance(self, point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
        """
        Compute minimum distance from a point to a line segment.
        
        Args:
            point: Query point (x, y)
            seg_start: Segment start (x, y)
            seg_end: Segment end (x, y)
        
        Returns:
            Minimum distance to segment
        """
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)
        
        if seg_len_sq < 1e-10:
            # Degenerate segment (point)
            return np.linalg.norm(point - seg_start)
        
        # Project point onto segment
        t = max(0, min(1, np.dot(point - seg_start, seg_vec) / seg_len_sq))
        projection = seg_start + t * seg_vec
        
        return np.linalg.norm(point - projection)
    
    def _steer(self, from_node: RRTNode, to_pos: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Steer from from_node toward to_pos, limited by step_size.
        
        Returns:
            new_pos: New position after steering
            distance: Actual distance traveled
        """
        direction = to_pos - from_node.pos
        distance = np.linalg.norm(direction)
        
        if distance < self.step_size:
            return to_pos, distance
        else:
            new_pos = from_node.pos + (direction / distance) * self.step_size
            return new_pos, self.step_size
    
    def _sample_random_position(self, goal_pos: np.ndarray) -> np.ndarray:
        """
        Sample a random position in the workspace.
        With probability goal_sample_rate, return goal_pos (for faster convergence).
        """
        if self.rng.rand() < self.goal_sample_rate:
            return goal_pos.copy()
        
        return np.array([
            self.rng.uniform(0, self.grid_width - 1),
            self.rng.uniform(0, self.grid_height - 1)
        ])
    
    def _get_nearest_node(self, tree: List[RRTNode], pos: np.ndarray) -> RRTNode:
        """Find the nearest node in the tree to the given position."""
        min_dist = float('inf')
        nearest = tree[0]
        
        for node in tree:
            dist = np.linalg.norm(node.pos - pos)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _get_nearby_nodes(self, tree: List[RRTNode], kdtree: KDTree, pos: np.ndarray) -> List[RRTNode]:
        """
        Get all nodes within search_radius of pos using KDTree.
        Limited to k_neighbors for computational efficiency.
        """
        indices = kdtree.query_ball_point(pos, self.search_radius)
        nearby = [tree[i] for i in indices]
        
        # Sort by cost and take top k_neighbors
        nearby.sort(key=lambda n: n.cost)
        return nearby[:self.k_neighbors]
    
    def _reconstruct_path(self, goal_node: RRTNode) -> np.ndarray:
        """
        Backtrack from goal_node to root to extract path.
        Interpolate to horizon waypoints to match DiffusionPolicy output format.
        """
        # Backtrack to get path
        path = []
        current = goal_node
        while current is not None:
            path.append(current.pos)
            current = current.parent
        
        path.reverse()
        path = np.array(path)
        
        # Interpolate to exactly horizon waypoints
        if len(path) < 2:
            # Degenerate case: repeat goal
            return np.tile(path[0], (self.horizon, 1))
        
        # Linear interpolation
        path_length = np.cumsum(np.r_[0, np.linalg.norm(np.diff(path, axis=0), axis=1)])
        total_length = path_length[-1]
        
        if total_length < 1e-6:
            return np.tile(path[0], (self.horizon, 1))
        
        # Sample horizon evenly-spaced points along path
        target_lengths = np.linspace(0, total_length, self.horizon)
        interpolated_path = np.zeros((32, 2))
        
        for i, target_len in enumerate(target_lengths):
            idx = np.searchsorted(path_length, target_len)
            if idx == 0:
                interpolated_path[i] = path[0]
            elif idx >= len(path):
                interpolated_path[i] = path[-1]
            else:
                # Linear interpolation between path[idx-1] and path[idx]
                alpha = (target_len - path_length[idx-1]) / (path_length[idx] - path_length[idx-1])
                interpolated_path[i] = (1 - alpha) * path[idx-1] + alpha * path[idx]
        
        return interpolated_path
    
    def predict_action(self, obs_dict: dict, cond=None) -> dict:
        """
        Main entry point called by HeRDPolicy integration.
        
        Args:
            obs_dict: Dictionary containing:
                - 'grid': (1, H, W) tensor - occupancy grid
                - 'robot_pos': (2,) array - robot position in grid coords
                - 'goal_pos': (2,) array - goal position in grid coords
                - 'box_positions': list of (2,) arrays - box positions in grid coords
                - 'receptacle_pos': (2,) array - receptacle position in grid coords
            cond: Unused (for API compatibility)
        
        Returns:
            Dictionary with 'action' key containing (1, horizon, 2) tensor of waypoints
        """
        import torch
        
        # Extract inputs
        grid = obs_dict['grid']
        if isinstance(grid, torch.Tensor):
            grid = grid.detach().cpu().numpy()[0]  # (H, W)
        
        robot_pos = np.array(obs_dict['robot_pos'], dtype=np.float64)
        goal_pos = np.array(obs_dict['goal_pos'], dtype=np.float64)
        receptacle_pos = np.array(obs_dict['receptacle_pos'], dtype=np.float64)
        
        box_positions = []
        if 'box_positions' in obs_dict:
            for bp in obs_dict['box_positions']:
                box_positions.append(np.array(bp, dtype=np.float64))
        
        # Call internal planning method
        waypoints = self.plan(robot_pos, goal_pos, box_positions, receptacle_pos, grid)

        # Optionally apply the same conditioning functions used by the diffusion policy
        path = []
        for wp_grid in waypoints:
            # wp_grid is [j, i] (x, y) in grid coordinates
            j = int(np.clip(wp_grid[0], 0, grid.shape[1] - 1))
            i = int(np.clip(wp_grid[1], 0, grid.shape[0] - 1))
            wp_world = obs_dict['pixel_indices_to_position'](i, j, grid.shape)
            path.append(wp_world)

        if self.condition_trajectory and self.conditioning_functions is not None:
            try:
                traj = torch.tensor(path, dtype=torch.float32).unsqueeze(0)  # [1, T, 2]
                for func in self.conditioning_functions:
                    traj = func(traj)
                path = traj.detach().cpu().numpy().squeeze(0)
            except Exception as e:
                if self.verbose or True:
                    print(f"Warning: conditioning functions failed: {e}")

        # Return in expected format: (1, horizon, 2) tensor
        return {'action': torch.tensor(path, dtype=torch.float32).unsqueeze(0)}
    
    def plan(
        self,
        robot_pos: np.ndarray,
        goal_pos: np.ndarray,
        box_positions: List[np.ndarray],
        receptacle_pos: np.ndarray,
        obstacle_map: np.ndarray
    ) -> np.ndarray:
        """
        Plan a path from robot_pos to goal_pos using RRT* with box-aware costs.
        
        Args:
            robot_pos: Current robot position (x, y) in grid coordinates
            goal_pos: Goal position (x, y) in grid coordinates
            box_positions: List of box positions [(x1, y1), (x2, y2), ...] in grid coords
            receptacle_pos: Receptacle position (x, y) in grid coordinates
            obstacle_map: Binary grid (1 = fixed obstacle, 0 = free)
        
        Returns:
            path: Numpy array of shape (horizon, 2) containing interpolated waypoints
        """
        start_time = time.time()
        
        # Store environment state
        self.obstacle_map = obstacle_map
        self.grid_height, self.grid_width = obstacle_map.shape
        self.box_positions = [np.array(bp) for bp in box_positions]
        self.receptacle_pos = np.array(receptacle_pos)
        
        # CRITICAL FIX: Inflate obstacles by robot_radius to prevent wall collisions!
        self.inflated_obstacle_map = self._inflate_obstacles(obstacle_map)
        
        # Validate start and goal
        if not self._is_valid(robot_pos):
            if self.verbose:
                print(f"Warning: Start position {robot_pos} is in collision! Returning straight line.")
            return np.linspace(robot_pos, goal_pos, self.horizon)
        
        if not self._is_valid(goal_pos):
            if self.verbose:
                print(f"Warning: Goal position {goal_pos} is in collision! Returning straight line.")
            return np.linspace(robot_pos, goal_pos, self.horizon)
        
        # Initialize RRT* tree
        root = RRTNode(pos=robot_pos.copy(), parent=None, cost=0.0)
        tree = [root]
        
        # RRT* expansion loop
        for iteration in range(self.max_iterations):
            # Sample random position
            random_pos = self._sample_random_position(goal_pos)
            
            # Find nearest node
            nearest_node = self._get_nearest_node(tree, random_pos)
            
            # Steer toward random position
            new_pos, _ = self._steer(nearest_node, random_pos)
            
            # Check if new position is valid
            if not self._is_valid(new_pos):
                continue
            
            # Check if edge is collision-free
            if not self._line_collision_check(nearest_node.pos, new_pos):
                continue
            
            # Compute edge cost using Deviation Heuristic
            edge_cost = self._compute_edge_cost(nearest_node.pos, new_pos)
            
            # Create new node
            new_node = RRTNode(
                pos=new_pos,
                parent=nearest_node,
                cost=nearest_node.cost + edge_cost
            )
            
            # RRT* rewiring: find nearby nodes and choose best parent
            if len(tree) > 1:
                # Build KDTree for efficient nearest neighbor search
                tree_positions = np.array([node.pos for node in tree])
                kdtree = KDTree(tree_positions)
                
                nearby_nodes = self._get_nearby_nodes(tree, kdtree, new_pos)
                
                # Find best parent among nearby nodes
                best_parent = nearest_node
                best_cost = nearest_node.cost + edge_cost
                
                for nearby_node in nearby_nodes:
                    if self._line_collision_check(nearby_node.pos, new_pos):
                        candidate_edge_cost = self._compute_edge_cost(nearby_node.pos, new_pos)
                        candidate_cost = nearby_node.cost + candidate_edge_cost
                        
                        if candidate_cost < best_cost:
                            best_parent = nearby_node
                            best_cost = candidate_cost
                
                # Update new node with best parent
                new_node.parent = best_parent
                new_node.cost = best_cost
            
            # Add to tree
            tree.append(new_node)
            new_node.parent.children.append(new_node)
            
            # Rewire nearby nodes if new_node provides better path
            if len(tree) > 1:
                tree_positions = np.array([node.pos for node in tree])
                kdtree = KDTree(tree_positions)
                nearby_nodes = self._get_nearby_nodes(tree, kdtree, new_pos)
                
                for nearby_node in nearby_nodes:
                    if nearby_node == new_node.parent:
                        continue
                    
                    if self._line_collision_check(new_node.pos, nearby_node.pos):
                        candidate_edge_cost = self._compute_edge_cost(new_node.pos, nearby_node.pos)
                        candidate_cost = new_node.cost + candidate_edge_cost
                        
                        if candidate_cost < nearby_node.cost:
                            # Rewire: update parent
                            old_parent = nearby_node.parent
                            if old_parent:
                                old_parent.children.remove(nearby_node)
                            
                            nearby_node.parent = new_node
                            nearby_node.cost = candidate_cost
                            new_node.children.append(nearby_node)
            
            # Check if we reached the goal
            if np.linalg.norm(new_pos - goal_pos) < self.step_size:
                # Found a path to goal!
                elapsed = time.time() - start_time
                if self.verbose:
                    print(f"RRT* found path in {elapsed:.2f}s ({iteration+1} iterations, {len(tree)} nodes)")
                
                return self._reconstruct_path(new_node)
        
        # Max iterations reached without finding goal
        # Return path to closest node to goal
        closest_node = min(tree, key=lambda n: np.linalg.norm(n.pos - goal_pos))
        elapsed = time.time() - start_time
        if self.verbose:
            print(f"RRT* timeout ({elapsed:.2f}s). Returning path to closest node (dist={np.linalg.norm(closest_node.pos - goal_pos):.2f})")
        
        return self._reconstruct_path(closest_node)