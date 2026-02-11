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

from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt

try:
    from diffusionPolicy.base_lowdim_policy import BaseLowdimPolicy
except Exception:
    class BaseLowdimPolicy:
        def predict_action(self, obs_dict: Dict[str, torch.Tensor], cond=None) -> Dict[str, torch.Tensor]:
            raise NotImplementedError()


class SamplingPushingPlanner(BaseLowdimPolicy):
    """
    RRT*-based planner with box-aware cost function for pushing tasks.
    
    The planner treats movable boxes as opportunities for advantageous pushes
    rather than static obstacles. A push is advantageous if it moves the box
    towards the receptacle (positive cosine similarity), and costly if it
    moves away (negative similarity).
    
    Usage:
        planner = SamplingPushingPlanner(
            horizon=32,
            max_iterations=5000,
            step_size=0.15,
            goal_sample_rate=0.15,
            rewire_radius_factor=2.0
        )
        
        obs_dict = {
            'grid': obstacle_map,  # (H, W) occupancy grid
            'robot_pos': np.array([x, y]),
            'goal_pos': np.array([gx, gy]),
            'box_positions': [np.array([bx1, by1]), ...],
            'receptacle_pos': np.array([rx, ry])
        }
        
        result = planner.predict_action(obs_dict)
        waypoints = result['action']  # shape (1, 32, 2)
    """
    
    # Push cost parameters
    PUSH_COST_ADVANTAGE = 0.5      # Cost multiplier for advantageous pushes
    PUSH_COST_PENALTY = 50.0       # Cost multiplier for detrimental pushes
    OBSTACLE_COST = 1e6            # Cost for hitting walls
    BASE_COST_MULTIPLIER = 2.0     # Multiplier for base edge cost
    PUSH_COST_THRESHOLD = 0.05     # Cosine similarity threshold for "advantage"
    
    def __init__(
        self,
        horizon: int = 32,
        max_iterations: int = 5000,
        step_size: float = 0.15,
        goal_sample_rate: float = 0.15,
        rewire_radius_factor: float = 2.0,
        collision_check_resolution: float = 0.05,
        box_radius: float = 0.1,
        verbose: bool = False
    ):
        """
        Initialize the SamplingPushingPlanner.
        
        Args:
            horizon: Number of output waypoints (default 32 for DiffusionPolicy)
            max_iterations: Max iterations of RRT* tree expansion
            step_size: Max distance per RRT* expansion step
            goal_sample_rate: Probability of sampling goal directly
            rewire_radius_factor: Multiplier for k-nearest rewiring radius
            collision_check_resolution: Distance between collision check points
            box_radius: Approximate radius of movable boxes
            verbose: Print debug information
        """
        self.horizon = horizon
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.rewire_radius_factor = rewire_radius_factor
        self.collision_check_resolution = collision_check_resolution
        self.box_radius = box_radius
        self.verbose = verbose
        
        # RRT* parameters
        self.gamma = 0.0  # Will be set based on free space area
        self.k_nearest_factor = 1.2  # For k = k_nearest_factor * log(n)
        
    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        cond=None
    ) -> Dict[str, torch.Tensor]:
        """
        Plan a path from robot_pos to goal_pos and return interpolated waypoints.
        
        Args:
            obs_dict: Dictionary containing:
                - 'grid': (H, W) obstacle map where 1=obstacle, 0=free
                - 'robot_pos': (2,) current robot position in world coords
                - 'goal_pos': (2,) goal position
                - 'box_positions': List of (2,) box positions
                - 'receptacle_pos': (2,) receptacle position
            cond: Optional conditioning (unused, for API compatibility)
            
        Returns:
            Dictionary with 'action' key containing waypoints of shape (1, horizon, 2)
        """
        # Extract inputs
        grid, robot_pos, goal_pos, box_positions, receptacle_pos = \
            self._extract_planning_inputs(obs_dict)
        
        if self.verbose:
            print(f"[SamplingPushingPlanner] Planning from {robot_pos} to {goal_pos}")
            print(f"[SamplingPushingPlanner] {len(box_positions)} boxes, receptacle at {receptacle_pos}")
        
        # Plan path using RRT*
        path_nodes = self._plan_rrt_star(
            grid, robot_pos, goal_pos, box_positions, receptacle_pos
        )
        
        # Interpolate to horizon waypoints
        if path_nodes is None or len(path_nodes) < 2:
            # Fallback: direct interpolation
            if self.verbose:
                print("[SamplingPushingPlanner] RRT* failed; using fallback linear path")
            path_nodes = [robot_pos, goal_pos]
        
        waypoints = self._interpolate_waypoints(path_nodes, self.horizon)
        
        # Return in expected format: (batch=1, horizon, 2)
        return {
            'action': torch.tensor(waypoints, dtype=torch.float32).unsqueeze(0)
        }
    
    # ===== Input/Output Processing =====
    
    def _extract_planning_inputs(
        self,
        obs_dict: Dict[str, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
        """Extract planning inputs from obs_dict."""
        
        # Extract grid
        if 'grid' in obs_dict:
            grid = obs_dict['grid']
            if isinstance(grid, torch.Tensor):
                grid = grid.detach().cpu().numpy()
            if grid.ndim == 3:
                grid = grid[0]
        else:
            raise ValueError("obs_dict must contain 'grid' key")
        
        # Extract positions
        robot_pos = self._to_numpy(obs_dict.get('robot_pos'))
        goal_pos = self._to_numpy(obs_dict.get('goal_pos'))
        receptacle_pos = self._to_numpy(obs_dict.get('receptacle_pos'))
        
        # Extract box positions
        box_positions = []
        if 'box_positions' in obs_dict:
            boxes = obs_dict['box_positions']
            
            # Handle list of numpy arrays or tensors
            if isinstance(boxes, list):
                for box in boxes:
                    box_np = self._to_numpy(box)
                    box_positions.append(box_np)
            else:
                # Handle tensor or numpy array
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.detach().cpu().numpy()
                if isinstance(boxes, np.ndarray):
                    if boxes.ndim == 2:
                        box_positions = [boxes[i] for i in range(boxes.shape[0])]
                    elif boxes.ndim == 1:
                        box_positions = [boxes]
        
        return grid, robot_pos, goal_pos, box_positions, receptacle_pos
    
    def _to_numpy(self, val) -> np.ndarray:
        """Convert tensor/list to numpy array."""
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().numpy()
        return np.asarray(val)
    
    def _interpolate_waypoints(
        self,
        path_nodes: List[np.ndarray],
        n_waypoints: int
    ) -> np.ndarray:
        """
        Interpolate path nodes to n_waypoints waypoints.
        
        Args:
            path_nodes: List of (2,) nodes
            n_waypoints: Target number of waypoints
            
        Returns:
            (n_waypoints, 2) array of interpolated waypoints
        """
        if len(path_nodes) < 2:
            # Degenerate case
            return np.repeat(path_nodes[0:1], n_waypoints, axis=0)
        
        path_nodes = np.array(path_nodes, dtype=np.float64)
        
        # Compute cumulative distances
        diffs = np.diff(path_nodes, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        distances = np.concatenate(([0], np.cumsum(segment_lengths)))
        total_dist = distances[-1]
        
        if total_dist < 1e-6:
            return np.repeat(path_nodes[0:1], n_waypoints, axis=0)
        
        # Sample waypoints uniformly along path
        sample_distances = np.linspace(0, total_dist, n_waypoints)
        waypoints = np.interp(
            sample_distances, distances, path_nodes[:, 0], left=path_nodes[0, 0]
        ).astype(np.float64)
        
        # Handle y-coordinate
        y_waypoints = np.interp(
            sample_distances, distances, path_nodes[:, 1], left=path_nodes[0, 1]
        ).astype(np.float64)
        
        return np.column_stack([waypoints, y_waypoints])
    
    # ===== RRT* Core Algorithm =====
    
    def _plan_rrt_star(
        self,
        grid: np.ndarray,
        start: np.ndarray,
        goal: np.ndarray,
        box_positions: List[np.ndarray],
        receptacle_pos: np.ndarray
    ) -> Optional[List[np.ndarray]]:
        """
        Plan using RRT* algorithm with box-aware cost function.
        
        Returns:
            List of waypoint arrays, or None if planning failed
        """
        # Estimate free space for rewiring radius
        free_cells = np.sum(grid < 0.5)
        area = float(free_cells)
        self.gamma = self.rewire_radius_factor * np.sqrt(area / np.pi)
        
        # Initialize tree
        nodes = [start.copy()]
        parents = [None]
        costs = [0.0]  # Cost from start to each node
        
        for iteration in range(self.max_iterations):
            # Sample random point
            if np.random.rand() < self.goal_sample_rate:
                rand_point = goal.copy()
            else:
                rand_point = self._sample_random_point(grid)
            
            # Find nearest node
            nearest_idx = self._find_nearest(nodes, rand_point)
            nearest_node = nodes[nearest_idx]
            
            # Steer towards random point
            direction = rand_point - nearest_node
            dist = np.linalg.norm(direction)
            
            if dist < 1e-6:
                continue
            
            new_node = nearest_node + (direction / dist) * min(dist, self.step_size)
            
            # Check collision
            if not self._is_collision_free(grid, nearest_node, new_node):
                continue
            
            # Compute cost: base distance + box-aware penalty
            edge_cost = self._compute_edge_cost(
                nearest_node, new_node, box_positions, receptacle_pos
            )
            new_cost = costs[nearest_idx] + edge_cost
            
            # Find k-nearest neighbors for rewiring
            k = max(1, int(self.k_nearest_factor * np.log(len(nodes) + 1)))
            nearest_indices = self._find_k_nearest(nodes, new_node, k)
            
            # Find best parent among k-nearest
            best_parent_idx = nearest_idx
            best_cost = new_cost
            
            for parent_idx in nearest_indices:
                parent_node = nodes[parent_idx]
                
                # Check if rewiring would create collision
                if not self._is_collision_free(grid, parent_node, new_node):
                    continue
                
                # Compute cost through this parent
                rewire_cost = self._compute_edge_cost(
                    parent_node, new_node, box_positions, receptacle_pos
                )
                candidate_cost = costs[parent_idx] + rewire_cost
                
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_parent_idx = parent_idx
            
            # Add new node with best parent
            nodes.append(new_node.copy())
            parents.append(best_parent_idx)
            costs.append(best_cost)
            new_node_idx = len(nodes) - 1
            
            # Rewire k-nearest if beneficial
            for neighbor_idx in nearest_indices:
                if neighbor_idx == best_parent_idx:
                    continue
                
                neighbor_node = nodes[neighbor_idx]
                
                # Check collision
                if not self._is_collision_free(grid, new_node, neighbor_node):
                    continue
                
                # Compute cost through new_node
                rewire_cost = self._compute_edge_cost(
                    new_node, neighbor_node, box_positions, receptacle_pos
                )
                candidate_cost = best_cost + rewire_cost
                
                if candidate_cost < costs[neighbor_idx]:
                    parents[neighbor_idx] = new_node_idx
                    costs[neighbor_idx] = candidate_cost
            
            # Check if reached goal
            dist_to_goal = np.linalg.norm(new_node - goal)
            if dist_to_goal < self.step_size:
                # Add goal node
                if self._is_collision_free(grid, new_node, goal):
                    goal_cost = self._compute_edge_cost(
                        new_node, goal, box_positions, receptacle_pos
                    )
                    goal_total_cost = best_cost + goal_cost
                    
                    nodes.append(goal.copy())
                    parents.append(new_node_idx)
                    costs.append(goal_total_cost)
                    
                    if self.verbose:
                        print(f"[RRT*] Goal reached at iteration {iteration}")
                    
                    # Reconstruct path
                    return self._reconstruct_path(nodes, parents)
        
        if self.verbose:
            print(f"[RRT*] Max iterations reached; returning best path to goal")
        
        # Find node closest to goal
        closest_idx = min(range(len(nodes)), key=lambda i: np.linalg.norm(nodes[i] - goal))
        return self._reconstruct_path(nodes, parents, end_idx=closest_idx)
    
    def _sample_random_point(self, grid: np.ndarray) -> np.ndarray:
        """Sample a random point in the free space."""
        H, W = grid.shape
        while True:
            i = np.random.randint(0, H)
            j = np.random.randint(0, W)
            if grid[i, j] < 0.5:  # Free space
                return np.array([float(j), float(i)])
    
    def _find_nearest(self, nodes: List[np.ndarray], point: np.ndarray) -> int:
        """Find index of nearest node to point."""
        if not nodes:
            return 0
        distances = [np.linalg.norm(n - point) for n in nodes]
        return int(np.argmin(distances))
    
    def _find_k_nearest(
        self,
        nodes: List[np.ndarray],
        point: np.ndarray,
        k: int
    ) -> List[int]:
        """Find indices of k nearest nodes using KDTree."""
        if len(nodes) < k:
            return list(range(len(nodes)))
        
        nodes_array = np.array(nodes)
        tree = KDTree(nodes_array)
        _, indices = tree.query(point, k=min(k, len(nodes)))
        
        if isinstance(indices, (list, np.ndarray)):
            return list(indices)
        else:
            return [indices]
    
    def _is_collision_free(
        self,
        grid: np.ndarray,
        node_a: np.ndarray,
        node_b: np.ndarray
    ) -> bool:
        """
        Check if path from node_a to node_b is collision-free.
        Only checks against walls, not boxes (boxes are handled in cost function).
        """
        H, W = grid.shape
        
        # Bresenham-like sampling
        direction = node_b - node_a
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return True
        
        n_steps = int(np.ceil(distance / self.collision_check_resolution)) + 1
        
        for step in range(n_steps):
            t = step / max(n_steps - 1, 1)
            point = node_a + t * direction
            
            i = int(np.clip(point[1], 0, H - 1))
            j = int(np.clip(point[0], 0, W - 1))
            
            if grid[i, j] > 0.5:  # Obstacle
                return False
        
        return True
    
    # ===== Box-Aware Cost Function =====
    
    def _compute_edge_cost(
        self,
        node_a: np.ndarray,
        node_b: np.ndarray,
        box_positions: List[np.ndarray],
        receptacle_pos: np.ndarray
    ) -> float:
        """
        Compute cost of traversing edge (node_a -> node_b).
        
        Cost = base_distance * (1 + box_penalty)
        
        where box_penalty is computed based on intersection with boxes:
        - Advantageous push (towards receptacle): low penalty
        - Detrimental push (away from receptacle): high penalty
        """
        base_distance = np.linalg.norm(node_b - node_a)
        
        if base_distance < 1e-6:
            return 0.0
        
        # Compute box penalties
        total_penalty = 0.0
        edge_direction = (node_b - node_a) / base_distance
        
        for box_pos in box_positions:
            # Check if edge is close to box
            dist_to_edge, closest_point = self._distance_point_to_segment(
                box_pos, node_a, node_b
            )
            
            if dist_to_edge > 2.0 * self.box_radius:
                continue  # Too far away
            
            # Compute push vector (from robot towards box)
            push_vec = box_pos - closest_point
            push_dist = np.linalg.norm(push_vec)
            
            if push_dist < 1e-6:
                # Robot passes through box center, treat as interaction
                push_vec = edge_direction
                push_dist = 1.0
            else:
                push_vec_normalized = push_vec / push_dist
                push_vec = push_vec_normalized
            
            # Compute target vector (from box to receptacle)
            target_vec = receptacle_pos - box_pos
            target_dist = np.linalg.norm(target_vec)
            
            if target_dist < 1e-6:
                continue  # Box already at receptacle
            
            target_vec_normalized = target_vec / target_dist
            
            # Compute alignment (cosine similarity)
            alignment = np.dot(push_vec, target_vec_normalized)
            
            # Proximity weight: higher penalty for closer interactions
            proximity_weight = max(0.0, 1.0 - dist_to_edge / (2.0 * self.box_radius))
            proximity_weight = proximity_weight ** 2
            
            # Apply penalty based on alignment
            if alignment > self.PUSH_COST_THRESHOLD:
                # Advantageous push: moving box towards receptacle
                penalty = self.PUSH_COST_ADVANTAGE * (1.0 - alignment) * proximity_weight
            else:
                # Detrimental push: moving box away from receptacle or perpendicular
                penalty = self.PUSH_COST_PENALTY * max(0.0, -alignment) * proximity_weight
            
            total_penalty += penalty
        
        edge_cost = base_distance * self.BASE_COST_MULTIPLIER * (1.0 + total_penalty)
        return edge_cost
    
    def _distance_point_to_segment(
        self,
        point: np.ndarray,
        seg_a: np.ndarray,
        seg_b: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compute distance from point to line segment [seg_a, seg_b].
        
        Returns:
            (distance, closest_point_on_segment)
        """
        seg_vec = seg_b - seg_a
        seg_length_sq = np.sum(seg_vec ** 2)
        
        if seg_length_sq < 1e-12:
            return np.linalg.norm(point - seg_a), seg_a.copy()
        
        # Project point onto segment
        t = np.dot(point - seg_a, seg_vec) / seg_length_sq
        t = np.clip(t, 0.0, 1.0)
        
        closest = seg_a + t * seg_vec
        distance = np.linalg.norm(point - closest)
        
        return distance, closest
    
    # ===== Path Reconstruction =====
    
    def _reconstruct_path(
        self,
        nodes: List[np.ndarray],
        parents: List[Optional[int]],
        end_idx: Optional[int] = None
    ) -> List[np.ndarray]:
        """Reconstruct path from start (index 0) to end_idx."""
        if end_idx is None:
            end_idx = len(nodes) - 1
        
        path = []
        idx = end_idx
        
        while idx is not None:
            path.append(nodes[idx].copy())
            idx = parents[idx]
        
        path.reverse()
        return path
