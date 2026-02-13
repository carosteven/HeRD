"""
Integrated SamplingPushingPlanner for HeRDPolicy.

This module provides a patched version of the HeRDPolicy.act() method 
that uses SamplingPushingPlanner instead of DiffusionPolicy.

The SamplingPushingPlanner uses RRT* with a sophisticated box-aware cost
function that evaluates pushing feasibility based on alignment with the
target receptacle.

Usage:
    from herd_policy_with_sampling_planner import HeRDPolicyWithSamplingPlanner
    
    policy = HeRDPolicyWithSamplingPlanner(cfg=cfg)
    path, spatial_action = policy.act(rl_obs, diff_obs, box_obs, robot_pose)
"""

import os
import sys
import torch
import numpy as np
from herd_policy import HeRDPolicy
from sampling_pushing_planner import SamplingPushingPlanner


class HeRDPolicyWithSamplingPlanner(HeRDPolicy):
    """
    HeRDPolicy variant that uses SamplingPushingPlanner instead of DiffusionPolicy
    for the low-level path generation.
    
    The sampling planner uses RRT* with a box-aware cost function that:
    1. Treats walls as forbidden (infinite cost)
    2. Evaluates box-pushing feasibility via "Deviation Heuristic"
    3. Uses cosine similarity between push and target vectors
    4. Applies low cost for advantageous pushes, high cost for detrimental ones
    """
    
    def __init__(self, cfg, job_id=None):
        super().__init__(cfg=cfg, job_id=job_id)
        
        # Replace diffusion policy with SamplingPushingPlanner
        if self.cfg.diffusion.use_diffusion_policy:
            print("HeRDPolicyWithSamplingPlanner: Using SamplingPushingPlanner (RRT* with box-aware costs)")
            self.diffusion_policy = SamplingPushingPlanner(
                horizon=self.cfg.diffusion.horizon,
                max_iterations=5000,
                step_size=0.15,
                goal_sample_rate=0.15,
                rewire_radius_factor=2.0,
                collision_check_resolution=0.05,
                box_radius=0.22,
                verbose=False,
                # Ensure RRT outputs go through the same feasibility conditioning
                condition_trajectory=True,
                conditioning_functions=[self.ensure_waypoint_feasibility, self.prune_by_distance, self.ensure_path_feasibility],
                robot_radius=0.66
            )
            # Don't need obs buffer for sampling planner
            if hasattr(self, 'obs_buffer'):
                del self.obs_buffer
    
    def act(self, rl_obs, diff_obs, box_obs, robot_pose, exploration_eps=None):
        """
        Act method with SamplingPushingPlanner integration.
        
        Uses RRT* with box-aware cost function for planning.
        """
        spatial_action, _ = self.rl_policy.predict(rl_obs, exploration_eps=exploration_eps)
        path, _ = self.env.position_controller.get_waypoints_to_spatial_action(
            robot_pose[0:2], robot_pose[2], spatial_action
        )

        if self.cfg.diffusion.use_diffusion_policy:
            box_in_path, _ = self.check_path_for_box_collision(path, box_obs)

            if box_in_path is None:
                target_pos = np.array(path[-1][0:2], dtype=np.float32)
                
                # === SAMPLING PUSHING PLANNER INTEGRATION ===
                try:
                    # Get occupancy grid from environment
                    grid = self.env.configuration_space.copy()  # Shape: (H, W)
                    
                    # Get receptacle position from environment
                    # (assuming it's available in the environment state)
                    receptacle_pos = self._get_receptacle_position()
                    
                    # Extract box positions in world coordinates
                    box_positions = self._extract_box_positions(box_obs)
                    
                    # Convert robot and goal positions from world to grid pixel indices
                    robot_pi, robot_pj = self.env.position_to_pixel_indices(
                        robot_pose[0], robot_pose[1], grid.shape
                    )
                    goal_pi, goal_pj = self.env.position_to_pixel_indices(
                        target_pos[0], target_pos[1], grid.shape
                    )
                    
                    # Convert receptacle position
                    receptacle_pi, receptacle_pj = self.env.position_to_pixel_indices(
                        receptacle_pos[0], receptacle_pos[1], grid.shape
                    )
                    
                    # Convert box positions to grid indices
                    box_positions_grid = []
                    for box_pos in box_positions:
                        bi, bj = self.env.position_to_pixel_indices(
                            box_pos[0], box_pos[1], grid.shape
                        )
                        box_positions_grid.append(np.array([bj, bi], dtype=np.float64))
                    
                    # Create observation dict for planner
                    grid_batch = torch.from_numpy(grid[np.newaxis, :, :]).float()
                    obs_dict = {
                        'grid': grid_batch,
                        'robot_pos': np.array([robot_pj, robot_pi], dtype=np.float64),
                        'goal_pos': np.array([goal_pj, goal_pi], dtype=np.float64),
                        'box_positions': box_positions_grid,
                        'receptacle_pos': np.array([receptacle_pj, receptacle_pi], dtype=np.float64),
                        'pixel_indices_to_position': self.env.pixel_indices_to_position,
                    }
                    
                    # Get waypoints from sampling planner
                    action_dict = self.diffusion_policy.predict_action(obs_dict)
                    # waypoints_grid = action_dict['action'].numpy()[0]  # Shape: (horizon, 2)
                    path = action_dict['action'].numpy()[0]  # Shape: (horizon, 2)
                    
                    # Convert waypoints from grid indices back to world coordinates
                    # path = []
                    # for wp_grid in waypoints_grid:
                    #     # wp_grid is [j, i] (x, y) in grid coordinates
                    #     j = int(np.clip(wp_grid[0], 0, grid.shape[1] - 1))
                    #     i = int(np.clip(wp_grid[1], 0, grid.shape[0] - 1))
                    #     wp_world = self.env.pixel_indices_to_position(i, j, grid.shape)
                    #     path.append(wp_world)
                    
                    path = np.array(path)
                    path = self.get_path_headings(path)
                    
                    if False:  # Enable for debugging
                        print(f"[SamplingPlanner] Generated {len(path)} waypoints")
                        print(f"[SamplingPlanner] Start: {path[0]}, End: {path[-1]}")
                    
                except Exception as e:
                    # Fallback: use original path from position controller
                    print(f"SamplingPushingPlanner error: {e}. Using fallback path.")
                    import traceback
                    traceback.print_exc()
                    path = self.get_path_headings(path)

        # when training rl policy, spatial action needs to be recorded in the replay buffer
        return path, spatial_action
    
    def _get_receptacle_position(self) -> np.ndarray:
        """
        Extract receptacle position from the environment.
        
        Returns:
            (2,) array of receptacle position in world coordinates
        """
        # Try multiple ways to access receptacle position
        if hasattr(self.env, 'receptacle_position'):
            pos = self.env.receptacle_position
            if hasattr(pos, '__len__'):
                return np.array(pos[0:2], dtype=np.float64)
        
        # Check pymunk space for receptacle body
        if hasattr(self.env, 'space'):
            for body in self.env.space.bodies:
                if hasattr(body, 'color') and 'receptacle' in str(body.color).lower():
                    return np.array([body.position.x, body.position.y], dtype=np.float64)
        
        # Fallback: return center of the environment
        H, W = self.env.configuration_space.shape
        center_i, center_j = H / 2, W / 2
        return self.env.pixel_indices_to_position(int(center_i), int(center_j), (H, W))
    
    def _extract_box_positions(self, box_obs: np.ndarray) -> list:
        """
        Extract box positions from the box observation.
        
        Args:
            box_obs: Box observation array (shape varies by observation type)
            
        Returns:
            List of (2,) arrays representing box positions in world coordinates
        """
        box_positions = []
        
        if box_obs is None or len(box_obs) == 0:
            return box_positions
        
        # Handle different box observation formats
        if isinstance(box_obs, np.ndarray):
            if box_obs.ndim == 1:
                # Single box: reshape to (1, len(box_obs))
                box_obs = box_obs.reshape(1, -1)
            
            if box_obs.ndim == 2:
                # Multiple boxes
                for i in range(box_obs.shape[0]):
                    # Assume first 2 elements are position
                    pos = box_obs[i, 0:2]
                    box_positions.append(np.array(pos, dtype=np.float64))
        
        # Try to get from environment directly
        if hasattr(self.env, 'box_bodies'):
            for body in self.env.box_bodies:
                if hasattr(body, 'position'):
                    box_positions.append(
                        np.array([body.position.x, body.position.y], dtype=np.float64)
                    )
        
        return box_positions
