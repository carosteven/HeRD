from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from skimage.draw import line

from environment import BoxDeliveryEnv
from greedy_heuristic_planner import GreedyHeuristicPlanner
from submodules.BenchNPIN.benchnpin.common.controller.position_controller import (
    pixel_indices_to_position,
    position_to_pixel_indices,
)


class GreedyHeuristicPolicy:
    """Classical non-ML baseline wrapper for evaluation in BoxDeliveryEnv."""

    def __init__(self, cfg):
        self.env = BoxDeliveryEnv(cfg)
        self.env.reset()
        self.cfg = self.env.cfg

        ppm = self.env.position_controller.local_map_pixels_per_meter
        self.min_waypoint_spacing = 0.2
        self.max_waypoints_per_step = 10

        # Geometric standoff so robot center stays behind the box while pushing.
        box_half_extent = float(self.cfg.boxes.box_size) / 2.0
        push_standoff = float(self.env.robot_radius) + box_half_extent + 0.05
        self.box_half_extent = box_half_extent
        self.push_standoff = push_standoff

        # Replan guards for stale push execution.
        self.box_contact_distance = float(self.env.robot_radius) + box_half_extent + 0.55
        self.target_track_lost_distance = 1.0
        self.push_goal_proximity = push_standoff + 0.6
        self.stale_push_miss_limit = 2
        self.push_progress_epsilon = 0.02
        self.push_engage_distance = float(self.env.robot_radius) + box_half_extent + 0.35
        self.push_release_distance = float(self.env.robot_radius) + box_half_extent + 0.75
        self.push_mode_min_steps = 3

        # Keep robot-center waypoints strictly inside the traversable room footprint.
        margin = float(self.env.robot_radius)
        half_len = float(self.env.room_length) / 2.0
        half_wid = float(self.env.room_width) / 2.0
        self.world_bounds = (
            -half_len + margin,
            half_len - margin,
            -half_wid + margin,
            half_wid - margin,
        )

        def world_to_grid(pos: Tuple[float, float], shape: Tuple[int, int]) -> Tuple[int, int]:
            i, j = position_to_pixel_indices(pos[0], pos[1], shape, ppm)
            return int(i), int(j)

        def grid_to_world(idx: Tuple[int, int], shape: Tuple[int, int]) -> Tuple[float, float]:
            x, y = pixel_indices_to_position(idx[0], idx[1], shape, ppm)
            return float(x), float(y)

        self.planner = GreedyHeuristicPlanner(
            pre_push_distance=push_standoff,
            stance_tolerance=0.25,
            obstacle_value=1,
            grid_resolution=1.0 / ppm,
            clearance_penalty=50.0,
            clearance_radius=0.7,
            push_step=0.12,
            allow_diagonal=True,
            world_to_grid_fn=world_to_grid,
            grid_to_world_fn=grid_to_world,
            world_bounds=self.world_bounds,
        )

    @staticmethod
    def _euclidean(p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p0[0], p1[1] - p0[1])

    @staticmethod
    def _select_target_box(
        robot_pos: Tuple[float, float],
        box_positions: Sequence[Tuple[float, float]],
        receptacle_pos: Tuple[float, float],
    ) -> Tuple[float, float]:
        best_box = box_positions[0]
        best_score = float("inf")
        for box in box_positions:
            score = math.hypot(robot_pos[0] - box[0], robot_pos[1] - box[1]) + math.hypot(
                box[0] - receptacle_pos[0], box[1] - receptacle_pos[1]
            )
            if score < best_score:
                best_score = score
                best_box = box
        return best_box

    @staticmethod
    def _compute_pre_push_stance(
        target_box: Tuple[float, float],
        receptacle_pos: Tuple[float, float],
        standoff: float,
    ) -> Tuple[float, float]:
        dx = target_box[0] - receptacle_pos[0]
        dy = target_box[1] - receptacle_pos[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-8:
            return target_box
        ux, uy = dx / norm, dy / norm
        return (target_box[0] + ux * standoff, target_box[1] + uy * standoff)

    def _fallback_path(
        self,
        robot_pos: Tuple[float, float],
        box_positions: Sequence[Tuple[float, float]],
        receptacle_pos: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        robot_pos = self._clip_to_bounds(robot_pos)
        receptacle_pos = self._clip_to_bounds(receptacle_pos)
        box_positions = [self._clip_to_bounds(p) for p in box_positions]

        if len(box_positions) == 0:
            return [robot_pos, receptacle_pos]

        target_box = self._select_target_box(robot_pos, box_positions, receptacle_pos)
        pre_push_stance = self._compute_pre_push_stance(
            target_box,
            receptacle_pos,
            self.planner.pre_push_distance,
        )
        pre_push_stance = self._clip_to_bounds(pre_push_stance)

        try:
            route = self.env.position_controller.shortest_path(
                source_position=robot_pos,
                target_position=pre_push_stance,
                check_straight=True,
                configuration_space=self.env.position_controller.configuration_space,
            )
            if len(route) >= 2:
                return [(float(p[0]), float(p[1])) for p in route]
        except Exception:
            pass

        return [robot_pos, pre_push_stance]

    def _clip_to_bounds(self, p: Tuple[float, float]) -> Tuple[float, float]:
        min_x, max_x, min_y, max_y = self.world_bounds
        return (
            min(max(float(p[0]), min_x), max_x),
            min(max(float(p[1]), min_y), max_y),
        )

    def evaluate(self, num_eps: int):
        eps_rewards = []
        eps_steps = []
        eps_distance = []
        eps_boxes = []
        eps_avg_box_distance = []

        for eps_idx in range(num_eps):
            print("Progress: ", eps_idx, " / ", num_eps, " episodes", end="\r")
            _, info = self.env.reset()
            done = truncated = False
            ep_steps = 0
            ep_reward = 0.0
            pending_path_xy: List[Tuple[float, float]] = []
            current_target_box: Optional[Tuple[float, float]] = None
            mode = "transit"
            push_mode_steps = 0
            prev_target_box_pos: Optional[Tuple[float, float]] = None
            box_not_moving_steps = 0
            box_movement_threshold = 0.01  # Box must move at least this far to count as moving

            while True:
                ep_steps += 1
                robot_pose = info["robot_pose"]
                robot_pos = (float(robot_pose[0]), float(robot_pose[1]))
                receptacle_pos = (
                    float(self.env.receptacle_position[0]),
                    float(self.env.receptacle_position[1]),
                )

                box_positions = self._extract_box_positions(info["box_obs"])
                obstacle_map = (1.0 - self.env.position_controller.configuration_space).astype(np.uint8)

                # Track the same target if possible; otherwise reselect greedily.
                if len(box_positions) == 0:
                    current_target_box = None
                    mode = "transit"
                    pending_path_xy = []
                    prev_target_box_pos = None
                    box_not_moving_steps = 0
                else:
                    if current_target_box is None:
                        current_target_box = self._select_target_box(robot_pos, box_positions, receptacle_pos)
                        prev_target_box_pos = current_target_box
                        box_not_moving_steps = 0
                    else:
                        nearest_result = self._nearest_box(current_target_box, box_positions)
                        if nearest_result is None or nearest_result[1] > self.target_track_lost_distance:
                            current_target_box = self._select_target_box(robot_pos, box_positions, receptacle_pos)
                            mode = "transit"
                            pending_path_xy = []
                            prev_target_box_pos = current_target_box
                            box_not_moving_steps = 0
                        else:
                            current_target_box = nearest_result[0]

                # Explicit FSM with hysteresis to avoid mode flapping.
                if current_target_box is not None:
                    box_robot_distance = self._euclidean(current_target_box, robot_pos)
                    if mode == "transit" and box_robot_distance <= self.push_engage_distance:
                        mode = "push"
                        push_mode_steps = 0
                        pending_path_xy = []
                        prev_target_box_pos = current_target_box
                        box_not_moving_steps = 0
                    elif mode == "push":
                        push_mode_steps += 1
                        
                        # Check if the box is actually moving while we're pushing
                        if prev_target_box_pos is not None:
                            box_displacement = self._euclidean(prev_target_box_pos, current_target_box)
                            if box_displacement < box_movement_threshold:
                                box_not_moving_steps += 1
                            else:
                                box_not_moving_steps = 0
                        prev_target_box_pos = current_target_box
                        
                        # Exit push mode if: box is too far OR box isn't moving OR enough push steps have passed
                        should_exit = False
                        if box_robot_distance >= self.push_release_distance and push_mode_steps >= self.push_mode_min_steps:
                            should_exit = True
                        elif box_not_moving_steps >= 5:  # Box hasn't moved for 5 consecutive steps
                            should_exit = True
                        
                        if should_exit:
                            mode = "transit"
                            pending_path_xy = []
                            push_mode_steps = 0
                            box_not_moving_steps = 0

                if mode == "push" and current_target_box is not None:
                    # Recompute push path each step from live box pose.
                    path_xy = self.planner._build_push_path(robot_pos, current_target_box, receptacle_pos)
                    pending_path_xy = list(path_xy)
                else:
                    # Transit mode uses queue/chunking and replans when exhausted.
                    if len(pending_path_xy) <= 1:
                        path_xy = self.planner.plan(
                            robot_pos=robot_pos,
                            box_positions=box_positions,
                            receptacle_pos=receptacle_pos,
                            obstacle_map=obstacle_map,
                        )

                        # Guard against degenerate planner output.
                        path_length = 0.0
                        for i in range(1, len(path_xy)):
                            path_length += self._euclidean(path_xy[i - 1], path_xy[i])
                        if len(path_xy) < 2 or path_length < 0.35:
                            path_xy = self._fallback_path(robot_pos, box_positions, receptacle_pos)

                        pending_path_xy = list(path_xy)

                path, consumed = self._chunk_xy_path_to_pose_path(
                    path_xy=pending_path_xy,
                    current_heading=float(robot_pose[2]),
                    max_waypoints=self.max_waypoints_per_step,
                )

                # Always anchor chunk start at current robot position to avoid turn-back artifacts.
                if len(path) > 0:
                    path[0, 0] = float(robot_pos[0])
                    path[0, 1] = float(robot_pos[1])

                # Apply the same feasibility conditioning to both push and transit paths
                # to ensure consistency with diffusion-based policies.
                path = self._apply_feasibility_pipeline(path)

                # Keep overlap at chunk boundaries for controller continuity.
                if consumed >= len(pending_path_xy):
                    pending_path_xy = []
                else:
                    next_start = max(0, consumed - 1)
                    pending_path_xy = pending_path_xy[next_start:]

                _, reward, done, truncated, info = self.env.step(path)
                ep_reward += reward
                if done or truncated:
                    break

            eps_steps.append(ep_steps)
            eps_rewards.append(ep_reward)
            eps_distance.append(info["cumulative_distance"])
            eps_boxes.append(info["cumulative_boxes"])
            eps_avg_box_distance.append(sum(self.env.box_distances.values()) / len(self.env.box_distances))

        eps_steps = np.array(eps_steps)
        eps_rewards = np.array(eps_rewards)
        eps_distance = np.array(eps_distance)
        eps_boxes = np.array(eps_boxes)
        eps_avg_box_distance = np.array(eps_avg_box_distance)

        print(f"Average eps_steps: {eps_steps.mean():.2f} ± {eps_steps.std():.2f}")
        print(f"Average eps_rewards: {eps_rewards.mean():.2f} ± {eps_rewards.std():.2f}")
        print(f"Average eps_distance: {eps_distance.mean():.2f} ± {eps_distance.std():.2f}")
        print(f"Average eps_boxes: {eps_boxes.mean():.2f} ± {eps_boxes.std():.2f}")
        print(
            f"Average eps_avg_box_distance: {eps_avg_box_distance.mean():.2f} ± {eps_avg_box_distance.std():.2f}"
        )

    @staticmethod
    def _extract_box_positions(box_obs: Sequence[np.ndarray]) -> List[Tuple[float, float]]:
        positions: List[Tuple[float, float]] = []
        for box_vertices in box_obs:
            if box_vertices is None:
                continue
            arr = np.asarray(box_vertices, dtype=np.float32)
            if arr.size == 0:
                continue
            center = arr.mean(axis=0)
            positions.append((float(center[0]), float(center[1])))
        return positions

    def _xy_path_to_pose_path(
        self,
        path_xy: Sequence[Tuple[float, float]],
        current_heading: float,
    ) -> np.ndarray:
        if len(path_xy) == 0:
            return np.array([[0.0, 0.0, current_heading]], dtype=np.float32)

        if len(path_xy) == 1:
            x, y = path_xy[0]
            return np.array([[x, y, current_heading], [x, y, current_heading]], dtype=np.float32)

        # Remove waypoints that are too close, preserving first/last points.
        pruned_xy = [tuple(path_xy[0])]
        for point in path_xy[1:-1]:
            last = pruned_xy[-1]
            if math.hypot(point[0] - last[0], point[1] - last[1]) >= self.min_waypoint_spacing:
                pruned_xy.append((float(point[0]), float(point[1])))
        pruned_xy.append(tuple(path_xy[-1]))

        # Ensure at least two distinct points for controller stability.
        if len(pruned_xy) < 2 or math.hypot(pruned_xy[-1][0] - pruned_xy[0][0], pruned_xy[-1][1] - pruned_xy[0][1]) < 1e-6:
            x0, y0 = float(pruned_xy[0][0]), float(pruned_xy[0][1])
            eps = self.min_waypoint_spacing
            pruned_xy = [(x0, y0), (x0 + eps * math.cos(current_heading), y0 + eps * math.sin(current_heading))]

        path_xy_np = np.asarray(pruned_xy, dtype=np.float32)
        if path_xy_np.shape[0] == 1:
            path_xy_np = np.vstack([path_xy_np, path_xy_np])
        return self.get_path_headings(path_xy_np, initial_heading=float(current_heading))

    def _chunk_xy_path_to_pose_path(
        self,
        path_xy: Sequence[Tuple[float, float]],
        current_heading: float,
        max_waypoints: int,
    ) -> Tuple[np.ndarray, int]:
        if len(path_xy) <= 2:
            pose_path = self._xy_path_to_pose_path(path_xy, current_heading)
            return pose_path, len(path_xy)

        max_waypoints = max(2, int(max_waypoints))
        end_idx = min(len(path_xy), max_waypoints)
        chunk_xy = path_xy[:end_idx]
        pose_path = self._xy_path_to_pose_path(chunk_xy, current_heading)
        return pose_path, end_idx

    def _nearest_box(
        self,
        anchor: Tuple[float, float],
        box_positions: Sequence[Tuple[float, float]],
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        if len(box_positions) == 0:
            return None
        best_box = min(box_positions, key=lambda p: self._euclidean(p, anchor))
        return best_box, self._euclidean(best_box, anchor)

    def _maybe_invalidate_stale_push_queue(
        self,
        pending_path_xy: List[Tuple[float, float]],
        current_target_box: Optional[Tuple[float, float]],
        robot_pos: Tuple[float, float],
        box_positions: Sequence[Tuple[float, float]],
        receptacle_pos: Tuple[float, float],
        stale_push_miss_count: int,
        prev_target_receptacle_distance: Optional[float],
    ) -> Tuple[List[Tuple[float, float]], Optional[Tuple[float, float]], int, Optional[float]]:
        if len(pending_path_xy) <= 1:
            return pending_path_xy, current_target_box, 0, None

        if current_target_box is None or len(box_positions) == 0:
            return [], None, 0, None

        nearest_result = self._nearest_box(current_target_box, box_positions)
        if nearest_result is None:
            return [], None, 0, None
        tracked_box, track_error = nearest_result

        # If we can no longer reliably identify the target box, force replan.
        if track_error > self.target_track_lost_distance:
            return [], None, 0, None

        current_target_box = tracked_box

        # If this queued path is a push-to-receptacle path but the box is not near the robot,
        # then the robot likely missed/slipped the box; invalidate immediately.
        queue_end = pending_path_xy[-1]
        queue_targets_receptacle = self._euclidean(queue_end, receptacle_pos) <= self.push_goal_proximity
        box_robot_distance = self._euclidean(current_target_box, robot_pos)
        target_receptacle_distance = self._euclidean(current_target_box, receptacle_pos)

        if not queue_targets_receptacle:
            return pending_path_xy, current_target_box, 0, None

        # Hysteresis: keep push mode if either the box is near robot OR the box is
        # still making progress toward the receptacle.
        in_contact_band = box_robot_distance <= self.box_contact_distance
        making_progress = False
        if prev_target_receptacle_distance is not None:
            making_progress = target_receptacle_distance < (prev_target_receptacle_distance - self.push_progress_epsilon)

        if in_contact_band or making_progress:
            stale_push_miss_count = 0
        else:
            stale_push_miss_count += 1

        if stale_push_miss_count >= self.stale_push_miss_limit:
            return [], None, 0, None

        return pending_path_xy, current_target_box, stale_push_miss_count, target_receptacle_distance

    def restrict_heading_range(self, heading: float) -> float:
        return np.mod(heading + np.pi, 2 * np.pi) - np.pi

    def get_path_headings(self, path: np.ndarray, initial_heading: Optional[float] = None) -> np.ndarray:
        # Mirrors HeRD heading computation while allowing first heading initialization.
        headings: List[float] = []
        if initial_heading is None:
            initial_heading = 0.0
        headings.append(float(self.restrict_heading_range(float(initial_heading))))
        for i in range(1, len(path)):
            x_diff = path[i][0] - path[i - 1][0]
            y_diff = path[i][1] - path[i - 1][1]
            waypoint_heading = self.restrict_heading_range(np.arctan2(y_diff, x_diff))
            headings.append(float(waypoint_heading))
        headings_np = np.array(headings, dtype=np.float32).reshape(-1, 1)
        return np.concatenate((path.astype(np.float32), headings_np), axis=1)

    def _ensure_path_feasibility_xy(self, path_xy: np.ndarray) -> np.ndarray:
        """
        Ensures path feasibility by detecting blocked segments and intelligently expanding them
        with intermediate waypoints. If a direct line is blocked by an obstacle, 
        we subdivide the segment and recursively check feasibility.
        """
        if path_xy.shape[0] <= 1:
            return path_xy

        new_points: List[np.ndarray] = [path_xy[0]]
        for i in range(1, path_xy.shape[0]):
            p1 = new_points[-1]
            p2 = path_xy[i]

            p1_pos = (float(p1[0]), float(p1[1]))
            p2_pos = (float(p2[0]), float(p2[1]))

            source_i, source_j = self.env.position_to_pixel_indices(p1_pos[0], p1_pos[1], self.env.configuration_space.shape)
            target_i, target_j = self.env.position_to_pixel_indices(p2_pos[0], p2_pos[1], self.env.configuration_space.shape)
            rr, cc = line(source_i, source_j, target_i, target_j)

            # Use thin configuration space for collision checking
            if (1 - self.env.configuration_space_thin[rr, cc]).sum() == 0:
                # Direct line is clear
                new_points.append(p2)
            else:
                # Direct line is blocked - add a midpoint and recursively check
                midpoint = (p1 + p2) / 2.0
                
                # Check p1 to midpoint
                mid_i, mid_j = self.env.position_to_pixel_indices(float(midpoint[0]), float(midpoint[1]), self.env.configuration_space.shape)
                rr_mid, cc_mid = line(source_i, source_j, mid_i, mid_j)
                
                if (1 - self.env.configuration_space_thin[rr_mid, cc_mid]).sum() == 0:
                    # p1 to midpoint is clear
                    new_points.append(midpoint)
                else:
                    # p1 to midpoint is blocked - project midpoint outward
                    closest_indices = self.env.closest_valid_cspace_indices(mid_i, mid_j)
                    midpoint_adjusted = self.env.pixel_indices_to_position(
                        closest_indices[0], closest_indices[1], self.env.configuration_space.shape
                    )
                    midpoint = np.array([midpoint_adjusted[0], midpoint_adjusted[1]], dtype=np.float32)
                    new_points.append(midpoint)
                
                # Check midpoint to p2
                mid_i, mid_j = self.env.position_to_pixel_indices(float(midpoint[0]), float(midpoint[1]), self.env.configuration_space.shape)
                rr_mid, cc_mid = line(mid_i, mid_j, target_i, target_j)
                
                if (1 - self.env.configuration_space_thin[rr_mid, cc_mid]).sum() == 0:
                    # midpoint to p2 is clear
                    new_points.append(p2)
                else:
                    # midpoint to p2 is blocked - project p2 outward
                    closest_indices = self.env.closest_valid_cspace_indices(target_i, target_j)
                    p2_adjusted = self.env.pixel_indices_to_position(
                        closest_indices[0], closest_indices[1], self.env.configuration_space.shape
                    )
                    new_points.append(np.array([p2_adjusted[0], p2_adjusted[1]], dtype=np.float32))

        return np.asarray(new_points, dtype=np.float32)

    def _prune_xy_by_distance(self, path_xy: np.ndarray, min_dist: Optional[float] = None) -> np.ndarray:
        if path_xy.shape[0] <= 2:
            return path_xy
        if min_dist is None:
            min_dist = self.min_waypoint_spacing

        pruned = [path_xy[0]]
        prev = path_xy[0]
        final = path_xy[-1]

        for i in range(1, path_xy.shape[0] - 1):
            point = path_xy[i]
            if np.linalg.norm(point - prev) >= min_dist and np.linalg.norm(point - final) >= min_dist:
                pruned.append(point)
                prev = point

        pruned.append(path_xy[-1])
        return np.asarray(pruned, dtype=np.float32)

    def _apply_feasibility_pipeline(self, path: np.ndarray) -> np.ndarray:
        if path is None or len(path) == 0:
            return path

        path_xy = path[:, :2].astype(np.float32)
        path_xy = self._ensure_path_feasibility_xy(path_xy)
        path_xy = self._prune_xy_by_distance(path_xy)

        if path_xy.shape[0] == 1:
            path_xy = np.vstack([path_xy, path_xy])

        init_heading = float(path[0, 2]) if path.shape[1] >= 3 else 0.0
        return self.get_path_headings(path_xy, initial_heading=init_heading)
