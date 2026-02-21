from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

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
        self.max_waypoints_per_step = 8

        # Geometric standoff so robot center stays behind the box while pushing.
        box_half_extent = float(self.cfg.boxes.box_size) / 2.0
        push_standoff = float(self.env.robot_radius) + box_half_extent + 0.05

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
        if len(box_positions) == 0:
            return [robot_pos, receptacle_pos]

        target_box = self._select_target_box(robot_pos, box_positions, receptacle_pos)
        pre_push_stance = self._compute_pre_push_stance(
            target_box,
            receptacle_pos,
            self.planner.pre_push_distance,
        )

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

                # Replan only when the current path queue is exhausted.
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

        path = []
        prev_heading = float(current_heading)
        for idx, (x, y) in enumerate(pruned_xy):
            if idx < len(pruned_xy) - 1:
                nx, ny = pruned_xy[idx + 1]
                dx, dy = nx - x, ny - y
                if abs(dx) + abs(dy) > 1e-8:
                    # Keep headings in [-pi, pi] to match environment comparisons.
                    prev_heading = math.atan2(dy, dx)
            path.append([float(x), float(y), float(prev_heading)])

        if len(path) == 1:
            path.append(path[0])

        return np.asarray(path, dtype=np.float32)

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
