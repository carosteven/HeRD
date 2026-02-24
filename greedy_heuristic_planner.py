from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np


Coordinate = Tuple[float, float]
GridIndex = Tuple[int, int]


class GreedyHeuristicPlanner:
    """
    Purely classical heuristic TAMP baseline planner (no machine learning).

    Finite-state behavior:
      - State A (Transit): A* to pre-push stance using clearance-aware cost map.
      - State B (Pushing): Straight-line path through target box to receptacle.

    Inputs are world coordinates, and `obstacle_map` is a 2D binary occupancy grid.
    Returned path is a list of `(x, y)` waypoints for direct loop integration.
    """

    def __init__(
        self,
        pre_push_distance: float = 0.5,
        stance_tolerance: float = 0.15,
        obstacle_value: int = 1,
        grid_resolution: float = 1.0,
        clearance_penalty: float = 30.0,
        clearance_radius: float = 0.6,
        push_step: float = 0.1,
        allow_diagonal: bool = True,
        world_to_grid_fn: Optional[Callable[[Coordinate, Tuple[int, int]], GridIndex]] = None,
        grid_to_world_fn: Optional[Callable[[GridIndex, Tuple[int, int]], Coordinate]] = None,
        world_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        self.pre_push_distance = float(pre_push_distance)
        self.stance_tolerance = float(stance_tolerance)
        self.obstacle_value = obstacle_value
        self.grid_resolution = float(grid_resolution)
        self.clearance_penalty = float(clearance_penalty)
        self.clearance_radius = float(clearance_radius)
        self.push_step = float(push_step)
        self.allow_diagonal = bool(allow_diagonal)
        self.world_to_grid_fn = world_to_grid_fn
        self.grid_to_world_fn = grid_to_world_fn
        self.world_bounds = world_bounds

        if self.grid_resolution <= 0:
            raise ValueError("grid_resolution must be > 0")
        if self.push_step <= 0:
            raise ValueError("push_step must be > 0")

    def plan(
        self,
        robot_pos: Sequence[float],
        box_positions: Sequence[Sequence[float]],
        receptacle_pos: Sequence[float],
        obstacle_map: np.ndarray,
    ) -> List[Coordinate]:
        """
        Generate `(x, y)` waypoints with greedy target selection and FSM planning.

        Args:
            robot_pos: Robot position `(x, y)`.
            box_positions: List of box positions.
            receptacle_pos: Receptacle position `(x, y)`.
            obstacle_map: 2D binary occupancy grid.

        Returns:
            List of `(x, y)` waypoints.
        """
        if obstacle_map.ndim != 2:
            raise ValueError("obstacle_map must be a 2D array")
        if len(box_positions) == 0:
            return [self._as_xy(robot_pos), self._as_xy(receptacle_pos)]

        robot_xy = self._as_xy(robot_pos)
        receptacle_xy = self._as_xy(receptacle_pos)
        boxes_xy = [self._as_xy(p) for p in box_positions]

        robot_xy = self._clip_world(robot_xy)
        receptacle_xy = self._clip_world(receptacle_xy)
        boxes_xy = [self._clip_world(p) for p in boxes_xy]

        target_idx = self._select_target_box(robot_xy, boxes_xy, receptacle_xy)
        target_box = boxes_xy[target_idx]

        pre_push_stance = self._compute_pre_push_stance(target_box, receptacle_xy)
        pre_push_stance = self._clip_world(pre_push_stance)

        # State B: if robot is already at/near pre-push stance, push directly.
        if self._euclidean(robot_xy, pre_push_stance) <= self.stance_tolerance:
            return self._build_push_path(robot_xy, target_box, receptacle_xy)

        # State A: transit to pre-push stance using clearance-aware A*.
        cost_map = self._build_cost_map(obstacle_map, boxes_xy, target_idx)
        start = self._nearest_free(self._world_to_grid(robot_xy, obstacle_map.shape), cost_map)
        goal = self._nearest_free(self._world_to_grid(pre_push_stance, obstacle_map.shape), cost_map)

        if start is None or goal is None:
            return [robot_xy]

        grid_path = self._astar(start, goal, cost_map)
        if not grid_path:
            return [robot_xy]

        waypoints = [self._grid_to_world(node, obstacle_map.shape) for node in grid_path]
        waypoints[0] = robot_xy
        waypoints[-1] = pre_push_stance
        return [self._clip_world(p) for p in waypoints]

    def _select_target_box(
        self,
        robot_pos: Coordinate,
        box_positions: Sequence[Coordinate],
        receptacle_pos: Coordinate,
    ) -> int:
        best_idx = 0
        best_score = float("inf")
        for idx, box in enumerate(box_positions):
            score = self._euclidean(robot_pos, box) + self._euclidean(box, receptacle_pos)
            if score < best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _compute_pre_push_stance(
        self,
        target_box: Coordinate,
        receptacle_pos: Coordinate,
    ) -> Coordinate:
        # Vector points from receptacle -> box, so stance is behind box wrt push direction.
        dx = target_box[0] - receptacle_pos[0]
        dy = target_box[1] - receptacle_pos[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-8:
            return target_box

        ux, uy = dx / norm, dy / norm
        return (
            target_box[0] + ux * self.pre_push_distance,
            target_box[1] + uy * self.pre_push_distance,
        )

    def _build_cost_map(
        self,
        obstacle_map: np.ndarray,
        box_positions: Sequence[Coordinate],
        target_idx: int,
    ) -> np.ndarray:
        # Base traversal cost (free cells)
        cost_map = np.ones_like(obstacle_map, dtype=np.float32)

        obstacle_mask = obstacle_map == self.obstacle_value
        cost_map[obstacle_mask] = np.inf

        radius_cells = max(1, int(round(self.clearance_radius / self.grid_resolution)))
        for idx, box in enumerate(box_positions):
            if idx == target_idx:
                continue
            center = self._world_to_grid(box, obstacle_map.shape)
            self._apply_radial_penalty(cost_map, center, radius_cells)

        return cost_map

    def _apply_radial_penalty(
        self,
        cost_map: np.ndarray,
        center: GridIndex,
        radius_cells: int,
    ) -> None:
        ci, cj = center
        h, w = cost_map.shape
        i0, i1 = max(0, ci - radius_cells), min(h - 1, ci + radius_cells)
        j0, j1 = max(0, cj - radius_cells), min(w - 1, cj + radius_cells)

        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                if not np.isfinite(cost_map[i, j]):
                    continue
                dist = math.hypot(i - ci, j - cj)
                if dist > radius_cells:
                    continue
                scaled = 1.0 - (dist / max(1.0, radius_cells))
                cost_map[i, j] += self.clearance_penalty * scaled

    def _astar(
        self,
        start: GridIndex,
        goal: GridIndex,
        cost_map: np.ndarray,
    ) -> List[GridIndex]:
        if start == goal:
            return [start]

        h, w = cost_map.shape
        g_score = np.full((h, w), np.inf, dtype=np.float32)
        parent_i = np.full((h, w), -1, dtype=np.int32)
        parent_j = np.full((h, w), -1, dtype=np.int32)
        closed = np.zeros((h, w), dtype=np.uint8)

        pq: List[Tuple[float, float, int, int]] = []
        si, sj = start
        gi, gj = goal

        g_score[si, sj] = 0.0
        start_h = self._heuristic(si, sj, gi, gj)
        heapq.heappush(pq, (start_h, 0.0, si, sj))

        neighbors = self._neighbors_8 if self.allow_diagonal else self._neighbors_4

        while pq:
            _, current_g, i, j = heapq.heappop(pq)
            if closed[i, j]:
                continue
            closed[i, j] = 1

            if i == gi and j == gj:
                return self._reconstruct_path((i, j), (si, sj), parent_i, parent_j)

            if current_g > g_score[i, j]:
                continue

            for ni, nj, step_dist in neighbors(i, j, h, w):
                if closed[ni, nj] or not np.isfinite(cost_map[ni, nj]):
                    continue

                tentative_g = g_score[i, j] + step_dist * float(cost_map[ni, nj])
                if tentative_g < g_score[ni, nj]:
                    g_score[ni, nj] = tentative_g
                    parent_i[ni, nj] = i
                    parent_j[ni, nj] = j
                    f = tentative_g + self._heuristic(ni, nj, gi, gj)
                    heapq.heappush(pq, (f, tentative_g, ni, nj))

        return []

    @staticmethod
    def _neighbors_4(i: int, j: int, h: int, w: int) -> Iterable[Tuple[int, int, float]]:
        if i > 0:
            yield i - 1, j, 1.0
        if i + 1 < h:
            yield i + 1, j, 1.0
        if j > 0:
            yield i, j - 1, 1.0
        if j + 1 < w:
            yield i, j + 1, 1.0

    @staticmethod
    def _neighbors_8(i: int, j: int, h: int, w: int) -> Iterable[Tuple[int, int, float]]:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    step = 1.41421356237 if di != 0 and dj != 0 else 1.0
                    yield ni, nj, step

    @staticmethod
    def _heuristic(i: int, j: int, gi: int, gj: int) -> float:
        # Euclidean admissible heuristic for weighted traversal costs >= 1.
        return math.hypot(gi - i, gj - j)

    @staticmethod
    def _reconstruct_path(
        end: GridIndex,
        start: GridIndex,
        parent_i: np.ndarray,
        parent_j: np.ndarray,
    ) -> List[GridIndex]:
        path: List[GridIndex] = [end]
        i, j = end
        while (i, j) != start:
            pi = int(parent_i[i, j])
            pj = int(parent_j[i, j])
            if pi < 0 or pj < 0:
                return []
            i, j = pi, pj
            path.append((i, j))
        path.reverse()
        return path

    def _nearest_free(self, seed: GridIndex, cost_map: np.ndarray) -> Optional[GridIndex]:
        si, sj = seed
        h, w = cost_map.shape
        if not (0 <= si < h and 0 <= sj < w):
            return None
        if np.isfinite(cost_map[si, sj]):
            return seed

        visited = np.zeros((h, w), dtype=np.uint8)
        q = deque([(si, sj)])
        visited[si, sj] = 1

        while q:
            i, j = q.popleft()
            for ni, nj, _ in self._neighbors_4(i, j, h, w):
                if visited[ni, nj]:
                    continue
                if np.isfinite(cost_map[ni, nj]):
                    return (ni, nj)
                visited[ni, nj] = 1
                q.append((ni, nj))

        return None

    def _build_push_path(
        self,
        robot_pos: Coordinate,
        target_box: Coordinate,
        receptacle_pos: Coordinate,
    ) -> List[Coordinate]:
        # Robot-center push trajectory: stay at a fixed standoff behind the box
        # along the push direction (box -> receptacle).
        push_dx = receptacle_pos[0] - target_box[0]
        push_dy = receptacle_pos[1] - target_box[1]
        push_norm = math.hypot(push_dx, push_dy)
        if push_norm < 1e-8:
            return [self._clip_world(robot_pos), self._clip_world(target_box)]

        ux = push_dx / push_norm
        uy = push_dy / push_norm
        standoff = self.pre_push_distance

        push_start = (target_box[0] - ux * standoff, target_box[1] - uy * standoff)
        push_end = (receptacle_pos[0] - ux * standoff, receptacle_pos[1] - uy * standoff)

        push_start = self._clip_world(push_start)
        push_end = self._clip_world(push_end)

        segment_1 = self._interpolate_line(robot_pos, push_start, self.push_step)
        segment_2 = self._interpolate_line(push_start, push_end, self.push_step)

        if segment_1 and segment_2 and segment_1[-1] == segment_2[0]:
            return [self._clip_world(p) for p in (segment_1 + segment_2[1:])]
        return [self._clip_world(p) for p in (segment_1 + segment_2)]

    @staticmethod
    def _interpolate_line(p0: Coordinate, p1: Coordinate, step: float) -> List[Coordinate]:
        dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        n = max(1, int(math.ceil(dist / step)))
        points: List[Coordinate] = []
        for k in range(n + 1):
            t = k / n
            points.append((p0[0] * (1 - t) + p1[0] * t, p0[1] * (1 - t) + p1[1] * t))
        return points

    def _world_to_grid(self, pos: Coordinate, shape: Tuple[int, int]) -> GridIndex:
        if self.world_to_grid_fn is not None:
            return self.world_to_grid_fn(pos, shape)
        h, w = shape
        i = int(round(pos[1] / self.grid_resolution))
        j = int(round(pos[0] / self.grid_resolution))
        i = max(0, min(h - 1, i))
        j = max(0, min(w - 1, j))
        return (i, j)

    def _grid_to_world(self, idx: GridIndex, shape: Tuple[int, int]) -> Coordinate:
        if self.grid_to_world_fn is not None:
            return self.grid_to_world_fn(idx, shape)
        _ = shape
        i, j = idx
        x = j * self.grid_resolution
        y = i * self.grid_resolution
        return (float(x), float(y))

    @staticmethod
    def _euclidean(p0: Coordinate, p1: Coordinate) -> float:
        return math.hypot(p1[0] - p0[0], p1[1] - p0[1])

    @staticmethod
    def _as_xy(p: Sequence[float]) -> Coordinate:
        if len(p) < 2:
            raise ValueError("Expected coordinate with at least 2 values")
        return (float(p[0]), float(p[1]))

    def _clip_world(self, p: Coordinate) -> Coordinate:
        if self.world_bounds is None:
            return p
        min_x, max_x, min_y, max_y = self.world_bounds
        x = min(max(float(p[0]), min_x), max_x)
        y = min(max(float(p[1]), min_y), max_y)
        return (x, y)
