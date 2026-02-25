# Greedy Heuristic Policy: Current Behavior

This document reflects the **current implementation** of the greedy heuristic baseline in:

- `greedy_heuristic_policy.py`
- `greedy_heuristic_planner.py`

It is intentionally code-aligned (not conceptual-only).

---

## 1) High-level structure

The baseline has two layers:

1. **`GreedyHeuristicPlanner`**
   - A* transit planner in grid space
   - Straight-line push path generator in world space

2. **`GreedyHeuristicPolicy`**
   - Episode loop / FSM (`transit` vs `push`)
   - Target tracking and mode switching
   - Waypoint chunking + feasibility conditioning

---

## 2) Mode logic (FSM)

### Transit mode
- Purpose: move to a pre-push stance for a selected target box.
- Entry: default mode, or after push exits.
- Replan: when queued path is exhausted.

### Push mode
- Purpose: push target toward receptacle.
- Enter when robot-target distance ≤ `push_engage_distance`.
- Exit when either:
  - robot-target distance ≥ `push_release_distance` and at least `push_mode_min_steps` elapsed, or
  - tracked box displacement is below threshold for 5 consecutive steps.

### Box-motion stall check
- During push, displacement is measured between consecutive tracked target positions.
- If displacement < `box_movement_threshold` (0.01), increment stall counter.
- Stall counter ≥ 5 forces exit to transit.

---

## 3) Target selection (current)

There are two layers of selection:

1. **Reachable pre-filter (`_select_reachable_target_box`)**
   - For each candidate box, compute pre-push stance.
   - Query `shortest_path` on `configuration_space_thin`.
   - Score feasible candidates by:
     - transit route length + box→receptacle Euclidean distance.
   - Fallback to pure Euclidean greedy if no feasible route exists.

2. **Transit replan robustness (`_plan_transit_reachable_path`)**
   - Sort candidate boxes by Euclidean greedy score.
   - For each candidate, call planner transit with that single box.
   - Pick first non-degenerate path (`len >= 2` and length ≥ 0.35).
   - If none pass, fall back to best candidate/fallback path.

This is the main anti-spin stability improvement: avoid repeatedly committing to a bad transit target/path.

---

## 4) Transit planning details (`GreedyHeuristicPlanner.plan`)

Given robot, boxes, receptacle, and obstacle map:

1. Choose planner target box by Euclidean greedy score.
2. Compute pre-push stance behind target relative to receptacle.
3. Build cost map:
   - obstacles → `inf`
   - free cells → base cost 1
   - radial penalties around non-target boxes
4. Run A* from nearest free start to nearest free goal.
5. Convert grid path back to world waypoints.

### Important current behavior
- Transit obstacle map comes from **`configuration_space_thin`** in policy.
- Diagonal corner-cutting is disallowed in A*.
- Planner no longer force-overwrites final waypoint to exact pre-push stance; it keeps the routed grid-derived endpoint.

---

## 5) Push path details (current)

`_build_push_path` now constructs a strict 2-segment line:

- `robot -> target_box`
- `target_box -> receptacle`

with interpolation step `push_step`.

Note: this is generated as a straight path, but the policy still runs feasibility conditioning afterward, so post-processed commands may deviate if collision repair is needed.

---

## 6) Waypoint processing pipeline

Each control step:

1. Build/update `pending_path_xy` (push or transit).
2. Chunk up to `max_waypoints_per_step` points.
3. Convert XY chunk to pose waypoints `(x, y, heading)`.
4. Anchor first waypoint to current robot position.
5. Run feasibility pipeline.
6. Send resulting path to environment/controller.

### Short-path corner preservation

In `_xy_path_to_pose_path`:
- For path length ≤ 3, waypoints are preserved (no distance-pruning).
- This prevents dropping critical middle corners that can create blocked direct segments.

---

## 7) Feasibility conditioning (applied in both modes)

`_apply_feasibility_pipeline`:

1. `._ensure_path_feasibility_xy`
   - Segment LOS check against `configuration_space_thin`.
   - If blocked, attempt midpoint subdivision and projection via nearest valid c-space index.
2. `._prune_xy_by_distance`
   - Remove dense interior points.
3. Recompute headings with `get_path_headings`.

This keeps push/transit conditioning consistent.

---

## 8) Key parameters used by current implementation

- `min_waypoint_spacing = 0.2`
- `max_waypoints_per_step = 10`
- `push_step = 0.12`
- `target_track_lost_distance = 1.0`
- `push_engage_distance = robot_radius + box_half_extent + 0.35`
- `push_release_distance = robot_radius + box_half_extent + 0.75`
- `push_mode_min_steps = 3`
- `box_movement_threshold = 0.01`

Planner defaults used by policy construction:

- `stance_tolerance = 0.25`
- `clearance_penalty = 50.0`
- `clearance_radius = 0.7`
- `allow_diagonal = True` (with corner-cut guard)

---

## 9) Summary (current)

The current greedy heuristic baseline:

- Uses reachable-aware target selection.
- Replans transit with candidate screening to avoid degenerate paths.
- Uses A* transit on thin c-space with diagonal corner-cut protection.
- Uses straight robot→box→receptacle push generation.
- Applies one shared feasibility pipeline in both transit and push.
- Uses push stall detection to exit failed pushes.

This is the implementation state to use for comparisons and debugging.
