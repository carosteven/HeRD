# Greedy Heuristic Policy - How It Works

This document explains the logic and implementation of the greedy heuristic baseline policy for the box delivery task.

---

## Overview

The greedy heuristic policy is a **classical (non-ML) baseline** that uses A* pathfinding and a finite-state machine (FSM) to navigate a robot to collect boxes and deliver them to a receptacle. Unlike the learned diffusion policy, this approach uses traditional planning algorithms and geometric reasoning.

### Key Components

1. **GreedyHeuristicPlanner** - Low-level path planner using A* search
2. **GreedyHeuristicPolicy** - High-level FSM controller that orchestrates planning and execution

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 GreedyHeuristicPolicy                   │
│                                                         │
│  ┌─────────────┐         ┌─────────────┐                │
│  │   Transit   │ ◄─────► │    Push     │                │
│  │    Mode     │         │    Mode     │                │
│  └─────────────┘         └─────────────┘                │
│                                                         │
│  Uses: GreedyHeuristicPlanner (A* search)               │
└─────────────────────────────────────────────────────────┘
```

---

## The Finite-State Machine (FSM)

The policy operates in two distinct modes:

### 🚶 **Transit Mode**
**Purpose:** Navigate to a box without pushing anything

**When active:**
- Robot is far from target box (> 0.35m)
- No active push in progress

**Behavior:**
- Use A* to plan path to "pre-push stance" (position behind box)
- Avoid obstacles and other boxes
- Once close enough to target box, switch to Push mode

### 📦 **Push Mode**  
**Purpose:** Push the target box toward the receptacle

**When active:**
- Robot is close to target box (≤ 0.35m)
- Maintains contact while pushing

**Behavior:**
- Generate straight-line push path: robot → box → receptacle
- Maintain fixed standoff distance behind box (robot center stays behind box)
- Monitor box movement to detect stalled pushes
- Exit when:
  - Box gets too far from robot (> 0.75m) - lost contact
  - Box hasn't moved for 5 consecutive steps - stuck
  - Minimum push duration met (≥ 3 steps)

### Hysteresis

The FSM uses **hysteresis** to prevent rapid mode flapping:
- Enter push: distance ≤ 0.35m
- Exit push: distance ≥ 0.75m OR box stalled OR target lost

This creates a "dead band" where the robot stays in push mode even if distance varies slightly.

---

## Path Planning Details

### Transit Planning (A* Search)

**Goal:** Find obstacle-avoiding path to pre-push stance

**Steps:**
1. **Select target box** - Choose nearest box using greedy heuristic: `distance(robot, box) + distance(box, receptacle)`

2. **Compute pre-push stance** - Position robot should reach before pushing:
   ```
   push_direction = (box_pos - receptacle_pos) / ||box_pos - receptacle_pos||
   pre_push_stance = box_pos + push_direction * standoff_distance
   ```
   This places robot **behind** the box relative to push direction.

3. **Build cost map:**
   - Obstacles: infinite cost
   - Free space: base cost = 1.0
   - Near other boxes: add radial penalty (expensive but traversable)
   
4. **Run A* from robot to pre-push stance** using cost map

5. **Convert grid path to world coordinates**

### Push Planning (Straight Line)

**Goal:** Generate path that maintains robot-box-receptacle alignment

**Calculation:**
```python
push_direction = (receptacle_pos - box_pos) / ||receptacle_pos - box_pos||
push_start = box_pos - push_direction * standoff
push_end = receptacle_pos - push_direction * standoff

path = [robot_pos → push_start → push_end]
```

This keeps robot at fixed offset behind box throughout push.

---

## Box Movement Detection

To prevent the robot from "pushing air" when it misses a box:

### Tracking Logic
```python
if mode == "push":
    box_displacement = distance(prev_box_pos, current_box_pos)
    
    if box_displacement < 0.01m:  # Movement threshold
        box_not_moving_steps += 1
    else:
        box_not_moving_steps = 0
    
    if box_not_moving_steps >= 5:
        # Box is stuck - exit push mode
        mode = "transit"
```

### Why This Matters
Without movement detection:
- Robot could continue "pushing" after missing box
- Wastes time following a stale push trajectory
- Box never reaches receptacle

With movement detection:
- If box doesn't move despite robot pushing, exit push mode
- Replan from transit mode to recover
- Improves robustness to planning errors

---

## Path Feasibility Pipeline

Both transit and push paths undergo the same post-processing:

### 1. Feasibility Check (`_ensure_path_feasibility_xy`)

**Problem:** Planner might generate paths with segments that pass through obstacles

**Solution:**
- Check each path segment using line-of-sight in configuration space
- If segment is blocked:
  - Add midpoint between endpoints
  - Recursively check sub-segments
  - Project blocked points to nearest valid space using `closest_valid_cspace_indices()`

### 2. Distance Pruning (`_prune_xy_by_distance`)

**Problem:** Too many waypoints close together

**Solution:**
- Remove waypoints closer than `min_waypoint_spacing` (0.2m)
- Always keep first and last waypoints

### 3. Heading Calculation (`get_path_headings`)

**Problem:** Controller needs (x, y, θ) waypoints

**Solution:**
- Compute heading for each waypoint as direction to next waypoint:
  ```python
  heading[i] = atan2(y[i+1] - y[i], x[i+1] - x[i])
  ```
- First waypoint uses current robot heading

---

## Execution Loop

Each timestep, the policy:

1. **Get current state:**
   - Robot pose (x, y, θ)
   - Box positions (from vertices observations)
   - Receptacle position
   - Obstacle map

2. **Track target box:**
   - If no target: select greedily
   - If have target: find nearest box to last target position
   - If target lost (>1.0m away): reselect greedily

3. **Update FSM mode:**
   - Check distance to target box
   - Apply hysteresis thresholds
   - Update push mode step counter
   - Check box movement (in push mode)

4. **Generate path:**
   - **Push mode:** Recompute push path from current positions every step
   - **Transit mode:** Use queued path; replan when queue depleted

5. **Chunk path:**
   - Take up to 10 waypoints from path
   - Convert to (x, y, θ) format
   - Anchor first waypoint at current robot position

6. **Apply feasibility pipeline:**
   - Ensure no collisions
   - Prune dense waypoints
   - Calculate headings

7. **Send to controller:**
   - Low-level MPC controller tracks waypoints
   - Return action, reward, done

8. **Update queue:**
   - Remove consumed waypoints
   - Keep 1 waypoint overlap for continuity

---

## Key Parameters

### FSM Thresholds
- `push_engage_distance = 0.35m` - Enter push mode
- `push_release_distance = 0.75m` - Exit push mode
- `push_mode_min_steps = 3` - Minimum duration in push
- `box_movement_threshold = 0.01m` - Box displacement to count as "moving"
- `target_track_lost_distance = 1.0m` - Max distance to track same box

### Geometric Constants
- `push_standoff = robot_radius + box_half_width + 0.05m` - Robot offset behind box
- `min_waypoint_spacing = 0.2m` - Minimum distance between waypoints
- `push_step = 0.12m` - Interpolation resolution for push paths

### Planning Parameters
- `clearance_penalty = 50.0` - Cost multiplier near non-target boxes
- `clearance_radius = 0.7m` - Radius of penalty zone around boxes
- `allow_diagonal = True` - Use 8-connected grid for A*

---

## Edge Cases Handled

### 1. **Target Box Lost**
- If tracked box moves >1.0m away, reselect target greedily
- Invalidate queued path and replan

### 2. **Box Not Moving (Stalled Push)**
- Count consecutive steps where box displacement < 0.01m
- If count ≥ 5, exit push mode and replan from transit

### 3. **Degenerate Planner Output**
- If A* returns path with <2 waypoints or <0.35m total length
- Fall back to `shortest_path()` directly to pre-push stance

### 4. **No Boxes Remaining**
- Set `current_target_box = None`
- Switch to transit mode
- Plan direct path to receptacle

### 5. **Blocked Start/Goal in A***
- Use BFS to find nearest free cell in cost map
- Start/end A* search from/to nearest valid location

---

## Differences from Diffusion Policy

| Aspect | Greedy Heuristic | Diffusion Policy |
|--------|------------------|------------------|
| **Planning** | Explicit A* search | Learned trajectory distribution |
| **Coordination** | Geometric pre-push stance | Learned approach behaviors |
| **Robustness** | FSM with explicit guards | Implicit from training data |
| **Computation** | Replans every step or when queue empty | Single forward pass per step |
| **Interpretability** | Fully transparent logic | Black-box neural network |
| **Adaptability** | Fixed geometric rules | Can generalize to new scenarios |

---

## Summary

The greedy heuristic policy is a classical baseline that:
1. **Selects** the nearest box using a simple distance heuristic
2. **Plans** collision-free paths using A* search with clearance penalties
3. **Executes** using a 2-mode FSM (transit vs. push)
4. **Monitors** box movement to detect and recover from failed pushes
5. **Processes** all paths through feasibility checks for consistency

It provides a strong non-learned baseline for comparison with learned policies like diffusion, demonstrating what can be achieved with traditional planning alone.
