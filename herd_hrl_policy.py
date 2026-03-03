import os
import numpy as np

from herd_policy import HeRDPolicy
from low_level_rl_policy import LowLevelRLPolicy


class HeRDHRLPolicy(HeRDPolicy):
    """
    Hierarchical policy that keeps the original high-level HeRD policy untouched.

    Flow:
      1) High-level RL picks a spatial action (subgoal in local-map coordinates)
      2) Low-level RL executes one primitive action at a time toward that subgoal
      3) Environment is stepped once per low-level action (closed-loop)
    """

    def __init__(self, cfg, job_id=None):
        super().__init__(cfg=cfg, job_id=job_id)
        self.low_level_policy = self.create_low_level_policy()

        low_level_cfg = getattr(self.cfg, "low_level", {})
        if isinstance(low_level_cfg, dict):
            self.ll_max_steps = int(low_level_cfg.get("max_steps", 64))
            self.ll_goal_threshold = float(low_level_cfg.get("goal_threshold_m", 0.2))
            self.ll_step_size_pixels = int(low_level_cfg.get("step_size_pixels", 1))
            self.ll_deterministic = bool(low_level_cfg.get("deterministic", True))
        else:
            self.ll_max_steps = int(getattr(low_level_cfg, "max_steps", 64))
            self.ll_goal_threshold = float(getattr(low_level_cfg, "goal_threshold_m", 0.2))
            self.ll_step_size_pixels = int(getattr(low_level_cfg, "step_size_pixels", 1))
            self.ll_deterministic = bool(getattr(low_level_cfg, "deterministic", True))

        self._directions = np.array(
            [
                [0, 1],
                [1, 1],
                [1, 0],
                [1, -1],
                [0, -1],
                [-1, -1],
                [-1, 0],
                [-1, 1],
            ],
            dtype=np.float32,
        )

    def create_low_level_policy(self):
        low_level_cfg = getattr(self.cfg, "low_level", {})
        if isinstance(low_level_cfg, dict):
            algorithm = str(low_level_cfg.get("algorithm", "auto")).lower()
            model_path = low_level_cfg.get("model_path", "models/rl_models/low_level_hrl_policy")
        else:
            algorithm = str(getattr(low_level_cfg, "algorithm", "auto")).lower()
            model_path = getattr(low_level_cfg, "model_path", "models/rl_models/low_level_hrl_policy")

        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), model_path)

        if algorithm == "auto":
            load_errors = []
            for candidate_algorithm in ["ppo", "dqn"]:
                try:
                    candidate_policy = LowLevelRLPolicy(algorithm=candidate_algorithm, device=str(self.device))
                    candidate_policy.load(model_path)
                    print(f"Loaded low-level RL policy with auto-detected algorithm: {candidate_algorithm}")
                    return candidate_policy
                except Exception as exc:
                    load_errors.append(f"{candidate_algorithm}: {exc}")
            raise RuntimeError(
                "Failed to load low-level model with auto-detect from "
                f"{model_path}. Errors: {' | '.join(load_errors)}"
            )

        policy = LowLevelRLPolicy(algorithm=algorithm, device=str(self.device))
        policy.load(model_path)
        return policy

    def _spatial_action_to_goal(self, robot_pose, spatial_action):
        waypoints, _ = self.env.position_controller.get_waypoints_to_spatial_action(
            robot_pose[0:2],
            robot_pose[2],
            int(spatial_action),
        )
        goal_position = np.asarray(waypoints[-1][0:2], dtype=np.float32)
        return self.low_level_policy.project_to_configuration_space(self.env, goal_position)

    def _distance_to_goal(self, robot_pose, goal_position):
        return float(np.linalg.norm(np.asarray(robot_pose[:2], dtype=np.float32) - np.asarray(goal_position[:2], dtype=np.float32)))

    def _goal_reached(self, robot_pose, goal_position):
        return self._distance_to_goal(robot_pose, goal_position) <= self.ll_goal_threshold

    def _current_robot_pose(self):
        return (
            float(self.env.robot.body.position.x),
            float(self.env.robot.body.position.y),
            float(self.env.restrict_heading_range(self.env.robot.body.angle)),
        )

    def execute_subgoal(self, rl_obs, info, goal_position):
        """
        Executes closed-loop low-level control toward goal_position.
        One low-level action is chosen and executed per environment step.
        """
        total_reward = 0.0
        low_level_steps = 0
        done = False
        truncated = False
        goal_position = self.low_level_policy.project_to_configuration_space(self.env, goal_position)

        while low_level_steps < self.ll_max_steps:
            current_robot_pose = self._current_robot_pose()
            if low_level_steps > 0 and self._goal_reached(current_robot_pose, goal_position):
                break

            ll_obs = self.low_level_policy.build_observation_for_goal(
                env=self.env,
                robot_position=np.asarray(current_robot_pose[:2], dtype=np.float32),
                robot_heading=float(current_robot_pose[2]),
                goal_position=np.asarray(goal_position[:2], dtype=np.float32),
            )
            ll_action = self.low_level_policy.act(ll_obs, deterministic=self.ll_deterministic)
            step_result = self.low_level_policy.execute_primitive_action(
                env=self.env,
                action=ll_action,
                step_size_pixels=self.ll_step_size_pixels,
            )

            rl_obs, reward, done, truncated, info = self.env.step(
                step_result["path"],
                already_moved=True,
                robot_initial_pose=current_robot_pose,
            )
            total_reward += reward
            low_level_steps += 1

            if self._goal_reached(self._current_robot_pose(), goal_position):
                break

            if done or truncated:
                break

        return rl_obs, info, done, truncated, total_reward, low_level_steps

    def evaluate(self, num_eps):
        eps_rewards = []
        eps_steps = []
        eps_distance = []
        eps_boxes = []
        eps_avg_box_distance = []

        for eps_idx in range(num_eps):
            print("Progress: ", eps_idx, " / ", num_eps, " episodes", end='\r')
            rl_obs, info = self.env.reset()
            done = truncated = False
            ep_steps = 0
            ep_reward = 0.0

            while not (done or truncated):
                spatial_action, _ = self.rl_policy.predict(rl_obs)
                goal_position = self._spatial_action_to_goal(info["robot_pose"], spatial_action)

                # Visualize current high-level subgoal in the environment renderer.
                self.env.goal_point = (float(goal_position[0]), float(goal_position[1]))

                rl_obs, info, done, truncated, reward_sum, ll_steps = self.execute_subgoal(
                    rl_obs=rl_obs,
                    info=info,
                    goal_position=goal_position,
                )
                ep_reward += reward_sum
                ep_steps += ll_steps

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

        print("*" * 80)
        print(f"Average eps_steps: {eps_steps.mean():.2f} ± {eps_steps.std():.2f}")
        print(f"Average eps_rewards: {eps_rewards.mean():.2f} ± {eps_rewards.std():.2f}")
        print(f"Average eps_distance: {eps_distance.mean():.2f} ± {eps_distance.std():.2f}")
        print(f"Average eps_boxes: {eps_boxes.mean():.2f} ± {eps_boxes.std():.2f}")
        print(f"Average eps_avg_box_distance: {eps_avg_box_distance.mean():.2f} ± {eps_avg_box_distance.std():.2f}")
