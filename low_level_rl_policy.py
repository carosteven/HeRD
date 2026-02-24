import os
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment import (
    BOX_SEG_INDEX,
    MAX_SEG_INDEX,
    OBSTACLE_SEG_INDEX,
    WAYPOINT_MOVING_THRESHOLD,
    BoxDeliveryEnv,
)


class LowLevelNavEnv(gym.Env):
    """
    Low-level navigation environment for HRL pre-training.

    Observation: 4-channel image
      1. Semantic segmentation (same as high-level)
      2. Robot mask (same as high-level)
      3. Distance transform to robot (same as high-level)
      4. Distance transform to subgoal (x_g, y_g)

    Action: Discrete(8) => 8 compass moves
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        cfg: Optional[Dict] = None,
        max_steps: int = 128,
        step_size_pixels: int = 1,
        goal_threshold_m: float = WAYPOINT_MOVING_THRESHOLD,
        min_start_goal_distance_m: float = 1.0,
    ):
        super().__init__()

        self.base_env = BoxDeliveryEnv(cfg=cfg)
        self.max_steps = max_steps
        self.goal_threshold_m = goal_threshold_m
        self.min_start_goal_distance_m = min_start_goal_distance_m

        self.step_size_m = step_size_pixels / self.base_env.local_map_pixels_per_meter

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.base_env.local_map_pixel_width,
                self.base_env.local_map_pixel_width,
                4,
            ),
            dtype=np.uint8,
        )

        self._directions = np.array(
            [
                [0, 1],    # N
                [1, 1],    # NE
                [1, 0],    # E
                [1, -1],   # SE
                [0, -1],   # S
                [-1, -1],  # SW
                [-1, 0],   # W
                [-1, 1],   # NW
            ],
            dtype=np.float32,
        )

        self._episode_steps = 0
        self._prev_dist = 0.0
        self._goal_position = np.zeros(2, dtype=np.float32)

    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1[:2] - p2[:2]))

    def _robot_position(self) -> np.ndarray:
        return np.array(
            [self.base_env.robot.body.position.x, self.base_env.robot.body.position.y],
            dtype=np.float32,
        )

    def _is_blocked_by_semantics(self, position_xy: np.ndarray, include_boxes: bool = True) -> bool:
        pixel_i, pixel_j = self.base_env.position_to_pixel_indices(
            position_xy[0],
            position_xy[1],
            self.base_env.global_overhead_map.shape,
        )
        value = self.base_env.global_overhead_map[pixel_i, pixel_j]

        obstacle_value = OBSTACLE_SEG_INDEX / MAX_SEG_INDEX
        if include_boxes:
            box_value = BOX_SEG_INDEX / MAX_SEG_INDEX
            return bool(
                np.isclose(value, obstacle_value, atol=1e-6)
                or np.isclose(value, box_value, atol=1e-6)
            )
        return bool(np.isclose(value, obstacle_value, atol=1e-6))

    def _is_valid_position(self, position_xy: np.ndarray, include_boxes: bool = True) -> bool:
        i, j = self.base_env.position_to_pixel_indices(
            position_xy[0],
            position_xy[1],
            self.base_env.configuration_space.shape,
        )
        if self.base_env.configuration_space[i, j] <= 0.5:
            return False

        self.base_env.update_global_overhead_map()
        if self._is_blocked_by_semantics(position_xy, include_boxes=include_boxes):
            return False

        return True

    def _sample_valid_position(self) -> np.ndarray:
        free_indices = np.argwhere(self.base_env.configuration_space > 0.5)
        if len(free_indices) == 0:
            raise RuntimeError("No free cells available in configuration space.")

        for _ in range(1000):
            k = self.np_random.integers(0, len(free_indices))
            i, j = free_indices[k]
            x, y = self.base_env.pixel_indices_to_position(
                int(i),
                int(j),
                self.base_env.configuration_space.shape,
            )
            candidate = np.array([x, y], dtype=np.float32)
            if self._is_valid_position(candidate, include_boxes=True):
                return candidate

        raise RuntimeError("Unable to sample valid position from free space.")

    def _create_global_shortest_path_to_subgoal_map(self) -> np.ndarray:
        return self.base_env.create_global_shortest_path_map(self._goal_position)

    def _build_observation(self) -> np.ndarray:
        self.base_env.update_global_overhead_map()

        robot_position = self.base_env.robot.body.position
        robot_heading = self.base_env.robot.body.angle

        channels = [
            self.base_env.get_local_map(
                self.base_env.global_overhead_map,
                robot_position,
                robot_heading,
            ),
            self.base_env.robot_state_channel,
            self.base_env.get_local_distance_map(
                self.base_env.create_global_shortest_path_map(robot_position),
                robot_position,
                robot_heading,
            ),
            self.base_env.get_local_distance_map(
                self._create_global_shortest_path_to_subgoal_map(),
                robot_position,
                robot_heading,
            ),
        ]

        obs = np.stack(channels, axis=2)
        return (obs * 255).astype(np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        obs_config = options.get("obs_config") if options else None
        self.base_env.reset(seed=seed, obs_config=obs_config)

        start_pos = options.get("start_pos") if options else None
        goal_pos = options.get("goal_pos") if options else None

        if start_pos is None:
            start = self._sample_valid_position()
        else:
            start = np.array(start_pos, dtype=np.float32)

        if goal_pos is None:
            goal = self._sample_valid_position()
            while self._distance(start, goal) < self.min_start_goal_distance_m:
                goal = self._sample_valid_position()
        else:
            goal = np.array(goal_pos, dtype=np.float32)

        self.base_env.robot.body.position = (float(start[0]), float(start[1]))
        self.base_env.robot.body.angle = 0.0

        self._goal_position = goal
        self.base_env.goal_point = (float(goal[0]), float(goal[1]))

        self._episode_steps = 0
        self._prev_dist = self._distance(self._robot_position(), self._goal_position)

        obs = self._build_observation()
        info = {
            "robot_pos": self._robot_position(),
            "goal_pos": self._goal_position.copy(),
            "distance_to_goal": self._prev_dist,
            "success": False,
            "collision": False,
        }
        return obs, info

    def step(self, action: int):
        self._episode_steps += 1

        terminated = False
        truncated = False
        collision = False
        success = False

        reward = -0.1

        action = int(action)
        current_pos = self._robot_position()
        current_heading = float(self.base_env.restrict_heading_range(self.base_env.robot.body.angle))

        direction = self._directions[action]
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction = direction / direction_norm
        target_heading = float(np.arctan2(direction[1], direction[0]))
        self.base_env.robot.body.angle = target_heading
        self.base_env.robot_hit_obstacle = False

        moved_total = 0.0
        max_internal_steps = 64
        speed = float(self.base_env.target_speed * 2.0)
        for _ in range(max_internal_steps):
            if moved_total >= self.step_size_m:
                break

            prev = self._robot_position()
            self.base_env.robot.body.angular_velocity = 0.0
            self.base_env.robot.body.velocity = (direction * speed).tolist()
            self.base_env.space.step(self.base_env.dt / self.base_env.steps)
            curr = self._robot_position()

            moved_step = float(np.linalg.norm(curr - prev))
            moved_total += moved_step

            if self.base_env.robot_hit_obstacle:
                break

            if moved_step < 1e-5:
                break

        self.base_env.robot.body.angular_velocity = 0.0
        self.base_env.robot.body.velocity = (0.0, 0.0)
        self.base_env.step_simulation_until_still()

        moved_vec = self._robot_position() - current_pos
        moved_norm = float(np.linalg.norm(moved_vec))
        if moved_norm > 1e-6:
            moved_heading = float(np.arctan2(moved_vec[1], moved_vec[0]))
            self.base_env.robot.body.angle = moved_heading

        if self.base_env.robot_hit_obstacle and moved_norm < 0.3 * self.step_size_m:
            collision = True
            reward -= 10.0
            terminated = True

        curr_dist = self._distance(self._robot_position(), self._goal_position)
        reward += self._prev_dist - curr_dist

        if curr_dist < self.goal_threshold_m:
            success = True
            reward += 50.0
            terminated = True

        if self._episode_steps >= self.max_steps and not terminated:
            truncated = True

        self._prev_dist = curr_dist

        obs = self._build_observation()
        info = {
            "robot_pos": self._robot_position(),
            "goal_pos": self._goal_position.copy(),
            "distance_to_goal": curr_dist,
            "success": success,
            "collision": collision,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        self.base_env.render()

    def close(self):
        self.base_env.close()


class SimpleCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        channels_first = observation_space.shape[0] <= 8
        n_input_channels = (
            observation_space.shape[0] if channels_first else observation_space.shape[-1]
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            if not channels_first:
                sample = sample.permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 4 and observations.shape[1] != 4 and observations.shape[-1] == 4:
            observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))


class SuccessThresholdCallback(BaseCallback):
    def __init__(
        self,
        eval_env: LowLevelNavEnv,
        eval_episodes: int,
        success_threshold: float,
        eval_freq: int,
        save_path: str,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_episodes = eval_episodes
        self.success_threshold = success_threshold
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.deterministic = deterministic
        self.reached_threshold = False
        self.last_success_rate = 0.0

    def _evaluate_success_rate(self) -> float:
        successes = 0
        for _ in range(self.eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            truncated = False
            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, _, done, truncated, info = self.eval_env.step(action)
            if info.get("success", False):
                successes += 1
        return successes / max(self.eval_episodes, 1)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        success_rate = self._evaluate_success_rate()
        self.last_success_rate = success_rate

        if self.verbose > 0:
            print(f"[low-level eval] steps={self.num_timesteps}, success_rate={success_rate:.3f}")

        if success_rate >= self.success_threshold:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self.model.save(self.save_path)
            self.reached_threshold = True
            if self.verbose > 0:
                print(f"Saved model to {self.save_path} (success_rate={success_rate:.3f})")
            return False

        return True


class LowLevelRLPolicy:
    """Reusable low-level RL policy wrapper for HRL (PPO or DQN)."""

    def __init__(
        self,
        algorithm: str = "ppo",
        device: str = "auto",
        policy_kwargs: Optional[Dict] = None,
    ):
        algorithm = algorithm.lower()
        if algorithm not in {"ppo", "dqn"}:
            raise ValueError("algorithm must be one of {'ppo', 'dqn'}")

        self.algorithm = algorithm
        self.device = device
        self.model = None

        if policy_kwargs is None:
            if self.algorithm == "ppo":
                policy_kwargs = dict(
                    features_extractor_class=SimpleCNNExtractor,
                    features_extractor_kwargs=dict(features_dim=256),
                    net_arch=dict(pi=[256, 128], vf=[256, 128]),
                )
            else:
                policy_kwargs = dict(
                    features_extractor_class=SimpleCNNExtractor,
                    features_extractor_kwargs=dict(features_dim=256),
                    net_arch=[256, 128],
                )

        self.policy_kwargs = policy_kwargs
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

    def _build_model(self, env: gym.Env, tensorboard_log: Optional[str] = None):
        if self.algorithm == "ppo":
            self.model = PPO(
                "CnnPolicy",
                env,
                policy_kwargs=self.policy_kwargs,
                n_steps=512,
                batch_size=128,
                n_epochs=10,
                learning_rate=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1,
                device=self.device,
                tensorboard_log=tensorboard_log,
            )
        else:
            self.model = DQN(
                "CnnPolicy",
                env,
                policy_kwargs=self.policy_kwargs,
                learning_rate=1e-4,
                buffer_size=100_000,
                learning_starts=10_000,
                batch_size=64,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=1_000,
                exploration_fraction=0.2,
                exploration_final_eps=0.05,
                verbose=1,
                device=self.device,
                tensorboard_log=tensorboard_log,
            )

    def train_until_success(
        self,
        train_env: LowLevelNavEnv,
        eval_env: LowLevelNavEnv,
        total_timesteps: int,
        save_path: str,
        success_threshold: float = 0.90,
        eval_freq: int = 10_000,
        eval_episodes: int = 100,
        tensorboard_log: Optional[str] = None,
        resume_from: Optional[str] = None,
        reset_num_timesteps: bool = False,
        checkpoint_freq: int = 0,
    ) -> Tuple[bool, float]:
        if resume_from is not None:
            self.load(resume_from, env=train_env)
            
            # Fix DQN exploration schedule on resume by setting exploration_rate
            # to what it should be based on num_timesteps already trained
            if self.algorithm == "dqn":
                current_timesteps = self.model.num_timesteps
                exploration_fraction = 0.2  # Must match _build_model parameter
                exploration_final_eps = 0.05  # Must match _build_model parameter
                exploration_initial_eps = 1.0
                
                # Calculate total exploration steps across ALL training
                # (not just this learn() call, which is the SB3 bug)
                total_exploration_steps = int(exploration_fraction * total_timesteps)
                
                # Set epsilon based on where we are in global training progress
                if current_timesteps >= total_exploration_steps:
                    # Already past exploration phase, use final epsilon
                    self.model.exploration_rate = exploration_final_eps
                else:
                    # Linearly interpolate based on global progress
                    progress = current_timesteps / total_exploration_steps
                    self.model.exploration_rate = (
                        exploration_initial_eps - (exploration_initial_eps - exploration_final_eps) * progress
                    )
                print(f"Resuming DQN from {current_timesteps} steps with epsilon={self.model.exploration_rate:.4f}")
                
        elif self.model is None:
            self._build_model(train_env, tensorboard_log=tensorboard_log)

        # Success threshold callback for early stopping
        success_callback = SuccessThresholdCallback(
            eval_env=eval_env,
            eval_episodes=eval_episodes,
            success_threshold=success_threshold,
            eval_freq=eval_freq,
            save_path=save_path,
        )
        
        # Combine callbacks
        callbacks = [success_callback]
        
        # Add checkpoint callback if requested
        if checkpoint_freq > 0:
            checkpoint_dir = os.path.join(os.path.dirname(save_path), "checkpoints")
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=checkpoint_dir,
                name_prefix="ckpt",
                save_replay_buffer=True,  # Important for DQN!
                save_vecnormalize=True,
            )
            callbacks.append(checkpoint_callback)
            print(f"Saving checkpoints every {checkpoint_freq} steps to {checkpoint_dir}")
        
        callback_list = CallbackList(callbacks)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            reset_num_timesteps=reset_num_timesteps,
        )

        if not success_callback.reached_threshold:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model.save(save_path)

        return success_callback.reached_threshold, success_callback.last_success_rate

    def load(self, model_path: str, env: Optional[gym.Env] = None):
        if self.algorithm == "ppo":
            self.model = PPO.load(model_path, env=env, device=self.device)
        else:
            self.model = DQN.load(model_path, env=env, device=self.device)

    def save(self, model_path: str):
        if self.model is None:
            raise RuntimeError("No model to save. Train or load a model first.")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)

    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        if self.model is None:
            raise RuntimeError("No model loaded. Train or load model before calling act().")
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)

    def build_observation_for_goal(
        self,
        env: BoxDeliveryEnv,
        robot_position: np.ndarray,
        robot_heading: float,
        goal_position: np.ndarray,
    ) -> np.ndarray:
        env.update_global_overhead_map()
        channels = [
            env.get_local_map(env.global_overhead_map, robot_position, robot_heading),
            env.robot_state_channel,
            env.get_local_distance_map(
                env.create_global_shortest_path_map(robot_position),
                robot_position,
                robot_heading,
            ),
            env.get_local_distance_map(
                env.create_global_shortest_path_map(goal_position),
                robot_position,
                robot_heading,
            ),
        ]
        obs = np.stack(channels, axis=2)
        return (obs * 255).astype(np.uint8)

    def rollout_subgoal_path(
        self,
        env: BoxDeliveryEnv,
        start_position: np.ndarray,
        start_heading: float,
        goal_position: np.ndarray,
        max_steps: int = 64,
        goal_threshold_m: float = 0.2,
        step_size_pixels: int = 1,
        deterministic: bool = True,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("No model loaded. Train or load model before rollout.")

        step_size_m = step_size_pixels / env.local_map_pixels_per_meter
        robot_position = np.array(start_position[:2], dtype=np.float32)
        robot_heading = float(start_heading)
        goal_position = np.array(goal_position[:2], dtype=np.float32)

        path = [robot_position.copy()]

        for _ in range(max_steps):
            if np.linalg.norm(robot_position - goal_position) < goal_threshold_m:
                break

            obs = self.build_observation_for_goal(
                env=env,
                robot_position=robot_position,
                robot_heading=robot_heading,
                goal_position=goal_position,
            )
            action = self.act(obs, deterministic=deterministic)

            direction = self._directions[int(action)]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm

            candidate = robot_position + direction * step_size_m

            i, j = env.position_to_pixel_indices(
                candidate[0],
                candidate[1],
                env.configuration_space.shape,
            )

            if env.configuration_space[i, j] <= 0.5:
                break

            robot_position = candidate
            path.append(robot_position.copy())

        path.append(goal_position.copy())

        path_with_heading = []
        prev = path[0]
        prev_heading = robot_heading
        for idx, point in enumerate(path):
            if idx == 0:
                heading = prev_heading
            else:
                delta = point - prev
                if np.linalg.norm(delta) > 1e-6:
                    heading = float(np.arctan2(delta[1], delta[0]))
                    prev_heading = heading
                else:
                    heading = prev_heading
            path_with_heading.append([float(point[0]), float(point[1]), float(heading)])
            prev = point

        return np.asarray(path_with_heading, dtype=np.float32)
