import argparse
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from low_level_rl_policy import LowLevelNavEnv, LowLevelRLPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train low-level HRL navigation policy.")
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--success-threshold", type=float, default=0.90)

    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--step-size-pixels", type=int, default=1)
    parser.add_argument("--goal-threshold-m", type=float, default=0.2)
    parser.add_argument("--min-start-goal-distance-m", type=float, default=1.0)

    parser.add_argument("--obstacle-config", type=str, default="large_columns")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--save-path",
        type=str,
        default="models/rl_models/low_level_hrl_policy",
        help="Path without extension for SB3 model save.",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default="scripts/per_logs/low_level_hrl",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to an existing SB3 checkpoint/model (without or with .zip) to resume from.",
    )
    parser.add_argument(
        "--reset-num-timesteps",
        action="store_true",
        help="Reset internal SB3 timestep counter when resuming. Default keeps counting.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=0,
        help="Save checkpoint every N steps. 0 disables periodic checkpoints (default: 0).",
    )

    return parser.parse_args()


def make_env_cfg(obstacle_config: str, seed: int):
    return {
        "render": {
            "show": True,
            "show_obs": False,
        },
        "env": {
            "obstacle_config": obstacle_config,
        },
        "misc": {
            "random_seed": seed,
        },
        'train': { 
            'random_env': True,
        },
    }


def main():
    args = parse_args()

    cfg = make_env_cfg(args.obstacle_config, args.seed)

    train_env = LowLevelNavEnv(
        cfg=cfg,
        max_steps=args.max_steps,
        step_size_pixels=args.step_size_pixels,
        goal_threshold_m=args.goal_threshold_m,
        min_start_goal_distance_m=args.min_start_goal_distance_m,
    )

    eval_env = LowLevelNavEnv(
        cfg=cfg,
        max_steps=args.max_steps,
        step_size_pixels=args.step_size_pixels,
        goal_threshold_m=args.goal_threshold_m,
        min_start_goal_distance_m=args.min_start_goal_distance_m,
    )

    train_env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed + 1)

    policy = LowLevelRLPolicy(algorithm=args.algorithm, device=args.device)

    save_path = args.save_path
    if not os.path.isabs(save_path):
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), save_path)

    tensorboard_log = args.tensorboard_log
    if not os.path.isabs(tensorboard_log):
        tensorboard_log = os.path.join(os.path.dirname(os.path.dirname(__file__)), tensorboard_log)

    resume_from = args.resume_from
    if resume_from is not None and not os.path.isabs(resume_from):
        resume_from = os.path.join(os.path.dirname(os.path.dirname(__file__)), resume_from)

    reached_threshold, last_success_rate = policy.train_until_success(
        train_env=train_env,
        eval_env=eval_env,
        total_timesteps=args.total_timesteps,
        save_path=save_path,
        success_threshold=args.success_threshold,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        tensorboard_log=tensorboard_log,
        resume_from=resume_from,
        reset_num_timesteps=args.reset_num_timesteps,
        checkpoint_freq=args.checkpoint_freq,
    )

    train_env.close()
    eval_env.close()

    print("=" * 80)
    print(f"Training finished. Model path: {save_path}")
    print(f"Reached threshold: {reached_threshold}")
    print(f"Last eval success rate: {last_success_rate:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
