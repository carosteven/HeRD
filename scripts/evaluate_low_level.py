import argparse
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from low_level_rl_policy import LowLevelNavEnv, LowLevelRLPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Visually evaluate a trained low-level HRL navigation policy.")
    parser.add_argument("--model-path", type=str, default="models/rl_models/low_level_hrl_policy")
    parser.add_argument("--algorithm", type=str, default="auto", choices=["auto", "ppo", "dqn"])

    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--step-size-pixels", type=int, default=1)
    parser.add_argument("--goal-threshold-m", type=float, default=0.2)
    parser.add_argument("--min-start-goal-distance-m", type=float, default=1.0)

    parser.add_argument("--obstacle-config", type=str, default="small_columns")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--sleep", type=float, default=0.03, help="Seconds to sleep between rendered steps.")
    parser.add_argument("--show-obs", action="store_true", help="Render observation channels alongside sim.")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic action sampling instead of deterministic.")

    return parser.parse_args()


def make_env_cfg(obstacle_config: str, seed: int, show_obs: bool):
    return {
        "render": {
            "show": True,
            "show_obs": show_obs,
        },
        "env": {
            "obstacle_config": obstacle_config,
        },
        "misc": {
            "random_seed": seed,
        },
    }


def main():
    args = parse_args()

    cfg = make_env_cfg(args.obstacle_config, args.seed, args.show_obs)

    env = LowLevelNavEnv(
        cfg=cfg,
        max_steps=args.max_steps,
        step_size_pixels=args.step_size_pixels,
        goal_threshold_m=args.goal_threshold_m,
        min_start_goal_distance_m=args.min_start_goal_distance_m,
    )

    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)

    selected_algorithm = args.algorithm
    policy = None
    if args.algorithm == "auto":
        load_errors = []
        for candidate_algorithm in ["ppo", "dqn"]:
            try:
                candidate_policy = LowLevelRLPolicy(algorithm=candidate_algorithm, device=args.device)
                candidate_policy.load(model_path)
                policy = candidate_policy
                selected_algorithm = candidate_algorithm
                print(f"Auto-detected algorithm: {selected_algorithm}")
                break
            except Exception as exc:
                load_errors.append(f"{candidate_algorithm}: {exc}")
        if policy is None:
            raise RuntimeError(
                "Failed to load model with auto-detect. Errors: " + " | ".join(load_errors)
            )
    else:
        policy = LowLevelRLPolicy(algorithm=args.algorithm, device=args.device)
        policy.load(model_path)

    deterministic = not args.stochastic

    successes = 0
    episode_rewards = []

    try:
        for episode_idx in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode_idx)
            env.render()

            done = False
            truncated = False
            ep_reward = 0.0
            ep_steps = 0

            while not (done or truncated):
                action = policy.act(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = env.step(action)
                ep_reward += reward
                ep_steps += 1

                env.render()
                if args.sleep > 0:
                    time.sleep(args.sleep)

            success = bool(info.get("success", False))
            successes += int(success)
            episode_rewards.append(ep_reward)

            print(
                f"Episode {episode_idx + 1}/{args.episodes} | "
                f"success={success} | "
                f"steps={ep_steps} | "
                f"reward={ep_reward:.2f} | "
                f"final_dist={float(info.get('distance_to_goal', -1.0)):.3f}"
            )

    finally:
        env.close()

    success_rate = successes / max(args.episodes, 1)
    avg_reward = sum(episode_rewards) / max(len(episode_rewards), 1)

    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Algorithm: {selected_algorithm}")
    print(f"Episodes: {args.episodes}")
    print(f"Success rate: {success_rate:.3f}")
    print(f"Average reward: {avg_reward:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
