#!/usr/bin/env python3
"""
Example evaluation script using SamplingPushingPlanner baseline.

This script evaluates the RRT*-based sampling planner with box-aware costs
(the "Deviation Heuristic") against the existing policies.

Note: This script requires the diffusionPolicy dependencies to be installed:
    pip install -e ./submodules/diffusionPolicy
    
Or install the full requirements:
    pip install -r requirements.txt

Usage:
    python3 benchmark/eval_with_sampling_planner.py --num_eps 10 --config small_columns
    python3 benchmark/eval_with_sampling_planner.py --num_eps 50 --config large_divider --verbose
"""

import sys
import os
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add diffusionPolicy submodule to path
diffusion_policy_path = os.path.join(project_root, 'submodules', 'diffusionPolicy')
if diffusion_policy_path not in sys.path:
    sys.path.insert(0, diffusion_policy_path)

import numpy as np
import torch

try:
    from submodules.BenchNPIN.benchnpin.common.utils.utils import DotDict
    from environment import BoxDeliveryEnv
    from benchmark.herd_policy_with_sampling_planner import HeRDPolicyWithSamplingPlanner
except ModuleNotFoundError as e:
    print(f"\nError: Missing dependencies!")
    print(f"Details: {e}")
    print(f"\nPlease install the required packages:")
    print(f"  pip install -r requirements.txt")
    print(f"\nOr install diffusionPolicy specifically:")
    print(f"  pip install -e ./submodules/diffusionPolicy")
    sys.exit(1)


def evaluate_sampling_planner(
    num_eps=5,
    obstacle_config='small_columns',
    verbose=False
):
    """
    Evaluate SamplingPushingPlanner baseline on the box delivery task.
    
    The planner uses RRT* with a box-aware cost function that:
    1. Treats walls as infinitely costly (forbidden)
    2. Evaluates box-pushing feasibility via the "Deviation Heuristic"
    3. Computes alignment (cosine similarity) between:
       - Push vector (robot position to box center)
       - Target vector (box to receptacle)
    4. Applies low cost for advantageous pushes (towards receptacle)
    5. Applies high cost for detrimental pushes (away from receptacle)
    
    Args:
        num_eps: Number of episodes to evaluate
        obstacle_config: One of 'small_empty', 'small_columns', 'large_columns', 'large_divider'
        verbose: Print detailed debugging information
    """
    
    print(f"\n{'='*70}")
    print(f"SamplingPushingPlanner (RRT* with Box-Aware Costs) Evaluation")
    print(f"{'='*70}")
    print(f"Episodes: {num_eps}")
    print(f"Obstacle Config: {obstacle_config}")
    print(f"Verbose: {verbose}")
    print(f"{'='*70}\n")
    
    # Configuration
    cfg = {
        'render': {
            'show': True,
            'show_obs': False,
        },
        'boxes': {
            'num_boxes_small': 10,
            'num_boxes_large': 20,
        },
        'env': {
            'obstacle_config': obstacle_config,
        },
        'misc': {
            'random_seed': 42,
        },
        'rl_policy': {
            'model_path': 'models/rl_models',
            'model_name': 'herd_rl_policy',
        },
        'train': {
            'train_mode': False,
            'final_exploration': 0.0,
        },
        'evaluate': {
            'final_exploration': 0.0,
        },
        'boxes': {
            'box_size': 0.44,
        },
        'rewards': {
            'max_distance_reward': True,
            'step_dist_penalty': True,
        },
        'demonstration': {
            'step_size': 0.1,
        },
        'diffusion': {
            'use_diffusion_policy': True,  # Enable to use SamplingPushingPlanner
            'model_name': 'herd_diffusion_model.ckpt',
            'obs_dim': 26,
            'obs_type': 'combo',
            'action_dim': 2,
            'horizon': 32,
            'n_obs_steps': 1,
            'n_action_steps': 32,
            'num_inference_steps': 20,
        }
    }
    cfg = DotDict.to_dot_dict(cfg)
    
    # Create policy (uses SamplingPushingPlanner instead of DiffusionPolicy)
    print(f"Initializing HeRDPolicy with SamplingPushingPlanner...")
    policy = HeRDPolicyWithSamplingPlanner(cfg=cfg)
    
    # Evaluation loop
    print(f"Running evaluation on {num_eps} episodes with obstacle config: {obstacle_config}")
    print(f"{'='*70}\n")
    
    eps_rewards = []
    eps_steps = []
    eps_distance = []
    eps_boxes_delivered = []
    
    for ep in range(num_eps):
        print(f"Episode {ep + 1}/{num_eps}", end=" ... ")
        sys.stdout.flush()
        
        rl_obs, info = policy.env.reset()
        done = truncated = False
        ep_reward = 0.0
        ep_steps = 0
        ep_boxes = 0
        
        while not (done or truncated):
            # Get observations from info dict
            diff_obs = info.get('obs_combo', info.get('diff_obs'))
            box_obs = info.get('box_obs')
            robot_pose = info.get('robot_pose')
            
            # Get action from policy
            try:
                path, spatial_action = policy.act(
                    rl_obs, diff_obs, box_obs, robot_pose
                )
            except Exception as e:
                if verbose:
                    print(f"\nPolicy error: {e}")
                    import traceback
                    traceback.print_exc()
                break
            
            # Execute path via step
            rl_obs, reward, done, truncated, info = policy.env.step(path)
            
            ep_reward += reward
            ep_steps += 1
            
            if 'cumulative_boxes' in info:
                ep_boxes = info['cumulative_boxes']
            elif 'boxes_delivered' in info:
                ep_boxes = info['boxes_delivered']
        
        eps_rewards.append(ep_reward)
        eps_steps.append(ep_steps)
        eps_boxes_delivered.append(ep_boxes)
        
        print(f"Reward: {ep_reward:8.2f}, Steps: {ep_steps:5d}, Boxes: {ep_boxes:2d}")
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print(f"Evaluation Complete")
    print(f"{'='*70}")
    print(f"Mean Reward: {np.mean(eps_rewards):8.2f} ± {np.std(eps_rewards):6.2f}")
    print(f"Mean Steps:  {np.mean(eps_steps):8.1f} ± {np.std(eps_steps):6.1f}")
    print(f"Mean Boxes Delivered: {np.mean(eps_boxes_delivered):6.2f}")
    print(f"Success Rate (≥1 box): {100.0 * np.sum(np.array(eps_boxes_delivered) > 0) / num_eps:.1f}%")
    print(f"{'='*70}\n")
    
    return {
        'rewards': eps_rewards,
        'steps': eps_steps,
        'boxes_delivered': eps_boxes_delivered,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate SamplingPushingPlanner baseline'
    )
    parser.add_argument(
        '--num_eps',
        type=int,
        default=5,
        help='Number of episodes to evaluate (default: 5)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='small_columns',
        choices=['small_empty', 'small_columns', 'large_columns', 'large_divider'],
        help='Obstacle configuration (default: small_columns)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed debugging information'
    )
    
    args = parser.parse_args()
    
    results = evaluate_sampling_planner(
        num_eps=args.num_eps,
        obstacle_config=args.config,
        verbose=args.verbose
    )
