"""
Experiment runner for HeRD (Human-Robot Delivery) navigation policies.

This script provides a unified interface to:
- Train RL policies with different configurations
- Evaluate trained policies across multiple environments
- Benchmark and compare different policy approaches
- Generate performance comparison plots
- Handle experiment configuration via files or inline parameters

Usage:
    # Train a new policy
    python run_experiments.py --config configs/train_config.yaml --job_id exp_001
    
    # Evaluate existing policies
    python run_experiments.py --config configs/eval_config.yaml
    
    # Run with inline configuration (see script for examples)
    python run_experiments.py --job_id exp_001
"""

import argparse
from train_rl_policy import DDQNTrainer
from herd_policy import HeRDPolicy
from submodules.BenchNPIN.benchnpin.common.metrics.base_metric import BaseMetric
from submodules.BenchNPIN.benchnpin.common.utils.utils import DotDict

def main(cfg, job_id):

    if cfg.train.train_mode:
        if cfg.train.resume_training:
            # model_name = cfg.train.job_id_to_resume
            model_name = f'{cfg.train.job_name}_{cfg.train.job_id_to_resume}'
        else:
            model_name = f'{cfg.train.job_name}_{job_id}'

        trainer = DDQNTrainer(model_name=model_name, cfg=cfg, job_id=job_id)
        trainer.train()
    
    if cfg.evaluate.eval_mode:
        benchmark_results = []
        num_eps = cfg.evaluate.num_eps
        model_path = cfg.evaluate.model_path
        for policy_type, action_type, model_name, obs_config, push_config, diffusion_config in zip(cfg.evaluate.policy_types, cfg.evaluate.action_types, cfg.evaluate.model_names, cfg.evaluate.obs_configs, cfg.evaluate.push_configs, cfg.evaluate.diffusion_configs):
            cfg.agent.action_type = action_type
            cfg.train.job_type = policy_type
            cfg.env.obstacle_config = obs_config
            cfg.ablation.better_pushing = push_config
            cfg.ablation.diffusion = diffusion_config

            policy = HeRDPolicy(cfg=cfg)
            benchmark_results.append(policy.evaluate(num_eps=num_eps))

        BaseMetric.plot_algs_scores(benchmark_results, save_fig_dir='./', plot_success=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and evaluate baselines for box pushing task'
    )

    parser.add_argument(
        '--config_file',
        type=str,
        help='path to the config file',
        default=None
    )

    parser.add_argument(
        '--job_id',
        type=str,
        help='slurm job id',
        default=None
    )

    job_id = parser.parse_args().job_id

    if parser.parse_args().config_file is not None:
        cfg = DotDict.load_from_file(parser.parse_args().config_file)


    else:
        # High level configuration for the box delivery task
        cfg={
            'render': {
                'show': True,           # if true display the environment
                'show_obs': False,       # if true show observation
            },
            'agent': {
                'action_type': 'position', # 'position', 'heading', 'velocity'
            },
            'boxes': {
                'num_boxes_small': 10,
                'num_boxes_large': 20,
            },
            'env': {
                'obstacle_config': 'small_columns', # options are small_empty, small_columns, large_columns, large_divider
            },
            'misc': {
                'random_seed': 0,
            },
            'train': { 
                'train_mode': False,
                'job_type': 'sam', # 'sam', 'ppo', 'sac'
                'job_name': 'diffusion_sam_sc',
                'log_dir': 'per_logs/',
                'resume_training': False,
                'job_id_to_resume': '16398526',
                'total_timesteps': 60000,
                # 'exploration_timesteps': 6000*2,
            },
            'evaluate': {
                'eval_mode': True,
                'num_eps': 20,
                'policy_types': ['sam', 'sam', 'sam', 'sam', 'sam'], # list of policy types to evaluate
                'action_types': ['position', 'position', 'position', 'position'], # list of action types to evaluate
                # 'model_names': ['newest_base_se2', 'newest_base_sc2', 'newest_base_lc2', 'newest_base_ld2'], # list of model names to evaluate
                'model_names': ['diffusion_8sdp_general', 'ablation_diffusion_16sdp_general', 'ablation_diffusion_4sdp_general', 'ablation_diffusion_16sdp_general', 'ablation_diffusion_4sdp_general', 'ablation_diffusion_16sdp_general'], # list of model names to evaluate
                'model_path': 'models/box_delivery/new_robot', # path to the models
                'obs_configs': ['small_empty', 'small_empty', 'large_divider', 'large_divider', 'large_divider', 'large_divider'], # list of observation configurations
                'push_configs': [False, False, False, False, False, False],
                # 'diffusion_configs': [False, False, False, False],
                'diffusion_configs': [True, True, True, True, True, True],
            },
            'rewards_sam': {
                'goal_reward': 1.0,
            },
            'ablation': {
                'half_action_space': False,
                'terminal_reward': 10,
                'step_penalty': 0,
                'max_distance_reward': True,
                'box_dist_penalty': False,
                'box_dist_penalty_scale': 0.35,
                'step_dist_penalty': True,
                'sparse': False,
                'per': True,
                'per_alpha': 0.6,
                'per_beta': 0.4,
                'curriculum': False,
                'better_pushing': True,
                'general': False,
                'diffusion': False,
                'average_filter': False,
            },
            'diffusion': {
                'model_name': 'herd_diffusion_model.ckpt',
                'obs_dim': 26, # for combo
                'obs_type': 'combo', # 'positions' or 'vertices'
            }
            
        }
        
        cfg = DotDict.to_dot_dict(cfg)

    main(cfg, job_id)
