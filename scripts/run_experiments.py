"""
Experiment runner for HeRD policies.

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
import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
from scripts.train_rl_policy import DDQNTrainer
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
        for model_name, obs_config, diffusion_config in zip(cfg.evaluate.model_names, cfg.evaluate.obs_configs, cfg.evaluate.diffusion_configs):
            cfg.env.obstacle_config = obs_config
            cfg.diffusion.use_diffusion_policy = diffusion_config

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
                'show': False,           # if true display the environment
                'show_obs': False,       # if true show observation
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
                'train_mode': True,
                'job_type': 'sam',
                'job_name': 'diffusion_sam_sc',
                'log_dir': 'per_logs/',
                'resume_training': False,
                'job_id_to_resume': '16398526',
                'total_timesteps': 60000,
            },
            'evaluate': {
                'eval_mode': False,
                'num_eps': 20,
                'model_names': ['diffusion_8sdp_general', 'ablation_diffusion_16sdp_general', 'ablation_diffusion_4sdp_general', 'ablation_diffusion_16sdp_general', 'ablation_diffusion_4sdp_general', 'ablation_diffusion_16sdp_general'], # list of model names to evaluate
                'model_path': 'models/box_delivery/new_robot', # path to the models
                'obs_configs': ['small_empty', 'small_empty', 'large_divider', 'large_divider', 'large_divider', 'large_divider'], # list of observation configurations
                'diffusion_configs': [True, True, True, True, True, True],
            },
            'ablation': {
                'max_distance_reward': True,
                'step_dist_penalty': True,
                'general': False,
                'diffusion': False,
            },
            'diffusion': {
                'use_diffusion_policy': False,
                'model_name': 'herd_diffusion_model.ckpt',
                'obs_dim': 26, # for combo
                'obs_type': 'combo', # 'positions' or 'vertices'
            }
            
        }
        
        cfg = DotDict.to_dot_dict(cfg)

    main(cfg, job_id)
