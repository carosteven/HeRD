"""
Experiment runner for HRL policy:
high-level RL picks spatial goal, low-level RL executes one action at a time.
"""

import argparse
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.train_rl_policy import DDQNTrainer
from herd_hrl_policy import HeRDHRLPolicy
from submodules.BenchNPIN.benchnpin.common.utils.utils import DotDict


def main(cfg, job_id):
    if cfg.train.train_mode:
        if cfg.train.resume_training:
            model_name = f"{cfg.train.job_name}_{cfg.train.job_id_to_resume}"
        else:
            model_name = f"{cfg.train.job_name}_{job_id}"

        trainer = DDQNTrainer(model_name=model_name, cfg=cfg, job_id=job_id)
        trainer.train()

    if cfg.evaluate.eval_mode:
        num_eps = cfg.evaluate.num_eps
        for model_name, obs_config in zip(cfg.evaluate.model_names, cfg.evaluate.obs_configs):
            cfg.rl_policy.model_name = model_name
            cfg.env.obstacle_config = obs_config
            cfg.diffusion.use_diffusion_policy = False

            policy = HeRDHRLPolicy(cfg=cfg)
            policy.evaluate(num_eps=num_eps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run HRL experiments for box delivery task')
    parser.add_argument('--config_file', type=str, help='path to the config file', default=None)
    parser.add_argument('--job_id', type=str, help='slurm job id', default=None)
    args = parser.parse_args()

    if args.config_file is not None:
        cfg = DotDict.load_from_file(args.config_file)
    else:
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
                'obstacle_config': 'small_columns',
            },
            'misc': {
                'random_seed': 1,
                'inactivity_cutoff': 10000,
            },
            'rl_policy': {
                'model_path': 'models/rl_models',
            },
            'train': {
                'train_mode': False,
                'job_name': 'hrl_sam',
                'log_dir': 'logs/',
                'resume_training': False,
                'job_id_to_resume': '',
                'total_timesteps': 60000,
                'random_env': False,
            },
            'evaluate': {
                'eval_mode': True,
                'num_eps': 20,
                'obs_configs': ['small_columns', 'small_columns', 'large_columns', 'large_divider'],
                'model_names': ['herd_rl_policy', 'herd_rl_policy', 'herd_rl_policy', 'herd_rl_policy'],
                'diffusion_configs': [False, False, False, False],
            },
            'rewards': {
                'max_distance_reward': True,
                'step_dist_penalty': True,
            },
            'diffusion': {
                'use_diffusion_policy': False,
                'model_name': 'herd_diffusion_model.ckpt',
                'obs_dim': 26,
                'obs_type': 'combo',
            },
            'low_level': {
                'algorithm': 'auto',
                'model_path': 'models/rl_models/low_level_hrl_policy',
                'max_steps': 150,
                'goal_threshold_m': 0.5,
                'step_size_pixels': 1,
                'deterministic': True,
            },
        }
        cfg = DotDict.to_dot_dict(cfg)

    main(cfg, args.job_id)
