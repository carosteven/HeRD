#!/usr/bin/env python
import sys
import numpy as np
from submodules.BenchNPIN.benchnpin.common.utils.utils import DotDict
from greedy_heuristic_policy import GreedyHeuristicPolicy

print("Testing with small_empty first...")
cfg = {
    'render': {'show': False, 'show_obs': False},
    'env': {'obstacle_config': 'small_empty'},
    'evaluate': {'num_eps': 1},
    'diffusion': {'use_diffusion_policy': False},
    'train': {'train_mode': False}
}
cfg = DotDict.to_dot_dict(cfg)

policy = GreedyHeuristicPolicy(cfg)
rewards, steps, distances, boxes, avg_distances = policy.evaluate(num_eps=1)
print(f'  small_empty: {boxes[0]:.0f}/10 boxes, {steps[0]:.0f} steps')

print("\nTesting with small_columns...")
cfg = {
    'render': {'show': False, 'show_obs': False},
    'env': {'obstacle_config': 'small_columns'},
    'evaluate': {'num_eps': 1},
    'diffusion': {'use_diffusion_policy': False},
    'train': {'train_mode': False}
}
cfg = DotDict.to_dot_dict(cfg)

policy = GreedyHeuristicPolicy(cfg)
rewards, steps, distances, boxes, avg_distances = policy.evaluate(num_eps=1)
print(f'  small_columns: {boxes[0]:.0f}/10 boxes, {steps[0]:.0f} steps')

if boxes[0] >= 9:
    print('\n✓ Feasibility properly adjusted paths around columns')
else:
    print(f'\n⚠ Issue: only {boxes[0]:.0f}/10 with columns')
