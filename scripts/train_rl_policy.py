from submodules.BenchNPIN.benchnpin.common.metrics.task_driven_metric import TaskDrivenMetric
from submodules.BenchNPIN.benchnpin.common.utils.utils import DotDict
from herd_policy import HeRDPolicy
import gymnasium as gym
from collections import namedtuple
import random
import os
import sys
import time
from datetime import datetime
import yaml
import numpy as np

import torch
import torch.optim as optim
from torch.nn.functional import smooth_l1_loss
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import logging

logging.getLogger('pymunk').propagate = False


# enable cuDNN auto-tuner to find the best algorithm to use for your hardware
torch.backends.cudnn.benchmark = True

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'ministeps', 'next_state'))

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Meters:
    def __init__(self):
        self.meters = {}

    def get_names(self):
        return self.meters.keys()

    def reset(self):
        for _, meter in self.meters.items():
            meter.reset()

    def update(self, name, val):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(val)

    def avg(self, name):
        return self.meters[name].avg

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.buffer)


class DDQNTrainer():
    def __init__(self, cfg, model_name='rl_model', model_path=None, job_id=None) -> None:
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('mps' if torch.backends.mps.is_available() else self.device)

        if model_path is None:
            self.model_path = os.path.join(os.path.dirname(__file__), 'models/')
        else:
            self.model_path = model_path

        self.model = None
        self.job_id = job_id

        # Check if preemption occurred and if so, use the config file from current run
        checkpoint_dir = os.path.join(os.path.dirname(__file__), f'checkpoint/{self.job_id}')
        # create checkpoint directory if it does not exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint')]
        if checkpoint_files: # if there exists a checkpoint file, this indicates a run has been interrupted
            config_path = f'{checkpoint_dir}/config.yaml'
            self.cfg = DotDict.load_from_file(config_path)
            self.model_name = f'{self.cfg.train.job_name}_{job_id}'
        else:
            self.cfg = cfg
            self.model_name = model_name


    def update_policy(self, policy_net, target_net, optimizer, batch, transform_func):
        state_batch = torch.cat([transform_func(s) for s in batch.state]).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        ministeps_batch = torch.tensor(batch.ministeps, dtype=torch.float32).to(self.device)
        non_final_next_states = torch.cat([transform_func(s) for s in batch.next_state if s is not None]).to(self.device, non_blocking=True)

        output = policy_net(state_batch)
        state_action_values = output.view(self.batch_size, -1).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_state_values = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool, device=self.device)

        # Double DQN
        with torch.no_grad():
            best_action = policy_net(non_final_next_states).view(non_final_next_states.size(0), -1).max(1)[1].view(non_final_next_states.size(0), 1)
            next_state_values[non_final_mask] = target_net(non_final_next_states).view(non_final_next_states.size(0), -1).gather(1, best_action).view(-1)

        expected_state_action_values = (reward_batch + torch.pow(self.gamma, ministeps_batch) * next_state_values)
        td_error = torch.abs(state_action_values - expected_state_action_values).detach()

        loss = smooth_l1_loss(state_action_values, expected_state_action_values)

        optimizer.zero_grad()
        loss.backward()
        if self.grad_norm_clipping is not None:
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), self.grad_norm_clipping)
        optimizer.step()

        train_info = {}
        train_info['q_value_min'] = output.min().item()
        train_info['q_value_max'] = output.max().item()
        train_info['td_error'] = td_error.mean()
        train_info['loss'] = loss

        return train_info


    def train(self) -> None:
        # policy
        policy = HeRDPolicy(self.cfg)
        self.cfg = policy.cfg # update cfg with env-specific config
        
        for key in self.cfg:
            print(f"{key}: {self.cfg[key]}")

        params = self.cfg['train']
        self.batch_size = params['batch_size']
        self.final_exploration = params['final_exploration']
        self.gamma = params['gamma']
        self.grad_norm_clipping = params['grad_norm_clipping']
        self.learning_rate = params['learning_rate']
        self.replay_buffer_size = params['replay_buffer_size']
        self.weight_decay = params['weight_decay']

        checkpoint_freq = params['checkpoint_freq']
        exploration_timesteps = params['exploration_timesteps']
        job_id_to_resume = params['job_id_to_resume']
        learning_starts = params['learning_starts']
        resume_training = params['resume_training']
        target_update_freq = params['target_update_freq']
        total_timesteps = params['total_timesteps']

        checkpoint_path = os.path.join(os.path.dirname(__file__), f'checkpoint/{self.job_id}/checkpoint-{self.model_name}.pt')

        log_dir = os.path.join(os.path.dirname(__file__), params['log_dir'])
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logging.basicConfig(filename=os.path.join(log_dir, f'{self.model_name}.log'), level=logging.DEBUG)
        logging.info("starting training...")
        logging.info(f"Job ID: {self.job_id}")

        # optimizer
        optimizer = optim.SGD(policy.rl_policy.policy_net.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)

        # replay buffer
        replay_buffer = ReplayBuffer(self.replay_buffer_size)

        # resume if possible
        start_timestep = 0
        episode = 0
        if os.path.exists(checkpoint_path) or resume_training:
            if resume_training:
                checkpoint_path_to_load = os.path.join(os.path.dirname(__file__), f'checkpoint/{job_id_to_resume}/checkpoint-{self.model_name}.pt')
            else:
                checkpoint_path_to_load = checkpoint_path
            checkpoint = torch.load(checkpoint_path_to_load)
            start_timestep = checkpoint['timestep']
            episode = checkpoint['episode']
            optimizer.load_state_dict(checkpoint['optimizer'])
            replay_buffer = checkpoint['replay_buffer']
            print(f"=> loaded checkpoint '{checkpoint_path}' (timestep: {start_timestep})")
            logging.info(f"=> loaded checkpoint '{checkpoint_path}' (timestep: {start_timestep})")
        else:
            print("=> no checkpoint detected, starting from initial state")
            logging.info("=> no checkpoint detected, starting from initial state")
        
        # target net
        target_net = policy.rl_policy.build_network()
        target_net.load_state_dict(policy.rl_policy.policy_net.state_dict())
        target_net.eval()

        # logging
        train_summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, f'_new_{self.model_name}'))
        meters = Meters()

        state, info = policy.env.reset()
        total_timesteps_with_warmup = total_timesteps + learning_starts
        for timestep in tqdm(range(start_timestep, total_timesteps_with_warmup),
                             initial=start_timestep, total=total_timesteps_with_warmup, file=sys.stdout):
            
            start_time = time.time()

            # select action
            if exploration_timesteps > 0:
                exploration_eps = 1 - min(max(timestep - learning_starts, 0) / exploration_timesteps, 1) * (1 - self.final_exploration)
            else:
                exploration_eps = self.final_exploration
            path, action = policy.act(state, info['obs_combo'], info['box_obs'], info['robot_pose'], exploration_eps=exploration_eps)

            # step the simulation
            next_state, reward, done, truncated, info = policy.env.step(path)
            ministeps = info['ministeps']

            # store in buffer
            replay_buffer.push(state, action, reward, ministeps, next_state)
            state = next_state

            # reset if episode ended
            if done:
                obs_config = None
                if self.cfg.train.random_env:
                    obs_config = random.choice(['large_columns', 'large_divider'])
                state, _ = policy.env.reset(obs_config = obs_config)
                episode += 1
                if truncated:
                    logging.info(f"Episode {episode} truncated. {info['cumulative_boxes']} in goal. Resetting environment...")
                else:
                    logging.info(f"Episode {episode} completed. Resetting environment...")
            
            # train network
            if timestep >= learning_starts:
                batch = replay_buffer.sample(self.batch_size)
                train_info = self.update_policy(policy.rl_policy.policy_net, target_net, optimizer, batch, policy.rl_policy.apply_transform)
            
            # update target network
            if (timestep + 1) % target_update_freq == 0:
                target_net.load_state_dict(policy.rl_policy.policy_net.state_dict())
            
            step_time = time.time() - start_time

            ################################################################################
            # Logging
            # meters
            meters.update('step_time', step_time)
            if timestep >= learning_starts:
                for name, val in train_info.items():
                    meters.update(name, val)
            
            if done:
                for name in meters.get_names():
                    train_summary_writer.add_scalar(name, meters.avg(name), timestep + 1)
                eta_seconds = meters.avg('step_time') * (total_timesteps_with_warmup - timestep)
                meters.reset()

                train_summary_writer.add_scalar('episodes', episode, timestep + 1)
                train_summary_writer.add_scalar('eta_hours', eta_seconds / 3600, timestep + 1)

                for name in ['cumulative_boxes', 'cumulative_distance', 'cumulative_reward']:
                    train_summary_writer.add_scalar(name, info[name], timestep + 1)

            ################################################################################
            # Checkpoint
            if (timestep + 1) % checkpoint_freq == 0 or timestep + 1 == total_timesteps_with_warmup:
                checkpoint_dir = os.path.dirname(checkpoint_path)
                model_path = f'{checkpoint_dir}/model-{self.model_name+str(timestep+1)}.pt'
                if not os.path.exists(checkpoint_dir):
                    try:
                        os.makedirs(checkpoint_dir, exist_ok=True)
                    except FileExistsError:
                        print(f"Directory {checkpoint_dir} already exists")
                        logging.info(f"Directory {checkpoint_dir} already exists")
                
                # Save the configuration file
                config_path = f'{checkpoint_dir}/config.yaml'
                with open(config_path, 'w') as file:
                    yaml.dump(dict(self.cfg), file, default_flow_style=False)
                
                # temp_model_path = f'{checkpoint_dir}/model-temp.pt'
                model = {
                    'timestep': timestep + 1,
                    'state_dict': policy.rl_policy.policy_net.state_dict(),
                }

                temp_checkpoint_path = f'{checkpoint_dir}/checkpoint-temp.pt'
                checkpoint = {
                    'timestep': timestep + 1,
                    'episode': episode,
                    'optimizer': optimizer.state_dict(),
                    'replay_buffer': replay_buffer,
                }

                # save model and checkpoint
                torch.save(model, model_path)
                torch.save(checkpoint, temp_checkpoint_path)

                # according to the GNU spec of rename, the state of checkpoint_path
                # is atomic, i.e. it will either be modified or not modified, but not in
                # between, during a system crash (i.e. preemtion)
                # os.replace(temp_model_path, model_path)
                os.replace(temp_checkpoint_path, checkpoint_path)
                msg = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": Checkpoint saved at " + checkpoint_path + self.model_name
                logging.info(msg)
