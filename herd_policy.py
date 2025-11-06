import os
import re
import torch
import numpy as np
import collections
import gymnasium as gym
from shapely.geometry import LineString, Polygon
from skimage.draw import line

from environment import BoxDeliveryEnv
from diffusionPolicy import DiffusionUnetLowdimPolicy
from submodules.BenchNPIN.benchnpin.baselines.box_delivery.SAM.policy import DenseActionSpacePolicy

def get_latest_model(model_dir, model_name):
    # List all files in the directory
    files = os.listdir(model_dir)
    
    # Regex to match the model files with step count
    pattern = re.compile(rf'model-{model_name}(\d+)\.pt')
    
    # Extract step counts and corresponding file names
    models = []
    for file in files:
        match = pattern.match(file)
        if match:
            step_count = int(match.group(1))
            models.append((step_count, file))
    
    # Sort by step count and get the latest model
    if models:
        latest_model = max(models, key=lambda x: x[0])[1]
        return latest_model.split('model-')[1].split('.pt')[0]
    else:
        return None


class HeRDPolicy():
    def __init__(self, cfg):
        self.env = BoxDeliveryEnv(cfg)
        self.env.reset()
        self.cfg = self.env.cfg # update cfg with env-specific config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('mps' if torch.backends.mps.is_available() else self.device)

        # RL Policy
        self.rl_policy = self.create_rl_policy()
        self.position_controller = self.env.position_controller

        # Diffusion Policy
        self.diffusion_policy = self.create_diffusion_policy()
        self.obs_buffer = collections.deque(maxlen=self.cfg.diffusion.n_obs_steps)

    
    def create_rl_policy(self):
        action_space = gym.spaces.Box(low=0, high=self.cfg.env.local_map_pixel_width * self.cfg.env.local_map_pixel_width, dtype=np.float32)
        num_channels = 4
        model_path = self.cfg.rl_policy.model_path
        model_name = self.cfg.rl_policy.model_name
        random_seed = self.cfg.misc.random_seed

        if self.cfg.train.train_mode:
            train = True
            evaluate = False
            final_exploration = self.cfg.train.final_exploration
            checkpoint_path = self.cfg.train.checkpoint_path
            resume_training = self.cfg.train.resume_training
            job_id_to_resume = self.cfg.train.job_id_to_resume
            # If models already exists in the checkpoint directory, use the latest model
            if resume_training:
                checkpoint_dir = os.path.join(os.path.dirname(__file__), f'checkpoint/{job_id_to_resume}/')
                model_name = get_latest_model(checkpoint_dir, model_name)
            elif os.path.exists(checkpoint_path):
                checkpoint_dir = os.path.dirname(checkpoint_path)
                model_name = get_latest_model(checkpoint_dir, model_name)

        else:
            train = False
            evaluate = True
            final_exploration = 0.0
            checkpoint_path = ''
            resume_training = False
            job_id_to_resume = None

        policy = DenseActionSpacePolicy(action_space,
                                        num_channels,
                                        final_exploration=final_exploration,
                                        train=train,
                                        checkpoint_path=checkpoint_path,
                                        resume_training=resume_training,
                                        evaluate=evaluate,
                                        job_id_to_resume=job_id_to_resume,
                                        random_seed=random_seed,
                                        model_name=model_name,
                                        model_dir=model_path,
                                        )

        return policy

    
    def create_diffusion_policy(self):
        # Model configuration
        model_config = {
            'input_dim': self.cfg.diffusion.action_dim,  # Only actions in trajectory for global conditioning
            'local_cond_dim': None,
            'global_cond_dim': self.cfg.diffusion.n_obs_steps * self.cfg.diffusion.obs_dim,  # n_obs_steps * obs_dim for global conditioning
            'diffusion_step_embed_dim': 256,
            'down_dims': [256, 512, 1024],
            'kernel_size': 5,
            'n_groups': 8,
            'cond_predict_scale': True
        }
        
        # Noise scheduler configuration
        scheduler_config = {
            'num_train_timesteps': 100,
            'beta_start': 0.0001,
            'beta_end': 0.02,
            'beta_schedule': 'squaredcos_cap_v2',
            'prediction_type': 'epsilon',
            'clip_sample': True,
            'variance_type': 'fixed_small',
        }
        
        # Policy configuration
        policy = DiffusionUnetLowdimPolicy(
            model=model_config,
            noise_scheduler=scheduler_config,
            horizon=self.cfg.diffusion.horizon,
            obs_dim=self.cfg.diffusion.obs_dim,
            action_dim=self.cfg.diffusion.action_dim,
            n_action_steps=self.cfg.diffusion.n_action_steps,
            n_obs_steps=self.cfg.diffusion.n_obs_steps,
            num_inference_steps=self.cfg.diffusion.num_inference_steps,
            obs_as_global_cond=True,  # Use global conditioning for box delivery
            pred_action_steps_only=True,  # Predict only action steps
            condition_trajectory=True, # TODO: make this configurable
            conditioning_functions=[self.ensure_waypoint_feasibility,
                                    self.prune_by_distance,
                                    self.ensure_path_feasibility],
            seed=self.cfg.misc.random_seed,
        ).to(self.device)

        # Load model weights
        path = self.cfg.diffusion.model_path + self.cfg.diffusion.model_name
        print(f"Loading diffusion checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        policy.load_state_dict(checkpoint['state_dicts']['model'])
        policy.eval()  # Set to evaluation mode

        return policy
    

    def check_path_for_box_collision(self, path, box_obs):
        """
        Checks the path for collision with any box. Returns the first box that intersects with the path,
        and the index along the path where the collision occurs.
        """
        # Construct LineString from path
        line_path = LineString([(pt[0], pt[1]) for pt in path])
        for i, box_vertices in enumerate(box_obs):
            polygon = Polygon(box_vertices).buffer(self.cfg.boxes.box_size / 2)
            if line_path.intersects(polygon):
                intersection_point = line_path.intersection(polygon)
                if intersection_point.is_empty:
                    return None, None

                if intersection_point.geom_type == 'Point':
                    collision_point = (intersection_point.x, intersection_point.y)

                elif intersection_point.geom_type == 'MultiPoint':
                    first = list(intersection_point.geoms)[0]
                    collision_point = (first.x, first.y)

                elif intersection_point.geom_type == 'LineString':
                    collision_point = list(intersection_point.coords)[0]

                elif intersection_point.geom_type == 'GeometryCollection':
                    for geom in intersection_point.geoms:
                        if geom.geom_type == "Point":
                            collision_point = (geom.x, geom.y)
                            break
                        elif geom.geom_type == "LineString":
                            collision_point = list(geom.coords)[0]
                            break
                    else:
                        return None, None  # no usable collision point

                else:
                    return None, None
                # Find closest point on path to collision point
                dists = [np.linalg.norm(np.array([pt[0], pt[1]]) - np.array(collision_point)) for pt in path]
                collision_idx = int(np.argmin(dists))
                return i, collision_idx
        return None, None

    
    def get_path_headings(self, path):
        # compute waypoint headings
        headings = [None]
        for i in range(1, len(path)):
            x_diff = path[i][0] - path[i - 1][0]
            y_diff = path[i][1] - path[i - 1][1]
            waypoint_headings = self.restrict_heading_range(np.arctan2(y_diff, x_diff))
            headings.append(waypoint_headings)

        headings = np.array(headings).reshape(-1, 1)
        path = np.concatenate((path, headings), axis=1)
        return path


    def restrict_heading_range(self, heading):
        return np.mod(heading + np.pi, 2 * np.pi) - np.pi
    

    def ensure_path_feasibility(self, trajectory):
        """ Wrapper for ensure_waypoint_feasibility with path_feasibility=True """
        return self.ensure_waypoint_feasibility(trajectory, path_feasibility=True)


    def ensure_waypoint_feasibility(self, trajectory, path_feasibility=False):
        trajectory = trajectory.squeeze(0)
        device = trajectory.device
        # always include first point
        new_trajectory = [trajectory[0]]
        is_last_point = False
        
        for t in range(1, trajectory.shape[0]):
            p1 = new_trajectory[-1]
            p2 = trajectory[t]

            p1_pos = (p1[0].item(), p1[1].item())
            p2_pos = (p2[0].item(), p2[1].item())

            # always include last point
            if t == trajectory.shape[0] - 1:
                is_last_point = True

            # check if the segment between p1 and p2 is valid
            source_i, source_j = self.env.position_to_pixel_indices(p1_pos[0], p1_pos[1], self.env.configuration_space.shape)
            target_i, target_j = self.env.position_to_pixel_indices(p2_pos[0], p2_pos[1], self.env.configuration_space.shape)
            rr, cc = line(source_i, source_j, target_i, target_j)
            if (1 - self.env.configuration_space_thin[rr, cc]).sum() == 0:
                # no obstacle in the way, continue
                new_trajectory.append(p2)
                continue
            elif not is_last_point and not path_feasibility: # we don't want to modify the last point
                # find closest valid indices in configuration space
                closest_indices = self.env.closest_valid_cspace_indices(target_i, target_j)
                # convert back to position
                p2_pos = self.env.pixel_indices_to_position(closest_indices[0], closest_indices[1], self.env.configuration_space.shape) 
                p2_feas = torch.tensor([p2_pos[0], p2_pos[1]], dtype=torch.float32, device=device)
                new_trajectory.append(p2_feas)

            if path_feasibility:
                shortest_path = self.env.shortest_path(p1_pos, p2_pos)
                # convert to tensor before adding
                shortest_path = [torch.tensor([pos[0], pos[1]], dtype=torch.float32, device=device) for pos in shortest_path]
                # avoid adding p1 again
                new_trajectory.extend(shortest_path[1:])
            
            elif is_last_point:
                new_trajectory.append(p2)
        
        new_trajectory = torch.stack(new_trajectory, dim=0).unsqueeze(0)
        return new_trajectory
    

    def prune_by_distance(self, path, min_dist=None):
        """
        Only used for diffusion policy.
        Remove waypoints that are too close together
        """
        if min_dist is None:
            min_dist = self.cfg.demonstration.step_size
        
        # Always include start and end points
        pruned = [path[:,0]]
        prev_point = path[:,0]
        final_point = path[:,-1]
    
        for i in range(1, path.shape[1] - 1):
            dist_from_prev = torch.norm(path[:,i] - prev_point)
            dist_from_final = torch.norm(path[:,i] - final_point)
            if dist_from_prev >= min_dist and dist_from_final >= min_dist:
                pruned.append(path[:,i])
                prev_point = path[:,i]

        pruned.append(path[:,-1]) # goal is always included
        pruned = torch.stack(pruned, dim=1)
        return pruned


    def act(self, rl_obs, diff_obs, box_obs, robot_pose, exploration_eps=None):
        spatial_action, _ = self.rl_policy.predict(rl_obs, exploration_eps=exploration_eps)
        path, _ = self.position_controller.get_waypoints_to_spatial_action(robot_pose[0:2], robot_pose[2], spatial_action)

        box_in_path, _ = self.check_path_for_box_collision(path, box_obs)

        if not box_in_path:
            target_pos = np.array(path[-1][0:2], dtype=np.float32)
            diff_obs = np.concatenate([diff_obs, target_pos], axis=-1)

            # Condition trajectory by inpainting
            cond = [torch.tensor(robot_pose[0:2]), torch.from_numpy(target_pos)]

            # Shape: [1, n_obs_steps=1, obs_dim]
            obs_tensor = torch.from_numpy(np.array(diff_obs)[np.newaxis, np.newaxis, ...]).float().to(self.device)
            obs_dict = {'obs': obs_tensor}

            with torch.no_grad():
                action_dict = self.diffusion_policy.predict_action(obs_dict, cond=cond)
            path = action_dict['action'].cpu().numpy()
            path = path.reshape(-1, 2)
            path = self.get_path_headings(path)

        if exploration_eps is not None:
            # when training rl policy, spatial action needs to be recorded in the replay buffer
            return path, spatial_action
        return path
    

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
            while True:
                ep_steps += 1
                path = self.act(rl_obs, info['obs_combo'], info['box_obs'], info['robot_pose'])
                rl_obs, reward, done, truncated, info = self.env.step(path)
                ep_reward += reward
                if done or truncated:
                    break
            eps_steps.append(ep_steps)
            eps_rewards.append(ep_reward)
            eps_distance.append(info['cumulative_distance'])
            eps_boxes.append(info['cumulative_boxes'])
            eps_avg_box_distance.append(sum(self.env.box_distances.values()) / len(self.env.box_distances))
        
        eps_steps = np.array(eps_steps)
        eps_rewards = np.array(eps_rewards)
        eps_distance = np.array(eps_distance)
        eps_boxes = np.array(eps_boxes)
        eps_avg_box_distance = np.array(eps_avg_box_distance)

        # Calculate stats
        avg_eps_steps = eps_steps.mean()
        std_dev_eps_steps = eps_steps.std()
        avg_eps_rewards = eps_rewards.mean()
        std_dev_eps_rewards = eps_rewards.std()
        avg_eps_distance = eps_distance.mean()
        std_dev_eps_distance = eps_distance.std()
        avg_eps_boxes = eps_boxes.mean()
        std_dev_eps_boxes = eps_boxes.std()
        avg_eps_avg_box_distance = eps_avg_box_distance.mean()
        std_dev_eps_avg_box_distance = eps_avg_box_distance.std()
        
        print(f"Average eps_steps: {avg_eps_steps:.2f} ± {std_dev_eps_steps:.2f}")
        print(f"Average eps_rewards: {avg_eps_rewards:.2f} ± {std_dev_eps_rewards:.2f}")
        print(f"Average eps_distance: {avg_eps_distance:.2f} ± {std_dev_eps_distance:.2f}")
        print(f"Average eps_boxes: {avg_eps_boxes:.2f} ± {std_dev_eps_boxes:.2f}")
        print(f"Average eps_avg_box_distance: {avg_eps_avg_box_distance:.2f} ± {std_dev_eps_avg_box_distance:.2f}")

    