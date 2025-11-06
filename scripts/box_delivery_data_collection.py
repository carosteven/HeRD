"""
A simple script to run a teleoperation pipeline for demonstration dataset collection on box delivery environments
'A': large left turn; 'D' large right turn
'Z': small left turn; 'C' small right turn
'W': start moving
'X': stop turning (note: this does not stop linear motion)
'esc': exit teleoperation
"""
import random

import gymnasium as gym
import numpy as np
import pickle
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
# import zarr
from pynput import keyboard
from os.path import dirname
from submodules.BenchNPIN.benchnpin.baselines.box_delivery.SAM.policy import BoxDeliverySAM
from diffusionPolicy.common.replay_buffer import ReplayBuffer
import pygame

WAYPOINT_MOVING_THRESHOLD = 0.6

FORWARD = 0
BACKWARD = 1
LEFT = 2
RIGHT = 3
STOP = 4
STOP_LINEAR = 5
STOP_TURNING = 6
SMALL_LEFT = 7
SMALL_RIGHT = 8
DELETE_PREV_DEMO = 9
DONE_DEMO = 10
BREAK_NONMOVEMENT = 11

command = STOP
# manual_stop = False

def on_press(key):
    global command
    try:
        if key.char == 'w':  # Move up
            command = FORWARD
        elif key.char == 's':  # Move down
            command = BACKWARD
        elif key.char == 'a':  # Move left
            command = LEFT
        elif key.char == 'd':  # Move right
            command = RIGHT
        elif key.char == 'e':  # Stop moving
            command = STOP
        elif key.char == 't': # Stop linear
            command = STOP_LINEAR
        elif key.char == 'r':  # Stop turning
            command = STOP_TURNING
        elif key.char == 'z':  # Move left slowly
            command = SMALL_LEFT
        elif key.char == 'c':  # Move right slowly
            command = SMALL_RIGHT
        elif key.char == 'q': # End current demonstration
            command = DONE_DEMO

    except AttributeError:
        if key == keyboard.Key.backspace: # Delete previous demonstration
            command = DELETE_PREV_DEMO
        elif key == keyboard.Key.space:
            command = BREAK_NONMOVEMENT # Action behind robot to break stuck cycle

def interpolate_trajectory(traj, target_len=32):
    """
    Interpolates to target_len points, preserving all original points exactly.
    Returns interpolated trajectory and valid_obs_mask.
    """
    N, D = traj.shape
    assert target_len >= N, f"target_len must be ≥ number of original points ({N})"

    # Step 1: original arc lengths
    deltas = np.diff(traj, axis=0)
    dist = np.linalg.norm(deltas, axis=1)
    arc_length = np.concatenate([[0], np.cumsum(dist)])
    arc_length /= arc_length[-1]

    # Step 2: Extra arc positions (excluding endpoints)
    num_extra = target_len - N
    extra_t = np.linspace(0, 1, num_extra + 2)[1:-1]  # avoid 0 and 1

    # Step 3: Interpolate values at extra arc positions
    extra_points = np.zeros((len(extra_t), D))
    for d in range(D):
        interp_fn = interp1d(arc_length, traj[:, d], kind='linear')
        extra_points[:, d] = interp_fn(extra_t)

    # Step 4: Combine original and extra points
    all_arc = np.concatenate([arc_length, extra_t])
    all_points = np.concatenate([traj, extra_points], axis=0)

    # Step 5: Sort by arc length
    sort_idx = np.argsort(all_arc)
    all_arc = all_arc[sort_idx]
    all_points = all_points[sort_idx]

    # Step 6: Choose evenly spaced indices that include all original points
    total_points = len(all_points)
    traj_interp = []
    valid_obs_mask = []

    # First: record arc positions of original points
    rounded_arc = np.round(arc_length, decimals=6)

    # Step: evenly select indices from sorted all_points
    selected_indices = np.linspace(0, total_points - 1, target_len).astype(int)

    # Compute index of original points in sorted list
    original_flags = np.array([True]*len(traj) + [False]*len(extra_points))
    sorted_flags = original_flags[sort_idx]

    for idx in selected_indices:
        point = all_points[idx]
        traj_interp.append(point)
        
        # Check if this index corresponds to an original point
        valid_obs_mask.append(sorted_flags[idx])

    traj_interp = np.array(traj_interp)
    valid_obs_mask = np.array(valid_obs_mask, dtype=bool)

    return traj_interp, valid_obs_mask

# def on_release(key):
#     global action, manual_stop
#     if key == keyboard.Key.esc:  # Stop teleoperation when ESC is pressed
#         manual_stop = True
#         return False

'''
Plan:
    - record high/low dim observations
        - high dim: channel 0 (224x224)
        - low dim: [agent_pos, [boxes_pos]]

    - record goal as xy
        - append goal to state

    * COORDINATES RELATIVE TO ROBOT FRAME *


    Initialize policy
    one demo = one step
    need to use 'demo_mode' but also use the threshold logic to terminate step when reach goal
    need to visualize goal --> destination from SAM
'''
def collect_demos():
    path = 'demo_data/box_delivery_teleop_demo_le_final.zarr'
    # path = 'demo_data/box_delivery_teleop_demo_nopush_le.zarr'
    replay_buffer = ReplayBuffer.create_from_path(path, mode='a')
    print(np.sum(replay_buffer['valid_obs_mask']))
    input()

    # ensure different environments
    seed = replay_buffer.n_episodes
    print(f'starting seed {seed}')

    cfg = {
        'render': {
                'show': True,
            },
        'env': {
            'obstacle_config': 'large_empty', # options are small_empty, small_columns, large_columns, large_divider
        },
        'boxes': {
            'num_boxes_small': 10,
        },
        'demonstration': {
            'demonstration_mode': True,
            'teleop_mode': True,
            'step_size': WAYPOINT_MOVING_THRESHOLD/2
        },
        'evaluate': {
            'final_exploration': 0.1,
        },
        'misc': {
            'inactivity_cutoff_sam': 100,  # set to a large number to avoid inactivity cutoff
            'inactivity_cutoff': 10000,  # set to a large number to avoid inactivity cutoff
            'random_seed': seed,
        },
        'ablation': {
            'better_pushing': True,
        }
    }
    env = gym.make('box-delivery-v0', cfg=cfg)
    env = env.unwrapped
    dummy_observation, _ = env.reset()

    # model_name = 'bp_per_qsdp_term_le'
    model_name = 'bp_qsdp_se'
    model_path = 'models/box_delivery/new_robot'

    # Initialize the policy
    policy = BoxDeliverySAM(cfg=env.cfg, model_name=model_name, model_path=model_path)
    policy.act(dummy_observation, env.action_space.high, env.num_channels)

    path_length = 0
    # step_size = 0.1
    # step_size = WAYPOINT_MOVING_THRESHOLD
    step_size = cfg['demonstration']['step_size']
    horizon = env.cfg.diffusion.horizon

    observation, info = env.reset()
    # record_transition(observation, observation, [info['state'][0], info['state'][1]], 0, False, False)
    # prev_state = [info['state'][0], info['state'][1]]

    terminated = False
    truncated = False

    manual_stop = False
    break_nonmovement_action = False

    episodes = []
    clock = pygame.time.Clock()
    # with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    if not cfg['demonstration']['teleop_mode']:
        num_demos = 0
        while num_demos <= 20000:
            terminated = False
            truncated = False
            while not terminated:
                # get action from policy
                goal_ravelled = policy.act(observation)

                # one step is travelling to the goal
                observation, reward, terminated, truncated, info = env.step(goal_ravelled)
                episodes.append(info['demonstration'])
                # num_demos += len(episodes[-1])
                num_demos += 1

            for episode in episodes:
                if len(episode) > 2 and len(episode) <= horizon:
                    robot_positions = np.array([step['action'] for step in episode]) # (N, 2)

                    # interpolate the path to make length of horizon
                    robot_positions_interp, valid_mask = interpolate_trajectory(robot_positions, target_len=horizon) # (horizon, 2)

                    # build a list of horizon steps
                    last_step = episode[-1].copy()
                    padded_episode = [last_step.copy() for _ in range(horizon)]

                    # copy original episode steps into valid positions
                    orig_idx = 0
                    for i in range(horizon):
                        if valid_mask[i]:
                            padded_episode[i] = episode[orig_idx].copy()
                            orig_idx += 1
                            
                    # fill in interpolated actions into the episode
                    for i in range(horizon):
                        padded_episode[i]['action'] = robot_positions_interp[i]

                    # replace episode with padded one
                    episode = padded_episode

                    data_dict = dict()
                    for key in episode[0].keys():
                        data_dict[key] = np.stack(
                            [x[key] for x in episode])
                    data_dict['valid_obs_mask'] = valid_mask
                    replay_buffer.add_episode(data_dict, compressors='default')
            episodes = []
            observation, _ = env.reset()
            print(num_demos)

    else:
        with keyboard.Listener(on_press=on_press) as listener:
            try:
                # episode-level while loop (an episode is travelling to the next point)
                while listener.running:  # While the listener is active
                    episode = list()

                    prev_state = [info['state'][0], info['state'][1]]
                    
                    # current robot pose
                    robot_current_position, robot_current_heading = env.robot.body.position, env.restrict_heading_range(env.robot.body.angle)
                    robot_current_position = list(robot_current_position)  

                    goal_ravelled = policy.act(observation)
                    if break_nonmovement_action:
                        # sometimes the goal can be really close to the robot such that it hits without moving
                        # if so then can break the cycle with an action far behind the robot
                        goal_ravelled = 95*94 + 45
                        break_nonmovement_action = False

                    goal = env.position_controller.get_target_position(robot_current_position, robot_current_heading, goal_ravelled) 

                    # display goal in environment
                    env.renderer.goal_point = goal

                    reached_goal = False
                    test = True
                    ignore_curr_demo = False
                    
                    t = 0
                    transition_count = 1        # start from 1 as we recorded the reset step   
                    global command
                    command = STOP

                    terminated = False
                    truncated = False

                    # step-level while loop
                    while not terminated or not truncated or not reached_goal:
                        # global command
                        if command == DELETE_PREV_DEMO:
                            print("\nCurrent demonstration ignored")
                            ignore_curr_demo = True
                            command = STOP
                        
                        elif command == DONE_DEMO:
                            # break
                            reached_goal = True

                        elif command == BREAK_NONMOVEMENT:
                            break_nonmovement_action = True
                            command = STOP

                        print("command: ", command, "; step: ", t, \
                            "; num completed: ", info['cumulative_boxes'],  end="\r")

                        if env.distance((info['state'][0], info['state'][1]), goal) < WAYPOINT_MOVING_THRESHOLD: # or env.robot_hit_obstacle:
                            reached_goal = True

                        # command = OTHER
                        # if t % 5 == 0:
                        #     env.render()

                        # only record points based on distance interval
                        if (((info['state'][0] - prev_state[0])**2 + (info['state'][1] - prev_state[1])**2)**(0.5) >= step_size) or terminated or truncated or reached_goal:
                            # goal = np.array(goal)
                            # action = np.array(info['state'][:2])
                            # data = {
                            #     'img': observation[0],
                            #     'state_vertices': np.float32(info['obs_vertices']),
                            #     'state_positions': np.float32(info['obs_positions']),
                            #     'goal': np.float32(goal),
                            #     'action': np.float32(action)
                            # }
                            data = env.get_demonstration_data([info['obs_vertices'], info['obs_positions'], info['obs_combo']], goal, info['state'][:2])
                            episode.append(data)

                            prev_state = [info['state'][0], info['state'][1]]
                            transition_count += 1
                        
                        observation, reward, terminated, truncated, info = env.step(command, reached_goal=test, goal=goal_ravelled)
                        test=False

                        if terminated or truncated or reached_goal:
                            print("\nterminated: ", terminated, "; truncated: ", truncated, "; reached goal: ", reached_goal)
                            path_length = transition_count
                            print()
                            print(transition_count)
                            if terminated:
                                observation, info = env.reset()
                            break

                        clock.tick(15)  # Limit the frame rate

                    t += 1

                    if not ignore_curr_demo:
                        episodes.append(episode)

                    if terminated:
                        # save episode buffer to replay buffer (on disk)
                        response = input("\nSave demonstrations? (y/n) ").strip().lower()[-1]
                        if response == 'y':
                            for episode in episodes:
                                if len(episode) > 2 and len(episode) <= horizon:
                                    robot_positions = np.array([step['action'] for step in episode]) # (N, 2)

                                    # interpolate the path to make length of horizon
                                    robot_positions_interp, valid_mask = interpolate_trajectory(robot_positions, target_len=horizon) # (horizon, 2)

                                    # build a list of horizon steps
                                    last_step = episode[-1].copy()
                                    padded_episode = [last_step.copy() for _ in range(horizon)]

                                    # copy original episode steps into valid positions
                                    orig_idx = 0
                                    for i in range(horizon):
                                        if valid_mask[i]:
                                            padded_episode[i] = episode[orig_idx].copy()
                                            orig_idx += 1
                                            
                                    # fill in interpolated actions into the episode
                                    for i in range(horizon):
                                        padded_episode[i]['action'] = robot_positions_interp[i]

                                    # replace episode with padded one
                                    episode = padded_episode

                                    data_dict = dict()
                                    for key in episode[0].keys():
                                        data_dict[key] = np.stack(
                                            [x[key] for x in episode])
                                    data_dict['valid_obs_mask'] = valid_mask
                                    replay_buffer.add_episode(data_dict, compressors='default')
                            
                            print("Demonstrations saved. Resetting environment...")
                        else:
                            print("Demonstrations ignored. Resetting environment...")

                        episodes = []
                 
                    # don't save the demo if this trial is truncated
                    # if manual_stop:
                    #     print("\nDemo manually stopped. Ignored")
                    #     return


            except KeyboardInterrupt:
                print("Exiting teleoperation.")

            finally:
                env.close()
    
    # don't save the demo if this trial is truncated
    if truncated:
        print("\n Demo truncated. Ignored")
        return

if __name__ == "__main__":
    collect_demos()
    # env.close()
