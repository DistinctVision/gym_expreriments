import yaml

from collections import deque

import numpy as np
import torch

from dqn_trainer import DqnTrainer, EpisodeDataRecorder
from data import ReplayBuffer

from stable_baselines3.common.env_util import make_vec_env


cfg = yaml.safe_load(open('dqn_cfg.yaml', 'r'))
game_name = str(cfg['game']['name'])

replay_buffer = ReplayBuffer()
trainer = DqnTrainer(cfg, replay_buffer)


if game_name == 'CartPole-v1':
    std_world_state = torch.tensor([2.5, 2.5, 0.3, 0.3], dtype=torch.float32)
    mean_world_state = torch.tensor([0.0, 0.0,  0.0,  0.0], dtype=torch.float32)
    n_envs = 1
elif game_name == 'LunarLander-v2':
    std_world_state = torch.tensor([1.5, 1.5, 5., 5., 3.1415927, 5., 1., 1.], dtype=torch.float32)
    mean_world_state = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    n_envs = 6
else:
    raise RuntimeError('Unknown game')


def preprocess(world_state: np.ndarray) -> torch.Tensor:
    world_state = torch.from_numpy(world_state)
    return (world_state - mean_world_state) / std_world_state


last_rewards = deque(maxlen=100)

ep_counter = 0

env = make_vec_env(game_name, n_envs=n_envs, seed=np.random.randint(0, 2 ** 16 - 1))
env.reset()

episode_data_recorders = [EpisodeDataRecorder(trainer) for _ in range(env.num_envs)]

train_counter = 0
train_freq = int(cfg['training']['train_freq'])

while True:
    cur_world_state_tensors = env.reset()
    cur_world_state_tensors = [preprocess(world_state_tensor) for world_state_tensor in cur_world_state_tensors]
    prev_world_state_tensors = cur_world_state_tensors
    dones = np.array([False for _ in range(env.num_envs)], dtype=bool)
    steps = 0
    ep_rewards = np.zeros((env.num_envs,), dtype=np.float32)
    
    while not dones.all():
        actions = [episode_data_recorder.get_action(prev_world_state_tensor, cur_world_state_tensor)
                   for episode_data_recorder, prev_world_state_tensor, cur_world_state_tensor in \
                       zip(episode_data_recorders, prev_world_state_tensors, cur_world_state_tensors)]
        env.step_async(actions)
        
        if len(replay_buffer) > 2048:
            train_counter += env.num_envs
            if train_counter >= train_freq:
                trainer.train_step()
                train_counter = 0
        
        new_world_state_tensors, rewards, next_dones, states = env.step_wait()
        new_world_state_tensors = [preprocess(world_state_tensor) for world_state_tensor in new_world_state_tensors]
        next_dones = np.logical_or(dones, next_dones)

        for episode_data_recorder, world_state_tensor, action_idx, reward, done, next_done in \
                zip(episode_data_recorders, cur_world_state_tensors, actions, rewards, dones, next_dones):
            if done:
                continue
            episode_data_recorder.record(world_state_tensor, action_idx, reward, next_done)
        
        dones = next_dones
        prev_world_state_tensors = cur_world_state_tensors
        cur_world_state_tensors = new_world_state_tensors
        ep_rewards += rewards
        steps += 1
    
    for reward in ep_rewards:
        last_rewards.append(reward)
    last_mean_reward = sum(last_rewards) / len(last_rewards)
    trainer.add_metric_value('reward', last_mean_reward)
    
    ep_counter += 1

    rp_size = len(replay_buffer)
    ep_rewards_str = ', '.join([f'{reward:.2f}' for reward in ep_rewards])
    print(f'Episode: {ep_counter} | Replay buffer size: {rp_size} | Mean rewards: {last_mean_reward:.2f} |'\
          f'Episode Rewards: {ep_rewards_str}')
