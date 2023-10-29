import typing as tp

import random
from collections import deque
import yaml

import numpy as np
import torch

from ppo_trainer import PpoTrainer
from rollout import RolloutBuffer, RolloutDataset
from actor_critic_policy import ActorCriticPolicy

from stable_baselines3.common.env_util import make_vec_env


cfg = yaml.safe_load(open('ppo_cfg.yaml', 'r'))
game_name = str(cfg['game']['name'])
rollout_cfg = dict(cfg['rollout'])

rollout_max_buffer_size = int(rollout_cfg['max_buffer_size'])
batch_size = int(cfg['training']['batch_size'])

trainer = PpoTrainer(cfg)
actor_critic_policy = trainer.models
device = trainer.device


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


last_rewards = deque(maxlen=10)

ep_counter = 0

env = make_vec_env(game_name, n_envs=n_envs, seed=np.random.randint(0, 2 ** 16 - 1))
env.reset()

rollout_buffers: tp.List[RolloutBuffer] = []


while True:
    cur_rollout_buffers = [RolloutBuffer(rollout_cfg, actor_critic_policy.value_net)
                           for _ in range(env.num_envs)]
    for cur_rollout_buffer in cur_rollout_buffers:
        cur_rollout_buffer.start()
    
    cur_world_state_tensors = env.reset()
    cur_world_state_tensors = [preprocess(world_state_tensor) for world_state_tensor in cur_world_state_tensors]
    prev_world_state_tensors = cur_world_state_tensors
    dones = np.array([False for _ in range(env.num_envs)], dtype=bool)
    steps = 0
    ep_rewards = np.zeros((env.num_envs,), dtype=np.float32)
    
    while not dones.all():
        action_dists = [actor_critic_policy.get_action_dist(cur_world_state_tensor)
                        for cur_world_state_tensor in cur_world_state_tensors]
        actions = [int(action_dict.sample()) for action_dict in action_dists]
        env.step_async(actions)
        next_world_state_tensors, rewards, next_dones, states = env.step_wait()
        next_world_state_tensors = [preprocess(world_state_tensor) for world_state_tensor in next_world_state_tensors]
        next_dones = np.logical_or(dones, next_dones)

        for cur_rollout_buffer, world_state_tensor, action, action_dist, reward, done, next_done in \
                zip(cur_rollout_buffers, cur_world_state_tensors, actions, action_dists, rewards, dones, next_dones):
            if done:
                continue
            cur_rollout_buffer.add(world_state_tensor, action,
                                   float(action_dist.log_prob(torch.tensor(action))),
                                   float(reward))
        
        for cur_rollout_buffer, next_world_state_tensor, next_done, state in \
                zip(cur_rollout_buffers, next_world_state_tensors, next_dones, states):
            if next_done:
                cur_rollout_buffer.finish(next_world_state_tensor, truncated=state['TimeLimit.truncated'])
        
        dones = next_dones
        prev_world_state_tensors = cur_world_state_tensors
        cur_world_state_tensors = next_world_state_tensors
        ep_rewards += rewards
        steps += 1
        
    assert all([cur_rollout_buffer.is_finished for cur_rollout_buffer in cur_rollout_buffers])
    rollout_buffers += cur_rollout_buffers
    
    for reward in ep_rewards:
        last_rewards.append(reward)
    last_mean_reward = sum(last_rewards) / len(last_rewards)
    
    trainer.set_ext_values(mean_reward=last_mean_reward)
    
    ep_counter += 1
    
    data_size = sum([len(rollout_buffer) for rollout_buffer in rollout_buffers])
    ep_rewards_str = ', '.join([f'{reward:.2f}' for reward in ep_rewards])
    print(f'Episode: {ep_counter} | Rollout buffer size: {data_size} | Mean rewards: {last_mean_reward:.2f} |'\
          f'Episode Rewards: {ep_rewards_str}')
        
    if data_size >= rollout_max_buffer_size:
        dataset, rollout_buffers = RolloutDataset.collect_data(rollout_max_buffer_size, batch_size, rollout_buffers)
        trainer.train(dataset)
