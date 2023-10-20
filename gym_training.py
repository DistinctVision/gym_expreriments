from pathlib import Path
import yaml
import time

from collections import deque

import numpy as np
import torch
import gymnasium as gym

import plotly.express as ex

from trainer import Trainer, EpisodeDataRecorder
from data import ReplayBuffer


cfg = yaml.safe_load(open(Path(__file__).parent / 'cfg.yaml', 'r'))
replay_buffer = ReplayBuffer()
trainer = Trainer(cfg, replay_buffer)
episode_data_recorder = EpisodeDataRecorder(trainer)


std_world_state = torch.tensor([0.0896, 0.5494, 0.0921, 0.8163], dtype=torch.float32)
mean_world_state = torch.tensor([0.0021, -0.0457,  0.0094,  0.0996], dtype=torch.float32)


def preprocess(world_state: np.ndarray) -> torch.Tensor:
    world_state = torch.from_numpy(world_state)
    return (world_state - mean_world_state) / std_world_state


last_rewards = deque(maxlen=100)

env = gym.make("CartPole-v1")
env.reset()

while True:
    world_state_tensor, state = env.reset()
    world_state_tensor = preprocess(world_state_tensor)
    done = False
    steps = 0
    ep_reward = 0
    t0 = time.time()
    
    data_is_updated = False
    
    while not done:
        action_idx = episode_data_recorder.get_action(world_state_tensor)
        new_world_state_tensor, reward, terminated, truncated, state = env.step(action_idx)
        new_world_state_tensor = preprocess(new_world_state_tensor)
        done = terminated or truncated
    
        data_is_updated_ = episode_data_recorder.record(world_state_tensor, action_idx, reward, done)
        data_is_updated = data_is_updated or data_is_updated_
        
        ep_reward += reward
        world_state_tensor = new_world_state_tensor
        steps += 1
        
        if len(replay_buffer) > 512:
            trainer.train_step()
    
    last_rewards.append(ep_reward)
    last_mean_reward = sum(last_rewards) / len(last_rewards)
    trainer.add_metric_value('reward', last_mean_reward)

    rp_size = len(replay_buffer)
    length = time.time() - t0
    print(f"Step time: {length / steps:1.5f} | Replay buffer size: {rp_size} | Mean rewards: {last_mean_reward:.2f} | Episode Rewards: {ep_reward}")

