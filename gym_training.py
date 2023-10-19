from pathlib import Path
import yaml
import time

import numpy as np
import torch
import gymnasium as gym

from trainer import Trainer, EpisodeDataRecorder
from data import SharedDataCollector


cfg = yaml.safe_load(open(Path(__file__).parent / 'cfg.yaml', 'r'))
data_collector = SharedDataCollector(cfg)
trainer = Trainer(cfg, data_collector)
episode_data_recorder = EpisodeDataRecorder(trainer)


std_world_state = torch.tensor([0.0896, 0.5494, 0.0921, 0.8163], dtype=torch.float32)
mean_world_state = torch.tensor([0.0021, -0.0457,  0.0094,  0.0996], dtype=torch.float32)


def preprocess(world_state: np.ndarray) -> torch.Tensor:
    world_state = torch.from_numpy(world_state)
    return (world_state - mean_world_state) / std_world_state


env = gym.make("CartPole-v1")
env.reset()

sum_ep_reward = 0.0
ep_counter = 0

while True:
    world_state_tensor, state = env.reset()
    world_state_tensor = preprocess(world_state_tensor)
    done = False
    steps = 0
    ep_reward = 0
    t0 = time.time()
    
    data_is_updated = False
    
    action_idx, data_is_updated = \
        episode_data_recorder.do_next_step(world_state_tensor, 0.0, False)
    
    while not done:
        new_world_state_tensor, reward, terminated, truncated, state = env.step(action_idx)
        new_world_state_tensor = preprocess(new_world_state_tensor)
        done = terminated or truncated
    
        action_idx, data_is_updated_ = episode_data_recorder.do_next_step(world_state_tensor,
                                                                          reward, not done)
        data_is_updated = data_is_updated or data_is_updated_
        
        ep_reward += reward
        world_state_tensor = new_world_state_tensor
        steps += 1
        
    sum_ep_reward += ep_reward
    ep_counter += 1

    length = time.time() - t0
    print("Step time: {:1.5f} | Episode time: {:.2f} | Episode Rewards: {:.2f}".format(length / steps, length, ep_reward))
    
    debug_info_str = f'Data size: {data_collector.data_size}, '\
                     f'sessions: {data_collector.n_episodes}, step: {trainer.step}\n'\
                     f'Current buffer size: {len(data_collector.current_rp_buffer)}'
    print(debug_info_str)
    
    if data_is_updated:
        avg_ep_reward = sum_ep_reward / ep_counter
        print(f'Avg episode reward: {avg_ep_reward:.2f}')
        sum_ep_reward = 0.0
        ep_counter = 0
        data_collector.save_rp_buffsers(trainer.log_writer.output_weights_folder)
        trainer.do_training()
