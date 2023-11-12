import typing as tp
import yaml
from collections import deque

import numpy as np
import torch
import cv2
import imageio

import gymnasium as gym
from model import CriticModel


cfg = yaml.safe_load(open('best_models\dqn_lunar_lander.yaml', 'r'))
# cfg = yaml.safe_load(open('best_models\dqn_cart_pole.yaml', 'r'))
game_name = str(cfg['game']['name'])


if game_name == 'CartPole-v1':
    std_world_state = torch.tensor([2.5, 2.5, 0.3, 0.3], dtype=torch.float32)
    mean_world_state = torch.tensor([0.0, 0.0,  0.0,  0.0], dtype=torch.float32)
    text_color = (0, 255, 0)
elif game_name == 'LunarLander-v2':
    std_world_state = torch.tensor([1.5, 1.5, 5., 5., 3.1415927, 5., 1., 1.], dtype=torch.float32)
    mean_world_state = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    text_color = (255, 255, 0)
else:
    raise RuntimeError('Unknown game')


def preprocess(world_state: np.ndarray) -> torch.Tensor:
    world_state = torch.from_numpy(world_state)
    return (world_state - mean_world_state) / std_world_state


env = gym.make(game_name, render_mode="rgb_array")

model = CriticModel.build_model(cfg['model'])
ckpt = torch.load(cfg['model']['critic_model_path'], map_location='cpu')
model.load_state_dict(ckpt)

last_rewards = deque(maxlen=100)
mean_reward: tp.Optional[float] = None

frames = []

for ep_idx in range(100):
    cur_world_state_tensor, info = env.reset()
    cur_world_state_tensor = preprocess(cur_world_state_tensor)
    prev_world_state_tensor = cur_world_state_tensor
    
    ep_reward = 0
    
    for _ in range(500):
        with torch.no_grad():
            pr_rewards: torch.Tensor = model(prev_world_state_tensor.unsqueeze(0).to(model.device),
                                             cur_world_state_tensor.unsqueeze(0).to(model.device))
            pr_rewards = pr_rewards.squeeze(0).cpu()
        action = int(pr_rewards.argmax(0))
        prev_world_state_tensor = cur_world_state_tensor
        cur_world_state_tensor, reward, terminated, truncated, info = env.step(action)
        cur_world_state_tensor = preprocess(cur_world_state_tensor)
        
        ep_reward += reward
        
        if ep_idx % 20 == 0:
            frame = env.render()
            if mean_reward is not None:
                cv2.putText(frame, f'Mean reward: {mean_reward:.2f}, episodes: {ep_idx}', (60, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, 2)
            frames.append(frame)
        
        if terminated or truncated:
            break
    
    last_rewards.append(ep_reward)
    mean_reward = sum(last_rewards) / len(last_rewards)
    
    print(f'Reward: {mean_reward}')

env.close()

imageio.mimsave(f'{game_name}-dqn.mp4', frames, fps=60)   
