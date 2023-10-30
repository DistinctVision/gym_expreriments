import yaml
from model import CriticModel

import numpy as np
import torch

import gymnasium as gym


cfg = yaml.safe_load(open('dqn_cfg.yaml', 'r'))
game_name = str(cfg['game']['name'])


if game_name == 'CartPole-v1':
    std_world_state = torch.tensor([0.0896, 0.5494, 0.0921, 0.8163], dtype=torch.float32)
    mean_world_state = torch.tensor([0.0021, -0.0457,  0.0094,  0.0996], dtype=torch.float32)
elif game_name == 'LunarLander-v2':
    std_world_state = torch.tensor([1.5, 1.5, 5., 5., 3.1415927, 5., 1., 1.], dtype=torch.float32)
    mean_world_state = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
else:
    raise RuntimeError('Unknown game')


def preprocess(world_state: np.ndarray) -> torch.Tensor:
    world_state = torch.from_numpy(world_state)
    return (world_state - mean_world_state) / std_world_state


env = gym.make(game_name, render_mode="human")

model = CriticModel.build_model(cfg['model'])
ckpt = torch.load(cfg['model']['critic_model_path'], map_location='cpu')
model.load_state_dict(ckpt)

while True:
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
        
        env.render()
        
        if terminated or truncated:
            break
    
    print(f'Reward: {ep_reward}')

env.close()