import typing as tp
import math
import functools
import torch
from torch.nn import functional as F


def get_model_num_params(model: torch.nn.Module) -> str:
    """
    Return the number of model parameters in text format 
    """
    n_params = sum(p.numel() for p in model.parameters())
    
    stage_postfixes = ['', ' M', ' B']
    stage_numbers = [1, 10 ** 6, 10 ** 9]
    
    stage_idx = 0
    for idx, stage_n in enumerate(stage_numbers):
        if n_params > stage_n:
            stage_idx = idx
    
    n_params /= stage_numbers[stage_idx]
    return f'{n_params}{stage_postfixes[stage_idx]}'

    
class CriticModel(torch.nn.Module):
    
    @staticmethod
    def build_model(critic_cfg: tp.Dict[str, tp.Union[int, float]]) -> 'CriticModel':
        return CriticModel(in_size=int(critic_cfg['in_size']),
                           out_size=int(critic_cfg['out_size']),
                           reward_decay=float(critic_cfg['reward_decay']),
                           layers=[int(layer) for layer in list(critic_cfg['layers'])],
                           dropout=float(critic_cfg['dropout']))
    
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 reward_decay: float,
                 layers: tp.List[int],
                 dropout: float = 0.1):
        super().__init__()
        self.reward_decay = reward_decay
        
        self.in_proj = torch.nn.Sequential(torch.nn.Linear(in_size, layers[0]), torch.nn.ReLU())
        
        block_layers = []
        for layer_in, layar_out in zip(layers[:-1], layers[1:]):
            block_layers.append(torch.nn.Sequential(torch.nn.Dropout(dropout),
                                                    torch.nn.Linear(layer_in, layar_out),
                                                    torch.nn.ReLU()))
        self.blocks = torch.nn.Sequential(*block_layers)
        self.out_proj = torch.nn.Linear(layers[-1], out_size, bias=True)
    
    @property
    def device(self) -> torch.device:
        return self.out_proj.weight.device
        
    def init_weights(self):
        torch.nn.init.kaiming_uniform_(self.in_proj[0].weight)
        torch.nn.init.uniform_(self.in_proj[0].bias, a=-5e-3, b=5e-3)
        
        def _init_fn(depth: int, module: torch.nn.Module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.uniform_(module.bias, a=-5e-3, b=5e-3)
                
        for depth, block in enumerate(self.blocks):
            block.apply(functools.partial(_init_fn, depth))
        
        torch.nn.init.kaiming_uniform_(self.out_proj.weight)
        torch.nn.init.uniform_(self.out_proj.bias, a=-5e-3, b=5e-3)
    
    def forward(self, batch_world_states_tensor: torch.Tensor) -> torch.Tensor:
        
        in_state = batch_world_states_tensor
        
        x: torch.Tensor = self.in_proj(in_state)
        x = self.blocks(x)
        x = self.out_proj(x)
        return x
