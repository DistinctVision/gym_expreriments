import typing as tp

import torch


def build_mlp_net(in_size: int, out_size: int,
                  layers: tp.List[int], dropout: float) -> torch.nn.Module:
    
    blocks = [torch.nn.Sequential(torch.nn.Linear(in_size, layers[0]),
                                  torch.nn.ReLU())]
    for layer_idx in range(len(layers) - 1):
        blocks.append(torch.nn.Sequential(torch.nn.Dropout(dropout),
                                          torch.nn.Linear(layers[layer_idx], layers[layer_idx+1]),
                                          torch.nn.ReLU()))
    blocks.append(torch.nn.Sequential(torch.nn.Dropout(dropout),
                                      torch.nn.Linear(layers[-1], out_size)))
    return torch.nn.Sequential(*blocks)


def xavier_init(module: torch.nn.Module, gain: float = 1, bias: float = 0, distribution: str = 'normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            torch.nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            torch.nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        torch.nn.init.constant_(module.bias, bias)
        

def orthogonal_init(module: torch.nn.Module, gain: float = 1):
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


def kaiming_init(module: torch.nn.Module,
                 a: float = 0,
                 mode: str = 'fan_out',
                 nonlinearity: str = 'relu',
                 bias: float = 0,
                 distribution: float = 'normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            torch.nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            torch.nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        torch.nn.init.constant_(module.bias, bias)


def constant_init(module: torch.nn.Module, val: float, bias: float = 0):
    if hasattr(module, 'weight') and module.weight is not None:
        torch.nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        torch.nn.init.constant_(module.bias, bias)
        

def init_weights(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            orthogonal_init(m)
        elif isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.GroupNorm)):
            constant_init(m, 1)


class ActorCriticPolicy(torch.nn.Module):
    
    @staticmethod
    def build(model_cfg: tp.Dict[str, tp.Union[int, float, tp.List[int]]]) -> 'ActorCriticPolicy':
        policy_net_layers = [int(layer) for layer in list(model_cfg['policy_net_layers'])]
        value_net_layers = [int(layer) for layer in list(model_cfg['value_net_layers'])]
        return ActorCriticPolicy(in_size=int(model_cfg['in_size']),
                                 out_size=int(model_cfg['out_size']),
                                 policy_net_layers=policy_net_layers,
                                 value_net_layers=value_net_layers,
                                 dropout=float(model_cfg['dropout']))
    
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 policy_net_layers: tp.List[int],
                 value_net_layers: tp.List[int],
                 dropout: float = 0.1):
        super().__init__()
        
        self.in_size = in_size
        self.out_size = out_size
        
        self.policy_net = build_mlp_net(in_size, out_size, policy_net_layers, dropout)
        self.value_net = build_mlp_net(in_size, 1, value_net_layers, dropout)
        
    @property
    def device(self) -> torch.device:
        first_param = next(iter(self.value_net.parameters()))
        return first_param.device
        
    def init_weights(self):
        init_weights(self.policy_net)
        init_weights(self.value_net)
    
    def get_action_dist(self, observation: torch.Tensor) -> torch.distributions.Categorical:
        device = observation.device
        logits = self.policy_net(observation.to(self.device)).to(device)
        action_prob = torch.nn.functional.softmax(logits, dim=-1)
        return torch.distributions.Categorical(action_prob)
    
    def evaluate_actions(self,
                         observations: torch.Tensor,
                         actions: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy, given the observations.

        Args:
            observations: Observations
            actions: Actions
            
        Returns:
            Estimated value, log likelihood of taking those actions and entropy of the action distribution.
        """
        
        logits = self.policy_net(observations)
        action_prob = torch.nn.functional.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_prob)
        action_log_prob = action_dist.log_prob(actions)
        values = self.value_net(observations)
        values = values.squeeze(1)
        return values, action_log_prob, action_dist.entropy()
    