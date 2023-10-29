import typing as tp

from pathlib import Path
import logging
import yaml
import math
from tqdm import tqdm
from contextlib import nullcontext
from dataclasses  import dataclass

import numpy as np
import torch
from tqdm import tqdm

from util import LogWriter, get_run_name, make_output_folder, BatchValueList
from model import get_model_num_params
from actor_critic_policy import ActorCriticPolicy, init_weights
from rollout import RolloutDataset


            
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


class PpoTrainer:
    

    def __init__(self,
                 cfg: tp.Dict[str, tp.Any]):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda')

        self.action_set = [action_idx for action_idx in range(int(self.cfg['model']['out_size']))]
        self.models: tp.Optional[ActorCriticPolicy] = None
        self.optimizer: tp.Optional[torch.optim.Optimizer] = None

        self.log_writer: tp.Optional[LogWriter] = None
        self._ext_values: tp.Dict[str, float] = {}

        self._init_log()
        self._init_models()
        
    def set_ext_values(self, **kwargs):
        self._ext_values = {value_name: value for value_name, value in kwargs.items()}
        
    def _init_log(self):
        training_cfg = self.cfg['training']

        run_name = get_run_name('ppo_%dt')
        run_output_folder = make_output_folder(training_cfg['output_folder'], run_name, False)
        
        logging.basicConfig(filename=run_output_folder / 'ppo.log',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            level=logging.INFO)
        self.cfg['meta'] = {'run_name': run_name}
        with open(run_output_folder / 'cfg.yaml', 'w') as cfg_file:
            yaml.safe_dump(self.cfg, cfg_file)
        output_weights_folder: Path = run_output_folder / 'weights'
        output_weights_folder.mkdir()
        self.log_writer = LogWriter(output_plot_folder=run_output_folder,
                                    project_name='RL', run_name=run_name,
                                    output_weights_folder=output_weights_folder,
                                    save_cfg=training_cfg['save'])

    def _init_models(self):
        model_cfg = dict(self.cfg['model'])
        training_cfg = dict(self.cfg['training'])
        
        self.models = ActorCriticPolicy.build(model_cfg)
        
        if 'models_path' in model_cfg:
            ckpt = torch.load(str(model_cfg['models_path']), map_location='cpu')
            self.models.load_state_dict(ckpt)
        else:
            self.models.init_weights
        
        self.models = self.models.to(self.device)
        
        self.logger.info(f'A size of the policy model: {get_model_num_params(self.models.policy_net)}')
        print(f'A size of the policy model: {get_model_num_params(self.models.policy_net)}')
        self.logger.info(f'A size of the value model: {get_model_num_params(self.models.value_net)}')
        print(f'A size of the value model: {get_model_num_params(self.models.value_net)}')
        
        non_frozen_actor_parameters = [param for param in self.models.policy_net.parameters() if param.requires_grad]
        non_frozen_critic_parameters = [param for param in self.models.value_net.parameters() if param.requires_grad]
        model_parameters = non_frozen_actor_parameters + non_frozen_critic_parameters
        self.optimizer = torch.optim.Adam(model_parameters,
                                          lr=float(training_cfg['lr']),
                                          betas=(0.9, 0.999), eps=1e-8)
        optimizer_path = model_cfg.get('optimizer_path', None)
        if optimizer_path is not None:
            optimizer_path = Path(optimizer_path)
            optimizer_ckpt = torch.load(optimizer_path)
            self.optimizer.load_state_dict(optimizer_ckpt)
            self.logger.info(f'Optimizer is loaded from "{optimizer_path}"')
        
    def train(self, dataset: RolloutDataset):
        
        training_cfg = dict(self.cfg['training'])
        max_grad_norm = training_cfg.get('max_grad_norm', None)
        n_epochs = int(training_cfg['n_epochs'])
        
        ppo_cfg = dict(training_cfg['ppo'])
        normalize_advantage = bool(ppo_cfg['normalize_advantage'])
        ppo_clip_range = float(ppo_cfg['clip_range'])
        entropy_coef = float(ppo_cfg['entropy_coef'])
        value_function_coef = float(ppo_cfg['value_function_coef'])
        
        explained_var = explained_variance(dataset.data.values.numpy(), dataset.data.returns.numpy())
        
        batch_value_list = self.log_writer.make_batch_value_list()
        self.models.train()
        
        progress_bar = tqdm(total=n_epochs * len(dataset), desc='Training')
        for _ in range(n_epochs):
            dataset.shuffle()
            
            for batch in dataset:
                batch.advantages = batch.advantages.detach()
                batch.returns = batch.returns.detach()
                
                batch.to_device(self.device)

                new_values, new_log_probs, new_entropy = self.models.evaluate_actions(batch.observations, batch.actions)

                advantages = batch.advantages
                if normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                prob_ratio = torch.exp(new_log_probs - batch.log_probs)

                policy_loss_1 = advantages * prob_ratio
                policy_loss_2 = advantages * torch.clamp(prob_ratio, 1 - ppo_clip_range, 1 + ppo_clip_range)
                policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()

                batch_value_list.add(float(policy_loss.cpu().item()), 'policy_loss')
                
                clip_fraction = torch.mean((torch.abs(prob_ratio - 1) > ppo_clip_range).float()).item()
                batch_value_list.add(clip_fraction, 'clip_fraction')

                value_loss = torch.nn.functional.mse_loss(batch.returns, new_values)
                batch_value_list.add(float(value_loss.cpu().item()), 'value_loss')

                entropy_loss = - torch.mean(new_entropy)
                batch_value_list.add(- float(entropy_loss.cpu().item()), 'entropy')

                loss = policy_loss + entropy_coef * entropy_loss + value_function_coef * value_loss
                batch_value_list.add(float(loss.cpu().item()), 'loss')

                # http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = new_log_probs - batch.log_probs
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu()
                    batch_value_list.add(float(approx_kl_div.cpu().item()), 'kl_divergence')

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.optimizer.parameters(), max_grad_norm)
                self.optimizer.step()
                
                progress_bar.update(1)
        
        batch_value_list.add(explained_var, 'explained_variance')
        for value_name, value in self._ext_values.items():
            batch_value_list.add(value, value_name)
            
        self.log_writer.add_batch_values(batch_value_list)
        self.log_writer.save_weights(self.models, self.optimizer)
        for subset_name, subset_values in self.log_writer.last_values.items():
            self.logger.info(f'Subset [{subset_name}]:')
            for metric_name, metric_value in subset_values.items():
                self.logger.info(f'  {metric_name}: {metric_value}')
        self.logger.info(' ')
        self.log_writer.save_plots()
        self.log_writer.save_weights(self.models, self.optimizer)
        self.log_writer.update_step()
                