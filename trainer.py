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
import einops

from util import LogWriter, get_run_name, make_output_folder, BatchValueList
from data import ReplayBuffer,  SeqRecords
from model import CriticModel, get_model_num_params


@dataclass
class TrainingData:
    grad_accum_counter: int
    batch_value_list: BatchValueList
    local_step: int = 0
    

class Trainer:

    def __init__(self,
                 cfg: tp.Dict[str, tp.Any],
                 replay_buffer: ReplayBuffer):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda:0')

        self.action_set = [action_idx for action_idx in range(int(self.cfg['model']['out_size']))]
        self.replay_buffer = replay_buffer
        self.model: tp.Optional[CriticModel] = None
        self.target_model: tp.Optional[CriticModel] = None
        self.optimizer: tp.Optional[torch.optim.Optimizer] = None

        self.log_writer: tp.Optional[LogWriter] = None

        self._init_models()
        self._init_log()
        
        self._training_data = TrainingData(0, self.log_writer.make_batch_value_list())
        
    @property
    def step(self) -> int:
        return self.log_writer.step
    
    def add_metric_value(self,  metric_name: str,  metric_value: float):
        self._training_data.batch_value_list.add(metric_value, metric_name)

    def _init_models(self):
        model_cfg = dict(self.cfg['model'])
        training_cfg = dict(self.cfg['training'])
        
        self.model = CriticModel.build_model(model_cfg)
        if 'critic_model_path' in model_cfg:
            ckpt = torch.load(str(model_cfg['critic_model_path']), map_location='cpu')
            self.model.load_state_dict(ckpt)
        else:
            self.model.init_weights()
        self.target_model = CriticModel.build_model(model_cfg)
        self.target_model.load_state_dict(self.model.state_dict())
        self.model = self.model.to(self.device).eval()
        self.target_model = self.target_model.to(self.device)
        self.target_model.train()
        
        self.logger.info(f'A size of the model: {get_model_num_params(self.model)}')
        print(f'A size of the model: {get_model_num_params(self.model)}')
        
        non_frozen_critic_parameters = [param for param in self.target_model.parameters() if param.requires_grad]
        self.optimizer = torch.optim.Adam(non_frozen_critic_parameters,
                                          lr=float(training_cfg['lr']),
                                          betas=(0.9, 0.999), eps=1e-8)
        optimizer_path = model_cfg.get('critic_optimizer_path', None)
        if optimizer_path is not None:
            optimizer_path = Path(optimizer_path)
            optimizer_ckpt = torch.load(optimizer_path)
            self.optimizer.load_state_dict(optimizer_ckpt)
            self.logger.info(f'Optimizer is loaded from "{optimizer_path}"')


    def _init_log(self):
        training_cfg = self.cfg['training']

        run_name = get_run_name('dqn_%dt')
        run_output_folder = make_output_folder(training_cfg['output_folder'], run_name, False)
        
        logging.basicConfig(filename=run_output_folder / 'dqn.log',
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
    
    
    def _soft_model_update(self, update_rate: float):
        with torch.no_grad():
            inv_update_rate = 1.0 - update_rate
            
            state_dict = self.target_model.state_dict()
            state_dict = tp.cast(tp.Dict[str, torch.Tensor], state_dict)
            for item_key, item in self.model.state_dict().items():
                item = tp.cast(torch.Tensor, item)
                if item.dtype.is_floating_point:
                    item = item * inv_update_rate + update_rate * state_dict[item_key].detach()
    
    
    def _get_eps_greedy_coeff(self) -> float:
        training_cfg = self.cfg['training']
        eps_greedy_cfg = dict(training_cfg['eps_greedy'])
        eps_from = float(eps_greedy_cfg['eps_from'])
        eps_to = float(eps_greedy_cfg['eps_to'])
        n_epochs_of_decays = int(eps_greedy_cfg['n_epochs_of_decays'])

        step_coeff = min(max(self.step / n_epochs_of_decays, 0.0), 1.0)
        eps_greedy_coeff = eps_from + (eps_to - eps_from) * step_coeff
        return eps_greedy_coeff
    
    
    def train_step(self):
        
        training_cfg: dict = self.cfg['training']
        
        n_grad_accum_steps: int = training_cfg.get('n_grad_accum_steps', 1)
        batch_size: int = training_cfg['batch_size']
        grad_norm: tp.Optional[float] = training_cfg.get('grad_norm', None)
            
        if bool(training_cfg['fp16']):
            precision_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
            grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            precision_ctx = nullcontext()
            grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
        
        batch_indices = self.replay_buffer.sample_batch_indices(batch_size)
        cur_records_batch, next_records_batch, mask_done = self.replay_buffer.sample_batch(batch_indices)
        sync_grad = (self._training_data.grad_accum_counter + 1) >= n_grad_accum_steps
                
        with precision_ctx:
            batch_world_states_tensor = next_records_batch.world_states.to(self.device)
            with torch.no_grad():
                next_pr_rewards = self.model(batch_world_states_tensor=batch_world_states_tensor)
                next_pr_rewards = tp.cast(torch.Tensor, next_pr_rewards)
                next_pr_rewards[mask_done][:] = 0.0
                next_pr_rewards = next_pr_rewards.max(-1)[0].detach()
                
            batch_world_states_tensor = cur_records_batch.world_states.to(self.device)
            rewards = (cur_records_batch.rewards.to(self.device) + next_pr_rewards * self.model.reward_decay)
            with nullcontext():
                pr_rewards: torch.Tensor = self.target_model(batch_world_states_tensor=batch_world_states_tensor)
                action_indices = cur_records_batch.action_indices.unsqueeze(1).long().to(self.device)
                pr_rewards = pr_rewards.gather(dim=1, index=action_indices).squeeze(1)
                
                loss = torch.nn.functional.smooth_l1_loss(pr_rewards, rewards)
                
                loss /= n_grad_accum_steps
                grad_scaler.scale(loss).backward()
                
                loss_value = float(loss.detach().cpu()) * n_grad_accum_steps
            
            if sync_grad:
                if grad_norm is not None:
                    grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.target_model.parameters(), max_norm=grad_norm)
                grad_scaler.step(self.optimizer)
                grad_scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self._training_data.grad_accum_counter = 0
            else:
                self._training_data.grad_accum_counter += 1
        self._training_data.batch_value_list.add(loss_value, 'loss')
        
        eps_greedy = self._get_eps_greedy_coeff()
        self._training_data.batch_value_list.add(eps_greedy, 'eps_greedy')
        
        self._training_data.local_step += 1
        n_local_steps = int(training_cfg['n_local_steps'])
        if self._training_data.local_step >= n_local_steps:
            self._training_data.local_step = 0
            self.log_writer.add_batch_values(self._training_data.batch_value_list)
            for subset_name, subset_values in self.log_writer.last_values.items():
                self.logger.info(f'Subset [{subset_name}]:')
                for metric_name, metric_value in subset_values.items():
                    self.logger.info(f'  {metric_name}: {metric_value}')
            self.logger.info(' ')
            self.log_writer.save_plots()
            self.log_writer.save_weights(self.model, self.optimizer)
            self.log_writer.update_step()
            self._training_data.batch_value_list = BatchValueList()
        
        model_update_cfg: dict = training_cfg['model_update']
        global_step = self.log_writer.step * n_local_steps + self._training_data.local_step
        if global_step % model_update_cfg['n_steps'] == 0:
            model_update_type: str =  model_update_cfg['type']
            if model_update_type == 'soft':
                self._soft_model_update(training_cfg)
            elif model_update_type == 'hard':
                self.model.load_state_dict(self.target_model.state_dict())
            else:
                raise RuntimeError(f'Unknown model update type: {model_update_type}')
        

class EpisodeDataRecorder:
    
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.current_episode = SeqRecords()
    
    def __len__(self) -> int:
        return len(self.current_episode)
    
    def get_action(self, world_state_tensor: torch.Tensor) -> int:
        return self._get_action_idx(world_state_tensor)
    
    def record(self,
               world_state_tensor: torch.Tensor,
               action: int, reward: float,
               episode_is_done: bool) -> bool:
        self.current_episode.add(world_state_tensor, action, reward)
        
        data_is_updated = False
        if episode_is_done:
            data_is_updated = self.trainer.replay_buffer.add_episode(self.current_episode)
            self.current_episode = SeqRecords()
            max_buffer_size = int(self.trainer.cfg['replay_buffer']['max_buffer_size'])
            while len(self.trainer.replay_buffer) > max_buffer_size:
                self.trainer.replay_buffer.remove_first_episode()
                
        return data_is_updated
    
    def _get_action_idx(self, world_state_tensor: torch.Tensor) -> int:
        eps_greedy_coeff = self.trainer._get_eps_greedy_coeff()
        if np.random.uniform(0, 1) < eps_greedy_coeff:
            action_idx = np.random.choice(self.trainer.action_set)
        else:
            model = self.trainer.model
            with torch.no_grad():
                pr_rewards: torch.Tensor = model(
                    batch_world_states_tensor=world_state_tensor.unsqueeze(0).to(model.device))
                pr_rewards = pr_rewards.squeeze(0).cpu()
            for action_idx in range(pr_rewards.shape[0]):
                if action_idx not in self.trainer.action_set:
                    pr_rewards[action_idx] = -10000
            action_idx = pr_rewards.argmax(0)
        return int(action_idx)