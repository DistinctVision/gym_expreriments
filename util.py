import typing as tp
from pathlib import Path
from collections import defaultdict
from copy import deepcopy
import os
import json
import shutil
import logging

from datetime import datetime

import torch
import torch.distributed as dist
import plotly.graph_objects as go

# import wandb


def get_run_name(train_tag: tp.Optional[str] = None) -> str:
    d_now = datetime.now()
    if train_tag is None:
        train_tag = '%dt'
    if '%dt' in train_tag:
        train_tag = train_tag.replace('%dt', d_now.strftime('%m_%d_%Y__%H_%M_%S'))
    if '%date' in train_tag:
        train_tag = train_tag.replace('%date', d_now.strftime('%m_%d_%Y'))
    if '%time' in train_tag:
        train_tag = train_tag.replace('%time', d_now.strftime('%H_%M_%S'))
    return train_tag


def make_output_folder(output_folder: tp.Union[Path, str],
                       run_name: str,
                       overwrite: bool) -> tp.Tuple[str, Path]:
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
        
    output_run_folder = output_folder / run_name
    if overwrite and output_run_folder.exists():
        shutil.rmtree(output_run_folder)
    output_run_folder.mkdir()
    return output_run_folder


class BatchValueList:
    def __init__(self, subsets: tp.Optional[tp.List[str]] = None):
        if subsets is None:
            subsets = ['common']
        self.data = {subset: defaultdict(list) for subset in subsets}

    def clear(self):
        self.data = {subset: defaultdict(list) for subset in self.data}

    def add(self, value: float, metric_name: str, subset: str = 'common'):
        if subset not in self.data:
            raise KeyError(f'subset "{subset}" is not exist')
        self.data[subset][metric_name].append(value)
        
    def __len__(self) -> int:
        size = -1
        for subset in self.data.values():
            for values in subset.values():
                size = max(size,  len(values))
        return size

    def get_values(self) -> tp.Dict[str, tp.Dict[str, float]]:
        out = {
            subset: {metric_name: sum(values) / len(values) for metric_name, values in self.data[subset].items()
                     if len(values) > 0}
            for subset in self.data
        }
        out = {subset: values for subset, values in out.items() if len(values) > 0}
        return out
    
    def ddp_gather(self):
        device = torch.device(f'cuda:{dist.get_rank()}')
        num_processes = dist.get_world_size()
        for subset in self.data.values():
            for metric_name in subset:
                values_tensor = torch.tensor(subset[metric_name], dtype=torch.float32, device=device)
                output_objects: tp.List[tp.Optional[torch.Tensor]] = [None for _ in range(num_processes)]
                dist.all_gather_object(output_objects, values_tensor)
                output_objects = [v_tensor.cpu().float().tolist() for v_tensor in output_objects]
                values = sum(output_objects, [])
                subset[metric_name] = values


class LogWriter:

    def __init__(self,
                 output_plot_folder: tp.Union[Path, str],
                 output_samples_folder: tp.Optional[tp.Union[Path, str]] = None,
                 output_tensorboard_folder: tp.Optional[tp.Union[Path, str]] = None,
                 use_wandb: bool = False,
                 project_name: tp.Optional[str] = None,
                 run_name: tp.Optional[str] = None,
                 subsets: tp.Optional[tp.List[str]] = None,
                 output_weights_folder: tp.Optional[tp.Union[Path, str]] = None,
                 save_cfg: tp.Optional[tp.Dict[str, int]] = None):
        self.output_plot_folder = Path(output_plot_folder)
        self.step = 1
        self._steps = []
        if subsets is None:
            subsets = ['common']
        self._data = {subset_name: {} for subset_name in subsets}
        self.output_samples_folder = Path(output_samples_folder) \
            if output_samples_folder is not None else None
        if output_tensorboard_folder is not None:
            self.output_tensorboard_folder = Path(output_tensorboard_folder)
            self.summary_writer = SummaryWriter(log_dir=str(self.output_tensorboard_folder))
        else:
            self.output_tensorboard_folder = None
            self.summary_writer = None
        if use_wandb:
            self.wandb = wandb.init(
                project=project_name,
                name=run_name,
            )
        else:
            self.wandb = None
        self.last_values: tp.Optional[tp.Dict[str, tp.Dict[str, float]]] = None
        self.save_cfg = save_cfg
        if output_weights_folder is not None:
            assert self.save_cfg is not None
            assert 'save_every_n_step' in self.save_cfg
            assert 'n_last_steps' in self.save_cfg
            
            self.save_cfg = save_cfg
            self.output_weights_folder = Path(output_weights_folder)
            self.best_weights: tp.Tuple[tp.Optional[float], tp.Optional[Path]] = (None, None)
            self.last_weights_paths: tp.List[Path] = []
            self.last_optimizer_paths: tp.List[Path] = []
        else:
            self.save_cfg = None
            self.output_weights_folder = None
            self.last_weights_paths = None
            self.best_weights = None
            self.last_optimizer_paths = None
            
    def make_batch_value_list(self) -> BatchValueList:
        return BatchValueList(list(self._data.keys()))

    def add_batch_values(self, batch_values: BatchValueList):
        default_value = 0.0

        metric_names = set()
        values = batch_values.get_values()
        self.last_values = deepcopy(values)
        for subset, subset_values in values.items():
            subset_data = self._data[subset]
            for metric_name, value in subset_values.items():
                if metric_name not in subset_data:
                    metric_values = [default_value] * len(self._steps)
                    subset_data[metric_name] = metric_values
                else:
                    metric_values = subset_data[metric_name]
                metric_values.append(value)
                metric_names.add(metric_name)
        if self.summary_writer is not None:
            for metric_name in metric_names:
                m_values = {}
                for subset in self._data:
                    if subset in values and metric_name in values[subset]:
                        m_values[subset] = values[subset][metric_name]
                if len(m_values) > 1:
                    self.summary_writer.add_scalars(metric_name, m_values, self.step)
                else:
                    subset, value = next(iter(m_values.items()))
                    self.summary_writer.add_scalar(metric_name, value, self.step)
        if self.wandb is not None:
            self.wandb.log(data=values, step=self.step)

    def update_step(self):
        self._steps.append(self.step)
        self.step += 1

    @staticmethod
    def _resolve_name(name: str) -> str:
        return name.replace('/', '__')

    def save_plots(self):
        steps = self._steps + [self.step]
        metric_names = set()
        scatters = {}
        for subset, subset_values in self._data.items():
            scatters[subset] = {}
            for metric_name, metric_values in subset_values.items():
                metric_names.add(metric_name)
                if len(steps) > 1:
                    scatter = go.Scatter(x=steps[1:], y=metric_values[1:], name=f'{subset}_{metric_name}')
                else:
                    scatter = go.Scatter(x=steps, y=metric_values, name=f'{subset}_{metric_name}')
                # scatter = go.Scatter(x=steps, y=metric_values, name=f'{subset}_{metric_name}')
                scatters[subset][metric_name] = scatter
        for metric_name in metric_names:
            figure = go.Figure()
            for subset in self._data.keys():
                if metric_name not in self._data[subset]:
                    continue
                figure.add_trace(scatters[subset][metric_name])
            out_plot_path = self.output_plot_folder / f'{self._resolve_name(metric_name)}.png'
            out_plot_path = out_plot_path.absolute()
            figure.write_image(out_plot_path)
        if len(metric_names) > 1:
            figure = go.Figure()
            for _, subset_scatters in scatters.items():
                for metric_name, metric_scatter in subset_scatters.items():
                    figure.add_trace(metric_scatter)
            out_plot_path = self.output_plot_folder / 'all.png'
            figure.write_image(out_plot_path)
            
    def save_weights(self, model: torch.nn.Module, optimizator: tp.Optional[torch.optim.Optimizer] = None):
        if self.save_cfg is None:
            return
        model_name = str(self.save_cfg.get('model_name', 'model'))
        cfg_target_metric = self.save_cfg.get('target_metric', None)
        if self.last_values is not None and cfg_target_metric is  not None:
            target_metric = cfg_target_metric.split('.')
            assert 1 <= len(target_metric) <= 2
            if len(target_metric) == 1:
                target_subset = 'common'
                target_metric = target_metric[0]
            elif len(target_metric) == 2:
                target_subset = target_metric[0]
                target_metric = target_metric[1]
            else:
                raise RuntimeError(f"The invalid target metric: {self.save_cfg['target_metric']}")
            if target_subset in self.last_values and target_metric in self.last_values[target_subset]:
                target_metric_value = self.last_values[target_subset][target_metric]
                if self.best_weights[0] is None or self.best_weights[0] > target_metric_value:
                    model_ckpt_path: Path = self.output_weights_folder / f'best_{model_name}.kpt'
                    print(f'Best model! {cfg_target_metric} = {target_metric_value:.4}')
                    torch.save(model.state_dict(), model_ckpt_path)
                    self.best_weights = (target_metric_value, model_ckpt_path)
                    best_metric_json_filepath = model_ckpt_path.parent / 'best_metrics.json'
                    with open(best_metric_json_filepath, 'w') as json_file:
                        json.dump(self.last_values, json_file, indent=4)
            else:
                logging.warning(f'"{cfg_target_metric}" is not found in metric values')
                    
        if self.step % int(self.save_cfg['save_every_n_step']) != 0:
            return
        n_last_steps = int(self.save_cfg['n_last_steps'])
        
        model_ckpt_path = self.output_weights_folder / f'{model_name}_{self.step}.kpt'
        torch.save(model.state_dict(), model_ckpt_path)
        self.last_weights_paths.append(model_ckpt_path)
        if len(self.last_weights_paths) > n_last_steps:
            del_paths = self.last_weights_paths[:len(self.last_weights_paths) - n_last_steps]
            for model_ckpt_path in del_paths:
                os.remove(model_ckpt_path)
            self.last_weights_paths = self.last_weights_paths[-n_last_steps:]
        
        if optimizator is not None:
            opt_ckpt_path = self.output_weights_folder / f'opt_{model_name}_{self.step}.kpt'
            torch.save(optimizator.state_dict(), opt_ckpt_path)
            self.last_optimizer_paths.append(opt_ckpt_path)
            
            if len(self.last_optimizer_paths) > n_last_steps:
                del_paths = self.last_optimizer_paths[:len(self.last_optimizer_paths) - n_last_steps]
                for opt_ckpt_path in del_paths:
                    os.remove(opt_ckpt_path)
                self.last_optimizer_paths = self.last_optimizer_paths[-n_last_steps:]
            
    def save_optimizator(self, optimizator: torch.optim.Optimizer, ext_name: tp.Optional[str] = None):
        if self.save_cfg is None:
            return
        if self.step % int(self.save_cfg['save_every_n_step']) != 0:
            return
        model_name = str(self.save_cfg.get('model_name', 'model'))
        n_last_steps = int(self.save_cfg['n_last_steps'])
        if ext_name is None:
            ext_name = ''
        
        opt_ckpt_path = self.output_weights_folder / f'opt_{model_name}_{self.step}{ext_name}.kpt'
        torch.save(optimizator.state_dict(), opt_ckpt_path)
        self.last_optimizer_paths.append(opt_ckpt_path)
        
        if len(self.last_optimizer_paths) > n_last_steps:
            del_paths = self.last_optimizer_paths[:len(self.last_optimizer_paths) - n_last_steps]
            for opt_ckpt_path in del_paths:
                os.remove(opt_ckpt_path)
            self.last_optimizer_paths = self.last_optimizer_paths[-n_last_steps:]

    @property
    def samples_are_saving(self) -> bool:
        if self.output_tensorboard_folder is None and self.output_samples_folder is None:
            return False
        return True

    def finish(self):
        if self.summary_writer is not None:
            self.summary_writer.close()
        if self.wandb is not None:
            self.wandb.finish()
