import typing as tp
from pathlib import Path
from collections import deque

from dataclasses import dataclass, field
from threading import RLock
import logging
import shutil

import random

import torch

from tqdm import tqdm


@dataclass
class Record:
    world_state_tensor: torch.Tensor
    action_index: int
    reward: float


@dataclass
class SeqRecords:
    world_state_tensors:  tp.List[torch.Tensor] = field(default_factory=lambda: [])
    acton_indices: tp.List[int] = field(default_factory=lambda: [])
    rewards: tp.List[float] = field(default_factory=lambda: [])
    
    def __len__(self) -> int:
        return len(self.world_state_tensors)

    def add(self,
            world_state_tensor: torch.Tensor,
            action_index: int,
            reward: float):
        self.world_state_tensors.append(world_state_tensor)
        self.acton_indices.append(action_index)
        self.rewards.append(reward)
    

@dataclass
class RecordArray:
    world_states: torch.Tensor
    action_indices: torch.Tensor
    rewards: torch.Tensor

    @staticmethod
    def from_seq(seq_data: SeqRecords) -> 'RecordArray':
        world_states = torch.stack(seq_data.world_state_tensors)
        action_indices = torch.tensor(seq_data.acton_indices, dtype=torch.int16)
        rewards = torch.tensor(seq_data.rewards, dtype=torch.float32)

        cached_array = RecordArray(world_states=world_states,
                                   action_indices=action_indices,
                                   rewards=rewards)
        return cached_array

    def __len__(self) -> int:
        return self.world_states.shape[0]


class ReplayBuffer:
    
    @staticmethod
    def load(folder_path: tp.Union[Path, str], progress_desc: str = 'Loading') -> 'ReplayBuffer':
        rp = ReplayBuffer()
        
        episode_paths: tp.List[Path] = []
        for ep_folder_path in folder_path.iterdir():
            if not ep_folder_path.is_dir():
                continue
            episode_paths.append(ep_folder_path)
        
        for ep_folder_path in tqdm(episode_paths,  desc=progress_desc):
            world_states = torch.load(ep_folder_path / 'world_states.pth')
            action_indices = torch.load(ep_folder_path / 'action_indices.pth')
            rewards = torch.load(ep_folder_path / 'rewards.pth')
            episode = RecordArray(world_states, action_indices, rewards)
            rp.buffer.append(episode)
        return rp
    
    def __init__(self):
        self.buffer: tp.Deque[RecordArray] = deque()
    
    def __len__(self) -> int:
        return sum([len(session) for session in self.buffer])
    
    def make_indices(self) -> tp.List[tp.Tuple[int, int]]:
        indices = []
        for session_idx, session in enumerate(self.buffer):
            for frame_idx in range(len(session)):
                indices.append((session_idx, frame_idx))
        return indices
    
    @property
    def num_episodes(self) -> int:
        return len(self.buffer)
    
    def num_records(self, episode_index: int) -> int:
        return len(self.buffer[episode_index])
    
    def get_record(self, episode_index: int, frame_index: int) -> Record:
        episode = self.buffer[episode_index]
        return Record(episode.world_states[frame_index,  :],
                      int(episode.action_indices[frame_index]),
                      float(episode.rewards[frame_index]))

    def add_episode(self, episode: SeqRecords):
        self.buffer.append(RecordArray.from_seq(episode))
        
    def clear(self):
        self.buffer = deque()
        
    def remove_first_episode(self):
        del self.buffer[0]
        
    def save(self, folder_path: tp.Union[Path, str], progress_desc: str = 'Saving'):
        folder_path = Path(folder_path)
        if not folder_path.exists():
            folder_path.mkdir(parents=True)
        for episode_idx, episode in enumerate(tqdm(self.buffer, desc=progress_desc)):
            episode_folder = folder_path / f'ep_{episode_idx}'
            episode_folder.mkdir()
            episode = tp.cast(RecordArray, episode)
            torch.save(episode.world_states, episode_folder / 'world_states.pth')
            torch.save(episode.action_indices, episode_folder / 'action_indices.pth')
            torch.save(episode.rewards, episode_folder / 'rewards.pth')
            
    def sample_batch_indices(self, batch_size: int) -> tp.List[int]:
        batch_indices = []
        for _ in range(batch_size):
            rand_ep_idx = random.randint(0, self.num_episodes - 1)
            rand_record_idx = random.randint(0, self.num_records(rand_ep_idx) - 1)
            batch_indices.append((rand_ep_idx, rand_record_idx))
        return batch_indices
    
    def get_seq_batch(self, batch_indices: tp.List[tp.Tuple[int,  int]],
                      sequence_size: int) -> tp.Tuple[RecordArray, RecordArray, torch.Tensor]:
        
        batch_episodes = [self.buffer[idx[0]] for idx in batch_indices]
        
        world_states = [episode.world_states[idx[1]:idx[1]+sequence_size]
                        for idx, episode in zip(batch_indices, batch_episodes)]
        action_indices = [episode.action_indices[idx[1]:idx[1]+sequence_size]
                         for idx, episode in zip(batch_indices, batch_episodes)]
        rewards = [episode.rewards[idx[1]:idx[1]+sequence_size]
                   for idx, episode in zip(batch_indices, batch_episodes)]
        for el_idx in range(len(batch_indices)):
            ext_size = sequence_size - world_states[el_idx].shape[0]
            if ext_size <= 0:
                continue
            ext_world_states = torch.stack([world_states[el_idx][0, :]] * ext_size)
            ext_action_indices = torch.stack([action_indices[el_idx][0]] * ext_size)
            ext_rewards = torch.stack([rewards[el_idx][0]] * ext_size)
            world_states[el_idx] = torch.cat([ext_world_states, world_states[el_idx]])
            action_indices[el_idx] = torch.cat([ext_action_indices, action_indices[el_idx]])
            rewards[el_idx] = torch.cat([ext_rewards, rewards[el_idx]])
        cur_batch = RecordArray(world_states=torch.stack(world_states),
                                action_indices=torch.stack(action_indices),
                                rewards=torch.stack(rewards))
        
        next_world_states = []
        next_action_indices = []
        next_rewards = []
        mask_done = []
        for idx, episode, cur_world_states, cur_action_indices, cur_rewards in \
                zip(batch_indices, batch_episodes, world_states, action_indices, rewards):
            done = (idx[1] + sequence_size) >= len(episode)
            if done:
                mask_done.append(True)
                next_world_states.append(torch.cat([cur_world_states[1:, :], cur_world_states[-1, :].unsqueeze(0)]))
                next_action_indices.append(torch.cat([cur_action_indices[1:], cur_action_indices[-1].unsqueeze(0)]))
                next_rewards.append(torch.cat([cur_rewards[1:], cur_rewards[-1].unsqueeze(0)]))
                continue
            mask_done.append(False)
            next_world_states.append(torch.cat([cur_world_states[1:, :],
                                                episode.world_states[idx[1] + sequence_size, :].unsqueeze(0)]))
            next_action_indices.append(torch.cat([cur_action_indices[1:],
                                                  episode.action_indices[idx[1] + sequence_size].unsqueeze(0)]))
            next_rewards.append(torch.cat([cur_rewards[1:], episode.rewards[idx[1] + sequence_size].unsqueeze(0)]))
        next_batch = RecordArray(world_states=torch.stack(next_world_states),
                                 action_indices=torch.stack(next_action_indices),
                                 rewards=torch.stack(next_rewards))
        
        return cur_batch, next_batch, torch.tensor(mask_done, dtype=torch.bool)
            
            

class ReplayBufferBatchSampler(torch.utils.data.BatchSampler):

    def __init__(self,
                 batch_size: int,
                 indices: tp.List[tp.Tuple[int, int]]):
        super().__init__(sampler=None, batch_size=batch_size, drop_last=True)
        self.indices = indices

    def __iter__(self) -> tp.Iterator[tp.List[tp.Tuple[int, int]]]:
        return (self.get_batch_indices(batch_idx) for batch_idx in range(len(self)))

    def __len__(self) -> int:
        return len(self.indices) // self.batch_size

    def get_batch_indices(self, batch_index: int) -> tp.List[tp.Tuple[int, int]]:
        offset = batch_index * self.batch_size
        return [self.indices[item_index] for item_index in range(offset, offset + self.batch_size)]

            
class Dataset:
    
    def __init__(self,
                 cached_replay_buffer: ReplayBuffer,
                 batch_sampler: ReplayBufferBatchSampler,
                 sequence_size: int):
        self.cached_replay_buffer = cached_replay_buffer
        self.batch_sampler = batch_sampler
        self.sequence_size = sequence_size
    
    def get_batch(self, index: int) -> tp.Tuple[RecordArray, RecordArray, torch.Tensor]:
        batch_indices = self.batch_sampler.get_batch_indices(index)
        return self.cached_replay_buffer.get_seq_batch(batch_indices, self.sequence_size)
        
    def __len__(self) -> int:
        return len(self.batch_sampler)
    
    def __iter__(self) -> tp.Iterator[tp.Tuple[RecordArray, RecordArray, torch.Tensor]]:
        return (self.get_batch(item_index) for item_index in range(len(self)))

    def __getitem__(self, index: int) -> tp.List[tp.Tuple[RecordArray, RecordArray, torch.Tensor]]:
        return self.get_batch(index)


class SharedDataCollector:
    
    def __init__(self, cfg: tp.Dict[str, tp.Any]):
        self.logger = logging.getLogger(__name__)
        self._locker = RLock()
        
        self.training_cfg = dict(cfg['training'])
        self.replay_buffer_cfg = dict(cfg['replay_buffer'])
        self.min_episode_size = int(self.replay_buffer_cfg['min_episode_size'])
        self.max_episode_size = int(self.replay_buffer_cfg['max_episode_size'])

        self.rp_buffers = {'train': ReplayBuffer(),
                           'val': ReplayBuffer()}
        self.current_rp_buffer = ReplayBuffer()
        
        if 'data_dir' in self.replay_buffer_cfg:
            self.load(str(self.replay_buffer_cfg['data_dir']))
            
            max_buffer_size = int(self.replay_buffer_cfg['max_buffer_size'])
            train_val_ratio = float(self.training_cfg['train_size']) / float(self.training_cfg['val_size'])
            cur_train_val_ratio = len(self.rp_buffers['train']) / len(self.rp_buffers['val'])
            
            while (len(self.rp_buffers['train']) + len(self.rp_buffers['val'])) > max_buffer_size:
                target_rp_buff = self.rp_buffers['val' if cur_train_val_ratio < train_val_ratio else 'train']
                target_rp_buff.buffer = target_rp_buff.buffer[1:]
                cur_train_val_ratio = len(self.rp_buffers['train']) / len(self.rp_buffers['val'])
    
    @property
    def data_size(self) -> int:
        return len(self.rp_buffers['train']) + len(self.rp_buffers['val'])
    
    @property
    def n_episodes(self) -> int:
        return self.rp_buffers['train'].num_episodes + self.rp_buffers['val'].num_episodes
    
    def add_episode(self, current_episode: SeqRecords) -> bool:
        if len(current_episode) < self.min_episode_size:
            return False
        
        with self._locker:
            
            max_buffer_size = int(self.replay_buffer_cfg['max_buffer_size'])
            
            if len(self.rp_buffers['val']) == 0:
                self.rp_buffers['val'].add_episode(current_episode)
                return False
            
            self.current_rp_buffer.add_episode(current_episode)
            
            min_epoch_data_size = int(self.training_cfg['min_epoch_data_size'])
            current_epoch_data_size = min(len(self.rp_buffers['train']) + len(self.rp_buffers['val']),
                                          int(self.training_cfg['max_epoch_data_size']))
            epoch_data_size = max(min_epoch_data_size, current_epoch_data_size)
            
            if current_epoch_data_size + len(self.current_rp_buffer) < min_epoch_data_size:
                return False
            if self.current_rp_buffer.num_episodes < 2:
                return False
            
            train_val_ratio = float(self.training_cfg['train_size']) / float(self.training_cfg['val_size'])
            
            for episode_idx in range(self.current_rp_buffer.num_episodes):
                cur_train_val_ratio = len(self.rp_buffers['train']) / len(self.rp_buffers['val'])
                if cur_train_val_ratio < train_val_ratio:
                    self.rp_buffers['train'].buffer.append(self.current_rp_buffer.buffer[episode_idx])
                else:
                    self.rp_buffers['val'].buffer.append(self.current_rp_buffer.buffer[episode_idx])
            self.current_rp_buffer.clear()
            
            while (len(self.rp_buffers['train']) + len(self.rp_buffers['val'])) > max_buffer_size:
                target_rp_buff = self.rp_buffers['val' if cur_train_val_ratio < train_val_ratio else 'train']
                target_rp_buff.buffer = target_rp_buff.buffer[1:]
                cur_train_val_ratio = len(self.rp_buffers['train']) / len(self.rp_buffers['val'])
                
            cur_train_val_ratio = len(self.rp_buffers['train']) / len(self.rp_buffers['val'])
            self.logger.info(f'Train / val ratio: {cur_train_val_ratio}')
            
            return True
        
    def __enter__(self):
        self._locker.acquire()
        return self
        
    def __exit__(self):
        self._locker.release()
    
    def load(self, in_folder: tp.Union[Path, str]):
        in_folder = Path(in_folder)
        for subset_name in self.rp_buffers:
            rp_folder = in_folder / f'rp_{subset_name}'
            if not rp_folder.exists():
                continue
            rp_buffer = ReplayBuffer.load(rp_folder, f'Loading [{subset_name}]')
            self.rp_buffers[subset_name] = rp_buffer
        
        
    def save_rp_buffsers(self, out_folder: tp.Union[Path, str]):
        out_folder = Path(out_folder)
        for subset_name, rp_buffer in self.rp_buffers.items():
            rp_folder = out_folder / f'rp_{subset_name}'
            if rp_folder.exists():
                shutil.rmtree(rp_folder)
            rp_folder.mkdir()
            rp_buffer.save(rp_folder, f'Saving [{subset_name}]')
