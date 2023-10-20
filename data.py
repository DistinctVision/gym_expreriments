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
    
    def sample_batch(self, batch_indices: tp.List[tp.Tuple[int,  int]]) -> tp.Tuple[RecordArray, RecordArray,  torch.Tensor]:
        cur_records: tp.List[Record] = [self.get_record(idx[0], idx[1])
                                        for idx in batch_indices]
        cur_batch = RecordArray(world_states=torch.stack([record.world_state_tensor
                                                          for record in cur_records]),
                                action_indices=torch.tensor([record.action_index for record in cur_records],
                                                            dtype=torch.int16),
                                rewards=torch.tensor([record.reward
                                                      for record in cur_records], dtype=torch.float32))
        
        next_records: tp.List[Record] = []
        mask_done: tp.List[bool] = []
        for r_idx,  idx in enumerate(batch_indices):
            if (idx[1] + 1) >= self.num_records(idx[0]):
                next_records.append(cur_records[r_idx])
                mask_done.append(True)
                continue
            next_records.append(self.get_record(idx[0], idx[1] + 1))
            mask_done.append(False)
        
        next_batch = RecordArray(world_states=torch.stack([record.world_state_tensor
                                                           for record in next_records]),
                                 action_indices=torch.tensor([record.action_index
                                                              for record in next_records],
                                                             dtype=torch.int16),
                                 rewards=torch.tensor([record.reward for record in next_records],
                                                       dtype=torch.float32))
        mask_done = torch.tensor(mask_done, dtype=torch.bool)
        return cur_batch, next_batch, mask_done
            
            

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
                 batch_sampler: ReplayBufferBatchSampler):
        self.cached_replay_buffer = cached_replay_buffer
        self.batch_sampler = batch_sampler
    
    def get_batch(self, index: int) -> tp.Tuple[RecordArray, RecordArray,  torch.Tensor]:
        batch_indices = self.batch_sampler.get_batch_indices(index)
        cur_records: tp.List[Record] = [self.cached_replay_buffer.get_record(idx[0], idx[1])
                                        for idx in batch_indices]
        cur_batch = RecordArray(world_states=torch.stack([record.world_state_tensor
                                                          for record in cur_records]),
                                           action_indices=torch.tensor([record.action_index for record in cur_records],
                                                                       dtype=torch.int16),
                                           rewards=torch.tensor([record.reward
                                                                 for record in cur_records], dtype=torch.float32))
        
        next_records: tp.List[Record] = []
        mask_done: tp.List[bool] = []
        for r_idx,  idx in enumerate(batch_indices):
            if (idx[1] + 1) >= self.cached_replay_buffer.num_records(idx[0]):
                next_records.append(cur_records[r_idx])
                mask_done.append(True)
                continue
            next_records.append(self.cached_replay_buffer.get_record(idx[0], idx[1] + 1))
            mask_done.append(False)
        
        next_batch = RecordArray(world_states=torch.stack([record.world_state_tensor
                                                           for record in next_records]),
                                 action_indices=torch.tensor([record.action_index
                                                              for record in next_records],
                                                             dtype=torch.int16),
                                 rewards=torch.tensor([record.reward for record in next_records],
                                                       dtype=torch.float32))
        mask_done = torch.tensor(mask_done, dtype=torch.bool)
        return cur_batch, next_batch, mask_done
        
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
