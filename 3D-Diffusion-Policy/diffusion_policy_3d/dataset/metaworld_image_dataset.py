from typing import Dict
import torch
import numpy as np
import copy
from tqdm import tqdm
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

class MetaworldDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            ):
        super().__init__()
        replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'img'])
        images = replay_buffer['img']
        images = images / 255.0
        images = np.transpose(images, (0, 3, 1, 2))
        # from moge.model import MoGeModel
        # img_encoder = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to('cuda')
        # features = list()
        # for i in tqdm(range(0, len(images), 64)):
        #     img = torch.from_numpy(images[i:i+64]).float().to('cuda')
        #     with torch.no_grad():
        #         feature = img_encoder.get_embedding(img)
        #     features.append(feature.cpu().numpy())

        # features = np.concatenate(features, axis=0)

        # debug
        features = np.random.randn(len(images), 1024).astype(np.float32)
        new_root = {
            'meta': replay_buffer.meta,
            'data': {**replay_buffer.data, 'feature': features}
        }
        self.replay_buffer = ReplayBuffer(root=new_root)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32)
        img = sample['img'][:,].astype(np.float32)
        feature = sample['feature'][:,].astype(np.float32)
        data = {
            'obs': {
                'img': img, 
                'agent_pos': agent_pos, 
                'feature': feature
            },
            'action': sample['action'].astype(np.float32)
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
