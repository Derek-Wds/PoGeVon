import os, pickle

import numpy as np
import pandas as pd

from lib import datasets_path
from .pd_dataset import PandasDataset
from ..utils import sample_mask

class Covid:
    def __init__(self, block=False):
        self.block = block
        data, mask, adjs, positions, adj_label = self.load()
        self.data = data
        self._mask = mask
        self.adjs = adjs
        self.positions = positions
        self.adj_label = adj_label
        self.df = pd.DataFrame(data, columns=range(data.shape[1]))
    
    def __len__(self):
        return len(self.data)

    def numpy(self, return_idx=False):
        if return_idx:
            return self.numpy(), np.array(range(len(self.data)))
        return self.data

    def load(self, impute_zeros=True):
        path = os.path.join(datasets_path['covid'], f'covid_ts.npy')
        data = np.load(path)[:, :, 0] # cases, 1 for deaths
        mask = ~np.isnan(data)
        try:
            if self.block:
                adjs = np.load(os.path.join(datasets_path['covid'], f'adjacency_block.npy'))
                positions = np.load(os.path.join(datasets_path['covid'], f'position_block.npy'))
                adj_label = np.load(os.path.join(datasets_path['covid'], f'adjacency_block_label.npy'))
            else:
                adjs = np.load(os.path.join(datasets_path['covid'], f'adjacency_point.npy'))
                positions = np.load(os.path.join(datasets_path['covid'], f'position_point.npy'))
                adj_label = np.load(os.path.join(datasets_path['covid'], f'adjacency_point_label.npy'))
        except OSError:
            print('Please generate the adjacency sequence first.')
            adjs, positions, adj_label =  None, None, None
        return data.astype('float32'), mask.astype('uint8'), adjs, positions, adj_label

    def get_similarity(self, type=None, thr=0.1, force_symmetric=False, sparse=False):
        return self.adjs[0]

    @property
    def mask(self):
        if self._mask is None:
            return self.df.values != 0.
        return self._mask


class MissingValuesCovid(Covid):
    SEED = 222

    def __init__(self, p_fault=0.0015, p_noise=0.05):
        super(MissingValuesCovid, self).__init__(block=p_fault!=0.0)
        self.rng = np.random.default_rng(self.SEED)
        self.p_fault = p_fault
        self.p_noise = p_noise
        eval_mask = sample_mask(self.numpy().shape,
                                p=p_fault,
                                p_noise=p_noise,
                                min_seq=12,
                                max_seq=12 * 4,
                                rng=self.rng)
        self.eval_mask = (eval_mask & self.mask).astype('uint8')

    @property
    def training_mask(self):
        return self.mask if self.eval_mask is None else (self.mask & (1 - self.eval_mask))

    def splitter(self, dataset, val_len=0, test_len=0, window=0):
        idx = np.arange(len(dataset))
        if test_len < 1:
            test_len = int(test_len * len(idx))
        if val_len < 1:
            val_len = int(val_len * (len(idx) - test_len))
        test_start = len(idx) - test_len
        val_start = test_start - val_len
        return [idx[:val_start - window], idx[val_start:test_start - window], idx[test_start:]]