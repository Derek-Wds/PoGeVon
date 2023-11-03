import os

import numpy as np
import pandas as pd

from lib import datasets_path
from .pd_dataset import PandasDataset
from ..utils import sample_mask

from math import radians, cos, sin, asin, sqrt

def distance(lat1, lat2, lon1, lon2):
     
    # The math module contains a function named
    # radians which converts from degrees to radians.
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
      
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
 
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers. Use 3956 for miles
    # r = 6371
    r = 3956
      
    # calculate the result
    return(c * r)

class Pems:
    def __init__(self, dataset='PEMS04', block=False):
        self.dataset = dataset
        self.block = block
        if dataset == 'PEMS03':
            self.threshold = 6
        elif dataset == 'PEMS04' or dataset == 'PEMS07':
            self.threshold = 15
        elif dataset == 'PEMS08':
            self.threshold = 25
        elif dataset == 'PEMS11':
            self.threshold = 10
        else:
            raise Exception('Invalid input dataset.')
        data, dist, mask, adjs, positions, adj_label = self.load()
        self.data = data
        self.dist = dist
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
        path = os.path.join(datasets_path['pems'], f'{self.dataset}/data.npy')
        data = np.load(path)[:, :, 0] # only take the first channel, flow rate
        # np.random.seed(22)
        # self.idxs = np.random.choice(data.shape[1], 64, replace=False)
        variance = np.var(data, axis=0)
        self.idxs = np.argsort(variance, axis=0)[-64:]
        data = data[:, self.idxs]
        mask = ~np.isnan(data)
        dist = self.load_distance_matrix()
        try:
            if self.block:
                adjs = np.load(os.path.join(datasets_path['pems'], f'{self.dataset}/adjacency_block.npy'))
                positions = np.load(os.path.join(datasets_path['pems'], f'{self.dataset}/position_block.npy'))
                adj_label = np.load(os.path.join(datasets_path['pems'], f'{self.dataset}/adjacency_block_label.npy'))
            else:
                adjs = np.load(os.path.join(datasets_path['pems'], f'{self.dataset}/adjacency_point.npy'))
                positions = np.load(os.path.join(datasets_path['pems'], f'{self.dataset}/position_point.npy'))
                adj_label = np.load(os.path.join(datasets_path['pems'], f'{self.dataset}/adjacency_point_label.npy'))
        except OSError:
            print('Please generate the adjacency sequence first.')
            adjs, positions, adj_label =  None, None, None
        return data.astype('float32'), dist, mask.astype('uint8'), adjs, positions, adj_label

    def load_distance_matrix(self):
        stations = pd.read_csv(os.path.join(datasets_path['pems'], f'{self.dataset}/station.csv'))

        # num_sensors = self.idxs.shape[0]
        num_sensors = len(self.idxs)
        dist = np.ones((num_sensors, num_sensors), dtype=np.float32) * np.inf

        # get the distance between sensors
        stations = stations.values[self.idxs]
        ds = list()
        for i in range(len(stations)):
            for j in range(i+1, len(stations)):
                lat1, lon1 = stations[i][1], stations[i][2]
                lat2, lon2 = stations[j][1], stations[j][2]
                d = distance(lat1, lat2, lon1, lon2)
                if d < self.threshold:
                    ds.append([i, j, d])

        # Fills cells in the matrix with distances.
        for row in ds:
            dist[int(row[0]), int(row[1])] = row[2]
        return dist

    def get_similarity(self, type=None, thr=0.1, force_symmetric=False, sparse=False):
        """
        Return similarity matrix among nodes. Implemented to match DCRNN.

        :param type: type of similarity matrix.
        :param thr: threshold to increase saprseness.
        :param trainlen: number of steps that can be used for computing the similarity.
        :param force_symmetric: force the result to be simmetric.
        :return: and NxN array representig similarity among nodes.
        """
        if type == 'dcrnn':
            finite_dist = self.dist.reshape(-1)
            finite_dist = finite_dist[~np.isinf(finite_dist)]
            sigma = finite_dist.std()
            adj = np.exp(-np.square(self.dist / sigma))
        elif type == 'stcn':
            sigma = 10
            adj = np.exp(-np.square(self.dist) / sigma)
        else:
            finite_dist = self.dist.reshape(-1)
            finite_dist = finite_dist[~np.isinf(finite_dist)]
            tmp_max = np.max(finite_dist)
            adj = self.dist / tmp_max
            adj[adj == np.inf] = 0
        # adj[adj < thr] = 0.
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)
        return adj

    @property
    def mask(self):
        if self._mask is None:
            return self.df.values != 0.
        return self._mask


class MissingValuesPems(Pems):
    SEED = 222

    def __init__(self, dataset='PEMS04', p_fault=0.0015, p_noise=0.05):
        super(MissingValuesPems, self).__init__(dataset=dataset, block=p_fault!=0.0)
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
