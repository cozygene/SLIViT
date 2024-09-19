import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from utils.slivit_auxiliaries import default_transform_gray


class SLIViTDataset(Dataset):
    def __init__(self, meta_data, label_name, path_col_name, transform=default_transform_gray):
        if not isinstance(meta_data, pd.DataFrame):
            # meta_data is a path to a csv file
            meta_data = pd.read_csv(meta_data)  # , index_col=0)
        self.labels = meta_data[label_name].values
        self.sample_paths = meta_data[path_col_name].values
        for p in self.sample_paths:
            assert os.path.exists(p), f'{p} do not exist'
        self.t = transform
        self.filter = lambda x: x

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample = self.sample_paths[idx]
        label = torch.FloatTensor([self.labels[idx]])  # unwrap two-dimensional array
        return sample, label  # TODO: Consider adding EHR info

    def load_scan(self, *args):
        raise NotImplementedError('load_scan method must be implemented in child class')

