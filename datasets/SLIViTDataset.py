import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class SLIViTDataset(Dataset):
    def __init__(self, meta_data, label_name, path_col_name, transform):
        if not isinstance(meta_data, pd.DataFrame):
            # meta_data is a path to a csv file
            meta_data = pd.read_csv(meta_data)  # , index_col=0)
        self.labels = meta_data[label_name].values
        self.scan_paths = meta_data[path_col_name].values
        for p in self.scan_paths:
            assert os.path.exists(p), f'{p} do not exist'
        self.t = transform
        self.filter = lambda x: x

    def __len__(self):
        return len(self.scan_paths)

    def __getitem__(self, idx):
        scan_path = self.scan_paths[idx]
        label = torch.FloatTensor(np.array([self.labels[idx]]))  # unwrap two-dimensional array
        return scan_path, label  # TODO: Consider adding EHR info

    def load_scan(self, *args):
        raise NotImplementedError('load_scan method must be implemented in child class')

    def get_num_classes(self):
        return len(np.unique(self.labels))
