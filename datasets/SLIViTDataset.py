import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class SLIViTDataset(Dataset):
    def __init__(self, meta, label_name, path_col_name, transform):
        if not isinstance(meta, pd.DataFrame):
            # meta is a path to a csv file
            meta = pd.read_csv(meta)  # , index_col=0)
        self.labels = meta[label_name].values
        self.scan_paths = meta[path_col_name].values#.apply(lambda x: f'/scratch/avram/Kermany/{x}').values #TOOD: delete this
        # print(meta[path_col_name].apply(lambda x: f'/scratch/avram/Kermany/{x}')[:10])
        for p in self.scan_paths:
            assert os.path.exists(p), f'{p} do not exist'
        self.t = transform
        self.filter = lambda x: x
        self.num_classes = len(self.labels[0])

    def __len__(self):
        return len(self.scan_paths)

    def __getitem__(self, idx):
        scan_path = self.scan_paths[idx]
        label = torch.FloatTensor(np.array([self.labels[idx]]))  # unwrap two-dimensional array
        return scan_path, label

    def load_scan(self, *args):
        raise NotImplementedError('load_scan method must be implemented in child class')

    def get_num_classes(self):
        return self.num_classes
