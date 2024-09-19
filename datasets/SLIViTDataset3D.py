import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from utils.slivit_auxiliaries import default_transform_gray


class SLIViTDataset3D(Dataset):
    def __init__(self, meta_data, label_name, path_col_name, num_slices_to_use,
                 sparsing_method, transform=default_transform_gray):
        if not isinstance(meta_data, pd.DataFrame):
            # meta_data is a path to a csv file
            meta_data = pd.read_csv(meta_data)  # , index_col=0)
        self.labels = meta_data[label_name].values
        self.sample_paths = meta_data[path_col_name].values
        for p in self.sample_paths:
            assert os.path.exists(p), f'{p} do not exist'
        self.num_slices_to_use = num_slices_to_use
        self.t = transform
        self.sparsing_method = sparsing_method
        self.filter = lambda x: x


    def __len__(self):
        return len(self.sample_paths)


    def __getitem__(self, idx):
        sample = self.sample_paths[idx]
        slice_idxs = self.get_slices_indexes(sample, self.num_slices_to_use)

        imgs = self.load_volume(sample, slice_idxs)

        imgs = torch.cat([self.t(im) for im in imgs], dim=-1)

        label = torch.FloatTensor([self.labels[idx]])  # unwrap two-dimensional array

        return imgs, label  # TODO: Consider adding EHR info

    def get_slices_indexes(self, vol_path, num_slices_to_use):
        total_num_of_slices = len(list(filter(self.filter, os.listdir(vol_path))))
        if self.sparsing_method == 'eq':
            # equally-spaced down sample the slices
            slc_idxs = np.linspace(0, total_num_of_slices - 1,
                                   num_slices_to_use).astype(int)  # down sample slices
        elif self.sparsing_method == 'mid':
            # middle down sample the slices
            middle = total_num_of_slices // 2
            slc_idxs = np.linspace(middle - num_slices_to_use // 2, middle + num_slices_to_use // 2,
                                   num_slices_to_use).astype(int)  # down sample slices
        elif self.sparsing_method == 'custom':
            # customized down sample method to be defined by the user
            raise NotImplementedError("Sparsing method not implemented")
        else:
            raise ValueError("Sparsing method not recognized")

        return slc_idxs


    def load_volume(self, *args):
        raise NotImplementedError('load_volume method must be implemented in child class')

