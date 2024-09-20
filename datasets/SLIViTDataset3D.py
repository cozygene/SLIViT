import os
import torch
import numpy as np
from datasets.SLIViTDataset import SLIViTDataset


class SLIViTDataset3D(SLIViTDataset):
    def __init__(self, meta_data, label_name, path_col_name, num_slices_to_use, sparsing_method, transformations):
        super().__init__(meta_data, label_name, path_col_name, transformations)
        self.num_slices_to_use = num_slices_to_use
        self.sparsing_method = sparsing_method

    def __getitem__(self, idx):
        scan_path, label = super().__getitem__(idx)
        slice_idxs = self.get_slices_indexes(scan_path, self.num_slices_to_use)  # TODO: consider moving to load_volume
        scan = self.load_scan(scan_path, slice_idxs)
        transformed_scan = torch.cat([self.t(im) for im in scan], dim=-1)
        return transformed_scan, label  # TODO: Consider adding EHR info

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

    def load_scan(self, *args):
        raise NotImplementedError('load_volume method must be implemented in child class')

