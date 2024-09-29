import os
import torch
import numpy as np
from auxiliaries.finetune import default_transform_gray
from datasets.SLIViTDataset import SLIViTDataset
from torchvision.transforms import ToTensor


class SLIViTDataset3D(SLIViTDataset):
    def __init__(self, meta, label_name, path_col_name, **kwargs):# num_slices_to_use, sparsing_method):
        super().__init__(meta, label_name, path_col_name, default_transform_gray)
        self.num_slices_to_use = kwargs.get('num_slices_to_use')
        self.sparsing_method = kwargs.get('sparsing_method')
        self.filter = lambda x: x.endswith(kwargs.get('img_suffix'))

    def __getitem__(self, idx):
        scan_path, label = super().__getitem__(idx)
        slice_idxs = self.get_slices_indexes(scan_path, self.num_slices_to_use)  # TODO: consider moving to load_volume
        scan = self.load_scan(scan_path, slice_idxs)
        transformed_scan = torch.cat([self.t(im) for im in scan], dim=-1)
        return transformed_scan, label.squeeze(0)  # TODO: Consider adding EHR info

    def get_slices_indexes(self, vol_path, num_slices_to_use):
        total_num_of_slices = len(list(filter(self.filter, os.listdir(vol_path))))
        assert total_num_of_slices > 0, f"No images found in {vol_path}"
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
            # customized sampling method to be defined by the user
            raise NotImplementedError("Sparsing method not implemented")
        else:
            raise ValueError("Sparsing method not recognized")

        return slc_idxs

    def load_scan(self, *args):
        raise NotImplementedError('load_volume method must be implemented in child class')

