import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.pretrain_auxiliaries import *
from utils.slivit_auxiliaries import default_transform_gray, totensor

class CustomDataset3D(Dataset):
    def __init__(self, metafile_path, label_name, num_slices_to_use,
                 sparsing_method, sample_paths='Path'):  # TODO: check equally_spaced_sparsing = False
        metadata = pd.read_csv(metafile_path)  # , index_col=0)
        self.labels = metadata[label_name].values
        self.sample_paths = metadata[sample_paths].values
        self.num_slices_to_use = num_slices_to_use
        self.t = default_transform_gray
        self.sparsing_method = sparsing_method

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        # torch tensor (image) or list of tensors (volume)
        imgs = self.load_custom(self.sample_paths[idx], self.num_slices_to_use, self.sparsing_method)
        # try:
        #     imgs = self.load_custom(self.sample_paths[idx], self.num_slices_to_use,
        #                             equally_spaced=self.equally_spaced_sparsing)
        # except:
        #     imgs = self.load_custom(self.sample_paths[idx][0], self.num_slices_to_use,
        #                             equally_spaced=self.equally_spaced_sparsing)

        imgs = torch.cat([self.t(im) for im in imgs], dim=-1)

        label = torch.FloatTensor([self.labels[idx]])  # unwrap two-dimensional array

        # print(imgs.size(1))# == expected_feature_size, "Feature size mismatch"
        # print(label.size(0))# == 1, "Label size mismatch"

        try:
            label = label.squeeze(1)
        except:
            label = label

        return imgs, label  # TODO ADD EHR info

    def load_custom(self, vol_path, num_slices_to_use, sparsing_method):  ##/scratch/avram/Amish/tiff

        vol = []
        try:
            slices = os.listdir(vol_path)
        except ValueError:
            slices = os.listdir(vol_path[0])
        # TODO: add upsampling capability
        if sparsing_method == 'eq':
            # equally-spaced down sample the slices
            slc_idxs = np.linspace(0, len(slices) - 1,
                                   num_slices_to_use).astype(int)  # down sample slices
        elif sparsing_method == 'mid':
            # middle down sample the slices
            middle = len(slices) // 2
            slc_idxs = np.linspace(middle - num_slices_to_use // 2, middle + num_slices_to_use // 2,
                                   num_slices_to_use).astype(int)  # down sample slices
        elif sparsing_method == 'custom':
            # customized down sample method to be defined by the user
            raise NotImplementedError("Sparsing method not implemented")
        else:
            raise ValueError("Sparsing method not recognized")

        for i, slice_name in enumerate(
                sorted(slices, key=lambda x: int(x.split('_')[-1].split('.')[0]))):  # sort by slice order
            #  Example image name of hadassah: bscan_12.tiff  ->  012
            #  Example image name of Houston: 12.tiff  ->  012
            if i in slc_idxs:
                try:
                    img = Image.open(f'{vol_path}/{slice_name}')
                except FileNotFoundError:
                    img = Image.open(f'{vol_path[0]}/{slice_name}')
                vol.append(totensor(img))

        return vol
