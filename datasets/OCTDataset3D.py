import os
from datasets.SLIViTDataset3D import SLIViTDataset3D, ToTensor
from auxiliaries.pretrain import *


class OCTDataset3D(SLIViTDataset3D):
    #  example image name of Hadassah: bscan_12.tiff  ->  012
    #  example image name of Houston: 12.tiff  ->  012

    def load_scan(self, vol_path, slc_idxs):

        slices = os.listdir(vol_path)
        vol = []
        # sort by slice order
        for i, slice_name in enumerate(sorted(slices, key=lambda x: int(x.split('_')[-1].split('.')[0]))):
            if i in slc_idxs:
                img = PIL.Image.open(f'{vol_path}/{slice_name}')
                vol.append(ToTensor()(img))
        return vol
