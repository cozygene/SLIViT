import os
from datasets.SLIViTDataset3D import SLIViTDataset3D
from utils.pretrain_auxiliaries import *
from utils.slivit_auxiliaries import default_transform_gray, totensor


class OCTDataset3D(SLIViTDataset3D):

    def __init__(self, meta_data, label_name, path_col_name, num_slices_to_use,
                 sparsing_method, transform=default_transform_gray):
        super().__init__(meta_data, label_name, path_col_name, num_slices_to_use, sparsing_method, transform)

        #  example image name of Hadassah: bscan_12.tiff  ->  012
        #  example image name of Houston: 12.tiff  ->  012
        self.filter = lambda x: x.endswith('tiff')


    def load_volume(self, vol_path, slc_idxs):

        slices = os.listdir(vol_path)
        vol = []
        # sort by slice order
        for i, slice_name in enumerate(sorted(slices, key=lambda x: int(x.split('_')[-1].split('.')[0]))):
            if i in slc_idxs:
                img = Image.open(f'{vol_path}/{slice_name}')
                vol.append(totensor(img))
        return vol
