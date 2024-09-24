import os
from datasets.SLIViTDataset3D import SLIViTDataset3D
from auxiliaries.pretrain_auxiliaries import *
from auxiliaries.slivit_auxiliaries import default_transform_gray
from auxiliaries.misc import to_tensor


class OCTDataset3D(SLIViTDataset3D):

    def __init__(self, meta_data, label_name, path_col_name, num_slices_to_use, sparsing_method):
        super().__init__(meta_data, label_name, path_col_name, num_slices_to_use, sparsing_method, default_transform_gray)

        #  example image name of Hadassah: bscan_12.tiff  ->  012
        #  example image name of Houston: 12.tiff  ->  012
        self.filter = lambda x: x.endswith('tiff')


    def load_scan(self, vol_path, slc_idxs):

        slices = os.listdir(vol_path)
        vol = []
        # sort by slice order
        for i, slice_name in enumerate(sorted(slices, key=lambda x: int(x.split('_')[-1].split('.')[0]))):
            if i in slc_idxs:
                img = Image.open(f'{vol_path}/{slice_name}')
                vol.append(to_tensor(img))
        return vol
