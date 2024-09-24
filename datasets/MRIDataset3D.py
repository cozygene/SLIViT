import os
import numpy as np
import pydicom as dicom
from datasets.SLIViTDataset3D import SLIViTDataset3D
from auxiliaries.slivit_auxiliaries import default_transform_gray
from auxiliaries.misc import to_tensor


class MRIDataset3D(SLIViTDataset3D):

    def __init__(self, meta_data, label_name, path_col_name, num_slices_to_use, sparsing_method):
        super().__init__(meta_data, label_name, path_col_name, num_slices_to_use, sparsing_method, default_transform_gray)

        # example image name: '1.3.12.2.1107.5.2.18.41754.2017082116102653036617510.dcm'
        self.filter = lambda x: x.endswith('dcm')

    def __getitem__(self, idx):
        return super().__getitem__(idx[0])

    def load_scan(self, path, slc_idxs):
        filtered = list(filter(self.filter, os.listdir(path)))  # exclude non-dicom files
        img_paths = [filtered[i] for i in slc_idxs]

        vol = []
        for img_name in img_paths:
            img = dicom.dcmread(f'{path}/{img_name}')
            vol.append(to_tensor(img.pixel_array.astype(np.float64)))

        return vol
