import os
import numpy as np
import pydicom as dicom
from datasets.SLIViTDataset3D import SLIViTDataset3D
from auxiliaries.slivit_auxiliaries import default_transform_gray
from auxiliaries.misc import to_tensor


class MRIDataset3D(SLIViTDataset3D):
    # example image name: '1.3.12.2.1107.5.2.18.41754.2017082116102653036617510.dcm'

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
