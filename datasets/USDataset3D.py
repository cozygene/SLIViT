from PIL import Image

from datasets.SLIViTDataset3D import SLIViTDataset3D
from utils.slivit_auxiliaries import totensor, default_transform_gray


class USDataset3D(SLIViTDataset3D):

    def __init__(self, meta_data, label_name, path_col_name, num_slices_to_use,
                 sparsing_method, transform=default_transform_gray):
        super().__init__(meta_data, label_name, path_col_name, num_slices_to_use, sparsing_method, transform)

        # example image name: '0003.tiff'
        self.filter = lambda x: x.endswith('tiff')

    def __getitem__(self, idx):
        t_imgs, label = super().__getitem__(idx[0])  #TODO: check why it's happenning: idx[0] (instead of idx)
        return t_imgs, label

    def load_scan(self, path, slice_idxs):
        frames_to_use = [str(x).zfill(4) for x in slice_idxs]

        vol = []
        for frame in frames_to_use:
            img = Image.open(f'{path}/{frame}.tiff')
            vol.append(totensor(img))
        return vol