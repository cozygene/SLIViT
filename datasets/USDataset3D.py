from PIL import Image

from datasets.SLIViTDataset3D import SLIViTDataset3D
from auxiliaries.slivit_auxiliaries import default_transform_gray
from auxiliaries.misc import to_tensor


class USDataset3D(SLIViTDataset3D):
    # example image name: '0003.tiff'

    def __getitem__(self, idx):
        return super().__getitem__(idx[0])  #TODO: check why it's happenning: idx[0] (instead of idx)

    def load_scan(self, path, slice_idxs):
        frames_to_use = [str(x).zfill(4) for x in slice_idxs]

        scan = []
        for frame in frames_to_use:
            frame = Image.open(f'{path}/{frame}.tiff')
            scan.append(to_tensor(frame))
        return scan