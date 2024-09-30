import PIL
from datasets.SLIViTDataset3D import SLIViTDataset3D, ToTensor


class USDataset3D(SLIViTDataset3D):
    # example image name: '0003.tiff'

    def __getitem__(self, idx):
        return super().__getitem__(idx[0])  #TODO: check why it's happenning: idx[0] (instead of idx)

    def load_scan(self, path, slice_idxs):
        frames_to_use = [str(x).zfill(4) for x in slice_idxs]

        scan = []
        for frame in frames_to_use:
            frame = PIL.Image.open(f'{path}/{frame}.tiff')
            scan.append(ToTensor()(frame))
        return scan