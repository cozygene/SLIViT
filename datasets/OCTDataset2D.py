from auxiliaries.pretrain import *
from datasets.SLIViTDataset import SLIViTDataset
from torchvision.transforms import PILToTensor


class OCTDataset2D(SLIViTDataset):
    def __init__(self, meta, label_name, path_col_name, **kwargs):
        super().__init__(meta, label_name, path_col_name, default_transform)

    def __getitem__(self, idx):
        sample_path, label = super().__getitem__(idx)
        scan = PILToTensor()(PIL.Image.open(sample_path))
        transformed_scan = self.t(scan)
        return transformed_scan, label.squeeze()
