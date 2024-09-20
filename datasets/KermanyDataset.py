import pandas as pd
import torch
from torch.utils.data import Dataset

from datasets.SLIViTDataset import SLIViTDataset
from auxiliaries.pretrain_auxiliaries import *
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor, PILToTensor

default_transform = Compose(
    [
        tf.ToPILImage(),
        tf.Resize((224, 224)),
        pil_contrast_strech(),
        ToTensor(),
        gray2rgb
    ]
)

# TODO: cleanup this class
class KermanyDataset(SLIViTDataset):
    def __init__(self, meta_data, label_name, path_col_name):
        super().__init__(meta_data, label_name, path_col_name, default_transform)

    def __getitem__(self, idx):
        sample_path, label = super().__getitem__(idx)
        scan = PILToTensor()(Image.open(sample_path))
        transformed_scan = self.t(scan)
        return transformed_scan, label.squeeze()  # TODO: Consider adding EHR info
