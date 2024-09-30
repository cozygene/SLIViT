import numpy as np
import torch
from torch.utils.data import Dataset
from auxiliaries.finetune import default_transform_gray

# TODO: clean up this class
class MedMNISTDataset3D(Dataset):
    def __init__(self, dataset, num_slices_to_use, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.num_slices_to_use = num_slices_to_use

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.num_slices_to_use == 1:
            vol = torch.FloatTensor(self.dataset[idx][0])[:, 13, :, :]
            vol = vol.reshape(1, 1, 28, 28)
        elif self.num_slices_to_use == 28:
            vol = torch.FloatTensor(self.dataset[idx][0])
        else:
            vol = torch.FloatTensor(self.dataset[idx][0])[:, np.linspace(0, 27, self.num_slices_to_use), :, :]

        vol = torch.cat([default_transform_gray(vol[:, i, :, :]) for i in range(vol.shape[1])], dim=-2)
        return vol, self.dataset[idx][1].astype(np.float32)
