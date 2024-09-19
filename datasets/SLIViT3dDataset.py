from utils.slivit_auxiliaries import *
import torch
from torch.utils.data import Dataset
from utils.slivit_auxiliaries import get_samples, get_label, totensor, default_transform_gray


class SLIViT3dDataset(Dataset):
    def __init__(self, metafile_path, pathologies, nslc, transform=default_transform_gray):
        self.metadata = pd.read_csv(metafile_path)
        self.pathologies = pathologies
        self.samples = get_samples(self.metadata)
        self.t = transform
        self.nslc = nslc
        self.label_reader = get_label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx[0]]
        imgs = self.load_volume(sample, self.nslc)
        label = self.label_reader(sample, self.metadata, self.pathologies)
        label = torch.FloatTensor(label)
        t_imgs = torch.cat([self.t(im) for im in imgs], dim=-1)
        return t_imgs, label

    def load_volume(self, *args):
        raise NotImplementedError('load_volume method must be implemented in child class')


