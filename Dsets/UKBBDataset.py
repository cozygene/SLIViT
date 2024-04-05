import numpy as np
from utils.slivit_auxiliaries import *
import pandas as pd
import torch
from fastai.vision.augment import aug_transforms
from torch.utils.data import Dataset
from fastai.vision import *


default_transform_gray = tf.Compose([
    tf.ToPILImage(),
    tf.Resize((256, 256)),
    #pil_contrast_strech(),
    tf.ToTensor(),
    gray2rgb
])
class UKBBDataset(Dataset):
    def __init__(self, metafile_path, annotations_path, pathologies, nslc,transform=default_transform_gray):
        self.metadata = pd.read_csv(metafile_path)
        self.annotations = pd.read_csv(annotations_path)
        self.pathologies = pathologies
        self.samples = get_samples(self.metadata, self.annotations, pathologies)
        self.t = transform
        self.data_reader = load_dcm
        self.label_reader = get_labels
        self.labels=[self.label_reader(self.samples[i], self.annotations, self.pathologies) for i in range(len(self.samples))]
        self.labels=torch.FloatTensor(self.labels)
        self.nslc=nslc

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx[0]]
        imgs = self.data_reader(sample,self.nslc)
        labels = self.label_reader(sample, self.annotations, self.pathologies)  
        labels = torch.FloatTensor(labels)
        t_imgs = torch.cat([self.t(im) for im in imgs], dim=-1)
        return t_imgs, labels.squeeze()





