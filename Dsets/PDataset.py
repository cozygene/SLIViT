import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.pretrain_auxiliaries import *

default_transform = tf.Compose(
    [
        tf.ToPILImage(),
        tf.Resize((224, 224)),
        pil_contrast_strech(),
        tf.ToTensor(),
        gray2rgb
    ]
)


class PDataset(Dataset):
    def __init__(self, metafile_path, annotations_path, data_dir, pathologies, transform=default_transform,
                 data_format='jpeg'):
        self.metadata = pd.read_csv(metafile_path)
        self.annotations = pd.read_csv(annotations_path)
        self.pathologies = pathologies
        self.samples = get_samples(self.metadata, self.annotations, pathologies)
        self.t = default_transform
        logger.info(f'{data_format.upper()} dataset loaded')
        self.data_reader = dict(
            jpeg=load_jpeg
        )[data_format]
        logger.info(f'Predicting {pathologies}')
        self.label_reader = get_labels
        self.labels = [self.label_reader(self.samples[i], self.annotations, self.pathologies)[0][0] for i in
                       range(len(self.samples))]
        self.labels = torch.FloatTensor(self.labels)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx[0]]
        imgs = self.data_reader(sample, self.data_dir)
        labels = self.label_reader(sample, self.annotations, self.pathologies)
        labels = torch.FloatTensor(labels)
        t_imgs = self.t(imgs)
        labels = labels.reshape(len(self.pathologies))

        return t_imgs, labels
