import torch
from torch.utils.data import Dataset
from torchvision import transforms as tf

from auxiliaries.pretrain import apply_contrast_stretch


class MedMNISTDataset2D(Dataset):
    def __init__(self, dataset, **kwargs):
        super()  # .__init__()
        self.dataset = dataset
        self.num_classes = len(self.dataset.info['label'])
        self.t = tf.Compose([  # TODO replace this with the one in pretrain.py
            apply_contrast_stretch,
            tf.ToTensor(),
            tf.Resize((224, 224)),  # if self.dataset[0][0].shape == (28, 28) else torch.nn.Identity(),
            tf.RandomResizedCrop((224, 224)),
            tf.Lambda(lambda x: x.expand(3, -1, -1))  # gray to rgb
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        scan, label = self.dataset[idx]
        return self.t(scan), torch.FloatTensor(label)

    def get_num_classes(self):
        return self.num_classes

