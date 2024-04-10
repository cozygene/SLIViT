
import numpy as np
import torch
from torch.utils.data import Dataset
from fastai.vision import *
from torchvision import transforms as tf
import torch
import PIL 
from skimage import exposure
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from torchvision.transforms import (
    RandomResizedCrop,
    ToTensor,
)
gray2rgb = tf.Lambda(lambda x: x.expand(3, -1, -1))

class truncate(object):
    def __init__(self, low=-1024, high=325,subtract=158.58,divide=324.70):
        self.low, self.high,self.subtract,self.divide = low, high,subtract,divide
    def __call__(self, CT):
        # Contrast stretching
        CT=np.array(CT)
        CT[np.where(CT <= self.low)] = self.low
        CT[np.where(CT >= self.high)] = self.high
        CT = CT - self.subtract
        CT = CT / self.divide

        return PIL.Image.fromarray(CT)

class pil_contrast_strech(object):

    def __init__(self, low=2, high=98):
        self.low, self.high = low, high

    def __call__(self, img):
        # Contrast stretching
        img = np.array(img)
        plow, phigh = np.percentile(img, (self.low, self.high))
        return PIL.Image.fromarray(exposure.rescale_intensity(img, in_range=(plow, phigh)))

transform_new = tf.Compose(
    [
        tf.Resize((256, 256)),
        pil_contrast_strech(),
        RandomResizedCrop((256,256)),
        ToTensor(),
        gray2rgb
    ])

class CMNIST(Dataset):
    def __init__(self,dataset):
        super()
        self.dataset=dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        if len(self.dataset[idx]) > 1:
            return torch.FloatTensor(transform_new(self.dataset[idx][0])), torch.FloatTensor(self.dataset[idx][1])






