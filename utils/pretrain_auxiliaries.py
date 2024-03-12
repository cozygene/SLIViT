import io
import json
import logging
import os
from zipfile import ZipFile, BadZipFile
import PIL
import numpy as np
from PIL import Image
from skimage import exposure
from torchvision import transforms as tf
import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)

logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

gray2rgb = tf.Lambda(lambda x: x.expand(3, -1, -1))
totensor = tf.Compose([
    tf.ToTensor(),
])


def get_labels(sample, labels, pathologies):
    label=labels[(labels.Path ==sample)][pathologies].values
    return label

def get_samples(metadata, labels, pathologies):
    samples = []
    label_to_count = {p: str((labels[p]==1).sum())+' Positive Scans, ' +str((labels[p]==0).sum())+ ' Negative Scans'  for p in pathologies}
    for sample in metadata.Path.values:
        sample_labels = get_labels(sample, labels, pathologies)
        samples.append(sample)
        logger.debug(sample)
    logger.info(f'Label counts is: {label_to_count}')
    print(f'Label counts is: {label_to_count}')
    return samples

class pil_contrast_strech(object):
    def __init__(self, low=2, high=98):
        self.low, self.high = low, high
    def __call__(self, img):
        # Contrast stretching
        img = np.array(img)
        plow, phigh = np.percentile(img, (self.low, self.high))
        return PIL.Image.fromarray(exposure.rescale_intensity(img, in_range=(plow, phigh)))
    
transform_t = transforms.Compose([
        transforms.PILToTensor()
    ])

def load_jpeg(vol_name,data_dir):
    img=Image.open(data_dir+vol_name)
    img_tensor=transform_t(img)
    return img_tensor
