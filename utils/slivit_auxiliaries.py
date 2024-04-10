import os
import PIL
import numpy as np
from skimage import exposure
from torchvision import transforms as tf
import pydicom as dicom
from PIL import Image

gray2rgb = tf.Lambda(lambda x: x.expand(3, -1, -1))
totensor = tf.Compose([
    tf.ToTensor(),
])

def get_labels(sample, labels, pathologies):
    label=labels[(labels.path==sample)][pathologies].values
    return label

def get_samples(metadata, labels, pathologies):
    samples = []
    #label_to_count = {p: {} for p in pathologies}
    for sample in metadata.path.values: #-2
        samples.append(sample)
    #print(f'Label counts is: {}')
    return samples
class pil_contrast_strech(object):

    def __init__(self, low=2, high=98):
        self.low, self.high = low, high

    def __call__(self, img):
        # Contrast stretching
        img = np.array(img)
        plow, phigh = np.percentile(img, (self.low, self.high))
        return PIL.Image.fromarray(exposure.rescale_intensity(img, in_range=(plow, phigh)))
