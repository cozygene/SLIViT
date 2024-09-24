import PIL
import numpy as np
from PIL import Image
from fastai.data.core import DataLoaders
from skimage import exposure
from torchvision import transforms as tf
from PIL import Image
from auxiliaries.misc import *
import torchvision.transforms as transforms

gray2rgb = tf.Lambda(lambda x: x.expand(3, -1, -1))


def get_labels(sample, labels, pathologies):
    label = labels[(labels.Path == sample)][pathologies].values
    return label


def get_samples(meta_data, labels, pathologies, path_col):
    samples = []
    label_to_count = {
        p: str((labels[p] == 1).sum()) + ' Positive Scans, ' + str((labels[p] == 0).sum()) + ' Negative Scans' for p in
        pathologies}
    for sample in meta_data[path_col].values:
        samples.append(sample)
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


def load_2dim(vol_name, data_dir):
    img = Image.open(data_dir + vol_name)
    img_tensor = transform_t(img)
    return img_tensor


default_transform = tf.Compose(
    [
        tf.ToPILImage(),
        tf.Resize((224, 224)),
        pil_contrast_strech(),
        ToTensor(),
        gray2rgb
    ]
)