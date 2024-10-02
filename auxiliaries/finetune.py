import sys
import PIL
from fastai.imports import *
from fastai.vision.all import *
from skimage import exposure
from torchvision import transforms as tf
from model.slivit import SLIViT
from auxiliaries.misc import *


if args.wandb_name is not None:
    from fastai.callback.wandb import *

gray2rgb = tf.Lambda(lambda x: x.expand(3, -1, -1))

def get_label(sample, labels, pathologies):
    label = labels[(labels.path == sample)][pathologies].values
    return label


def get_samples(meta):
    samples = []
    # label_to_count = {p: {} for p in pathologies}
    for sample in meta.path.values:  # -2
        samples.append(sample)
    # print(f'Label counts is: {}')
    return samples


# class pil_contrast_strech(object):
#
#     def __init__(self, low=2, high=98):
#         self.low, self.high = low, high
#
#     def __call__(self, img):
#         # Contrast stretching
#         img = np.array(img)
#         plow, phigh = np.percentile(img, (self.low, self.high))
#         return PIL.Image.fromarray(exposure.rescale_intensity(img, in_range=(plow, phigh)))


default_transform_gray = tf.Compose([
    tf.ToPILImage(),
    tf.Resize((256, 256)),
    # pil_contrast_strech(),
    tf.ToTensor(),
    gray2rgb
])

