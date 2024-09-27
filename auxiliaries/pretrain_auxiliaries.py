import PIL
import numpy as np
from PIL import Image
from fastai.data.core import DataLoaders
from skimage import exposure
from torchvision import transforms as tf
from PIL import Image
from auxiliaries.misc import *
import torchvision.transforms as transforms

#TODO: check if can be removed (and use the one in MNISTDataset2D)
default_transform = tf.Compose(
    [
        tf.ToPILImage(),
        tf.Resize((224, 224)),
        # pil_contrast_strech(),
        apply_contrast_stretch,
        ToTensor(),
        tf.Lambda(lambda x: x.expand(3, -1, -1))  # gray to rgb
    ]
)

