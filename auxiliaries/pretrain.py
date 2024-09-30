import PIL
import numpy as np
from skimage import exposure
from torchvision import transforms as tf


def apply_contrast_stretch(img, low=2, high=98):
    img = np.array(img)
    plow, phigh = np.percentile(img, (low, high))
    return PIL.Image.fromarray(exposure.rescale_intensity(img, in_range=(plow, phigh)))


#TODO: check if can be removed (and use the one in MedMNISTDataset2D)
default_transform = tf.Compose(
    [
        tf.ToPILImage(),
        tf.Resize((224, 224)),
        # pil_contrast_strech(),
        apply_contrast_stretch,
        tf.ToTensor(),
        tf.Lambda(lambda x: x.expand(3, -1, -1))  # gray to rgb
    ]
)

