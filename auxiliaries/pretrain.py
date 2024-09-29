from auxiliaries.misc import *
from torchvision import transforms as tf


#TODO: check if can be removed (and use the one in MedMNISTDataset2D)
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

