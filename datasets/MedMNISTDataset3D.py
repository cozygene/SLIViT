import numpy as np
import torch
from torch.utils.data import Dataset
from auxiliaries.finetune import default_transform_gray
# from torchvision.transforms import (
#     Compose,
#     Normalize,
#     RandomHorizontalFlip,
#     RandomResizedCrop,
#     ToTensor,
# )


# class pil_contrast_strech(object):
#     def __init__(self, low=2, high=98):
#         self.low, self.high = low, high
#
#     def __call__(self, img):
#         # Contrast stretching
#         img = np.array(img)
#         plow, phigh = np.percentile(img, (self.low, self.high))
#         return PIL.Image.fromarray(exposure.rescale_intensity(img, in_range=(plow, phigh)))


# transform_new = tf.Compose(
#     [
#         tf.ToPILImage(),
#         tf.Resize((256, 256)),
#         # pil_contrast_strech(),
#         # RandomResizedCrop((256,256)),
#         # RandomHorizontalFlip(),
#         ToTensor(),
#         gray2rgb
#     ])

# TODO: clean up this class
class MedMNISTDataset3D(Dataset):
    def __init__(self, dataset, num_slices_to_use, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.num_slices_to_use = num_slices_to_use

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.num_slices_to_use == 1:
            vol = torch.FloatTensor(self.dataset[idx][0])[:, 13, :, :]
            vol = vol.reshape(1, 1, 28, 28)
        elif self.num_slices_to_use == 28:
            vol = torch.FloatTensor(self.dataset[idx][0])
        else:
            vol = torch.FloatTensor(self.dataset[idx][0])[:, np.linspace(0, 27, self.num_slices_to_use), :, :]

        vol = torch.cat([default_transform_gray(vol[:, i, :, :]) for i in range(vol.shape[1])], dim=-2)
        return vol, self.dataset[idx][1].astype(np.float32)
