import torch
from torch.utils.data import Dataset
from torchvision import transforms as tf

# from auxiliaries.misc import to_tensor

# gray2rgb = tf.Lambda(lambda x: x.expand(3, -1, -1))

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
#

# transform_new = tf.Compose(
#     [  # TODO: check why here no need for ToPILImage in contrast to the 3D version
#         # tf.Resize((256, 256)),
#         # tf.Resize((224, 224)),
#         # pil_contrast_strech(),
#         # RandomResizedCrop((256, 256)),
#         # ToTensor(),
#         # gray2rgb
#     ])
'''
epoch     train_loss  valid_loss  roc_auc_score  average_precision_score  time    
0         0.485304    0.322785    0.518210       0.069442                 00:12     
Better model found at epoch 0 with valid_loss value: 0.3227846324443817.
'''


class MNISTDataset2D(Dataset):
    def __init__(self, dataset, **kwargs):
        super()  # .__init__()
        self.dataset = dataset
        self.num_classes = len(self.dataset.info['label'])
        self.t = tf.Compose([
            # pil_contrast_strech(),
            tf.ToTensor(),
            # resize is
            tf.Resize((224, 224)),  # if self.dataset[0][0].shape == (28, 28) else torch.nn.Identity(),
            # RandomResizedCrop((224, 224)),
            tf.Lambda(lambda x: x.expand(3, -1, -1))  # gray to rgb
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        scan, label = self.dataset[idx]
        return self.t(scan), torch.FloatTensor(label)

    def get_num_classes(self):
        return self.num_classes

'''
epoch     train_loss  valid_loss  roc_auc_score  average_precision_score  time    
0         0.492240    0.325151    0.500485       0.077712                 00:12     
Better model found at epoch 0 with valid_loss value: 0.3251511752605438.
'''
