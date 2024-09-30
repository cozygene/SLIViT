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


def store_predictions(learner, test_loader, meta, pathology, results_file, split_col='Split'):
    logger.info(f'Computing predictions...')
    preds = learner.get_preds(dl=test_loader)

    df = pd.read_csv(meta).iloc[test_loader.indices.squeeze()]
    assert df[split_col].str.contains('test', case=False).all()

    with open(results_file, 'w') as f:
        f.write(f'{",".join(df.columns.to_list() + ["Pred"])}\n')
        for i in range(len(preds[1])):
            if pathology == 'CRORA':  # and len(df.columns) > 1:
                assert df.CRORA.iloc[i] == preds[1][i].item(), f'CRORA does not contain the true label!! Check failed at index {df.index[i]}'
            # record = f'{df.index[i]},' + df.iloc[i].to_csv(header=False, index=False).rstrip().replace('\n', ',')
            record = df.iloc[i].to_csv(header=False, index=False).rstrip().replace('\n', ',')
            f.write(f'{record},{preds[0][i].item()}\n')

    logger.info(f'Predictions are saved at:\n{results_file}')
    return preds
