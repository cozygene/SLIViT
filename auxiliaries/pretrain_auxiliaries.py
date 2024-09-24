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


def setup_dataloaders(args, out_dir):
    dataset_class, mnist = get_dataset_class(args.dataset)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset_class, args, out_dir, mnist)
    dls = DataLoaders(train_loader, valid_loader)
    dls.c = 2
    return dls, test_loader


def get_2d_dataloaders(dataset_class, args, out_dir, mnist=None):
    msg = ''
    if mnist is not None:
        train_subset = dataset_class(mnist(split="train", download=True), args.slices)
        valid_subset = dataset_class(mnist(split="val", download=True), args.slices)
        test_subset = dataset_class(mnist(split="test", download=True), args.slices)

        if args.mnist_mocks is not None:
            msg += f'Running a mock version of the dataset with {args.mnist_mocks} samples only!!'

            train_subset = Subset(train_subset, np.arange(0, args.mnist_mocks))
            valid_subset = Subset(valid_subset, np.arange(0, args.mnist_mocks))
            test_subset = Subset(test_subset, np.arange(0, args.mnist_mocks))

    else:
        train_indices, valid_indices, test_indices = get_split_indices(args.meta_data, out_dir,
                                                                       args.split_ratio, args.label3d,
                                                                       args.split_col, args.pid_col)
        dataset = dataset_class(args.meta_data,
                                args.label3d,
                                args.path_col,
                                args.slices,
                                args.sparsing_method)

        train_subset = Subset(dataset, train_indices)
        valid_subset = Subset(dataset, valid_indices)
        if len(test_indices) > 0:
            # assert args.split_ratio[2] > 0, 'Test set is not empty but split_ratio[2] is 0'
            # internal test set
            msg = f'Using internal test set for final model evaluation'
            if args.split_ratio[2] == 0:
                msg += ' (split_ratio 0 is overridden by a pre-defined split)'
            test_subset = Subset(dataset, test_indices)
        else:
            # external test set
            assert args.split_ratio[2] == 0, 'Test set is empty but split_ratio[2] is not 0'
            if args.test_csv is None:
                msg = 'No model evaluation will be done (test ratio was set to 0 and no test_csv was provided).'
                test_subset = Subset(dataset, [])  # empty test set
            else:
                msg = f'Using external test set for final model evaluation from:\n{args.test_csv}'
                test_df = pd.read_csv(args.test_csv)
                test_subset = Subset(dataset_class(test_df, args.label3d, args.path_col, args.slices,
                                                   args.sparsing_method), np.arange(0, len(test_df)))

    if msg and get_script_name() != 'evaluate':
        logger.info('\n\n' + '*' * 100 + f'\n{msg}\n' + '*' * 100 + '\n')

    logger.info(f'Num of cpus is {args.cpus}')

    train_loader = DataLoader(train_subset, batch_size=args.batch, num_workers=args.cpus, drop_last=True)
    logger.info(f'# of train batches is {len(train_loader)}')
    valid_loader = DataLoader(valid_subset, batch_size=args.batch, num_workers=args.cpus, drop_last=True)
    logger.info(f'# of validation batches is {len(valid_loader)}')
    test_loader = DataLoader(test_subset, batch_size=args.batch, num_workers=args.cpus, drop_last=True)
    logger.info(f'# of Test batches is {len(test_loader)}\n')
    return train_loader, valid_loader, test_loader

default_transform = tf.Compose(
    [
        tf.ToPILImage(),
        tf.Resize((224, 224)),
        pil_contrast_strech(),
        ToTensor(),
        gray2rgb
    ]
)