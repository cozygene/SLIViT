from fastai.data.load import DataLoader
from torch.utils.data import Subset
from fastai.imports import *
import torch
from utils.options_parser import args
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

import sys
if args.wandb_name is not None:
    import wandb

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from torchvision.transforms import ToTensor

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        logging.FileHandler('./log.txt'),
                        logging.StreamHandler()  # Log messages to the console
                    ])

logger = logging.getLogger()


def to_tensor(x):
    return ToTensor()(x)


def get_script_name():
    return sys.argv[0].split('/')[-1].split('.')[0]


def get_split_indices(meta, out_dir, split_ratio, pathology, split_col, pid_col):

    df = pd.read_csv(meta)
    if split_col in df.columns:
        # Get indices for train, val and test
        train_idx = np.argwhere(df[split_col].str.contains('train', case=False))
        val_idx = np.argwhere(df[split_col].str.contains('val', case=False))
        test_idx = np.argwhere(df[split_col].str.contains('test', case=False))
    else:
        if split_ratio[-1] < 1:
            logger.info(f'Creating split indices according to the provided split ratio: {split_ratio}')
            # Draw indices for train, val and test
            # First split
            gss = GroupShuffleSplit(n_splits=1, train_size=split_ratio[0])
            train_idx, temp_idx = next(gss.split(df,
                                                 y=None if get_script_name() == 'pretrain' else df[pathology],
                                                 groups=df[pid_col]))  # split by patient

            if split_ratio[2] == 0:
                # using external test set
                val_idx = temp_idx
                test_idx = temp_idx[[]]
            else:
                test_val_df = df.iloc[temp_idx]
                # Second split
                gss_temp = GroupShuffleSplit(n_splits=1, train_size=split_ratio[1] / sum(split_ratio[1:]))
                val_idx, test_idx = next(gss_temp.split(test_val_df, test_val_df[pathology],
                                                        test_val_df[pid_col]))  # split by patient

                # Map temporary indices back to original indices
                val_idx = temp_idx[val_idx]
                test_idx = temp_idx[test_idx]

            # Store split indices
            df[split_col] = 'train'
            df.loc[val_idx, split_col] = 'val'
            df.loc[test_idx, split_col] = 'test'
            df.to_csv(f'{out_dir}/{os.path.split(meta)[-1]}', index=False)
        else:
            assert get_script_name() == 'evaluate', \
                '--split_ratio was set inappropriately (empty train split is only allowed for evaluate.py)'
            # evaluation_only
            train_idx = np.array([])
            val_idx = np.array([])
            test_idx = np.arange(0, len(df))

    return train_idx, val_idx, test_idx


def get_dataloaders(dataset_class, args, out_dir, mnist=None):
    msg = ''
    if mnist is not None:
        # TODO: make sure test returns empty when pretraining (use all samples for pretraining)
        train_subset = dataset_class(mnist(split="train", download=True, root=args.mnist_root,
                                           size=28 if args.mnist_mocks else 224),
                                     num_slices_to_use=args.slices)
        valid_subset = dataset_class(mnist(split="val", download=True, root=args.mnist_root,
                                           size=28 if args.mnist_mocks else 224),
                                     num_slices_to_use=args.slices)
        test_subset = dataset_class(mnist(split="test", download=True, root=args.mnist_root,
                                          size=28 if args.mnist_mocks else 224),
                                    num_slices_to_use=args.slices)

        if args.mnist_mocks is not None:
            msg += f'Running a mock version of the dataset with {args.mnist_mocks} samples only!!'

        train_subset = Subset(train_subset, np.arange(args.mnist_mocks if args.mnist_mocks else len(train_subset)))
        valid_subset = Subset(valid_subset, np.arange(args.mnist_mocks if args.mnist_mocks else len(valid_subset)))
        test_subset = Subset(test_subset, np.arange(args.mnist_mocks if args.mnist_mocks else len(test_subset)))

        # dataset  = ConcatDataset([train_subset, valid_subset, test_subset])
    else:
        train_indices, valid_indices, test_indices = get_split_indices(args.meta, out_dir,
                                                                       args.split_ratio, args.label,
                                                                       args.split_col, args.pid_col)
        dataset = dataset_class(args.meta,
                                args.label,#.split(','),
                                args.path_col,
                                # **kwargs
                                num_slices_to_use=args.slices,
                                sparsing_method=args.sparsing_method,
                                img_suffix=args.img_suffix)

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
            if args.test_meta is None:
                msg = 'No model evaluation will be done (test ratio was set to 0 and no test_meta was provided).'
                test_subset = Subset(dataset, [])  # empty test set
            else:
                msg = f'Using external test set for final model evaluation from:\n{args.test_meta}'
                test_df = pd.read_csv(args.test_meta)
                test_subset = Subset(dataset_class(test_df, args.label, args.path_col,
                                                   #**kwargs
                                                   num_slices_to_use=args.slices,
                                                   sparsing_method=args.sparsing_method,
                                                   img_suffix=args.img_suffix),
                                     np.arange(len(test_df)))

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


def get_dataset_class(dataset_name):
    mnist = None
    if dataset_name == 'xray2d':
        from medmnist import ChestMNIST as mnist
        from datasets.MNISTDataset2D import MNISTDataset2D as dataset_class
    elif dataset_name == 'ct3d':
        from medmnist import NoduleMNIST3D as mnist
        from datasets.MNISTDataset3D import MNISTDataset3D as dataset_class
    elif dataset_name == 'oct2d':
        from datasets.OCTDataset2D import OCTDataset2D as dataset_class
    elif dataset_name == 'oct3d':
        from datasets.OCTDataset3D import OCTDataset3D as dataset_class
    elif dataset_name == 'us3d':
        from datasets.USDataset3D import USDataset3D as dataset_class
    elif dataset_name == 'mri3d':
        from datasets.MRIDataset3D import MRIDataset3D as dataset_class
    elif dataset_name == 'custom2d':
        from datasets.CustomDataset2D import CustomDataset2D as dataset_class
    elif dataset_name == 'custom3d':
        from datasets.CustomDataset3D import CustomDataset3D as dataset_class
    else:
        raise ValueError('Unknown dataset option. Please choose from: '
                         'xray2d, ct3d, oct2d, oct3d, us3d, mri3d, custom2d, custom3d.')

    return dataset_class, mnist


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if args.seed is not None:
    set_seed(args.seed)
