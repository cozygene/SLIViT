from fastai.data.load import DataLoader
from torch.utils.data import Subset

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


def get_split_indices(meta_data, out_dir, split_ratio, pathology, split_col, pid_col):

    df = pd.read_csv(meta_data)
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
            df.to_csv(f'{out_dir}/{os.path.split(meta_data)[-1]}', index=False)
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
                                args.label3d,#.split(','),
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
            if args.test_csv is None:
                msg = 'No model evaluation will be done (test ratio was set to 0 and no test_csv was provided).'
                test_subset = Subset(dataset, [])  # empty test set
            else:
                msg = f'Using external test set for final model evaluation from:\n{args.test_csv}'
                test_df = pd.read_csv(args.test_csv)
                test_subset = Subset(dataset_class(test_df, args.label3d, args.path_col,
                                                   #**kwargs
                                                   num_slices_to_use=args.slices,
                                                   sparsing_method=args.sparsing_method,
                                                   img_suffix=args.img_suffix),
                                     np.arange(0, len(test_df)))

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
    if dataset_name == 'chestmnist':
        from medmnist import ChestMNIST as mnist
        from datasets.ChestMNIST import MNISTDataset2D as dataset_class
    elif dataset_name == 'ct':
        from medmnist import NoduleMNIST3D as mnist
        from datasets.MNISTDataset3D import MNISTDataset3D as dataset_class
    elif dataset_name == 'kermany':
        from datasets.KermanyDataset import KermanyDataset as dataset_class
    elif dataset_name == 'oct':
        from datasets.OCTDataset3D import OCTDataset3D as dataset_class
    elif dataset_name == 'ultrasound':
        from datasets.USDataset3D import USDataset3D as dataset_class
    elif dataset_name == 'mri':
        from datasets.MRIDataset3D import MRIDataset3D as dataset_class
    elif dataset_name == 'custom2d':
        from datasets.CustomDataset2D import CustomDataset2D as dataset_class
    elif dataset_name == 'custom3d':
        from datasets.CustomDataset3D import CustomDataset3D as dataset_class
    else:
        raise ValueError('Unknown dataset option')

    return dataset_class, mnist