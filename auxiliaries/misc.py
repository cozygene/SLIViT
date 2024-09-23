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
            train_idx, temp_idx = next(gss.split(df, df[pathology],
                                                 df[pid_col]))  # split by patient

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
