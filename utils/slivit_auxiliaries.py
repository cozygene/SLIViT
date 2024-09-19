from utils.options_parser import args
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

import sys
import wandb
import torch
import PIL
from fastai.data.core import DataLoaders
from fastai.data.load import DataLoader
from fastai.imports import *
from fastai.learner import Learner
from fastai.metrics import RocAucMulti, APScoreMulti, R2Score, ExplainedVariance, PearsonCorrCoef
from fastai.vision.all import *
from fastai.callback.wandb import *
from skimage import exposure
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Subset
from torchvision import transforms as tf
from PIL import Image
import logging
from slivit import SLIViT
from utils.load_backbone import load_backbone

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        logging.FileHandler('./log.txt'),
                        logging.StreamHandler()  # Log messages to the console
                    ])

logger = logging.getLogger()

gray2rgb = tf.Lambda(lambda x: x.expand(3, -1, -1))
totensor = tf.Compose([
    tf.ToTensor(),
])


def get_label(sample, labels, pathologies):
    label = labels[(labels.path == sample)][pathologies].values
    return label


def get_samples(metadata):
    samples = []
    # label_to_count = {p: {} for p in pathologies}
    for sample in metadata.path.values:  # -2
        samples.append(sample)
    # print(f'Label counts is: {}')
    return samples


class pil_contrast_strech(object):

    def __init__(self, low=2, high=98):
        self.low, self.high = low, high

    def __call__(self, img):
        # Contrast stretching
        img = np.array(img)
        plow, phigh = np.percentile(img, (self.low, self.high))
        return PIL.Image.fromarray(exposure.rescale_intensity(img, in_range=(plow, phigh)))


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


default_transform_gray = tf.Compose([
    tf.ToPILImage(),
    tf.Resize((256, 256)),
    # pil_contrast_strech(),
    tf.ToTensor(),
    gray2rgb
])


def get_split_indices(meta_csv, out_dir, split_ratio, pathology,
                      split_col, pid_col):
    # try:
    #     df = pd.read_csv(split_file_path)
    #     train_idx = np.argwhere(df[split_col].str.contains('train', case=False))
    #     val_idx = np.argwhere(df[split_col].str.contains('val', case=False))
    #     test_idx = np.argwhere(df[split_col].str.contains('test', case=False))
    #     return train_idx, val_idx, test_idx
    # except FileNotFoundError:
    #     pass

    df = pd.read_csv(meta_csv)
    if split_col in df.columns:
        # Get indices for train, val and test
        train_idx = np.argwhere(df[split_col].str.contains('train', case=False))
        val_idx = np.argwhere(df[split_col].str.contains('val', case=False))
        test_idx = np.argwhere(df[split_col].str.contains('test', case=False))
    else:
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
        df.to_csv(f'{out_dir}/{os.path.split(meta_csv)[-1]}', index=False)

    return train_idx, val_idx, test_idx


def store_predictions(learner, test_loader, meta_csv, pathology, results_file, split_col='Split'):
    logger.info(f'Computing predictions...')
    preds = learner.get_preds(dl=test_loader)

    df = pd.read_csv(meta_csv).iloc[test_loader.indices.squeeze()]
    assert df[split_col].str.contains('test', case=False).all()

    with open(results_file, 'w') as f:
        f.write(f'{",".join(df.columns.to_list() + ["Pred"])}\n')
        for i in range(len(preds[1])):
            if pathology == 'CRORA':  # and len(df.columns) > 1:
                assert df.CRORA.iloc[i] == preds[1][i].item(), \
                    ValueError(f'CRORA does not contain the true label!! Check failed at index {df.index[i]}')
            # record = f'{df.index[i]},' + df.iloc[i].to_csv(header=False, index=False).rstrip().replace('\n', ',')
            record = df.iloc[i].to_csv(header=False, index=False).rstrip().replace('\n', ',')
            f.write(f'{record},{preds[0][i].item()}\n')

    logger.info(f'Predictions are saved at:\n{results_file}')
    return preds


def evaluate_model(learner, evaluation_loader, out_dir, preds=None):
    if preds is None:
        logger.info(f'Computing scores...')
        metric_scores = learner.validate(dl=evaluation_loader)

        logger.info('\n' + '*' * 100 + f'\nModel evaluation performance on test set is:')
        metric_names = ['loss_score'] + [m.name for m in learner.metrics]  # loss is not included in the metrics
    else:
        # TODO: implement score computation from predictions (instead of re-running inference)
        pass

    for metric_score, metric_name in zip(metric_scores, metric_names):
        logger.info(f'{metric_name}: {metric_score:.5f}')
        with open(f'{out_dir}/{metric_name}.txt', 'w') as f:
            f.write(f'{metric_score:.5f}\n')
    logger.info('\n' + '*' * 100 + f'\nScores are saved at:\n{out_dir}')


def evaluate_and_store_results(learner, data_loader, model_name, meta_csv, pathology, out_dir):
    print()
    if hasattr(data_loader, 'indices') and len(data_loader.indices) > 0 or \
            hasattr(data_loader, 'get_idxs') and len(data_loader.get_idxs()) > 0:
        # evaluate the best model
        learner.load(model_name)
        results_file = f'{out_dir}/predicted_scores.csv'
        # preds = store_predictions(learner, data_loader, meta_csv, pathology, results_file)
        evaluate_model(learner, data_loader, out_dir)
    else:
        # evaluation_loader is empty
        logger.info('Evaluation loader is empty. No evaluation is performed.')


def get_volumetric_dataloaders(dataset_class, args, out_dir, mnist=None):
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
        train_indices, valid_indices, test_indices = get_split_indices(args.meta_csv, out_dir,
                                                                       args.split_ratio, args.label3d,
                                                                       args.split_col, args.pid_col)
        dataset = dataset_class(args.meta_csv,
                                args.label3d,
                                args.slices,
                                args.sparsing_method,
                                args.path_col)

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
                test_subset = Subset(dataset_class(test_df, args.label3d, args.slices,
                                                   args.sparsing_method, args.path_col), torch.arange(0, len(test_df)))

    if msg:
        logger.info('\n\n' + '*' * 100 + f'\n{msg}\n' + '*' * 100 + '\n')

    logger.info(f'Num of cpus is {args.cpus}')

    train_loader = DataLoader(train_subset, batch_size=args.batch, num_workers=args.cpus, drop_last=True)
    logger.info(f'# of train batches is {len(train_loader)}')
    valid_loader = DataLoader(valid_subset, batch_size=args.batch, num_workers=args.cpus, drop_last=True)
    logger.info(f'# of validation batches is {len(valid_loader)}')
    test_loader = DataLoader(test_subset, batch_size=args.batch, num_workers=args.cpus, drop_last=True)
    logger.info(f'# of Test batches is {len(test_loader)}\n')
    return train_loader, valid_loader, test_loader


def save_options(options_file, args):
    arguments = []
    for arg in vars(args):
        value = getattr(args, arg)
        if isinstance(value, bool):
            if value:
                arguments.append(f'--{arg}')
        else:  # if value is not None:
            arguments.append(f'--{arg} "{value}"')

    command = f"python {sys.argv[0]} {' '.join(arguments)}"

    with open(options_file, 'w') as f:
        f.write(' '.join(sys.argv) + "\n\n")
        f.write(command + "\n")

    logger.info(f'Running configuration is saved at:\n{options_file}\n')


def get_dataset_class(dataset_name):
    mnist = None
    if dataset_name == 'oct':
        from datasets.OCTDataset3D import OCTDataset3D as dataset_class
    elif dataset_name == 'ultrasound':
        from datasets.USDataset3D import USDataset3D as dataset_class
    elif dataset_name == 'mri':
        from datasets.MRIDataset3D import MRIDataset3D as dataset_class
    elif dataset_name == 'ct':
        from medmnist import NoduleMNIST3D as mnist
        from datasets.MNISTDataset3D import MNISTDataset3D as dataset_class
    elif dataset_name == 'custom':
        from datasets.CustomDataset3D import CustomDataset3D as dataset_class
    else:
        raise ValueError('Unknown dataset option')

    return dataset_class, mnist


def get_loss_and_metrics(task):
    if task == 'cls':
        loss_f = torch.nn.BCEWithLogitsLoss()
        # metrics = [RocAucMulti(average=None), APScoreMulti(average=None)]
        metrics = [RocAucMulti(), APScoreMulti()]
    elif task == 'reg':
        loss_f = torch.nn.L1Loss()
        metrics = [R2Score(), ExplainedVariance(), PearsonCorrCoef()]
    else:
        raise ValueError('Unknown task option')

    return loss_f, metrics


def get_script_name():
    return sys.argv[0].split('/')[-1].split('.')[0]


def wrap_up(out_dir, msg=''):
    with open(f'{out_dir}/done_{get_script_name()}', 'w') as f:
        pass
    if msg:
        with open(f'{out_dir}/error_{get_script_name()}', 'w') as f:
            f.write(msg)
    logger.info('Done!')
    logger.info('_' * 100 + '\n\n\n')


def setup_slivit(args):
    slivit = SLIViT(backbone=load_backbone(args.fe_classes, args.fe_path),
                    fi_dim=args.vit_dim, fi_depth=args.vit_depth, heads=args.heads, mlp_dim=args.mlp_dim,
                    num_vol_frames=args.slices, dropout=args.dropout, emb_dropout=args.emb_dropout)
    slivit.to(device='cuda')
    return slivit


def setup_dataloaders(args, out_dir):
    dataset_class, mnist = get_dataset_class(args.dataset)
    train_loader, valid_loader, test_loader = get_volumetric_dataloaders(dataset_class, args, out_dir, mnist)
    dls = DataLoaders(train_loader, valid_loader)
    dls.c = 2
    return dls, test_loader


def init_out_dir(args):
    out_dir = args.out_dir.rstrip('/')
    if not args.drop_default_suffix:
        # by default, the csv file name (or dataset name in case of mnist) is added to the output directory
        if args.meta_csv is not None:
            # example args.meta_csv:
            # /meta_file_folder_path/ultrasound.csv
            csv_file_name = os.path.splitext(args.meta_csv.split("/")[-1])[0]  # remove extension
            out_dir = f'{out_dir}/{csv_file_name}'
        else:
            out_dir = f'{out_dir}/{"mock_" if args.mnist_mocks else ""}{args.dataset}'

    if args.out_suffix is not None:
        # subfolders for hp search
        out_dir += f'/{args.out_suffix}'

    logger.info(f'\nOutput direcory is\n{out_dir}\n')
    os.makedirs(out_dir, exist_ok=True)

    options_file = f'{out_dir}/options_{get_script_name()}.txt'
    save_options(options_file, args)

    return out_dir


def create_learner(slivit, dls, out_dir, args):
    best_model_name = f'slivit_{args.dataset}'
    loss_f, metrics = get_loss_and_metrics(args.task)
    learner = Learner(dls, slivit, model_dir=out_dir, loss_func=loss_f, metrics=metrics,
                      cbs=[SaveModelCallback(fname=best_model_name),
                           WandbCallback(),
                           CSVLogger(),
                           EarlyStoppingCallback(monitor='valid_loss', min_delta=args.min_delta,
                                                 patience=args.patience)])
    return learner, best_model_name


if args.seed is not None:
    set_seed(args.seed)