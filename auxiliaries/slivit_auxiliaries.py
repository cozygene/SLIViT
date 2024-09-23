import sys
import torch
import PIL
from fastai.data.core import DataLoaders
from fastai.data.load import DataLoader
from fastai.imports import *
from fastai.learner import Learner
from fastai.metrics import RocAucMulti, APScoreMulti, R2Score, ExplainedVariance, PearsonCorrCoef
from fastai.vision.all import *
from skimage import exposure
from torch.utils.data import Subset
from torchvision import transforms as tf
from PIL import Image
from slivit import SLIViT
from auxiliaries.misc import *
from utils.load_backbone import load_backbone
if args.wandb_name is not None:
    from fastai.callback.wandb import *


gray2rgb = tf.Lambda(lambda x: x.expand(3, -1, -1))


def get_label(sample, labels, pathologies):
    label = labels[(labels.path == sample)][pathologies].values
    return label


def train_and_evaluate(learner, out_dir, best_model_name, args, test_loader=None):
    gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
    for gpu in range(len(gpus)):
        try:
            # Set the current GPU
            torch.cuda.set_device(gpu)  # Switch to the current GPU
            learner.model.to(f'cuda:{gpu}')  # Move model to the current GPU

            # Release previous GPU's memory if not on the first GPU
            if gpu > 0:
                torch.cuda.set_device(gpu - 1)  # Switch to the previous GPU
                torch.cuda.empty_cache()  # Release the memory of the previous GPU
                torch.cuda.set_device(gpu)  # Switch back to the current GPU

            # Train or fine-tune the model
            if args.fine_tune:
                learner.fine_tune(args.epochs, args.lr)
            else:
                learner.fit(args.epochs, args.lr)

            logger.info(f'Best model is stored at:\n{out_dir}/{best_model_name}.pth')

            # Evaluate the model on the test set if provided
            if len(test_loader):
                evaluate_and_store_results(learner, test_loader, best_model_name, args.meta_data, args.label3d, out_dir)
            else:
                logger.info('No test set provided. Skipping evaluation...')

            # successful running
            return

        except RuntimeError as e:
            if 'out of memory' in e.args[0]:
                logger.error(f'GPU {gpus[gpu]} ran out of memory. Trying next GPU...\n')
            else:
                logger.error(f'Unrecognized errro occurred: {e.args[0]}. Exiting...\n')
                raise e

    # Handle failure case where all GPUs run out of memory or error out
    logger.error('Out of memory error occurred on all GPUs. Exiting...\n')
    raise e  # Re-raise the exception for proper handling outside this function


def get_samples(meta_data):
    samples = []
    # label_to_count = {p: {} for p in pathologies}
    for sample in meta_data.path.values:  # -2
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


def store_predictions(learner, test_loader, meta_data, pathology, results_file, split_col='Split'):
    logger.info(f'Computing predictions...')
    preds = learner.get_preds(dl=test_loader)

    df = pd.read_csv(meta_data).iloc[test_loader.indices.squeeze()]
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
    logger.info('\n' + '*' * 100 + f'\nRunning result is saved at:\n{out_dir}')


def evaluate_and_store_results(learner, data_loader, model_name, meta_data, pathology, out_dir):
    print()
    if hasattr(data_loader, 'indices') and len(data_loader.indices) > 0 or \
            hasattr(data_loader, 'get_idxs') and len(data_loader.get_idxs()) > 0:
        # evaluate the best model
        learner.load(model_name)
        results_file = f'{out_dir}/predicted_scores.csv'
        # preds = store_predictions(learner, data_loader, meta_data, pathology, results_file)
        evaluate_model(learner, data_loader, out_dir)
    else:
        # evaluation_loader is empty
        logger.info('Evaluation loader is empty. No evaluation is performed.')


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


def wrap_up(out_dir, msg=''):
    with open(f'{out_dir}/done_{get_script_name()}', 'w') as f:
        pass
    if msg:
        with open(f'{out_dir}/error_{get_script_name()}', 'w') as f:
            f.write(msg)
        logger.info('Encountered an error!')
    else:
        logger.info('Done!')
    logger.info('_' * 100 + '\n\n')


def setup_dataloaders(args, out_dir):
    dataset_class, mnist = get_dataset_class(args.dataset)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset_class, args, out_dir, mnist)
    dls = DataLoaders(train_loader, valid_loader)
    dls.c = 2
    return dls, test_loader


def init_out_dir(args):
    out_dir = args.out_dir.rstrip('/')
    if not args.drop_default_suffix:
        # by default, the csv file name (or dataset name in case of mnist) is added to the output directory
        if args.meta_data is not None:
            # example args.meta_data:
            # /meta_file_folder_path/ultrasound.csv
            csv_file_name = os.path.splitext(args.meta_data.split("/")[-1])[0]  # remove extension
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
                           EarlyStoppingCallback(monitor='valid_loss', min_delta=args.min_delta,
                                                 patience=args.patience),
                           CSVLogger()] + [WandbCallback()] if (args.wandb_name is not None and
                                                                get_script_name() != 'evaluate') else [])
    return learner, best_model_name


if args.seed is not None:
    set_seed(args.seed)