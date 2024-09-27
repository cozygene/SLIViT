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
script_name = get_script_name()


def get_label(sample, labels, pathologies):
    label = labels[(labels.path == sample)][pathologies].values
    return label


def train_and_evaluate(learner, out_dir, best_model_name, args, test_loader=None):
    gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
    for gpu in range(len(gpus)):
        try:
            # Set the current GPU
            logger.info(f'Trying GPU {gpus[gpu]}\n')
            torch.cuda.set_device(gpu)  # Switch to the current GPU
            learner.model.to(f'cuda:{gpu}')  # Move model to the current GPU

            # Release previous GPU's memory if not on the first GPU
            if gpu > 0:
                torch.cuda.set_device(gpu - 1)  # Switch to the previous GPU
                torch.cuda.empty_cache()  # Release the memory of the previous GPU
                torch.cuda.set_device(gpu)  # Switch back to the current GPU

            # Train or fine-tune the model
            if args.finetune:
                learner.fine_tune(args.epochs, args.lr)
            else:
                learner.fit(args.epochs, args.lr)

            logger.info(f'Best model is stored at:\n{out_dir}/{best_model_name}.pth')

            # Evaluate the model on the test set if provided
            if len(test_loader):
                evaluate_and_store_results(learner, test_loader, best_model_name, args.meta, args.label, out_dir)
            else:
                logger.info('Skipping evaluation... (test set was not provided)')

            # successful running
            return

        except RuntimeError as e:
            if 'out of memory' in e.args[0]:
                if gpu < len(gpus) - 1:
                    logger.error(f'GPU ran out of memory. Trying next GPU...\n')
                else:
                    # Handle failure case where all GPUs run out of memory or error out
                    logger.error('Out of memory error occurred on all GPUs.\n'
                                 'You may want to try reducing the batch size or using a larger GPU.'
                                 'Exiting...\n')
                    # Re-raise the exception for proper handling outside this function
                    raise e
            else:
                logger.error(f'Unrecognized error occurred: {e.args[0]}. Exiting...\n')
                raise e


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
        logger.info(f'{metric_name}: {metric_score:.5f}' + (('\n' + '*' * 100)
                                                            if metric_name == metric_names[-1] else ''))
        with open(f'{out_dir}/{metric_name}.txt', 'w') as f:
            f.write(f'{metric_score:.5f}\n')
    logger.info(f'Running result is saved at:\n{out_dir}\n')


def evaluate_and_store_results(learner, data_loader, model_name, meta, pathology, out_dir):
    print()
    if hasattr(data_loader, 'indices') and len(data_loader.indices) > 0 or \
            hasattr(data_loader, 'get_idxs') and len(data_loader.get_idxs()) > 0:
        # evaluate the best model
        learner.load(model_name)
        results_file = f'{out_dir}/predicted_scores.csv'
        # preds = store_predictions(learner, data_loader, meta, pathology, results_file)
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


def get_loss_and_metrics(task):
    if task == 'cls':
        loss_f = torch.nn.BCEWithLogitsLoss()  #TODO: consider using CrossEntropyLoss
        # metrics = [RocAucMulti(average=None), APScoreMulti(average=None)]
        metrics = [RocAucMulti(), APScoreMulti()]
    elif task == 'reg':
        loss_f = torch.nn.L1Loss()
        metrics = [R2Score(), ExplainedVariance(), PearsonCorrCoef()]
    else:
        raise ValueError('Unknown task option')

    return loss_f, metrics


def wrap_up(out_dir, e=None):
    with open(f'{out_dir}/done_{script_name}', 'w') as f:
        if e is None:
            # done file should be empty when successful
            logger.info('Done successfully!')
            logger.info('_' * 100 + '\n')
        else:
            f.write(f'{e}\n')
            raise e


def setup_dataloaders(args, out_dir):
    dataset_class, mnist = get_dataset_class(args.dataset)
    assert args.meta is not None or \
           mnist is not None, \
        'Meta file is required for non-mnist datasets. Please provide the meta file path.'
    train_loader, valid_loader, test_loader = get_dataloaders(dataset_class, args, out_dir, mnist)
    dls = DataLoaders(train_loader, valid_loader)
    dls.c = 2
    return dls, test_loader, mnist


def init_out_dir(args):
    out_dir = args.out_dir.rstrip('/')
    if not args.drop_default_suffix:
        # by default, the csv file name (or dataset name in case of mnist) is added to the output directory
        if args.meta is not None:
            # example args.meta:
            # ./meta/echonet.csv
            csv_file_name = os.path.splitext(args.meta.split("/")[-1])[0]  # remove extension
            out_dir = f'{out_dir}/{csv_file_name}' + (f'_{args.label}' if len(args.label.split(',')) == 1 else '')
        else:
            out_dir = f'{out_dir}/{"mock_" if args.mnist_mocks else ""}{args.dataset}'

    if args.out_suffix is not None:
        # subfolders for hp search
        out_dir += f'/{args.out_suffix}'

    logger.info(f'\nOutput direcory is\n{out_dir}\n')
    os.makedirs(out_dir, exist_ok=True)

    options_file = f'{out_dir}/options_{script_name}.txt'
    save_options(options_file, args)

    return out_dir


def create_learner(slivit, dls, out_dir, args, mnist):
    best_model_name = f'slivit_{args.dataset}' + (f'_{args.label}' if mnist is None and len(args.label.split(',')) == 1 else '')
    loss_f, metrics = get_loss_and_metrics(args.task)
    learner = Learner(dls, slivit, model_dir=out_dir, loss_func=loss_f, metrics=metrics,
                      cbs=[SaveModelCallback(fname=best_model_name),
                           EarlyStoppingCallback(min_delta=args.min_delta, patience=args.patience),
                           CSVLogger()] +
                          ([WandbCallback()] if (args.wandb_name is not None and script_name != 'evaluate') else []))
    return learner, best_model_name
