import os
import logging
import wandb
from Options.bb_train_options import args
from utils.pretrain_auxiliaries import get_pathologies
from fastai.callback.wandb import *

wandb.init()

if args.verbose:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    logger = logging.getLogger('main')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    from torch.utils.data import Subset
    from fastai.vision.all import *
    from fastai.data.transforms import TrainTestSplitter
    from Dsets.PDataset import PDataset
    from transformers import AutoModelForImageClassification

    pathologies = get_pathologies(args.meta_csv)

    dataset = PDataset(args.meta_csv,
                       args.meta_csv,
                       args.data_dir,
                       data_format='jpeg',
                       pathologies=pathologies
                       )

    batch_size = args.b_size
    num_workers = args.n_cpu
    logger.info(f'Num of cpus is {args.n_cpu}')
    logger.info(f'Number of samples is {len(pd.read_csv(args.meta_csv))}')
    logger.info(f'Batch size is {args.b_size}')

    df = pd.read_csv(args.meta_csv)
    splts = [f.split('/')[1] for f in df['Path'].values]

    valid_dataset = Subset(dataset, np.argwhere(np.array(splts) == 'test'))
    train_dataset = Subset(dataset, np.argwhere(np.array(splts) == 'train'))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)

    logger.info(f'# of train batches is {len(train_loader)}')
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    logger.info(f'# of validation batches is {len(valid_loader)}')
    dls = DataLoaders(train_loader, valid_loader)
    dls.c = 2

    model2 = AutoModelForImageClassification.from_pretrained("facebook/convnext-base-224", return_dict=False,
                                                             num_labels=len(pathologies),
                                                             ignore_mismatched_sizes=True)


    class ConvNext(nn.Module):
        def __init__(self, model):
            super(ConvNext, self).__init__()
            self.model = model

        def forward(self, x):
            x = self.model(x)[0]
            return x


    model = ConvNext(model2)
    model.to(device='cuda')
    learner = Learner(dls, model, model_dir=args.out_dir,
                      loss_func=torch.nn.BCEWithLogitsLoss(), cbs=WandbCallback())

    learner.metrics = [RocAucMulti(average=None), APScoreMulti(average=None)]

    learner.fit_one_cycle(n_epoch=args.n_epochs, cbs=SaveModelCallback(fname='convnext_bb'))
