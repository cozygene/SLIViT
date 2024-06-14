import os
from Options.slivit_train_options import TrainOptions
opt = TrainOptions().parse()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
# Set seed for reproducibility
from utils.slivit_auxiliaries import set_seed
set_seed(42)


if __name__ == '__main__':
    from fastai.vision.all import *
    from fastai.callback.wandb import *
    from torch.utils.data import Subset
    from model.slivit import SLIViT
    from utils.load_backbone import load_backbone

    torch.manual_seed(0)

    batch_size = opt.b_size
    num_workers = opt.n_cpu
    print(f'Num of cpus is {opt.n_cpu}')
    warnings.filterwarnings('ignore')

    if opt.dataset3d == 'nodulemnist':
        from medmnist import NoduleMNIST3D
        from Dsets.NoduleMNISTDataset import NoduleMNISTDataset

        train_dataset = NoduleMNISTDataset(NoduleMNIST3D(split="train", download=True), opt.nslc)
        valid_dataset = NoduleMNISTDataset(NoduleMNIST3D(split="val", download=True), opt.nslc)
        test_dataset = NoduleMNISTDataset(NoduleMNIST3D(split="test", download=True), opt.nslc)

        print('Running a mock version of the dataset with 20 samples only!!')
        train_dataset = Subset(train_dataset, np.arange(0, 20))
        valid_dataset = Subset(valid_dataset, np.arange(0, 20))
        test_dataset = Subset(test_dataset, np.arange(0, 20))
    else:
        if opt.dataset3d == 'ukbb':
            pathology = 'PDFF'
            from Dsets.UKBBDataset import UKBBDataset as dataset_class

        if opt.dataset3d == 'ultrasound':
            pathology = 'EF_b'
            from Dsets.USDataset import USDataset as dataset_class

        if opt.dataset3d == 'custom':
            pathology = custom_pathology
            from Dsets.CustomDataset import CustomDataset as dataset_class

        meta = pd.read_csv(opt.meta_csv)
        train_indices = np.argwhere(meta['Split'].str.contains('train', case=False))
        valid_indices = np.argwhere(meta['Split'].str.contains('val', case=False))
        test_indices = np.argwhere(meta['Split'].str.contains('test', case=False))
        dataset = dataset_class(opt.meta_csv,
                                opt.meta_csv,
                                nslc=opt.nslc,
                                pathologies=pathology)
        train_dataset = Subset(dataset, train_indices)
        valid_dataset = Subset(dataset, valid_indices)
        test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    print(f'# of train batches is {len(train_loader)}')
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    print(f'# of validation batches is {len(valid_loader)}')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    print(f'# of Test batches is {len(test_loader)}')

    dls = DataLoaders(train_loader, valid_loader)
    dls.c = 2
    backbone = load_backbone(opt.gpu_id, opt.bbpath, opt.nObb_feat)
    model = SLIViT(backbone=backbone, image_size=(768, 64), patch_size=64, num_classes=1, dim=opt.dim, depth=opt.depth,
                   heads=opt.heads,
                   mlp_dim=opt.dim, channels=opt.nslc, dropout=opt.dropout, emb_dropout=opt.emb_dropout)
    model.to(device='cuda')

    if opt.task == 'classification':
        loss_f = torch.nn.BCEWithLogitsLoss()

    elif opt.task == 'regression':
        loss_f = torch.nn.L1Loss()

    learner = Learner(dls, model, model_dir=opt.out_dir,
                      loss_func=loss_f)

    fp16 = MixedPrecision()

    if opt.task == 'classification':
        learner.metrics = [RocAucMulti(average=None), APScoreMulti(average=None)]

    elif opt.task == 'regression':
        learner.metrics = [R2Score(), ExplainedVariance(), PearsonCorrCoef()]

    learner.fit(lr=1e-4, n_epoch=opt.n_epochs, cbs=[SaveModelCallback(fname='slivit_' + opt.dataset3d),
                                                    EarlyStoppingCallback(monitor='valid_loss', min_delta=opt.min_delta
                                                                          , patience=opt.patience)])
