#Set seed for reproducibility
#import torch
#torch.manual_seed(0)
from Options.slivit_train_options import TrainOptions
import os

if __name__ == '__main__':
    opt =  TrainOptions().parse()  
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id) 
    from fastai.vision.all import *
    from fastai.callback.wandb import *
    from Dsets.NoduleMNISTDataset import NoduleMNISTDataset
    from fastai.callback.wandb import *
    from torch.utils.data import Subset
    from fastai.vision.all import *
    from fastai.callback.wandb import *
    from model.slivit import SLIViT
    from utils.load_backbone import load_backbone
    from fastai.callback.wandb import *
    from medmnist import NoduleMNIST3D
    from Dsets.UKBBDataset import UKBBDataset
    from Dsets.CustomDataset import CustomDataset
    from Dsets.USDataset import USDataset
    import torch
    torch.manual_seed(0)
    
    batch_size = opt.b_size
    num_workers = opt.n_cpu
    print(f'Num of cpus is {opt.n_cpu}')
    warnings.filterwarnings('ignore')

    if opt.dataset3d == 'nodulemnist':   
        test_dataset = NoduleMNIST3D(split="test", download=True)
        train_dataset = NoduleMNIST3D(split="train", download=True)
        valid_dataset = NoduleMNIST3D(split="val", download=True)
        test_dataset = NoduleMNISTDataset(test_dataset,opt.nslc)
        train_dataset = NoduleMNISTDataset(train_dataset,opt.nslc)
        valid_dataset = NoduleMNISTDataset(valid_dataset,opt.nslc)

    elif opt.dataset3d == 'ukbb':   
        meta=pd.read_csv(opt.meta_csv)
        train_indices =np.argwhere(meta['Split'].values=='train')
        test_indices = np.argwhere(meta['Split'].values=='test')
        valid_indices = np.argwhere(meta['Split'].values=='valid')
        dataset = UKBBDataset(opt.meta_csv,
                            opt.meta_csv,
                            nslc=opt.nslc,
                            pathologies='PDFF')
        train_dataset = Subset(dataset, train_indices)
        valid_dataset = Subset(dataset, valid_indices)
        test_dataset = Subset(dataset, test_indices)

    elif opt.dataset3d == 'ultrasound':
        meta=pd.read_csv(opt.meta_csv)
        train_indices =np.argwhere(meta['Split'].values=='TRAIN')
        test_indices = np.argwhere(meta['Split'].values=='TEST')
        valid_indices = np.argwhere(meta['Split'].values=='VAL')
        dataset = USDataset(opt.meta_csv,
                            opt.meta_csv,
                            nslc=opt.nslc,
                            pathologies='EF_b')
        train_dataset = Subset(dataset, train_indices)
        valid_dataset = Subset(dataset, valid_indices)
        test_dataset = Subset(dataset, test_indices)   
  
    elif opt.dataset3d == 'custom':
        test_dataset = CustomDataset(opt.test_dir,opt.nslc)
        train_dataset = CustomDataset(opt.train_dir,opt.nslc)
        valid_dataset = CustomDataset(opt.valid_dir,opt.nslc)


    print()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    print(f'# of train batches is {len(train_loader)}')
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    print(f'# of validation batches is {len(valid_loader)}')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    print(f'# of Test batches is {len(test_loader)}')

    dls = DataLoaders(train_loader, valid_loader)
    dls.c = 2
    backbone=load_backbone(opt.gpu_id,opt.bbpath,opt.nObb_feat)
    model = SLIViT(backbone=backbone, image_size=(768, 64), patch_size=64, num_classes=1, dim=opt.dim, depth=opt.depth, heads=opt.heads,
                    mlp_dim=opt.dim, channels=opt.nslc, dropout=opt.dropout, emb_dropout=opt.emb_dropout)
    model.to(device='cuda')

    if opt.task == 'classification':
        loss_f=torch.nn.BCEWithLogitsLoss()

    elif opt.task == 'regression':
        loss_f=torch.nn.L1Loss()

    learner = Learner(dls, model, model_dir=opt.out_dir ,
                loss_func=loss_f )
    
    fp16 = MixedPrecision()
    
    if opt.task == 'classification':
        learner.metrics = [RocAucMulti(average=None), APScoreMulti(average=None)]

    elif opt.task == 'regression':
        learner.metrics =  [R2Score(),ExplainedVariance(),PearsonCorrCoef()]

    learner.fit(lr=1e-4,n_epoch=opt.n_epochs, cbs=[SaveModelCallback(fname='slivit_'+opt.dataset3d),EarlyStoppingCallback(monitor='valid_loss', min_delta=opt.min_delta
                        , patience=opt.patience)])
    

