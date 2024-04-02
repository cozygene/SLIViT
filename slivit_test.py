from Options.slivit_test_options import TestOptions
from utils.load_backbone import load_backbone
import os

if __name__ == '__main__':
    opt =  TestOptions().parse()  
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id) 
    from fastai.vision.all import *
    from fastai.callback.wandb import *
    from Dsets.NDataset import NDataset
    from fastai.callback.wandb import *
    from torch.utils.data import Subset
    from fastai.vision.all import *
    from fastai.callback.wandb import *
    from model.slivit import SLIViT
    from fastai.callback.wandb import *
    from medmnist import NoduleMNIST3D
    from Dsets.UKBBDataset import UKBBDataset
    import torch
    import sklearn
    
    batch_size = opt.b_size
    num_workers = opt.n_cpu
    print(f'Num of cpus is {opt.n_cpu}')
    warnings.filterwarnings('ignore')

    if opt.dataset3d == 'nodulemnist':    
        test_dataset = NoduleMNIST3D(split="test", download=True)
        train_dataset = NoduleMNIST3D(split="train", download=True)
        valid_dataset = NoduleMNIST3D(split="val", download=True)

        test_dataset = NDataset(test_dataset,opt.nslc)
        train_dataset = NDataset(train_dataset,opt.nslc)
        valid_dataset = NDataset(valid_dataset,opt.nslc)
    elif opt.dataset3d == 'ukbb':   
        meta=pd.read_csv(opt.meta_csv)
        train_indices =np.argwhere(meta['Split'].values=='train')
        test_indices = np.argwhere(meta['Split'].values=='test')
        valid_indices = np.argwhere(meta['Split'].values=='valid')
        dataset = UKBBDataset(opt.meta_csv,
                            opt.meta_csv,
                            data_format='dcm',
                            pathologies='PDFF')
        train_dataset = Subset(dataset, train_indices)
        valid_dataset = Subset(dataset, valid_indices)
        test_dataset = Subset(dataset, test_indices)
  
    elif opt.dataset3d == 'custom3d':
        test_dataset = NDataset(opt.test_dir,opt.nslc)
        train_dataset = NDataset(opt.train_dir,opt.nslc)
        valid_dataset = NDataset(opt.valid_dir,opt.nslc)


    print()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    print(f'# of train batches is {len(train_loader)}')

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    print(f'# of validation batches is {len(valid_loader)}')

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    print(f'# of Test batches is {len(test_loader)}')
    dls = DataLoaders(train_loader, valid_loader)
    dls.c = 2

    backbone=load_backbone(gpu_id=opt.gpu_id)
    model = SLIViT(backbone=backbone, image_size=(768, 64), patch_size=64, num_classes=1, dim=opt.dim, depth=opt.depth, heads=opt.heads,
                    mlp_dim=opt.dim, channels=opt.nslc, dropout=opt.dropout, emb_dropout=opt.emb_dropout)
    model.to(device='cuda')

    if opt.metric =='roc-auc' or opt.metric =='pr-auc':
        loss_f=torch.nn.BCEWithLogitsLoss()
    elif opt.metric =='r2':
        loss_f=torch.nn.L1Loss()

    learner = Learner(dls, model, model_dir=opt.out_dir ,
                loss_func=loss_f )
    fp16 = MixedPrecision()
    
    if opt.metric =='roc-auc' or opt.metric =='pr-auc':
        learner.metrics = [RocAucMulti(average=None), APScoreMulti(average=None)]
    elif opt.metric =='r2':
        learner.metrics =  [R2Score(),ExplainedVariance(),PearsonCorrCoef()]

    t_model = learner.load('./slivit'  )
    valid_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    print(f'# of Test batches is {len(valid_loader)}')
    res = learner.get_preds(dl=valid_loader)
    act=nn.Sigmoid()

    if opt.metric =='roc-auc':
        score = sklearn.metrics.roc_auc_score(res[1], act(res[0]))
    elif opt.metric =='pr-auc':
        score = sklearn.metrics.roc_auc_score(res[1], act(res[0]))
    elif opt.metric =='r2':
        score = sklearn.metrics.r2_score(res[1], res[0])
    print('  Performance: ' + str(score))









            