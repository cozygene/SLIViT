import time
from Options.bb_train_options import TrainOptions
import numpy as np
from transformers import ConvNextFeatureExtractor, ConvNextModel
import pandas as pd
import os


if __name__ == '__main__':
    opt =  TrainOptions().parse()  

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id) 
    from torch.utils.data import Subset
    from fastai.vision.all import *
    from fastai.data.transforms import TrainTestSplitter
    from fastai.callback.wandb import *
    from Dsets.PDataset import PDataset
    from fastai.callback.wandb import *
    from transformers import AutoModelForImageClassification
    import torchvision.models as tmodels
    dataset = PDataset(opt.meta_csv,
                           opt.meta_csv,
                           opt.data_dir,
                        data_format='jpeg',
                        pathologies= [p for p in opt.pathologies.split(',')] )

    batch_size = opt.b_size
    num_workers = opt.n_cpu
    print(f'Num of cpus is {opt.n_cpu}')
    print(f'Number of samples is {len(pd.read_csv(opt.meta_csv))}')
    print(f'Batch size is {opt.b_size}')

    df=pd.read_csv(opt.meta_csv)
    splts=[f.split('/')[1] for f in df['Path'].values]
    
    valid_dataset = Subset(dataset,np.argwhere(np.array(splts)=='test') )
    train_dataset = Subset(dataset, np.argwhere(np.array(splts)=='train'))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)

    print(f'# of train batches is {len(train_loader)}')
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True) 
    print(f'# of validation batches is {len(valid_loader)}')
    dls = DataLoaders(train_loader, valid_loader)
    dls.c = 2

    model2 = AutoModelForImageClassification.from_pretrained("facebook/convnext-base-224",return_dict=False,num_labels=len(opt.pathologies.split(',')),
                                                            ignore_mismatched_sizes=True)
    class ConvNext(nn.Module):
        def __init__(self, model):
            super(ConvNext, self).__init__()
            self.model=model

        def forward(self, x):
            x = self.model(x)[0]
            return x
        
    model=ConvNext(model2)
    model.to(device='cuda')
    learner = Learner(dls, model, model_dir=opt.out_dir,
                    loss_func=torch.nn.BCEWithLogitsLoss())
    
    learner.metrics = [RocAucMulti(average=None), APScoreMulti(average=None)]

    learner.fit_one_cycle(n_epoch=opt.n_epochs, cbs=SaveModelCallback(fname='convnext_bb' ))


