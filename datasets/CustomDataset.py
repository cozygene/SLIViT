import numpy as np
import pandas as pd
import torch
from fastai.vision.augment import aug_transforms
from torch.utils.data import Dataset
from fastai.vision import *
from utils.pretrain_auxiliaries import *
from torchvision.transforms import Compose,ToTensor



## Custom Dataset

class CustomDataset(Dataset):

    def __init__(self,meta):
        
        #Initialize meta_csv, pathologies 
        self.metadata = meta
        #Initialize data reader and label reader

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        # Get Sample
        # Read Sample & Label
        # Augment the Sample
        #return label and scan
        return
