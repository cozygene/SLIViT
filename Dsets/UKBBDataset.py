import numpy as np
from utils.slivit_auxiliaries import *
import pandas as pd
import torch
from fastai.vision.augment import aug_transforms
from torch.utils.data import Dataset
from fastai.vision import *


default_transform_gray = tf.Compose([
    tf.ToPILImage(),
    tf.Resize((256, 256)),
    #pil_contrast_strech(),
    tf.ToTensor(),
    gray2rgb
])
class UKBBDataset(Dataset):
    def __init__(self, metafile_path, annotations_path, pathologies, nslc,transform=default_transform_gray):
        self.metadata = pd.read_csv(metafile_path)
        self.annotations = pd.read_csv(annotations_path)
        self.pathologies = pathologies
        self.samples = get_samples(self.metadata, self.annotations, pathologies)
        self.t = transform
        self.data_reader = self.load_dcm
        self.label_reader = get_labels
        self.labels=[self.label_reader(self.samples[i], self.annotations, self.pathologies) for i in range(len(self.samples))]
        self.labels=torch.FloatTensor(self.labels)
        self.nslc=nslc

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx[0]]
        imgs = self.data_reader(sample,self.nslc)
        labels = self.label_reader(sample, self.annotations, self.pathologies)  
        labels = torch.FloatTensor(labels)
        t_imgs = torch.cat([self.t(im) for im in imgs], dim=-1)
        return t_imgs, labels.squeeze()
    
    def load_dcm(self,path,nslc):
        vol=[]
        img_paths = os.listdir(path)
        filtered = filter(lambda img_path: img_path.split('.')[-1] == 'dcm', img_paths)
        img_paths = list(filtered)
        if len(img_paths) == nslc:
            for img_name in img_paths:
                img=dicom.dcmread(f'{path}/{img_name}')
                vol.append(totensor(img.pixel_array.astype(np.float64)))
        else:
            i=0
            idx_smpl=np.linspace(0, len(img_paths)-1, nslc).astype(int)
            for img_name in img_paths:
                if i in idx_smpl:
                    img=dicom.dcmread(f'{path}/{img_name}')
                    vol.append(totensor(img.pixel_array.astype(np.float64)))
                i+=1
        return vol





