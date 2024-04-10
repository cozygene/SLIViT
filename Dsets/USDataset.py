import pandas as pd
import torch
from utils.slivit_auxiliaries import *
from torch.utils.data import Dataset
from fastai.vision import *



default_transform_gray = tf.Compose([
    tf.ToPILImage(),
    tf.Resize((256, 256)),
    #pil_contrast_strech(),
    tf.ToTensor(),

    gray2rgb
])
class USDataset(Dataset):
    def __init__(self, metafile_path, annotations_path, pathologies, nslc=32,transform=default_transform_gray):
        self.metadata = pd.read_csv(metafile_path)
        self.annotations = pd.read_csv(annotations_path)
        self.pathologies = pathologies
        self.samples = get_samples(self.metadata, self.annotations, pathologies)
        self.t = transform
        self.nslc=nslc
        self.data_reader =self.load_tiff
        self.label_reader = get_labels
        self.labels=[self.label_reader(self.samples[i], self.annotations, self.pathologies) for i in range(len(self.samples))]
        self.labels=torch.FloatTensor(self.labels)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx[0]]
        imgs = self.data_reader(sample,self.nslc)
        labels = self.label_reader(sample, self.annotations, self.pathologies)  
        labels = torch.FloatTensor(labels)
        t_imgs = torch.cat([self.t(im) for im in imgs], dim=-1)
        return t_imgs, labels
    
    def load_tiff(self,path,nslc):
        vol=[]
        img_paths = os.listdir(path)
        filtered = filter(lambda img_path: img_path.split('.')[-1] == 'tiff', img_paths)
        img_paths = list(filtered)
        slc_idxs=np.linspace(0, len(img_paths), nslc+1).astype(int)
        for img_name in img_paths:
            try:
                if int(img_name.split('.')[0]) in slc_idxs  :
                    img = Image.open(f'{path}/{img_name}')
                    vol.append(totensor(img))
            except EOFError:
                print(f'Error reading {img_name}')
                continue
        return vol
    
