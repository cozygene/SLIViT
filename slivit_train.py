import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils.data import Subset
from fastai.vision.all import *
from fastai.data.transforms import TrainTestSplitter
from fastai.callback.wandb import *
from model.slivit import load_backbone
from model.slivit import SLIViT
from fastai.callback.wandb import *
from medmnist import NoduleMNIST3D
from Dsets.NDataset import NDataset
from torchvision import transforms as tf
import numpy as np
import torch



warnings.filterwarnings('ignore')
shuffles=np.arange(1)
splitter = TrainTestSplitter(test_size=0.40)
splitter2 = TrainTestSplitter(test_size=0.5)

test_dataset = NoduleMNIST3D(split="test", download=True)
train_dataset = NoduleMNIST3D(split="train", download=True)
valid_dataset = NoduleMNIST3D(split="val", download=True)

test_dataset=NDataset(test_dataset,4)
train_dataset = NDataset(train_dataset,4)
valid_dataset = NDataset(valid_dataset,4)

batch_size = 4
num_workers = 64


print()

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
print(f'# of train batches is {len(train_loader)}')

valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
print(f'# of validation batches is {len(valid_loader)}')

test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
print(f'# of Test batches is {len(test_loader)}')
dls = DataLoaders(train_loader, valid_loader)
dls.c = 2
multi_gpu = False

backbone = load_backbone('convnext')

model = SLIViT(backbone=backbone, image_size=(768, 64), patch_size=64, num_classes=1, dim=64, depth=3, heads=10,
                    mlp_dim=64, channels=4, dropout=0.2, emb_dropout=0.1)

model.to(device='cuda')
learner = Learner(dls, model, model_dir=f'/home/berkin/projects/hyper_rev2/' ,
            loss_func=torch.nn.BCEWithLogitsLoss()
            #loss_func= nn.CrossEntropyLoss(reduction='mean')
            )

fp16 = MixedPrecision()
learner.metrics = [RocAucMulti(average=None), APScoreMulti(average=None)]
print('Searching for learning rate...')
learner.fit_one_cycle( lr=1e-5,n_epoch=10, cbs=SaveModelCallback(fname='slivit'))





# t_model = learner.load('/home/berkin/projects/hyper_rev2/' +dis_n )
# print('Required Task has Started')
# valid_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
# print(f'# of Test batches is {len(valid_loader)}')
# xx1 = learner.get_preds(dl=valid_loader)
# # np.savez('/home/berkin/projects/Amish_hybrid/Houston_slivit_kermany_new2/' + mod + '_' + label_t[
# #  0] + '_nOslc_' + str(nOslices[kkk]) + '.npz', np.array(xx1))
# act=nn.Sigmoid()
# p2 = sklearn.metrics.roc_auc_score(xx1[1], act(xx1[0]))
# per_test_l2.append(p2)
# #np.savez('/scratch/dberkin/ORCID_new/Ricord_split_' + str(spl)+'/slivit_'+label_t[0] + '.npz', np.array(xx1))
# #np.savez('/scratch/dberkin/ORCID_new/Ricord_split_' + str(spl) + '/slivit_roc' + '.npz', np.array(p2))
# print('  Performance: ' + str(p2))

# print(per_test_l2)
# #print('Prevalance: '+ str(np.mean(xx1[1])))
# scr=bt_trap(xx1)
# np.savez('/home/berkin/projects/hyper_rev2/' +dis_n+'.npz' ,scr)
# print(np.mean(scr))





# if cntr ==0:
#     dictty={'model':[dis_n]*100,'auc':scr.reshape(100),'Imaging Modality':['CT']*100}
#     dfp=pd.DataFrame(dictty)
# else:
#     dictty={'model':[dis_n]*100,'auc':scr.reshape(100),'Imaging Modality':['CT']*100}
#     #dfp2=pd.DataFrame(dictty)
#     dfp=pd.concat([dfp,pd.DataFrame(dictty)])

# cntr+=1


# print()
# plt.figure()
# plt.ylim(0.5,1)
# sns.boxplot(x='Imaging Modality', y='auc', data=dfp,hue='model',width=0.9,linewidth=0.5,showfliers = False,whis=[5,95]) ##dff whis=[5,95]
# plt.legend(loc='lower right')
# plt.show()
# plt.savefig('slivit_figure.png',dpi=200)








            