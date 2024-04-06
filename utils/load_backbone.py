import os
from torch import nn

class ConvNext(nn.Module):
        def __init__(self, model):
            super(ConvNext, self).__init__()
            self.model=model

        def forward(self, x):
            x = self.model(x)[0]

            return x

def load_backbone(gpu_id,bb_path='./Checkpoints/convnext_bb_kermany.pth',nOfeat=4):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) 
    import torch
    from transformers import AutoModelForImageClassification
    kermany_pretrained_weights = bb_path
    model2 = AutoModelForImageClassification.from_pretrained("facebook/convnext-tiny-224", return_dict=False,
                                                                num_labels=nOfeat, ignore_mismatched_sizes=True)
    model = ConvNext(model2)
    model_weights = kermany_pretrained_weights
    model.load_state_dict(torch.load(model_weights, map_location=torch.device("cuda")))
    model_tmp = list(model.children())[0]
    model = torch.nn.Sequential(
        *[list(list(model_tmp.children())[0].children())[0], list(list(model_tmp.children())[0].children())[1]])
    return model
