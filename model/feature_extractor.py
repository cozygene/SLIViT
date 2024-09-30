import torch
from torch import nn
from transformers import AutoModelForImageClassification

class CustomHuggingFaceModel(nn.Module):
    def __init__(self, hugging_face_model):
        super().__init__()
        self.model = hugging_face_model

    def forward(self, x):
        # Get logits from the Hugging Face model
        return self.model(x).logits

def get_feature_extractor(num_labels, pretrained_weights=''):
    hugging_face_model = AutoModelForImageClassification.from_pretrained("facebook/convnext-tiny-224", return_dict=False,
                                                             num_labels=num_labels, ignore_mismatched_sizes=True)

    # weights from the Hugging Face model cannot be correctly loaded into the fastai model due to mismatched layers
    # so we wrap the Hugging Face model in a custom model that only returns the logits
    model = CustomHuggingFaceModel(hugging_face_model)

    if pretrained_weights:
        model.load_state_dict(torch.load(pretrained_weights, map_location=torch.device("cuda")))

    model_tmp = list(model.children())[0]
    model = torch.nn.Sequential(
        *[list(list(model_tmp.children())[0].children())[0], list(list(model_tmp.children())[0].children())[1]])
    return model
