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
    chf = CustomHuggingFaceModel(hugging_face_model)

    if pretrained_weights:
        chf.load_state_dict(torch.load(pretrained_weights, map_location=torch.device("cuda")))

    nested_model = list(chf.model.children())[0]

    return torch.nn.Sequential(*list(nested_model.children())[:2])  # drop last LayerNorm layer
