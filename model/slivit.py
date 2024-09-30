import torch
from torch import nn
from vit_pytorch.vit import ViT, Transformer
from einops.layers.torch import Rearrange


class ConvNext(nn.Module):
    def __init__(self, model):
        super(ConvNext, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)[0]
        return x


class SLIViT(ViT):
    def __init__(self, *, feature_extractor, vit_dim, vit_depth, heads, mlp_dim,
                 num_of_patches, dropout=0., emb_dropout=0., patch_height=768,
                 patch_width=64, rnd_pos_emb=False, num_classes=1, dim_head=64):

        super().__init__(image_size=(patch_height * num_of_patches, patch_width),
                         patch_size=(patch_height, patch_width),
                         num_classes=num_classes, dim=vit_dim, depth=vit_depth,
                         heads=heads, mlp_dim=mlp_dim, channels=1,  # Adjust if necessary
                         dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout)

        # SLIViT-specific attributes
        self.feature_extractor = feature_extractor  # Initialize the feature_extractor
        self.num_patches = num_of_patches

        # Override random positional embedding initialization (by default)
        if not rnd_pos_emb:
            self.pos_embedding = nn.Parameter(
                torch.arange(self.num_patches + 1).repeat(vit_dim, 1).t().unsqueeze(0).float()
            )

        # Override the patch embedding layer to handle feature-map patching (rather than the standard image patching)
        self.to_patch_embedding[0] = Rearrange('b c (h p1) (w p2) -> b (c h w) (p1 p2)',
                                               p1=patch_height, p2=patch_width)


    def forward(self, x):
        x = self.feature_extractor(x).last_hidden_state
        x = x.reshape((x.shape[0], self.num_patches, 768, 64))
        return super().forward(x)

    # TODO: move load_backbone here

2024-09-29 15:26:23 - INFO - Computing scores...
Better model found at epoch 0 with valid_loss value: 0.6785714285714285.
2024-09-29 15:26:25 - INFO - 
****************************************************************************************************
Model evaluation performance on test set is:
2024-09-29 15:26:25 - INFO - loss_score: 0.37672
2024-09-29 15:26:25 - INFO - roc_auc_score: 0.67857
2024-09-29 15:26:25 - INFO - average_precision_score: 0.25000
'''

'''
WandbCallback was not able to prepare a DataLoader for logging prediction samples -> 
epoch     train_loss  valid_loss  r2_score  explained_variance_score  pearsonr  time    
0         1.805435    1.725075    0.463685  0.466094                  0.693834  28:35                                                                     
Better model found at epoch 0 with valid_loss value: 1.725075125694275.
1         1.456722    1.452665    0.626510  0.626919                  0.796963  28:39                                                                     
Better model found at epoch 1 with valid_loss value: 1.4526647329330444.

'''
