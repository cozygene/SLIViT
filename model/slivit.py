import torch
from torch import nn
from vit_pytorch.vit import ViT
from einops.layers.torch import Rearrange


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

    # TODO: consider moving get_feature_extractor here
