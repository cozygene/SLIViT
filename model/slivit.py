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


## TODO: check for code reuse for constructor
# class SLIViT(ViT):
#     def __init__(self, *, backbone, fi_dim, fi_depth, heads, mlp_dim, num_vol_frames, patch_height=768, patch_width=64, rnd_pos_emb=False, num_classes=1, dim_head=64, dropout=0., emb_dropout=0.):
#         super().__init__(
#             image_size=(patch_height, patch_width),
#             patch_size=(patch_height, patch_width),  # Assuming you have similar args
#             num_classes=num_classes,
#             dim=fi_dim,
#             depth=fi_depth,  # Adjust based on original ViT arguments
#             heads=heads,  # Adjust based on original ViT arguments
#             mlp_dim=mlp_dim,  # Adjust based on original ViT arguments
#             pool='cls',
#             channels=3,  # This might need adjustment if different
#             dim_head=dim_head,
#             dropout=dropout,
#             emb_dropout=emb_dropout
#         )
#         patch_dim = patch_height * patch_width  # 768 * 64
#
#         self.backbone = backbone
#         self.num_patches = num_vol_frames
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (c h w) (p1 p2)', p1=patch_height, p2=patch_width),
#             nn.Linear(patch_dim, fi_dim),
#         )


class SLIViT(nn.Module):
    # TODO: inherit from ViT as much as possible
    def __init__(self, *, backbone, vit_dim, vit_depth, heads, mlp_dim,
                 num_of_patches, patch_height=768, patch_width=64, rnd_pos_emb=False,
                 num_classes=1, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.backbone = backbone  # TODO: call load_backbone here
        self.num_patches = num_of_patches

        # Override random positional embedding initialization (by default)
        if not rnd_pos_emb:
            self.pos_embedding = nn.Parameter(
                torch.arange(self.num_patches + 1).repeat(vit_dim, 1).t().unsqueeze(0).float()
            )

        # Override the patch embedding layer to handle feature-map patching (rather than image pathcing)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c h w) (p1 p2)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_height * patch_width),
            nn.Linear(patch_height * patch_width, vit_dim),
            nn.LayerNorm(vit_dim),
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.feature_extractor(x).last_hidden_state
        x = x.reshape((x.shape[0], self.num_patches, 768, 64))
        return super().forward(x)

    # TODO: move load_backbone here

