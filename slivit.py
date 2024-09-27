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

    def __init__(self, *, backbone, fi_dim, fi_depth, heads, mlp_dim,
                 num_vol_frames, patch_height=768, patch_width=64, rnd_pos_emb=False,
                 num_classes=1, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.backbone = backbone  # TODO: call load_backbone here
        self.num_patches = num_vol_frames
        patch_dim = patch_height * patch_width  # 768 * 64
        # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c h w) (p1 p2)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, fi_dim),
        )
        # tmpp = torch.zeros((1, fi_width))
        # positional_embeddings = torch.arange(self.num_patches) + 1
        # for i in range(self.num_patches): tmpp = torch.concat([tmpp, torch.ones((1, fi_width)) * positional_embeddings[i]], axis=0)
        # self.pos_embedding = nn.Parameter(tmpp.reshape((1, tmpp.shape[0], tmpp.shape[1])))  # .cuda()

        if rnd_pos_emb:
            # initialize positional embeddings randomly
            # TODO: investigate if this is the best way to initialize positional embeddings
            # TODO: it seems to work better for external crora
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, fi_dim))
        else:
            # initialize positional embeddings with the slice number
            self.pos_embedding = nn.Parameter(torch.arange(self.num_patches+1).repeat(fi_dim, 1).t().unsqueeze(0).float())  # require

        self.cls_token = nn.Parameter(torch.randn(1, 1, fi_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(fi_dim, fi_depth, heads, dim_head, mlp_dim, dropout)
        self.pool = 'cls'
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(fi_dim),
            nn.Linear(fi_dim, num_classes)
            #,nn.LayerNorm(num_classes)  #TODO: check if this is needed
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x).last_hidden_state
        # print(x.shape)
        # x = x.last_hidden_state
        # print(x.shape)
        x = x.reshape((x.shape[0], self.num_patches, 768, 64))
        # print(x.shape)
        return ViT.forward(self, x)
        # x = self.to_patch_embedding(x)
        # b, n, _ = x.shape
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        # x = self.dropout(x)
        # x = self.transformer(x)
        # # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        # x = x[:, 0]
        # # x = self.to_latent(x)
        # x = self.mlp_head(x)
        # return x

    # TODO: move load_backbone here

