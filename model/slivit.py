from vit_pytorch import vit
from einops import repeat
from einops.layers.torch import Rearrange
from torchvision import transforms as tf
import torch
from torch import nn


class ConvNext(nn.Module):
    def __init__(self, model):
        super(ConvNext, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)[0]

        return x


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


gray2rgb = tf.Lambda(lambda x: x.expand(3, -1, -1))


class SLIViT(nn.Module):
    def __init__(self, *, backbone, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.backbone = backbone
        self.channels = channels
        image_height, image_width = pair(image_size)
        _, patch_width = pair(patch_size)
        patch_height = 12 * patch_width
        num_patches = (image_height // patch_height) * (image_width // patch_width) * channels
        patch_dim = patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c h w) (p1 p2)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )
        tmpp = torch.zeros((1, dim))
        tmp = torch.arange(num_patches) + 1
        for i in range(num_patches): tmpp = torch.concat([tmpp, torch.ones((1, dim)) * tmp[i]], axis=0)
        self.pos_embedding = nn.Parameter(tmpp.reshape((1, tmpp.shape[0], tmpp.shape[1])))  # .cuda()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = vit.Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        x = x.last_hidden_state
        x = x.reshape((x.shape[0], self.channels, 768, 64))
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        x = self.mlp_head(x)
        return x
