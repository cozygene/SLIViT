from vit_pytorch import ViT
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from sklearn.metrics import precision_recall_curve
from utils.slivit_auxiliaries import *
import cv2
from tqdm import tqdm
import os
from torch import nn
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)
from torchvision import transforms as tf
import torch
from transformers import AutoModelForImageClassification
from torch import nn


class ConvNext(nn.Module):
    def __init__(self, model):
        super(ConvNext, self).__init__()
        self.model=model

    def forward(self, x):
        x = self.model(x)[0]

        return x

def pair(t):
    return t if isinstance(t, tuple) else (t, t)
gray2rgb = tf.Lambda(lambda x: x.expand(3, -1, -1))
transform_new = tf.Compose(
    [

        tf.ToPILImage(),
        tf.Resize((256, 256)),
        pil_contrast_strech(),
        #RandomResizedCrop((256,256)),
        #RandomHorizontalFlip(),
        ToTensor(),
        #normalize,
        gray2rgb
])
# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            #nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SLIViT(nn.Module):
    def __init__(self, *, backbone, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.backbone = backbone
        self.channels=channels
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
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
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
