# --------------------------------------------------------
# Copyright (C) 2022 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2022 paper
# A-ViT: Adaptive Tokens for Efficient Vision Transformer
# Hongxu Yin, Arash Vahdat, Jose M. Alvarez, Arun Mallya, Jan Kautz,
# and Pavlo Molchanov
# --------------------------------------------------------

import sys
from functools import partial

import torch
import torch.nn as nn

sys.path.append("A-ViT")
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import VisionTransformer, _cfg

__all__ = ["avit_tiny_patch16_224", "avit_small_patch16_224", "vit_tiny_patch8", ""]


@register_model
def tvit_tiny_patch8(pretrained=False, **kwargs):
    from timm.models.te_vision_transformer import VisionTransformer

    print("Loading TViT Tiny Patch8")

    model = VisionTransformer(
        patch_size=8,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@register_model
def vit_tiny_patch8(pretrained=False, **kwargs):
    from timm.models.vision_transformer import VisionTransformer

    model = VisionTransformer(
        patch_size=8,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@register_model
def avit_tiny_patch16_224(pretrained=False, **kwargs):
    from timm.models.act_vision_transformer import VisionTransformer

    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        # note that this part loads DEIT weights, not A-ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def avit_small_patch16_224(pretrained=False, **kwargs):
    from timm.models.act_vision_transformer import VisionTransformer

    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )

    model.default_cfg = _cfg()
    if pretrained:
        # note that this part loads DEIT weights, not A-ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


### added


@register_model
def avit_tiny_patch8(pretrained=False, **kwargs):
    from timm.models.act_vision_transformer import VisionTransformer

    model = VisionTransformer(
        patch_size=8,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    return model
