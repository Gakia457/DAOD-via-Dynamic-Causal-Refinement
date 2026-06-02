"""RegionCLIP-Style ResNet backbones and FPN wrappers for Detectron2 integration."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from collections import OrderedDict

from detectron2.modeling.backbone import Backbone, BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from detectron2.layers import ShapeSpec, FrozenBatchNorm2d
import fvcore.nn.weight_init as weight_init


__all__ = ["ModifiedResNet", "build_clip_resnet_backbone", "build_clip_resnet_fpn_backbone"]

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, norm_type='FronzenBN'):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn1 = FrozenBatchNorm2d(planes)
        elif norm_type == 'SyncBN':
            self.bn1 = nn.SyncBatchNorm(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn2 = FrozenBatchNorm2d(planes)
        elif norm_type == 'SyncBN':
            self.bn2 = nn.SyncBatchNorm(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn3 = FrozenBatchNorm2d(planes * self.expansion)
        elif norm_type == 'SyncBN':
            self.bn3 = nn.SyncBatchNorm(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            downsample_layers = []
        
            if stride > 1:
                downsample_layers.append(nn.AvgPool2d(stride))
            else:
                downsample_layers.append(nn.AvgPool2d(1)) 
            
            if inplanes != planes * Bottleneck.expansion:
                downsample_layers.append(
                    nn.Conv2d(inplanes, planes * self.expansion, 
                             kernel_size=1, stride=1, bias=False)
                )
            
            if norm_type == 'FronzenBN':
                this_norm = FrozenBatchNorm2d(planes * self.expansion)
            elif norm_type == 'SyncBN':
                this_norm = nn.SyncBatchNorm(planes * self.expansion)
                
            downsample_layers.append(this_norm)
            self.downsample = nn.Sequential(*downsample_layers)

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ModifiedResNet(Backbone):
    """
    CLIP-style ResNet adapted for detection.
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, 
        out_features=None, freeze_at=0, depth=None, pool_vec=False, create_att_pool=False, norm_type='FronzenBN'):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.norm_type = norm_type

        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn1 = FrozenBatchNorm2d(width // 2)
        elif norm_type == 'SyncBN':
            self.bn1 = nn.SyncBatchNorm(width // 2)
        
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn2 = FrozenBatchNorm2d(width // 2)
        elif norm_type == 'SyncBN':
            self.bn2 = nn.SyncBatchNorm(width // 2)
        
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn3 = FrozenBatchNorm2d(width)
        elif norm_type == 'SyncBN':
            self.bn3 = nn.SyncBatchNorm(width)
        
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        
        self.pool_vec = False

        self._out_features = out_features if out_features else []
        if depth in [50, 101]:
            self._out_feature_channels = {
                'stem': width, 'res2': width * 4, 'res3': width * 8, 
                'res4': width * 16, 'res5': width * 32
            }
            self._out_feature_strides = {
                'stem': 4, 'res2': 4, 'res3': 8, 'res4': 16, 'res5': 32
            }
        elif depth in [200]:
            self._out_feature_channels = {
                'stem': 80, 'res2': 320, 'res3': 640, 'res4': 1280, 'res5': 2560
            }
            self._out_feature_strides = {
                'stem': 4, 'res2': 4, 'res3': 8, 'res4': 16, 'res5': 32
            }
        
        self.freeze(freeze_at)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride, norm_type=self.norm_type)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, norm_type=self.norm_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x
        
        assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        
        if "stem" in self._out_features:
            outputs["stem"] = x
            
        x = self.layer1(x)
        if "res2" in self._out_features:
            outputs['res2'] = x
            
        x = self.layer2(x)
        if "res3" in self._out_features:
            outputs['res3'] = x
            
        x = self.layer3(x)
        if "res4" in self._out_features:
            outputs['res4'] = x
            
        x = self.layer4(x)
        if "res5" in self._out_features:
            outputs['res5'] = x
        
        return outputs

    def freeze(self, freeze_at=0):
        """
        Freeze early stages of the backbone.
        """
        def cnnblockbase_freeze(nn_module):
            """Disable gradients and convert BatchNorm layers to frozen BN."""
            for p in nn_module.parameters():
                p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(nn_module)
        
        if freeze_at >= 1:
            cnnblockbase_freeze(self.conv1)
            cnnblockbase_freeze(self.bn1)
            cnnblockbase_freeze(self.conv2)
            cnnblockbase_freeze(self.bn2)
            cnnblockbase_freeze(self.conv3)
            cnnblockbase_freeze(self.bn3)
        for idx, stage in enumerate([self.layer1, self.layer2, self.layer3, self.layer4], start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    cnnblockbase_freeze(block)
        return self

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_clip_resnet_backbone(cfg, input_shape: ShapeSpec):
    """Build a CLIP ResNet backbone from config."""
    depth = cfg.MODEL.RESNETS.DEPTH
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES

    layers_map = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        200: [4, 6, 10, 6],
    }
    
    width_map = {
        50: 64,
        101: 64,
        200: 80,
    }
    
    output_dim_map = {
        50: 1024,
        101: 512,
        200: 640,
    }

    if depth not in layers_map:
        raise ValueError(f"Unsupported CLIP ResNet depth: {depth}. Choose from {list(layers_map.keys())}")

    return ModifiedResNet(
        layers=layers_map[depth],
        output_dim=output_dim_map[depth],
        heads=width_map[depth] * 32 // 64,
        width=width_map[depth],
        out_features=out_features,
        freeze_at=freeze_at,
        depth=depth,
        pool_vec=False, 
        create_att_pool=False 
    )


@BACKBONE_REGISTRY.register()
def build_clip_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """Build a CLIP ResNet backbone wrapped by FPN."""
    bottom_up = build_clip_resnet_backbone(cfg, input_shape)
    
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    
    return backbone


def build_clip_resnet50_fpn(out_channels: int = 256, freeze_at: int = 0):
    """Convenience builder for CLIP ResNet-50 with FPN."""
    bottom_up = ModifiedResNet(
        layers=[3, 4, 6, 3],
        output_dim=1024,
        heads=32,
        width=64,
        out_features=["res2", "res3", "res4", "res5"],
        freeze_at=freeze_at,
        depth=50,
        pool_vec=False,
        create_att_pool=False
    )
    
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=["res2", "res3", "res4", "res5"],
        out_channels=out_channels,
        norm="",
        top_block=LastLevelMaxPool(),
        fuse_type="sum"
    )
    
    return backbone
