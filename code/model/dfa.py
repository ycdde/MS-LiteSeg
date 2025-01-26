import torch
import torch.nn as nn
from mmseg.registry import MODELS

class DFAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(DFAttention, self).__init__()
        self.local_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        local_feat = self.local_attention(x)
        global_context = self.global_attention(x)
        out = local_feat * global_context
        return x + out

@MODELS.register_module()
class DFANeck(nn.Module):
    def __init__(self, in_channels=[64, 128, 256, 512], enhance_stages=[2, 3]):
        super(DFANeck, self).__init__()
        self.enhance_stages = enhance_stages
        enhance_modules = {}
        for stage in enhance_stages:
            enhance_modules[f'dfa{stage}'] = DFAttention(
                in_channels=in_channels[stage]
            )
        self.enhance_modules = nn.ModuleDict(enhance_modules)

    def forward(self, xlist):
        output = list(xlist)
        for stage in self.enhance_stages:
            output[stage] = self.enhance_modules[f'dfa{stage}'](output[stage])
        return tuple(output)
