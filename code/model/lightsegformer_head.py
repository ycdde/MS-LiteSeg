import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize


@MODELS.register_module()
class LightSegformerHead(BaseDecodeHead):
    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        outs = []

        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            x_conv = conv(x)
            x_resized = resize(
                input=x_conv,
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners)
            outs.append(x_resized)

        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.cls_seg(out)
        out = resize(
            input=out,
            scale_factor=8,
            mode=self.interpolate_mode,
            align_corners=self.align_corners)

        return out
