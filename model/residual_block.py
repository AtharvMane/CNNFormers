from typing import Optional, Any, Callable

import torch
import torch.nn as nn
from model.attention import PatchUnPatchMHSA

from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck

class TransBasicBlock(nn.Module):
  expansion: int = 1
  def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: int = 0.3,
        drop_key: int = 0.1
    ):
    super(TransBasicBlock, self).__init__()
    self.resnet_basic_block = BasicBlock(
        inplanes=inplanes,
        planes=planes,
        stride=stride,
        downsample=downsample,
        groups=groups,
        base_width=base_width,
        dilation=dilation,
        norm_layer=norm_layer
    )
    self.resnet_basic_block.relu = nn.SiLU()
    self.self_attn = PatchUnPatchMHSA(8, planes, 8*planes, planes, drop_key=drop_key, dropout=dropout)

  def forward(self, x):
    x = self.resnet_basic_block(x)
    x = self.self_attn(x)
    return x