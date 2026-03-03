from typing import Optional, Any, Callable

from torch import dropout
import torch.nn as nn
from model.attention import PatchUnPatchMHSA

from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck

class TransBasicBlock(BasicBlock):
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
    super(TransBasicBlock, self).__init__(
        inplanes=inplanes,
        planes=planes,
        stride=stride,
        downsample=downsample,
        groups=groups,
        base_width=base_width,
        dilation=dilation,
        norm_layer=norm_layer
    )
    self.relu = nn.SiLU()
    self.self_attn = PatchUnPatchMHSA(8, planes, 8*planes, planes, drop_key=drop_key, dropout=dropout)
  
  def forward(self, x):
    x = super().forward(x)
    x = self.self_attn(x)
    return x