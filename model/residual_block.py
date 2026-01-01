from typing import Optional, Any, Callable

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
    
    self.self_attn = PatchUnPatchMHSA(4, planes, planes if planes<=128 else planes//2, planes)
  
  def forward(self, x):
    x = super().forward(x)
    x = self.self_attn(x)
    return x