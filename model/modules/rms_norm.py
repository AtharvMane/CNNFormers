import torch
import torch.nn as nn
from jaxtyping import jaxtyped, Float
from beartype import beartype as typechecker

class RMSNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        """
        RMSNorm tailored for image tensors of shape (B, C, H, W)
        """
        super().__init__()
        self.eps = eps
        # Learnable affine parameter (gamma)
        self.weight = nn.Parameter(torch.ones(num_channels))

    @jaxtyped(typechecker=typechecker)
    def forward(
            self,
            x: Float[torch.Tensor, "B C H W"]
        ) -> Float[torch.Tensor, "B C H W"]:
        # 1. Calculate RMS over the channel dimension (dim=1)
        rms = torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        
        # 2. Normalize and apply learnable weight broadcasted to (1, C, 1, 1)
        return (x / rms) * self.weight.view(1, -1, 1, 1).contiguous()