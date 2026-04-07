import torch
import torch.nn as nn

from jaxtyping import jaxtyped, Float
from beartype import beartype
from model.modules.rms_norm import RMSNorm2d

class PatchUnPatchMHSA(nn.Module):
  def __init__(
      self,
      patch_size: int,
      input_dim: int,
      embed_dim: int,
      dropout: float,
      upscaler_kernel_size: int = 5,
      dims_per_head: int = 64
    ):
    super(PatchUnPatchMHSA, self).__init__()
    assert patch_size%4==0, "patch size needs to be divisible by 4"
    assert upscaler_kernel_size%2==1, "upscaler kernel needs to be an odd number."
    self.in_projection = nn.Sequential(
      nn.Conv2d(
          input_dim,
          input_dim,
          kernel_size = (patch_size,patch_size),
          stride=(patch_size//2, patch_size//2),
          groups=input_dim,
          padding=patch_size//4,
          padding_mode='reflect'
      ),
      nn.Conv2d(
          input_dim,
          embed_dim,
          kernel_size = (1,1),
          stride=(1,1),
      ),

    )

    self.self_attn = nn.MultiheadAttention(
      embed_dim=embed_dim,
      num_heads = int(embed_dim/dims_per_head),
      batch_first=True,
      dropout=dropout
    )

    output_dim = input_dim*(patch_size//2)**2

    self.upscaler = nn.Sequential(
        nn.Conv2d(
          in_channels=embed_dim,
          out_channels=output_dim,
          kernel_size=(upscaler_kernel_size, upscaler_kernel_size),
          stride=(1,1),
          padding=upscaler_kernel_size//2,
        ),

        nn.PixelShuffle(upscale_factor=patch_size//2)
    )

    self.activation_fn = nn.GELU()
    self.patch_size = patch_size
    self.rms_norm_attn = RMSNorm2d(embed_dim)
    self.rms_norm_out = RMSNorm2d(input_dim)

  def forward(self, feats):
    x_q=self.in_projection(feats)
    b, c, h, w = x_q.shape
    x = self.activation_fn(x_q).flatten(2).permute(0,2,1).contiguous()
    dmsa_mask = torch.eye(x.shape[1], device=feats.device)
    x, _ = self.self_attn(
      x,x,x,
      attn_mask=dmsa_mask
    )

    x = x.permute(0, 2, 1).reshape(b, c, h, w).contiguous()
    x = self.activation_fn(self.rms_norm_attn((x+x_q)))

    x = self.upscaler(x)
    x = self.activation_fn(x)
    return self.activation_fn(self.rms_norm_out((x+feats)))