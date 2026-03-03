import torch
import torch.nn as nn

from jaxtyping import jaxtyped, Float
from beartype import beartype

class PatchUnPatchMHSA(nn.Module):
  def __init__(self, patch_size: int, input_dim: int, embed_dim: int, output_dim: int, drop_key:int, dropout: float):
    super(PatchUnPatchMHSA, self).__init__()
    assert patch_size%2==0, "patch size needs to be divisible by 2"
    self.in_projection = nn.Sequential(
      nn.Conv2d(
          input_dim,
          input_dim,
          kernel_size = (patch_size,patch_size),
          stride=(patch_size//2, patch_size//2),
          groups=input_dim,
          padding=2,
          padding_mode='reflect'
      ),
      nn.Conv2d(
          input_dim,
          embed_dim,
          kernel_size = (1,1),
          stride=(1,1),
      ),

    )


    self.self_attn = nn.MultiheadAttention(embed_dim, num_heads = int(embed_dim/64), batch_first=True, dropout=dropout)

    self.upscaler = nn.Sequential(
        nn.ConvTranspose2d(
          in_channels=embed_dim,
          out_channels=embed_dim,
          kernel_size=(patch_size,patch_size),
          stride=(patch_size//2, patch_size//2),
          groups=embed_dim,
          padding=2,
        ),

        nn.Conv2d(
          embed_dim,
          out_channels=output_dim,
          kernel_size=(1,1),
          stride=(1,1)
        )
    )

    self.dropkey = drop_key
    self.activation_fn = nn.GELU()
    self.patch_size = patch_size
    self.rms_norm_attn = nn.RMSNorm(embed_dim)
    self.rms_norm_out = nn.RMSNorm(output_dim)
  
  def forward(self, feats):
    x_q=self.in_projection(feats)
    b, c, h, w = x_q.shape

    x = self.activation_fn(x_q).flatten(2).permute(0,2,1)
    drop_key_mask = (torch.rand(x.shape[:-1], device=feats.device)<self.dropkey)
    dmsa_mask = torch.eye(x.shape[1], device=x.device, dtype=drop_key_mask.dtype)
    x, _ = self.self_attn(
      x,x,x,
      # key_padding_mask=drop_key_mask,
      attn_mask=dmsa_mask
    )

    x = x.permute(0, 2, 1).reshape(b, c, h, w)
    x = self.activation_fn(self.rms_norm_attn((x+x_q).permute(0,2,3,1))).permute(0,3,1,2)
    x = self.upscaler(x)
    x = self.activation_fn(x)
    return self.activation_fn(self.rms_norm_out((x+feats).permute(0,2,3,1))).permute(0,3,1,2)


    # b, c, h, w = x.shape
    # x = self.patchify(x)
    # x, _ = self.self_attn(x, x, x)
    # x = self.rms_norm_attn(x)
    # x = self.activation_fn(x)
    # x = self.unpatchify(x, h, w, c)
    # x = self.out_projection(x)
    # return self.activation_fn(self.rms_norm_out((x+feats).permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
