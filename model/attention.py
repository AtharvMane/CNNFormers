import torch
import torch.nn as nn

from jaxtyping import jaxtyped, Float
from beartype import beartype

class PatchUnPatchMHSA(nn.Module):
  def __init__(self, patch_size: int, input_dim: int, embed_dim: int, output_dim: int):
    super(PatchUnPatchMHSA, self).__init__()
    self.in_projection = nn.Conv2d(input_dim, embed_dim, (1,1))
    self.self_attn = nn.MultiheadAttention(embed_dim*patch_size*patch_size, embed_dim, batch_first=True)
    self.out_projection = nn.Conv2d(embed_dim, output_dim, (1,1))
    self.activation_fn = nn.GELU()
    self.patch_size = patch_size
    self.rms_norm_attn = nn.RMSNorm(embed_dim*patch_size*patch_size )
    self.rms_norm_out = nn.RMSNorm(output_dim)
  
  @jaxtyped(typechecker=beartype)
  def patchify(self, inp_feats: Float[torch.Tensor, "batch channels height width"]):
    b, c, h, w = inp_feats.shape
    assert w%self.patch_size==0 and h%self.patch_size==0, "The image must be patchifyable"
    x = inp_feats.view(b, c, h//self.patch_size,
                       self.patch_size,
                       w//self.patch_size,
                       self.patch_size).permute(0,1,2,4,3,5).permute(0,2,3,1,4,5)

    x = x.flatten(3).reshape(b,-1, c*self.patch_size*self.patch_size)
    return x

  @jaxtyped(typechecker=beartype)
  def unpatchify(self, feats: Float[torch.Tensor, "batch hw ppc"], out_h: int, out_w: int, out_c:int ):
    b = feats.shape[0]
    x = feats.reshape(b, out_h//self.patch_size, out_w//self.patch_size, -1)
    x = x.reshape(b, out_h//self.patch_size,
              out_w//self.patch_size,
              self.patch_size,
              self.patch_size,
              out_c).permute(0, 3, 1, 2, 4,5).permute(0, 1, 2, 4, 3, 5).reshape(b, out_c, out_h, out_w)
    return x
  
  def forward(self, feats):
    x = self.in_projection(feats)
    x = self.activation_fn(x)
    b, c, h, w = x.shape
    x = self.patchify(x)
    x, _ = self.self_attn(x, x, x)
    x = self.rms_norm_attn(x)
    x = self.activation_fn(x)
    x = self.unpatchify(x, h, w, c)
    x = self.out_projection(x)
    return self.activation_fn(self.rms_norm_out((x+feats).permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
