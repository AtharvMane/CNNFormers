import torch
import torch.nn as nn
from torch.nn import functional as F

from jaxtyping import jaxtyped, Float
from beartype import beartype as typechecker
from model.modules.rms_norm import RMSNorm2d

class PatchUnPatchMHSA(nn.Module):
  def __init__(
      self,
      patch_size: int,
      input_dim: int,
      embed_dim: int,
      dropout: float,
      upscaler_kernel_size: int = 5,
      dims_per_head: int = 64,
      is_dmsa: bool | None = None
    ):
    super(PatchUnPatchMHSA, self).__init__()
    assert patch_size%4==0, "patch size needs to be divisible by 4"
    assert upscaler_kernel_size%2==1, "upscaler kernel needs to be an odd number."
    self.embed_dim = embed_dim
    self.patch_size = patch_size

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
          in_channels=2*embed_dim,
          out_channels=output_dim,
          kernel_size=(upscaler_kernel_size, upscaler_kernel_size),
          stride=(1,1),
          padding=upscaler_kernel_size//2,
        ),

        nn.PixelShuffle(upscale_factor=patch_size//2)
    )

    self.projector = nn.Parameter(
      torch.rand(2*input_dim, input_dim)
    )
    self.activation_fn = nn.GELU()
    self.rms_norm_attn = RMSNorm2d(2*embed_dim)
    self.rms_norm_attn_global = RMSNorm2d(embed_dim)

    self.rms_norm_out = RMSNorm2d(input_dim)
    self.is_dmsa = is_dmsa

  @jaxtyped(typechecker=typechecker)
  def forward(
      self,
      feats: Float[torch.Tensor, "batch_size C_in H_in W_in"],
      cls_token: Float[torch.Tensor, "batch_size {self.embed_dim}"]
    )->tuple[
      Float[torch.Tensor, "batch_size C_in H_in W_in"],
      Float[torch.Tensor, "batch_size H_in*W_in/4+1 H_in*W_in/4+1"],
      Float[torch.Tensor, "batch_size {self.embed_dim}"]
    ]:
    x_q = self.get_inprojection(feats)
    b, c, h, w = x_q.shape
    attention_tokens = self.prepare_attention_tokens(cls_token, x_q)
    cls_token_new, x, attention = self.get_attention_outputs(attention_tokens)
    x = x.permute(0, 2, 1).reshape(b, c, h, w).contiguous()
    
    x = self.skipConnectionWithNormAndAct(
      x_q, x, self.rms_norm_attn
    )


    x = self.upscaler(x)
    x = self.activation_fn(x)
    x = torch.cat([x, feats], dim=1)
    x = torch.einsum("co, bchw->bohw", self.projector, x)

    cls_token = self.skipConnectionWithNormAndActGlobal(
      cls_token,
      cls_token_new,
      self.rms_norm_attn_global
    )
    return self.activation_fn(self.rms_norm_out(x)), attention, cls_token
  

  @jaxtyped(typechecker=typechecker)
  def skipConnectionWithNormAndAct(
    self,
    identity_path: Float[torch.Tensor, "b c h w"],
    layer_path: Float[torch.Tensor, "b c h w"],
    rms_module: RMSNorm2d,
  )->Float[torch.Tensor, "b 2*c h w"]:
    x = torch.cat([layer_path, identity_path], dim=1)
    return self.activation_fn(rms_module(x))
    
  
  @jaxtyped(typechecker=typechecker)
  def skipConnectionWithNormAndActGlobal(
    self,
    identity_path: Float[torch.Tensor, "b {self.embed_dim}"],
    layer_path: Float[torch.Tensor, "b {self.embed_dim}"],
    rms_module: RMSNorm2d,
  ):
    return self.activation_fn(
      rms_module(
        (identity_path+layer_path)[:,:, None, None]
      )
    )[:,:,0,0]


  @jaxtyped(typechecker=typechecker)
  def get_inprojection(
    self,
    feats: Float[torch.Tensor, "batch_size C_in H_in W_in"],
  )->Float[
    torch.Tensor,
    "batch_size {self.embed_dim} H_in*2//{self.patch_size} W_in*2//{self.patch_size}"
  ]:
    x = self.in_projection(feats)
    return self.activation_fn(x)
  

  @jaxtyped(typechecker=typechecker)
  def prepare_attention_tokens(
    self,
    cls_token: Float[torch.Tensor, "batch_size {self.embed_dim}"],
    patches: Float[
      torch.Tensor,
      "batch_size {self.embed_dim} H W"
    ]
  )->Float[torch.Tensor, "batch_size H*W+1 {self.embed_dim}"]:
    x = patches.flatten(2).permute(0,2,1).contiguous()
    return torch.cat([cls_token[:, None], x], dim = 1)


  @jaxtyped(typechecker=typechecker)
  def get_attention_outputs(
    self,
    input_tokens: Float[torch.Tensor, "batch_size N {self.embed_dim}"],
  )->tuple[
    Float[torch.Tensor, "batch_size {self.embed_dim}"],
    Float[torch.Tensor, "batch_size N-1 {self.embed_dim}"],
    Float[torch.Tensor, "batch_size N N"],
  ]:
    dmsa_mask = torch.eye(input_tokens.shape[1], device=input_tokens.device)
    x = input_tokens
    x, attention = self.self_attn(
      x,x,x,
      attn_mask= dmsa_mask if self.is_dmsa else None
    )

    return  x[:, 0], x[:,1:], attention
    