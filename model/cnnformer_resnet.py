import torch
import torch.nn as nn

from transformers import ResNetModel
from transformers.modeling_outputs import BaseModelOutput, BackboneOutput
from transformers.utils.generic import ModelOutput
from transformers import PreTrainedModel
from transformers.backbone_utils import BackboneMixin

from model.modules.attention import PatchUnPatchMHSA
from model.config.cnnformer_config import CNNFormerConfig

from dataclasses import dataclass

from jaxtyping import Float, jaxtyped
from beartype import beartype as typechecker

from model.modules.loss import FeatureComparisonLoss


@dataclass
class DenseWithGlobalOutput(ModelOutput):
  last_dense_hidden_state: torch.Tensor
  last_global_hidden_state: torch.Tensor = None
  dense_hidden_states: torch.Tensor = None
  global_hidden_states: torch.Tensor = None
  dense_attentions: torch.Tensor = None



class CNNFormerPretrainedModel(PreTrainedModel):
  config: CNNFormerConfig
  base_model_prefix = "cnnformer"
  main_input_name = "pixel_values"
  input_modalities = ("image",)


class CNNFormerResNetModel(CNNFormerPretrainedModel):
  def __init__(
      self,
      config: CNNFormerConfig
    ):
    super(CNNFormerResNetModel, self).__init__(config = config)
    self.config = config

    self.cls_token = nn.Parameter(torch.rand(self.config.attention_embed_dim))

    resnet = ResNetModel(config)

    self.stem = resnet.embedder.embedder
    self.self_attn_stem = PatchUnPatchMHSA(
      patch_size=4,
      input_dim=resnet.embedder.embedder.convolution.out_channels, 
      embed_dim=config.attention_embed_dim,
      dropout=config.dropout,
      upscaler_kernel_size=config.upscaler_kernel_size,
      dims_per_head=config.dims_per_multi_attention_head
    )

    self.resnet_stages = nn.ModuleList([])
    self.attention_stages = nn.ModuleList([])
    
    for idx, resnet_stage in enumerate(resnet.encoder.stages):
      self.resnet_stages.append(resnet_stage)
      self.attention_stages.append(
        PatchUnPatchMHSA(
          patch_size=4,
          input_dim=self.config.hidden_sizes[idx],
          embed_dim=config.attention_embed_dim,
          dropout=config.dropout,
          upscaler_kernel_size=config.upscaler_kernel_size,
          dims_per_head=config.dims_per_multi_attention_head
        )
      )

  @jaxtyped(typechecker=typechecker)
  def forward(
      self,
      pixel_values: Float[torch.Tensor, "batch_size C H W"],
      output_hidden_states: bool | None = None,
      return_dict: bool | None = None,
      output_attentions: bool | None = None,
      **kwargs
    )->DenseWithGlobalOutput:
    return_dict = return_dict if return_dict is not None else self.config.return_dict
    output_hidden_states = (
      output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    cls_token = self.cls_token[None].repeat(pixel_values.shape[0], 1)
    hidden_state, attention, cls_token = self.self_attn_stem(self.stem(pixel_values), cls_token = cls_token)
    hidden_states = (hidden_state,)
    attentions = (attention,)

    for idx, (resnet_stage, attention_stage) in enumerate(zip(self.resnet_stages, self.attention_stages)):
      hidden_state = resnet_stage(hidden_states[-1])
      hidden_state, attention, cls_token = attention_stage(hidden_state, cls_token = cls_token)
      hidden_states+=(hidden_state,)
      attentions+=(attention,)

    if not return_dict:
      return (hidden_states, attentions)

    return DenseWithGlobalOutput(
      last_global_hidden_state = cls_token,
      last_dense_hidden_state = hidden_states[-1],
      dense_hidden_states = hidden_states if output_hidden_states else None,
      dense_attentions = attentions if output_attentions else None
    )


class CNNFormerResNetBackBone(CNNFormerPretrainedModel, BackboneMixin):
  def __init__(self, config):
    super(CNNFormerResNetBackBone, self).__init__(config)
    self.backbone = ResNetModel(config)

  def forward(
      self,
      pixel_values: Float[torch.Tensor, "batch_size C H W"],
      output_hidden_states: bool | None = None,
      return_dict: bool | None = None,
      output_attentions: bool | None = None,
  ):
    return_dict = return_dict if return_dict is not None else self.config.return_dict
    output_hidden_states = (
      output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    output_attentions = output_attentions if output_attentions is not None else self.config.output_hidden_states


    backbone_out = self.backbone(
      pixel_values,
      output_hidden_states = True,
      return_dict = True, 
      output_attentions = output_attentions
    )

    for idx, hidden_state in enumerate(backbone_out.hidden_states):
      if idx in self.out_indices:
        feature_maps+= (hidden_state,)

    if not return_dict:
      return (feature_maps, backbone_out.hidden_states, backbone_out.attentions)
    
    return BackboneOutput(
      feature_maps = feature_maps,
      hidden_states = backbone_out.hidden_states if output_hidden_states else None,
      attentions = backbone_out.attentions if output_attentions else None
    )

