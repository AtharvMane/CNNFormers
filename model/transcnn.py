import torch
import torch.nn as nn
from transformers import ResNetModel

from transformers.modeling_outputs import ImageClassifierOutput, BackboneOutput
from transformers import PreTrainedModel
from transformers.backbone_utils import BackboneMixin

from model.modules.attention import PatchUnPatchMHSA
from model.config.cnnformer_config import CNNFormerConfig


class CNNFormerPretrainedModel(PreTrainedModel):
  config: CNNFormerConfig
  base_model_prefix = "cnnformer"
  main_input_name = "pixel_values"
  input_modalities = ("image", )

class TransResNet(CNNFormerPretrainedModel, BackboneMixin):
  def __init__(
      self,
      config: CNNFormerConfig
    ):
    super(TransResNet, self).__init__(config = config)
    self.config = config
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

    stages = []
    for idx, resnet_stage in enumerate(resnet.encoder.stages):
      stages.append(
         nn.Sequential(
            resnet_stage,
            PatchUnPatchMHSA(
              patch_size=4,
              input_dim=self.config.hidden_sizes[idx],
              embed_dim=config.attention_embed_dim,
              dropout=config.dropout,
              upscaler_kernel_size=config.upscaler_kernel_size,
              dims_per_head=config.dims_per_multi_attention_head
            )
         )
      )
    self.stages = nn.ModuleList(stages)

  def forward(
      self,
      pixel_values,
      output_hidden_states: bool | None = None,
      return_dict: bool | None = None,
      **kwargs
    )->BackboneOutput:
    return_dict = return_dict if return_dict is not None else self.config.return_dict
    output_hidden_states = (
      output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    feature_maps = ()
    hidden_outputs = (self.self_attn_stem(self.stem(pixel_values)),)

    for idx, stage in enumerate(self.stages):
      hidden_outputs+=(stage(hidden_outputs[-1]),)

    for idx, hidden_output in enumerate(hidden_outputs):
      if idx in self.out_indices:
        feature_maps+= (hidden_output,)

    if not return_dict:
      return (feature_maps, hidden_outputs)

    return BackboneOutput(
       feature_maps=feature_maps,
       hidden_states=hidden_outputs if output_hidden_states else None,
       attentions=None
    )
