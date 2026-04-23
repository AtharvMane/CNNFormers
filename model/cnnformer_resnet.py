import os

import torch
import torch.nn as nn

from transformers import ResNetModel
from transformers.modeling_outputs import ImageClassifierOutput, BackboneOutput
from transformers.utils.generic import ModelOutput
from transformers import PreTrainedModel
from transformers.backbone_utils import BackboneMixin

from model.modules.attention import PatchUnPatchMHSA
from model.modules.rms_norm import RMSNorm2d
from model.config.cnnformer_config import CNNFormerConfig
from model.modules.loss import FeatureComparisonLoss, InfoNCELoss

from dataclasses import dataclass

from jaxtyping import Float, jaxtyped
from beartype import beartype as typechecker

import kornia.augmentation as K

@dataclass
class DenseWithGlobalOutput(ModelOutput):
  last_dense_hidden_state: torch.Tensor
  last_global_hidden_state: torch.Tensor = None
  dense_hidden_states: torch.Tensor = None
  global_hidden_states: torch.Tensor = None
  dense_attentions: torch.Tensor = None


@dataclass
class DenseContrastiveOutput(ModelOutput):
  loss: torch.FloatTensor | None = None
  teacher_output: DenseWithGlobalOutput | None = None
  student_output: DenseWithGlobalOutput | None = None


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
      patch_size=config.attention_patch_size,
      input_dim=resnet.embedder.embedder.convolution.out_channels, 
      embed_dim=config.attention_embed_dim,
      dropout=config.dropout,
      upscaler_kernel_size=config.upscaler_kernel_size,
      dims_per_head=config.dims_per_multi_attention_head,
      is_dmsa=config.attention_is_dmsa
    )

    self.resnet_stages = nn.ModuleList([])
    self.attention_stages = nn.ModuleList([])
    
    for idx, resnet_stage in enumerate(resnet.encoder.stages):
      self.resnet_stages.append(resnet_stage)
      self.attention_stages.append(
        PatchUnPatchMHSA(
          patch_size=config.attention_patch_size,
          input_dim=self.config.hidden_sizes[idx],
          embed_dim=config.attention_embed_dim,
          dropout=config.dropout,
          upscaler_kernel_size=config.upscaler_kernel_size,
          dims_per_head=config.dims_per_multi_attention_head,
          is_dmsa=config.attention_is_dmsa
        )
      )
    
    self.cnn_global_projector = nn.Sequential(
      nn.AdaptiveAvgPool2d((1,1)),
      nn.Conv2d(
        in_channels=config.hidden_sizes[-1],
        out_channels=config.attention_embed_dim,
        kernel_size=(1,1)
      ),
      RMSNorm2d(
        num_channels=config.attention_embed_dim
      ),
      nn.GELU()
    )

    self.global_projector = nn.Sequential(
      nn.Conv2d(
        in_channels=2*config.attention_embed_dim,
        out_channels=config.attention_embed_dim,
        kernel_size=(1,1)
      ),
      RMSNorm2d(config.attention_embed_dim),
      nn.GELU()
    )
    self.post_init()

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

    cnn_cls_token = self.cnn_global_projector(
      hidden_states[-1]
    ).squeeze([2,3])
    
    global_cls_token = torch.cat([cnn_cls_token, cls_token], dim = 1)
    
    global_cls_token = self.global_projector(global_cls_token[:,:, None, None]).squeeze([2,3])

    if not return_dict:
      return (hidden_states, attentions)

    return DenseWithGlobalOutput(
      last_global_hidden_state = global_cls_token,
      last_dense_hidden_state = hidden_states[-1],
      dense_hidden_states = hidden_states if output_hidden_states else None,
      dense_attentions = attentions if output_attentions else None
    )


class CNNFormerResNetBackBone(CNNFormerPretrainedModel, BackboneMixin):
  def __init__(self, config):
    super(CNNFormerResNetBackBone, self).__init__(config)
    self.backbone = ResNetModel(config)

  @jaxtyped(typechecker=typechecker)
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

class CNNFormerResNetForPixelLevelRepresentationModeling(CNNFormerPretrainedModel):
  def __init__(
    self,
    config: CNNFormerConfig
  ):
    super(CNNFormerResNetForPixelLevelRepresentationModeling, self).__init__(config = config)

    self.backbone_student = CNNFormerResNetModel(config)
    self.backbone_teacher = CNNFormerResNetModel(config)

    self.teacher_projectors = nn.ModuleList([])
    self.student_projectors = nn.ModuleList([])

    self.losses = nn.ModuleList([])

    scales = [2,4] if self.config.downsample_in_first_stage else [2,2]
    for i in range(len(self.config.depths)-1):
      scales.append(2**(i+2))

    for i in range(config.num_loss_stages):
      self.student_projectors.append(
        nn.Sequential(
          nn.Conv2d(
            config.hidden_sizes[-i-1],
            config.dense_ssl_projection_dim,
            kernel_size=(1,1),
            stride=(1,1)
          ),
          RMSNorm2d(config.dense_ssl_projection_dim),
          nn.GELU(),
          nn.Conv2d(
            in_channels=config.dense_ssl_projection_dim,
            out_channels=config.dense_ssl_projection_dim,
            kernel_size=(1,1),
            stride=(1,1)
          )
        )
      )
      self.teacher_projectors.append(
        nn.Sequential(
          nn.Conv2d(
            config.hidden_sizes[-i-1],
            config.dense_ssl_projection_dim,
            kernel_size=(1,1),
            stride=(1,1)
          ),
          RMSNorm2d(config.dense_ssl_projection_dim),
          nn.GELU(),
          nn.Conv2d(
            in_channels=config.dense_ssl_projection_dim,
            out_channels=config.dense_ssl_projection_dim,
            kernel_size=(1,1),
            stride=(1,1)
          )
        )
      )

      self.student_global_ssl_projector = nn.Sequential(
        nn.Linear(in_features=config.attention_embed_dim, out_features=config.dense_ssl_projection_dim),
        nn.GELU(),
        nn.Linear(in_features=config.dense_ssl_projection_dim, out_features=config.num_labels)
      )

      self.teacher_global_ssl_projector = nn.Sequential(
        nn.Linear(in_features=config.attention_embed_dim, out_features=config.dense_ssl_projection_dim),
        nn.ReLU(),
        nn.Linear(in_features=config.dense_ssl_projection_dim, out_features=config.num_labels)
      )

      self.losses.append(
        FeatureComparisonLoss(scale_factor=scales[-i-1], temperature=config.loss_temperature)
      )
    
    self.global_loss = InfoNCELoss(temperature=config.loss_temperature)
    self.initialize_teacher()

    with torch.device('cpu'):
      self._build_transform()

    self.post_init()


  @torch.no_grad()
  def _build_transform(self):
    self.transform = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=self.config.horizontal_flip_probability),
        K.RandomRotation(self.config.random_rotation_max_angle_degrees),
        K.RandomResizedCrop(
          size = self.config.random_resize_crop_size,
          scale = self.config.random_resize_crop_scale,
          p = self.config.random_resize_crop_probability
        ),
        K.ColorJitter(
          brightness = self.config.color_jitter_brightness,
          contrast = self.config.color_jitter_contrast,
          saturation = self.config.color_jitter_saturation,
          hue = self.config.color_jitter_hue,
          p = self.config.color_jitter_probability
        ),
        K.RandomGrayscale(
          p=self.config.random_grayscale_probability,
          rgb_weights=torch.tensor(self.config.normalize_mean)
        ),
        K.RandomGaussianBlur(
          kernel_size=self.config.gaussian_blur_kernel_size,
          sigma=self.config.gaussian_blur_sigma,
          p=self.config.gaussian_blur_probability
        ),
        K.RandomSolarize(p=self.config.solarize_probability),
        K.Normalize(mean=self.config.normalize_mean, std=self.config.normalize_std),
        data_keys=['image']
      )

  @torch.no_grad()
  def initialize_teacher(self):
    for parameter_teacher, parameter_student in zip(self.backbone_teacher.parameters(), self.backbone_student.parameters()):
      parameter_teacher.data = parameter_student.data.clone()
      parameter_teacher.requires_grad_(False)
    
    for parameter_teacher, parameter_student in zip(self.teacher_projectors.parameters(), self.student_projectors.parameters()):
      parameter_teacher.data = parameter_student.data.clone()
      parameter_teacher.requires_grad_(False)
  
  @torch.no_grad()
  def update_teacher(self):
    for parameter_teacher, parameter_student in zip(self.backbone_teacher.parameters(), self.backbone_student.parameters()):
      parameter_teacher.data = (
        self.config.teacher_training_lambda*parameter_student.data+(1-self.config.teacher_training_lambda)*parameter_teacher.data
      )
    
    for parameter_teacher, parameter_student in zip(self.teacher_projectors.parameters(), self.student_projectors.parameters()):
      parameter_teacher.data = (
        self.config.teacher_training_lambda*parameter_student.data+(1-self.config.teacher_training_lambda)*parameter_teacher.data
      )

  @torch.no_grad()
  def teacher_forward(self, pixel_values):
    return self.backbone_teacher(pixel_values, output_hidden_states=True)

  def forward(
      self,
      pixel_values,
      labels = None
  )->DenseContrastiveOutput:
    curr_dtype = pixel_values.dtype
    self.update_teacher()

    # if self.transform is None:
      

    with torch.autocast(device_type=pixel_values.device.type, dtype=torch.float32, enabled=False):
      pixel_values_1 = self.transform(pixel_values)
      pix_transform_1 = self.transform.transform_matrix
      pixel_values_2 = self.transform(pixel_values)
      pix_transform_2 = self.transform.transform_matrix

    student_outs = self.backbone_student(pixel_values_1.to(curr_dtype), output_hidden_states=True)
    teacher_outs = self.teacher_forward(pixel_values_2.to(curr_dtype))

    student_global_outs = self.student_global_ssl_projector(
      student_outs.last_global_hidden_state
    )

    teacher_global_outs = self.teacher_global_ssl_projector(
      teacher_outs.last_global_hidden_state
    )

    loss = self.global_loss(
      student_global_outs[None],
      teacher_global_outs[None]
    )

    for idx, (loss_fn, student_projector, teacher_projector) in enumerate(
      zip(self.losses, self.student_projectors, self.teacher_projectors)
    ): 
      student_features = student_projector(student_outs.dense_hidden_states[-idx-1])
      teacher_features = teacher_projector(teacher_outs.dense_hidden_states[-idx-1])
      loss+=loss_fn(
        features_1 = student_features,
        features_2 = teacher_features,
        transform_matrix_1 = pix_transform_1,
        transform_matrix_2 = pix_transform_2
      )

    return DenseContrastiveOutput(
      loss=loss,
      student_output=student_outs,
      teacher_output=teacher_outs
    )

  def save_pretrained(self, save_directory, **kwargs):
    super().save_pretrained(save_directory, **kwargs)
    save_dir_backbone = os.path.join(save_directory, 'backbone')
    self.backbone_student.save_pretrained(save_directory=save_dir_backbone, **kwargs)
    
class CNNFormerForImageClassification(CNNFormerPretrainedModel):
  def __init__(self, config: CNNFormerConfig):
    super().__init__(config=config)
    self.backbone = CNNFormerResNetModel(config)
    
    if config.freeze_backbone:
      with torch.no_grad():
        for param in self.backbone.parameters():
          param.requires_grad_(False)
    
    self.projector = nn.Sequential(
      nn.Linear(
        in_features=config.dense_ssl_projection_dim,
        out_features=config.dense_ssl_projection_dim,
      ),
      nn.GELU(),
      nn.Linear(
        in_features=config.dense_ssl_projection_dim,
        out_features=config.num_labels
      )
    )
    self.loss_fct = nn.CrossEntropyLoss()
  def forward(
      self,
      pixel_values,
      labels = None
    ):
    backbone_outs = self.backbone.forward(pixel_values)
    hidden_state = backbone_outs.last_global_hidden_state
    logits = self.projector(hidden_state)
    loss = None
    if labels is not None:
      loss = self.loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
    return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    load_backbone_only = kwargs.pop('load_backbone_only', False)
    if load_backbone_only:
      config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
      model = cls(config)

      model.backbone = CNNFormerResNetModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
    else:
      return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)