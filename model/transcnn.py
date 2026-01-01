import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet

from transformers.modeling_outputs import ImageClassifierOutput
from transformers import PretrainedConfig

from model.residual_block import TransBasicBlock

class TransResNet(nn.Module):
  def __init__(self, layers, num_classes):
    super(TransResNet, self).__init__()
    self.model = ResNet(TransBasicBlock, layers = layers, num_classes = num_classes)
    self.num_classes = num_classes
    self.config = PretrainedConfig(num_labels=self.num_classes)
    self.loss_fct = nn.CrossEntropyLoss()
  def forward(self, pixel_values, labels=None):
    logits = self.model(pixel_values)
    loss = None
    if labels is not None:
      loss = self.loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
    return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )