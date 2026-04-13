import torch
from torch import nn

class FeatureComparisonLoss(nn.Module):
    def __init__(
            self,
            scale_factor:int = 2,
        ):
        super().__init__()
        self.scale_factor=scale_factor
    
    def forward(
        self,
        features_1,
        features_2,
        transform_matrxi_1,
        transform_matrix_2
    ):
        pass

