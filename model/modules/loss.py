import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple

from jaxtyping import jaxtyped, Float, Bool, Int
from beartype import beartype as typechecker


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        embeddings1: Float[torch.Tensor, "num_embeddings embed_dim"],
        embeddings2: Float[torch.Tensor, "num_embeddings embed_dim"],
        invalid_embeddings_mask1: Bool[torch.Tensor, "num_embeddings 1"] | None = None,
        invalid_embeddings_mask2: Bool[torch.Tensor, "num_embeddings 1"] | None = None,
    ):
        embeddings1 = F.normalize(embeddings1, dim = 1)
        embeddings2 = F.normalize(embeddings2, dim = 1)
        if invalid_embeddings_mask1 is None or invalid_embeddings_mask2 is None:
            ignore_mask = None
        else:
            ignore_mask = (invalid_embeddings_mask1 | invalid_embeddings_mask2).squeeze(dim=1)

        labels = self.get_labels(
            ignore_mask=ignore_mask,
            num_labels = embeddings1.shape[0],
            device = embeddings1.device,
        )


        scores_ab = torch.matmul(embeddings1, embeddings2.transpose(0,1))/self.temperature

        if invalid_embeddings_mask1 is not None and invalid_embeddings_mask2 is not None:
            scores_ab = scores_ab.masked_fill_(
                invalid_embeddings_mask1, -1e9
            ).masked_fill_(
                invalid_embeddings_mask2.transpose(0,1), -1e9
            )
        loss = self.ce_loss(scores_ab, labels) + self.ce_loss(scores_ab.transpose(0,1), labels)
        return torch.nan_to_num(loss, nan=0.0)

    @jaxtyped(typechecker=typechecker)
    @torch.no_grad()
    def get_labels(
            self,
            num_labels:int,
            device: torch.device,
            ignore_mask: Bool[torch.Tensor, "{num_labels}"] | None
        )->Int[torch.Tensor, "{num_labels}"]:
        labels = torch.arange(
            num_labels, device=device
        )
        if ignore_mask is not None:
            labels.masked_fill_(ignore_mask, -100)
        return labels



class FeatureComparisonLoss(nn.Module):
    def __init__(
            self,
            temperature: float,
        ):
        super().__init__()
        self.info_nce_loss = InfoNCELoss(temperature=temperature)
    
    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        features_1: Float[torch.Tensor, "batch_size channels height width"],
        features_2: Float[torch.Tensor, "batch_size channels height width"],
        transformed_coords_1: Float[torch.Tensor, "batch_size num_points 1 2"],
        transformed_coords_2: Float[torch.Tensor, "batch_size num_points 1 2"],
        mask1: Bool[torch.Tensor, "batch_size num_points 1"],
        mask2: Bool[torch.Tensor, "batch_size num_points 1"],
    ):
        sampled_features_1 = self.sample_grids(transformed_coords_1, features_1)
        sampled_features_2 = self.sample_grids(transformed_coords_2, features_2)

        loss = self.info_nce_loss(
            embeddings1 = sampled_features_1.flatten(0,1),
            embeddings2 = sampled_features_2.flatten(0,1),
            invalid_embeddings_mask1 = mask1.flatten(0,1),
            invalid_embeddings_mask2 = mask2.flatten(0,1)
        )

        return loss

    @jaxtyped(typechecker=typechecker)
    def sample_grids(
        self,
        transformed_coords: Float[torch.Tensor, "batch_size num_points 1 2"],
        features: Float[torch.Tensor,  "batch_size C H W"]
    )->Float[torch.Tensor, "batch_size num_points C"]:
        grid_sample = F.grid_sample(features, transformed_coords, align_corners=False)[:,:,:,0]
        return grid_sample.transpose(1,2).contiguous()
