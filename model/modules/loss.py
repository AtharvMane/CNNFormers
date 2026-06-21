import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple

from jaxtyping import jaxtyped, Float, Bool, Int
from beartype import beartype as typechecker


class SSLImgProcUtils:
    @staticmethod
    @jaxtyped(typechecker=typechecker)
    @torch.no_grad()
    def get_masks(
            transformed_coords_1: Float[torch.Tensor, "batch_size num_points 1 2"],
            transformed_coords_2: Float[torch.Tensor, "batch_size num_points 1 2"]
    )->tuple[
        Bool[torch.Tensor, "batch_size num_points 1"],
        Bool[torch.Tensor, "batch_size num_points 1"]
    ]:
        mask1 = (transformed_coords_1[:,:,:,0].abs()<1)&(transformed_coords_1[:,:,:,1].abs()<1)
        mask2 = (transformed_coords_2[:,:,:,0].abs()<1)&(transformed_coords_2[:,:,:,1].abs()<1)
        return ~mask1, ~mask2



class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        embeddings1: Float[torch.Tensor, "batch_size num_embeddings embed_dim"],
        embeddings2: Float[torch.Tensor, "batch_size num_embeddings embed_dim"],
        invalid_embeddings_mask1: Bool[torch.Tensor, "batch_size num_embeddings 1"] | None = None,
        invalid_embeddings_mask2: Bool[torch.Tensor, "batch_size num_embeddings 1"] | None = None,
    ):
        
        if invalid_embeddings_mask1 is None or invalid_embeddings_mask2 is None:
            ignore_mask = None
        else:
            ignore_mask = (invalid_embeddings_mask1 | invalid_embeddings_mask2).squeeze(dim=2)

        labels = self.get_labels(
            batch_size = embeddings1.shape[0],
            ignore_mask=ignore_mask,
            num_labels = embeddings1.shape[1],
            device = embeddings1.device,
        )


        scores_ab = torch.bmm(embeddings1, embeddings2.transpose(1,2))/self.temperature

        if invalid_embeddings_mask1 is not None or invalid_embeddings_mask2 is not None:
            scores_ab = scores_ab.masked_fill_(
                invalid_embeddings_mask1, -1e9
            ).masked_fill_(
                invalid_embeddings_mask2.transpose(1,2), -1e9
            )
        loss = self.ce_loss(scores_ab, labels) + self.ce_loss(scores_ab.transpose(1,2), labels)
        return torch.nan_to_num(loss, nan=0.0)

    @jaxtyped(typechecker=typechecker)
    @torch.no_grad()
    def get_labels(
            self,
            num_labels:int,
            batch_size : int,
            device: torch.device,
            ignore_mask: Bool[torch.Tensor, "batch_size {num_labels}"] | None
        )->Int[torch.Tensor, "batch_size {num_labels}"]:
        labels = torch.arange(
            num_labels, device=device
        )[None].repeat(batch_size,1)
        if ignore_mask is not None:
            labels.masked_fill_(ignore_mask, -100)
        return labels



class FeatureComparisonLoss(nn.Module):
    def __init__(
            self,
            temperature: float,
            scale_factor:int = 2,
        ):
        super().__init__()
        self.scale_factor=scale_factor
        self.info_nce_loss = InfoNCELoss(temperature=temperature)
    
    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        features_1: Float[torch.Tensor, "batch_size channels height width"],
        features_2: Float[torch.Tensor, "batch_size channels height width"],
        transform_matrix_1: Float[torch.Tensor, "batch_size 2 3"],
        transform_matrix_2: Float[torch.Tensor, "batch_size 2 3"]
    ):
        sampled_features_1, transformed_coords_1 = self.sample_grids(transform_matrix_1, features_1)
        sampled_features_2, transformed_coords_2 = self.sample_grids(transform_matrix_2, features_2)


        mask1, mask2 = SSLImgProcUtils.get_masks(
            transformed_coords_1=transformed_coords_1,
            transformed_coords_2=transformed_coords_2
        )

        loss = self.info_nce_loss(
            embeddings1 = sampled_features_1,
            embeddings2 = sampled_features_2,
            invalid_embeddings_mask1 = mask1,
            invalid_embeddings_mask2 = mask2
        )

        return loss

    @jaxtyped(typechecker=typechecker)
    def sample_grids(
        self,
        transform_matrix: Float[torch.Tensor, "batch_size 2 3"],
        features: Float[torch.Tensor,  "batch_size C H W"]
    )->Tuple[
        Float[torch.Tensor, "batch_size num_points C"],
        Float[torch.Tensor, "batch_size num_points 1 2"]
    ]:
        coords = F.affine_grid(
            transform_matrix,
            features.shape,
            align_corners=False
        ).flatten(1,2).unsqueeze(2)

        grid_sample = F.grid_sample(features, coords, align_corners=False)[:,:,:,0]
        return F.normalize(grid_sample, dim = 1).transpose(1,2).contiguous(), coords.contiguous()
