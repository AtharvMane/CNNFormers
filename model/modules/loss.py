import torch
from torch import nn
from torch.nn import functional as F

from jaxtyping import jaxtyped, Float, Bool, Int
from beartype import beartype as typechecker

from matplotlib import pyplot as plt


class SSLImgProcUtils:
    @staticmethod
    @jaxtyped(typechecker=typechecker)
    @torch.no_grad()
    def get_coords(
        scale_factor,
        dtype,
        device,
        batch_size,
        height,
        width
    )->Float[torch.Tensor, "{batch_size} num_points 3"]:
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(width, dtype = dtype, device=device),
                torch.arange(height, dtype = dtype, device=device),
                indexing='xy'
            )
        ).flatten(1,2).t()*scale_factor
        coords = torch.cat([coords, torch.ones(len(coords),1, device=device)], dim=1)
        coords = coords.repeat(batch_size, 1, 1)
        return coords
    
    @staticmethod
    @jaxtyped(typechecker=typechecker)
    @torch.no_grad()
    def transform_coords(
        coords: Float[torch.Tensor, "batch_size num_points 3"],
        transformation_matrix: Float[torch.Tensor, "batch_size 3 3"],
        height: int,
        width: int,
        scale_factor: int
    )->Float[torch.Tensor, "batch_size num_points 1 2"]:
        transformed_coords = torch.einsum("nij, nkj->nki", transformation_matrix, coords)
        transformed_coords[:,:,:2] = transformed_coords[:,:,:2]/transformed_coords[:,:,2][:,:,None]

        transformed_coords = transformed_coords[:,:,:2]

        transformed_coords[:,:,0] = transformed_coords[:,:,0]*2/(width*scale_factor)-1
        transformed_coords[:,:,1] = transformed_coords[:,:,1]*2/(height*scale_factor)-1
        return transformed_coords[:,:,None,:]
    

    @jaxtyped(typechecker=typechecker)
    @torch.no_grad()
    def get_masks(
            transformed_coords_1: Float[torch.Tensor, "batch_size num_points 1 2"],
            transformed_coords_2: Float[torch.Tensor, "batch_size num_points 1 2"]
    )->tuple[
        Bool[torch.Tensor, "batch_size num_points 1"],
        Bool[torch.Tensor, "batch_size 1 num_points"],
        Bool[torch.Tensor, "batch_size num_points"]
    ]:
        mask1 = (transformed_coords_1[:,:,:,0].abs()<1)*(transformed_coords_1[:,:,:,1].abs()<1)
        mask2 = (transformed_coords_2[:,:,:,0].abs()<1)*(transformed_coords_2[:,:,:,1].abs()<1)
        combined_mask = mask1*mask2
        mask2 = mask2.transpose(1,2)
        return mask1, mask2, ~combined_mask[:,:,0]



class FeatureComparisonLoss(nn.Module):
    def __init__(
            self,
            scale_factor:int = 2,
        ):
        super().__init__()
        self.scale_factor=scale_factor
        self.temperature = 0.07
        self.ce_loss = nn.CrossEntropyLoss()
    
    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        features_1: Float[torch.Tensor, "batch_size channels height width"],
        features_2: Float[torch.Tensor, "batch_size channels height width"],
        transform_matrix_1: Float[torch.Tensor, "batch_size 3 3"],
        transform_matrix_2: Float[torch.Tensor, "batch_size 3 3"]
    ):
        B, N, H1, W1 = features_1.shape
        coords = SSLImgProcUtils.get_coords(
            scale_factor=self.scale_factor,
            dtype=features_1.dtype,
            device=features_1.device,
            batch_size=B,
            height=H1,
            width=W1
        )

        transformed_coords_1 = SSLImgProcUtils.transform_coords(
            coords,
            transform_matrix_1,
            W1,
            H1,
            self.scale_factor
        )

        transformed_coords_2 = SSLImgProcUtils.transform_coords(
            coords,
            transform_matrix_2,
            W1,
            H1,
            self.scale_factor
        )


        sampled_features_1 = self.sample_grids(transformed_coords_1, features_1)
        sampled_features_2 = self.sample_grids(transformed_coords_2, features_2)


        mask1, mask2, ignore_mask = SSLImgProcUtils.get_masks(
            transformed_coords_1=transformed_coords_1,
            transformed_coords_2=transformed_coords_2
        )

        labels = self.get_labels(
            combined_mask=ignore_mask,
            num_labels = H1*W1,
            device = ignore_mask.device,
        )

        if (labels==-100).all():
            return torch.tensor(0.0, device=labels.device, dtype=features_1.dtype)

        scores = torch.bmm(sampled_features_1.transpose(1,2), sampled_features_2)/self.temperature
        scores_ab = scores.clone()
        scores_ab = scores_ab.masked_fill_(~mask1, -1e9).masked_fill_(~mask2, -1e9)
        loss = self.ce_loss(scores_ab, labels)
        loss += self.ce_loss(scores_ab.transpose(1,2), labels)
        return loss

    @jaxtyped(typechecker=typechecker)
    def sample_grids(
        self,
        coords: Float[torch.Tensor, "batch_size num_points 1 2"],
        features: Float[torch.Tensor,  "batch_size C H W"]
    )->Float[torch.Tensor, "batch_size C num_points"]:
        grid_sample = F.grid_sample(features, coords, align_corners=False)[:,:,:,0]
        return F.normalize(grid_sample, dim = 1)


    @jaxtyped(typechecker=typechecker)
    @torch.no_grad()
    def get_labels(
            self,
            combined_mask: Bool[torch.Tensor, "batch_size {num_labels}"],
            num_labels:int,
            device: torch.device
        )->Int[torch.Tensor, "batch_size {num_labels}"]:
        labels = torch.arange(
            num_labels, device=device
        )[None].repeat(combined_mask.shape[0],1)
        labels.masked_fill_(combined_mask, -100)
        return labels

