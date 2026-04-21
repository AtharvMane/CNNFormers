import torch
from torch import nn
from torch.nn import functional as F

from jaxtyping import jaxtyped, Float, Bool, Int
from beartype import beartype as typechecker


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

        if (labels==-100).all():
            return 0.0*embeddings1.sum()

        scores = torch.bmm(embeddings1, embeddings2.transpose(1,2))/self.temperature
        scores_ab = scores.clone()

        if invalid_embeddings_mask1 is not None or invalid_embeddings_mask2 is not None:
            scores_ab = scores_ab.masked_fill_(
                invalid_embeddings_mask1, -1e9
            ).masked_fill_(
                invalid_embeddings_mask2.transpose(1,2), -1e9
            )

        loss = self.ce_loss(scores_ab, labels)
        loss = loss + self.ce_loss(scores_ab.transpose(1,2), labels)

        return loss

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
        transform_matrix_1: Float[torch.Tensor, "batch_size 3 3"],
        transform_matrix_2: Float[torch.Tensor, "batch_size 3 3"]
    ):
        B, _, H1, W1 = features_1.shape
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
            transformation_matrix=transform_matrix_1,
            height=H1,
            width=W1,
            scale_factor=self.scale_factor
        )

        transformed_coords_2 = SSLImgProcUtils.transform_coords(
            coords,
            transformation_matrix=transform_matrix_2,
            height=H1,
            width=W1,
            scale_factor=self.scale_factor
        )


        sampled_features_1 = self.sample_grids(transformed_coords_1, features_1)
        sampled_features_2 = self.sample_grids(transformed_coords_2, features_2)


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
        coords: Float[torch.Tensor, "batch_size num_points 1 2"],
        features: Float[torch.Tensor,  "batch_size C H W"]
    )->Float[torch.Tensor, "batch_size num_points C"]:
        grid_sample = F.grid_sample(features, coords, align_corners=False)[:,:,:,0]
        return F.normalize(grid_sample, dim = 1).transpose(1,2)
