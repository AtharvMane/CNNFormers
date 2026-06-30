import torch
import kornia.augmentation as K
from model.config.cnnformer_config import CNNFormerConfig
import torch.nn.functional as F
from typing import Tuple
from jaxtyping import jaxtyped, Float, Bool
from beartype import beartype as typechecker

class SSLImgProcUtils:
    @staticmethod
    @jaxtyped(typechecker=typechecker)
    def get_mask(
        transformed_coords: Float[torch.Tensor, "batch_size num_points 2"]
    )->Bool[torch.Tensor, "batch_size num_points"]:
        mask1 = (transformed_coords[:,:,0].abs()<1)&(transformed_coords[:,:,1].abs()<1)
        return ~mask1
    
    @jaxtyped(typechecker=typechecker)
    @staticmethod
    def get_coords_with_masks_and_labels(
        matrix1: Float[torch.Tensor, "batch_size 3 3"],
        matrix2: Float[torch.Tensor, "batch_size 3 3"],
        batch_size: int,
        scale: int,
        num_points: int
    )->Tuple[
        Float[torch.Tensor, "batch_size num_points 1 2"],
        Float[torch.Tensor, "batch_size num_points 1 2"],
        Bool[torch.Tensor, "batch_size num_points 1"],
        Bool[torch.Tensor, "batch_size num_points 1"]
    ]:
        selector_weights = torch.rand(batch_size, scale*scale, device=matrix1.device)
        view1_coords = F.affine_grid(matrix1[:,:2], (batch_size, 1, scale, scale)).flatten(1,2)
        view2_coords = F.affine_grid(matrix2[:,:2], (batch_size, 1, scale, scale)).flatten(1,2)
        mask1 = SSLImgProcUtils.get_mask(view1_coords)
        mask2 = SSLImgProcUtils.get_mask(view2_coords)

        selector_weights.masked_fill_(mask1, -1)
        selector_weights.masked_fill_(mask2, -1)
        selector_weights.masked_fill_(mask1&mask2, -2)

        _, indices = selector_weights.topk(num_points, dim=-1)
        batch_indices = torch.arange(batch_size, device=matrix1.device).unsqueeze(-1)

        view1_coords = view1_coords[batch_indices, indices].unsqueeze(2)
        view2_coords = view2_coords[batch_indices, indices].unsqueeze(2)
        mask1 = mask1[batch_indices, indices].unsqueeze(2)
        mask2 = mask2[batch_indices, indices].unsqueeze(2)

        return view1_coords, view2_coords, mask1, mask2
        


class KorniaGPUTransform:
    def __init__(
            self,
            config: CNNFormerConfig,
            device: torch.device
        ):
        self.device = device
        
        # Build the sequence directly from your serialized config
        # and immediately map it to the target GPU/XLA device!
        self.transform = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=config.horizontal_flip_probability),
            # K.RandomRotation(config.random_rotation_max_angle_degrees),
            K.RandomResizedCrop(
                size=config.random_resize_crop_size,
                scale=config.random_resize_crop_scale,
                p=config.random_resize_crop_probability
            ),
            K.ColorJitter(
                brightness=config.color_jitter_brightness,
                contrast=config.color_jitter_contrast,
                saturation=config.color_jitter_saturation,
                hue=config.color_jitter_hue,
                p=config.color_jitter_probability
            ),
            K.RandomGrayscale(
                p=config.random_grayscale_probability,
                rgb_weights=torch.tensor(config.normalize_mean)
            ),
            K.RandomGaussianBlur(
                kernel_size=config.gaussian_blur_kernel_size,
                sigma=config.gaussian_blur_sigma,
                p=config.gaussian_blur_probability
            ),
            K.RandomSolarize(p=config.solarize_probability),
            K.Normalize(mean=config.normalize_mean, std=config.normalize_std),
            data_keys=['image']
        ).to(self.device) 

        H, W = config.random_resize_crop_size
        self.un_normalizer = torch.tensor(
            [
                [(W-1)/2, 0.0, (W-1)/2],
                [0, (H-1)/2, (H-1)/2],
                [0,      0,       1]
            ],
            device=self.device
        )
        self.noralizer = torch.tensor(
            [
                [2/(W-1), 0.0, -1],
                [0, 2/(H-1), -1],
                [0,      0,  1]
            ],
            device=self.device
        )

        scales = [H//2,H//4] if config.downsample_in_first_stage else [H//2,H//2]
        for i in range(len(config.depths)-1):
            scales.append(H//2**(i+2))
        
        self.scales = scales[-config.num_loss_stages:]
        self.num_points = config.num_loss_points

    def __call__(self, batch):
        pixel_values = batch["pixel_values"].to(self.device, dtype=torch.float32)

        # 2. Generate Views on GPU (No autograd needed for augmentations)
        with torch.no_grad():
            view1 = self.transform(pixel_values)
            matrix1 = self.transform.transform_matrix.clone()

            view2 = self.transform(pixel_values)
            matrix2 = self.transform.transform_matrix.clone()

        matrix1 = self.noralizer @ matrix1 @ self.un_normalizer
        matrix2 = self.noralizer @ matrix2 @ self.un_normalizer

        views1_coords = []
        masks1 = []
        views2_coords = []
        masks2 = []

        B, C, H, W = view1.shape
        for idx, scale in enumerate(self.scales):
            view1_coords, view2_coords, mask1, mask2 = SSLImgProcUtils.get_coords_with_masks_and_labels(
                matrix1,
                matrix2,
                B,
                scale,
                self.num_points*(idx+1)
            )

            views1_coords.append(view1_coords)
            views2_coords.append(view2_coords)
            masks1.append(mask1)
            masks2.append(mask2)            


        batch["pixel_values_1"] = view1#.to(memory_format=torch.channels_last)
        batch["pixel_values_2"] = view2#.to(memory_format=torch.channels_last)
        batch["pixel_values_1_coords"] = views1_coords
        batch["pixel_values_2_coords"] = views2_coords
        batch["pixel_values_1_masks"] = masks1
        batch["pixel_values_2_masks"] = masks2
        # Drop the original pixel_values to save VRAM before hitting the model
        del batch["pixel_values"]

        return batch
