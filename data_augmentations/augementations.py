import torch
import kornia.augmentation as K
from model.config.cnnformer_config import CNNFormerConfig
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
        matrix1 = matrix1[:, :2, :]
        matrix2 = matrix2[:, :2, :]
        # 3. Inject into the batch dictionary
        batch["pixel_values_1"] = view1.to(memory_format=torch.channels_last)
        batch["pixel_values_2"] = view2.to(memory_format=torch.channels_last)
        batch["transform_matrix_1"] = matrix1
        batch["transform_matrix_2"] = matrix2

        # Drop the original pixel_values to save VRAM before hitting the model
        del batch["pixel_values"]

        return batch
