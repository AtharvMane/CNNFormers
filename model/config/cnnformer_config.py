from transformers import ResNetConfig
import torch

class CNNFormerConfig(ResNetConfig):
    def __init__(
        self,
        # Model Arch Hyperparams
        ## ResNet Params
        depths: list[int] | tuple[int, ...] = (3, 4, 6, 3),
        hidden_sizes: list[int] | tuple[int, ...] = (256, 512, 1024, 2048),

        ## PatchUnpatchMHSA Params
        attention_patch_size: int = 8,
        attention_embed_dim: int = 384,
        upscaler_kernel_size: int = 4,
        dropout: float = 30,
        dims_per_multi_attention_head: int = 64,
        attention_is_dmsa: bool | None = None,

        #Backbone Params
        output_indices: list[int] | None = None,
        output_features: list[str] | None = None,

        # Self Supervised Training Model/update Hyperparams
        num_loss_stages: int = 3, #Tells what layers to compare for dense SSL Loss
        dense_ssl_projection_dim: int = 384,
        teacher_training_lambda: int = 0.99,

        # Self Supervised Training Augmentation Params
        horizontal_flip_probability: float = 0.5,

        random_rotation_max_angle_degrees: float | None = 180,

        random_resize_crop_size: tuple[int, int] = (256, 256),
        random_resize_crop_scale: tuple[int, int] = (0.08, 1.0),
        random_resize_crop_probability: float = 0.9,

        color_jitter_brightness: float | tuple[float, float] | list[float] = 0.4,
        color_jitter_contrast: float | tuple[float, float] | list[float] = 0.4,
        color_jitter_saturation: float | tuple[float, float] | list[float] = 0.4,
        color_jitter_hue: float | tuple[float, float] | list[float] = 0.1,
        color_jitter_probability: float = 0.75,

        random_grayscale_probability: float = 0.1,

        gaussian_blur_kernel_size: float | tuple[float, float] | list[float] = (23, 23),
        gaussian_blur_sigma: float | tuple[float, float] | list[float] = (0.1, 2.0),
        gaussian_blur_probability: float = 1.0,

        solarize_probability: float = 0.1,

        normalize_mean: float | tuple[float, float, float] | list[float] = (0.485, 0.456, 0.406),
        normalize_std:float | tuple[float, float, float] | list[float]= (0.229, 0.224, 0.225),

        # SSL Loss Params
        loss_temperature: float = 0.07,
        **kwargs
    ):
        assert len(hidden_sizes)==len(depths), "Recieved unequal number of depths and hidden_sizes. Specify a hidden size for reach element in depth"
        super().__init__(depths=depths, hidden_sizes=hidden_sizes, **kwargs)
        self.attention_patch_size = attention_patch_size
        self.attention_embed_dim = attention_embed_dim
        self.upscaler_kernel_size = upscaler_kernel_size
        self.dropout = dropout
        self.dims_per_multi_attention_head = dims_per_multi_attention_head
        self.attention_is_dmsa = attention_is_dmsa

        self.stage_names = ['stem_out'] + [f"layer_{i}_out" for i in range(1, len(depths)+1)]
        self.set_output_features_output_indices(out_features=output_features, out_indices=output_indices)
        self.num_loss_stages = num_loss_stages
        self.dense_ssl_projection_dim = dense_ssl_projection_dim
        self.teacher_training_lambda = teacher_training_lambda

        self.horizontal_flip_probability = horizontal_flip_probability

        self.random_rotation_max_angle_degrees =  random_rotation_max_angle_degrees

        self.random_resize_crop_size =  random_resize_crop_size
        self.random_resize_crop_scale =  random_resize_crop_scale
        self.random_resize_crop_probability =  random_resize_crop_probability

        self.color_jitter_brightness = color_jitter_brightness
        self.color_jitter_contrast = color_jitter_contrast
        self.color_jitter_saturation = color_jitter_saturation
        self.color_jitter_probability = color_jitter_probability
        self.color_jitter_hue = color_jitter_hue

        self.random_grayscale_probability =  random_grayscale_probability

        self.gaussian_blur_kernel_size = gaussian_blur_kernel_size
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.gaussian_blur_probability = gaussian_blur_probability

        self.solarize_probability =  solarize_probability
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        self.loss_temperature = loss_temperature