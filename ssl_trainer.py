from transformers import Trainer
from data_augmentations.augementations import KorniaGPUDataLoaderWrapper
import torch

class SSLTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize accumulators
        self._image_level_loss_sum = None
        self._patch_level_loss_sum = None
        self._custom_loss_count = 0

    @torch._dynamo.disable
    def _accumulate_custom_metrics(self, img_loss, patch_loss):
        self._custom_loss_count += 1

        if self._image_level_loss_sum is None:
            # Drop the .item() here! Just detach.
            self._image_level_loss_sum = img_loss.detach()
            self._patch_level_loss_sum = patch_loss.detach()
        else:
            self._image_level_loss_sum += img_loss.detach()
            self._patch_level_loss_sum += patch_loss.detach()

    def get_train_dataloader(self):
        """
        Override the default train dataloader to ensure that it is compatible with our custom loss accumulation.
        """
        dataloader = super().get_train_dataloader()
        return KorniaGPUDataLoaderWrapper(dataloader, self.model.config, self.args.device)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        We override this to intercept our custom losses.
        """

        outputs = model(**inputs)
        
        # Extract total loss (required by Trainer for backprop)
        loss = outputs.get("loss")
        
        # Accumulate our custom losses (only during training)
        if model.training:
            # Call our uncompiled helper method
            self._accumulate_custom_metrics(
                outputs.get("image_level_loss"), 
                outputs.get("patch_level_loss")
            )       
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, start_time: float | None = None):
        """
        Intercept the standard logging mechanism to inject our averaged custom losses.
        """
        # If we have accumulated losses, average them and add them to the log dictionary
        if self._custom_loss_count > 0:
            logs["image_level_loss"] = (self._image_level_loss_sum / self._custom_loss_count).item()
            logs["patch_level_loss"] = (self._patch_level_loss_sum / self._custom_loss_count).item()

            # Reset the accumulators for the next logging window
            self._image_level_loss_sum = None
            self._patch_level_loss_sum = None
            self._custom_loss_count = 0
            
        # Call the standard logging method so it gets pushed to WandB, TensorBoard, etc.
        super().log(logs, start_time)


class SSLTrainerXLA(SSLTrainer):
  def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
  
  def _wrap_model(self, model, training=True, **kwargs):
        # Let the native Trainer handle standard wrapper setups (like DDP/FSDP if any)
        model = super()._wrap_model(model, training=training)
        
        # Now that the model and optimizer are fully prepared by accelerate, 
        # compile it specifically for OpenXLA!
        if training:
            print("[INFO] Explicitly compiling model with torch.compile for OpenXLA...")
            model = torch.compile(model, backend="openxla")
            
        return model