from transformers import Trainer
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
            # Explicitly clone and detach outside the graph tracking
            self._image_level_loss_sum = img_loss.detach().item()
            self._patch_level_loss_sum = patch_loss.detach().item()
        else:
            self._image_level_loss_sum = self._image_level_loss_sum + img_loss.detach().item()
            self._patch_level_loss_sum = self._patch_level_loss_sum + patch_loss.detach().item()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        We override this to intercept our custom losses.
        """
        pixel_values = inputs["pixel_values"]
        curr_dtype = pixel_values.dtype
        pixel_values = pixel_values.float()

        view1 = model.transform(pixel_values)
        matrix1 = model.transform.transform_matrix.clone()

        view2 = model.transform(pixel_values)
        matrix2 = model.transform.transform_matrix.clone()

        model_inputs = {
            'pixel_values_1': view1.to(curr_dtype),
            'pixel_values_2': view2.to(curr_dtype),
            'transform_matrix_1': matrix1.to(curr_dtype),
            'transform_matrix_2': matrix2.to(curr_dtype),
        }
        model.update_teacher()
        outputs = model(**model_inputs)
        
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
            logs["image_level_loss"] = self._image_level_loss_sum / self._custom_loss_count
            logs["patch_level_loss"] = self._patch_level_loss_sum / self._custom_loss_count
            
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