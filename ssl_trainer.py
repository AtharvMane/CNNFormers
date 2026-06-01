from transformers import Trainer


class SSLTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize accumulators
        self._image_level_loss_sum = 0.0
        self._patch_level_loss_sum = 0.0
        self._custom_loss_count = 0

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
            # Use .detach().item() to prevent memory leaks!
            self._image_level_loss_sum += outputs.get("image_level_loss").detach().item()
            self._patch_level_loss_sum += outputs.get("patch_level_loss").detach().item()
            self._custom_loss_count += 1
            
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
            self._image_level_loss_sum = 0.0
            self._patch_level_loss_sum = 0.0
            self._custom_loss_count = 0
            
        # Call the standard logging method so it gets pushed to WandB, TensorBoard, etc.
        super().log(logs, start_time)