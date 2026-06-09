from transformers import Trainer


class SSLTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize accumulators
        self._image_level_loss_sum = None
        self._patch_level_loss_sum = None
        self._custom_loss_count = 0


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        We override this to intercept our custom losses.
        """
        pixel_values = inputs["pixel_values"]
        pixel_values = pixel_values.float()

        view1 = model.transform(pixel_values)
        matrix1 = model.transform.transform_matrix.clone()

        view2 = model.transform(pixel_values)
        matrix2 = model.transform.transform_matrix.clone()

        model_inputs = {
            'pixel_values_1': view1,
            'pixel_values_2': view2,
            'transform_matrix_1': matrix1,
            'transform_matrix_2': matrix2,
        }
        model.update_teacher()
        outputs = model(**model_inputs)
        
        # Extract total loss (required by Trainer for backprop)
        loss = outputs.get("loss")
        
        # Accumulate our custom losses (only during training)
        if model.training:
            self._custom_loss_count += 1
            img_loss = outputs.get("image_level_loss").detach()
            patch_loss = outputs.get("patch_level_loss").detach()

            if self._image_level_loss_sum is None:
                self._image_level_loss_sum = img_loss
                self._patch_level_loss_sum = patch_loss
            else:
                self._image_level_loss_sum = self._image_level_loss_sum + img_loss
                self._patch_level_loss_sum = self._patch_level_loss_sum + patch_loss
            
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, start_time: float | None = None):
        """
        Intercept the standard logging mechanism to inject our averaged custom losses.
        """
        # If we have accumulated losses, average them and add them to the log dictionary
        if self._custom_loss_count > 0:
            logs["image_level_loss"] = self._image_level_loss_sum.item() / self._custom_loss_count
            logs["patch_level_loss"] = self._patch_level_loss_sum.item() / self._custom_loss_count
            
            # Reset the accumulators for the next logging window
            self._image_level_loss_sum = None
            self._patch_level_loss_sum = None
            self._custom_loss_count = 0
            
        # Call the standard logging method so it gets pushed to WandB, TensorBoard, etc.
        super().log(logs, start_time)