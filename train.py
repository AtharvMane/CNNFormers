import datasets
import transformers
import evaluate
from transformers import TrainingArguments, Trainer
import torch

from model.cnnformer_resnet import CNNFormerResNet
from model.config.cnnformer_config import CNNFormerConfig
import torchvision.transforms as transforms
from safetensors.torch import load_file
import numpy as np

# import os
# os.environ["TORCHINDUCTOR_CACHE_DIR"] = "./torch_compile_cache"

# Metrics
accuracy_metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    """Called by the Trainer to compute metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)



# function to apply transforms
def apply_transforms(examples, transform):
    examples['pixel_values'] = [train_transforms(image.convert('RGB')) for image in examples['image']]
    examples['labels'] = examples.pop('label')
    del examples['features']
    del examples["image"]
    return examples



if __name__=="__main__":
	# Load the "320px" subset of Imagenette
    ds = datasets.load_dataset("data/essence_of_imagenet_512")

    # Get label info
    labels_list = ds['train'].features['label'].names
    num_labels = len(labels_list)
    # print(f"Found {num_labels} labels: {labels_list}")

    train_transforms = transforms.Compose([
        transforms.RandAugment(num_ops = 4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply preprocessing
    train_ds = ds['train'].with_transform(lambda x: apply_transforms(x, train_transforms))
    val_ds = ds['validation'].with_transform(lambda x: apply_transforms(x, val_transforms))

    config = CNNFormerConfig(
        depths=[2,2,2,2,2],
        hidden_sizes = [64, 128, 256, 512, 1024],
        hidden_act = "silu",
        attention_embed_dim=384,
        upscaler_kernel_size=5,
        dropout=0.3,
        dims_per_multi_attention_head=64,
        output_features=['stem_out', 'layer_1_out', 'layer_3_out', 'layer_5_out']
    )
    model = CNNFormerResNet(config=config)
    # state_dict = load_file("./checkpoints_essence_of_imagenet_with_conv_unconv_former_tbs384/model_new.safetensors")
    # model.load_state_dict(state_dict)
  
    training_args = TrainingArguments(
      output_dir="./checkpoints_essence_of_imagenet_with_conv_unconv_former_tbs128",
      per_device_train_batch_size=2,
      per_device_eval_batch_size=2,
      eval_strategy="epoch",            # Run evaluation every epoch
      save_strategy="epoch",            # Save checkpoint every epoch
      report_to="wandb",
      num_train_epochs=300,             # Total number of epochs (use more for real training)
      learning_rate=1e-4,
      load_best_model_at_end=True,      # Load the best model at the end
      metric_for_best_model="accuracy", # Use accuracy to find the best model
      logging_dir='./logs',
      logging_steps=50,
      warmup_steps=100,
      run_name="cnnformer_bigger_batch_Run_essence_of_imagenet_with_dropout",
      weight_decay=1e-4,
      remove_unused_columns=False,
      bf16=True,
      # torch_compile=True,
      # torch_compile_backend="inductor",
      # torch_compile_mode="reduce-overhead",
      tf32=False,
      optim="adamw_torch"
    )
    
    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_ds,
      eval_dataset=val_ds,
      compute_metrics=compute_metrics,
    )

    trainer.train(
      # resume_from_checkpoint="./checkpoints_essence_of_imagenet_with_conv_unconv_former_tbs128/checkpoint-4000/"
    )
