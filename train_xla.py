import datasets
import transformers
import evaluate
from transformers import TrainingArguments, Trainer
import torch

from model.transcnn import TransResNet
import torchvision.transforms as transforms
from safetensors.torch import load_file
import numpy as np


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
        transforms.RandAugment(num_ops = 6),
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

    model = TransResNet([2, 2, 2, 2], num_classes=num_labels)
    # state_dict = load_file("./checkpoints_essence_of_imagenet_with_conv_unconv_former_tbs384/model_new.safetensors")
    # model.load_state_dict(state_dict)
    model._keys_to_ignore_on_save = None
    # compiled_model = torch.compile(
    #   model, 
    #   backend="openxla", 
    #   mode="default",    # Change to "max-autotune" for final runs
    #   dynamic=True       # Crucial for dynamic shapes!
    # )

    training_args = TrainingArguments(
      output_dir="./checkpoints_essence_of_imagenet_with_conv_unconv_former_tbs256",
      per_device_train_batch_size=256,
      per_device_eval_batch_size=64,
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
      # torch_compile_backend="openxla",
      # torch_compile_mode="default",
      tf32=False,
      optim="adamw_torch_xla",
      dataloader_num_workers=12
    )
    
    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_ds,
      eval_dataset=val_ds,
      compute_metrics=compute_metrics,
    )

    trainer.train(
      resume_from_checkpoint="./checkpoints_essence_of_imagenet_with_conv_unconv_former_tbs256/checkpoint-42000/"
    )
