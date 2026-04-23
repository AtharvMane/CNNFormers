import datasets
from transformers import TrainingArguments, Trainer

from model.cnnformer_resnet import CNNFormerResNetForPixelLevelRepresentationModeling
from model.config.cnnformer_config import CNNFormerConfig
import torchvision.transforms as transforms



# function to apply transforms
def apply_transforms(examples, transform):
    examples['pixel_values'] = [transform(image.convert('RGB')) for image in examples['image']]
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

    train_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Apply preprocessing
    train_ds = ds['train'].with_transform(lambda x: apply_transforms(x, train_transforms))
    val_ds = ds['validation'].with_transform(lambda x: apply_transforms(x, val_transforms))

    config = CNNFormerConfig(
        depths=[2,2,2,2],
        hidden_sizes = [64, 128, 256, 512],
        hidden_act = "silu",
        layer_type = "basic",
        num_loss_stages = 2,
        attention_patch_size=8,
        attention_embed_dim=384,
        upscaler_kernel_size=5,
        dropout=0.3,
        dims_per_multi_attention_head=64,
    )
    model = CNNFormerResNetForPixelLevelRepresentationModeling(config=config)
  
    training_args = TrainingArguments(
      output_dir="./checkpoints_cnn_former_ssl",
      per_device_train_batch_size=40,
      per_device_eval_batch_size=40,
      eval_strategy="no",
      do_eval=False,
      save_strategy="steps",
      save_steps=1000,
      save_total_limit=3,
      report_to="wandb",
      num_train_epochs=300,
      learning_rate=1e-4,
      load_best_model_at_end=False,      # Load the best model at the end
      logging_dir='./logs',
      logging_steps=50,
      warmup_steps=100,
      run_name="cnnformer_ssl",
      weight_decay=1e-4,
      remove_unused_columns=False,
      bf16=True,
      tf32=False,
      optim="adamw_torch"
    )
    
    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_ds,
      eval_dataset=val_ds,
    )

    trainer.train(
      # resume_from_checkpoint="./checkpoints_essence_of_imagenet_with_conv_unconv_former_tbs128/checkpoint-4000/"
    )
