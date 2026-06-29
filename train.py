import os
import glob
import shutil
# ── Torch compile cache ────────────────────────────────────────────────────────
# Must be set BEFORE torch is imported so CUDA and Inductor pick them up.
_COMPILE_CACHE = "/content/torch_compile_cache_a100"
os.environ["TORCHINDUCTOR_CACHE_DIR"]    = f"{_COMPILE_CACHE}/inductor"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"   # persist compiled FX graphs
os.environ["TRITON_CACHE_DIR"]           = f"{_COMPILE_CACHE}/triton"  # persist autotuned kernel configs
# ──────────────────────────────────────────────────────────────────────────────

import torch
import torch._inductor.config as inductor_config
inductor_config.fx_graph_cache = True   # belt-and-suspenders alongside the env var

import datasets
from transformers import TrainingArguments
import torchvision.transforms as transforms

from model.config.cnnformer_config import CNNFormerConfig
from ssl_trainer import SSLTrainer
from model.cnnformer_resnet import CNNFormerResNetForPixelLevelRepresentationModeling
from callbacks.update_teacher_callback import TeacherEMACallback

torch.set_float32_matmul_precision('high')

# function to apply transforms
def apply_transforms(examples, transform):
    examples['pixel_values'] = [transform(image.convert('RGB')) for image in examples['image']]
    examples['labels'] = examples.pop('label')
    examples.pop('features', None)
    examples.pop('image', None)
    return examples



if __name__=="__main__":
	# Load the "320px" subset of Imagenette
    ds = datasets.load_dataset("benjamin-paine/imagenet-1k-256x256", cache_dir = "imagenet-1k-256x256")

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
    model = CNNFormerResNetForPixelLevelRepresentationModeling.from_pretrained("./checkpoints_cnn_former_ssl_corrected_momentum_a100/checkpoint-81649/")
    # model = CNNFormerResNetForPixelLevelRepresentationModeling(config=config)
  
    training_args = TrainingArguments(
      output_dir="./checkpoints_cnn_former_ssl_corrected_momentum_a100",
      per_device_train_batch_size=200,
      per_device_eval_batch_size=200,
      eval_strategy="no",
      do_eval=False,
      save_strategy="steps",
      save_steps=5000,
      report_to="wandb",
      num_train_epochs=300,
      learning_rate=1e-4,
      load_best_model_at_end=False,      # Load the best model at the end
      logging_dir='./logs',
      logging_steps=16,
      warmup_steps=500,
      run_name="cnnformer_ssl",
      weight_decay=1e-4,
      remove_unused_columns=False,
      tf32=False,
      bf16=True,
      optim="adamw_torch",
      lr_scheduler_type="cosine",
      torch_compile=True,
      torch_compile_backend="inductor",
      torch_compile_mode="max-autotune",
      dataloader_num_workers=12,
      dataloader_pin_memory=True,
      dataloader_drop_last=True,         # CUDA graphs require a static batch size;
                                         # without this the last partial batch triggers a recompile
    )
    
    trainer = SSLTrainer(
      model=model,
      args=training_args,
      train_dataset=train_ds,
      eval_dataset=val_ds,
      callbacks=[TeacherEMACallback()]
    )

    try:
      trainer.train(
        resume_from_checkpoint="./checkpoints_cnn_former_ssl_corrected_momentum_a100/checkpoint-81649"
      )
    except KeyboardInterrupt:
      print("\n[!] Training manually interrupted. Initiating emergency save and upload...")
      
      # 1. Force the trainer to save a full checkpoint (model + optimizer + scheduler + state)
      # We use the internal method _save_checkpoint to ensure it formats exactly like a standard step checkpoint
      trainer._save_checkpoint(model=trainer.model, trial=None)
