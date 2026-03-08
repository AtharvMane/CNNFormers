import torch
import os
import json
from safetensors.torch import load_file, save_file

def convert_to_bf16(checkpoint_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Convert config.json (Update the dtype metadata if it exists)
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        config["torch_dtype"] = "bfloat16"
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        print("Updated config.json")

    # 2. Convert Model Weights (Safetensors or PyTorch Bin)
    # Check for safetensors first (Hugging Face default)
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    
    if os.path.exists(safetensors_path):
        print("Converting model.safetensors...")
        state_dict = load_file(safetensors_path)
        for k, v in state_dict.items():
            if v.is_floating_point():
                state_dict[k] = v.to(torch.bfloat16)
        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
        
    elif os.path.exists(bin_path):
        print("Converting pytorch_model.bin...")
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        for k, v in state_dict.items():
            if v.is_floating_point():
                state_dict[k] = v.to(torch.bfloat16)
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    # 3. Convert Optimizer States
    optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
    if os.path.exists(optimizer_path):
        print("Converting optimizer.pt...")
        # weights_only=False is required for optimizer states as they contain non-tensor Python objects
        opt_dict = torch.load(optimizer_path, map_location="cpu", weights_only=False)
        
        # The optimizer state is usually a nested dictionary under the 'state' key
        if 'state' in opt_dict:
            for param_id, state in opt_dict['state'].items():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.is_floating_point():
                        state[k] = v.to(torch.bfloat16)
        
        torch.save(opt_dict, os.path.join(output_dir, "optimizer.pt"))

    # 4. Copy any remaining essential trainer files (scheduler, rng state, etc.)
    for file in os.listdir(checkpoint_dir):
        if file not in ["config.json", "model.safetensors", "pytorch_model.bin", "optimizer.pt"]:
            src = os.path.join(checkpoint_dir, file)
            dst = os.path.join(output_dir, file)
            if os.path.isfile(src):
                # Just read and write binary for other generic files
                with open(src, 'rb') as f_src, open(dst, 'wb') as f_dst:
                    f_dst.write(f_src.read())
                print(f"Copied {file}")

    print("Conversion complete!")

# --- RUN THE CONVERSION ---
# Replace these paths with your actual directories
input_checkpoint = "./checkpoints_essence_of_imagenet_with_conv_unconv_former_tbs128/checkpoint-60000" 
output_checkpoint = "./checkpoints_essence_of_imagenet_with_conv_unconv_former_tbs128/checkpoint-60000-bf16"

convert_to_bf16(input_checkpoint, output_checkpoint)
