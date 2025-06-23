import os
import json
import random
import torch
import types
from datetime import datetime
from PIL import Image
from diffusers import FluxPipeline

# Configuration
guidance_scale = 5.0
skip_layer_indices = [6, 7, 13, 18]
num_prompts = 10
steps = 20

# Load dataset
dataset_path = os.path.join(os.getcwd(), "failure_case_prompts.json")
with open(dataset_path, "r") as f:
    dataset = json.load(f)

# Load model
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
device = torch.device("cuda")

# Results directory
skip_tag = "_".join(map(str, skip_layer_indices))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(os.getcwd(), f"flux_customcfg_layers_{skip_tag}_g{guidance_scale}_{timestamp}")
os.makedirs(results_dir, exist_ok=True)

# Select prompts
selected_prompts = random.sample(dataset, min(num_prompts, len(dataset)))

with torch.no_grad():
    for idx, prompt in enumerate(selected_prompts):
        print(f"\n[{idx+1}/{len(selected_prompts)}] Prompt: {prompt}")

        # Standard image
        image_standard = pipe(prompt=prompt, num_inference_steps=2).images[0]
        image_standard.save(os.path.join(results_dir, f"prompt_{idx}_standard.png"))

        # Custom CFG with layer skipping
        image_guided = run_custom_cfg(pipe, prompt, skip_layer_indices, guidance_scale=guidance_scale, num_inference_steps=2)
        image_guided.save(os.path.join(results_dir, f"prompt_{idx}_customcfg.png"))

        print(f"Saved standard + guided images for prompt {idx}")