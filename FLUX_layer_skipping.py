"""
FLUX[dev] - Skip Layer Guidance Exploration
See https://huggingface.co/stabilityai/stable-diffusion-3-medium

This script generates images for pairs of prompts that differ only 
in text being present in the image. We then compare intermediate 
outputs of the model to extract information about which layers 
correspond most to text being present in the image.
"""

import gc
import json 
import inspect
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from diffusers import FluxPipeline
import random
import inspect
import torch
import numpy as np
import types
import torch.nn.functional as F
from PIL import Image
from datetime import datetime

def implement_skip_layer_guidance(pipe, text_prompt, skip_layers, device="cuda"):
    print(f"Skip layers: {skip_layers}")
    original_forwards = {}

    for layer_idx in skip_layers:
        block = pipe.transformer.transformer_blocks[layer_idx]
        original_forwards[layer_idx] = block.forward

        def create_skip_forward(orig_idx):
            def skip_forward(self, hidden_states, encoder_hidden_states=None, *args, **kwargs):
                return encoder_hidden_states, hidden_states
            return skip_forward

        block.forward = types.MethodType(create_skip_forward(layer_idx), block)

    print(f"Generate image with skipped layers {skip_layers}...")
    modified_image = pipe(
        prompt=text_prompt,
        num_inference_steps=20,
    ).images[0]

    for layer_idx, forward_func in original_forwards.items():
        pipe.transformer.transformer_blocks[layer_idx].forward = forward_func

    return modified_image

# Parameters
num_prompts = 10  # Change this to however many prompts you want to test
skip_layer_indices = [6,7,13,18]  # Specify which layers to skip here

# Load dataset
dataset_path = os.path.join(os.getcwd(), "failure_case_prompts.json")
with open(dataset_path, "r") as f:
    dataset = json.load(f)

# Load pipeline
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(os.getcwd(), f"flux_skip_results_{timestamp}")
os.makedirs(results_dir, exist_ok=True)

# Select prompts
selected_prompts = random.sample(dataset, min(num_prompts, len(dataset)))

# Process prompts
with torch.no_grad():
    for idx, prompt in enumerate(selected_prompts):
        print(f"\nProcessing prompt {idx+1}/{len(selected_prompts)}: {prompt}")

        # Unmodified generation
        image_original = pipe(prompt=prompt, num_inference_steps=20).images[0]
        image_original_path = os.path.join(results_dir, f"prompt_{idx}_original.png")
        image_original.save(image_original_path)

        # Modified generation with skip layers
        image_modified = implement_skip_layer_guidance(pipe, prompt, skip_layer_indices)
        image_modified_path = os.path.join(results_dir, f"prompt_{idx}_modified.png")
        image_modified.save(image_modified_path)

        print(f"Saved: {image_original_path}, {image_modified_path}")