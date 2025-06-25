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
import os

from SD3_custom_pipeline import StableDiffusion3Pipeline
import random
import torch
from datetime import datetime

# Parameters
num_prompts = 10  # Change this to however many prompts you want to test
skip_layer_indices = [2, 14]  # Specify which layers to skip here

# Load dataset
dataset_path = os.path.join(os.getcwd(), "prompt_datasets/failure_case_prompts.json")
with open(dataset_path, "r") as f:
    dataset = json.load(f)

# Load pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16)
pipe.to("cuda")

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
        pipe.skip_layers = None
        image_original = pipe(prompt=prompt, num_inference_steps=20).images[0]
        image_original_path = os.path.join(results_dir, f"prompt_{idx}_original.png")
        image_original.save(image_original_path)

        # Modified generation with skip layers
        pipe.skip_layers = skip_layer_indices
        image_modified = pipe(prompt=prompt, num_inference_steps=20).images[0]
        image_modified_path = os.path.join(results_dir, f"prompt_{idx}_modified.png")
        image_modified.save(image_modified_path)

        print(f"Saved: {image_original_path}, {image_modified_path}")